# 足式与轮足机器人强化学习仿真平台 (Isaac Gym + Legged Gym + RSL-RL)

本项目工作空间整合了基于 **NVIDIA Isaac Gym**、**ETH Zurich Legged Gym** 以及 **RSL-RL** 的机器人强化学习端到端训练框架，支持四足机器人、双足机器人及轮足机器人的大规模 GPU 并行仿真与运动控制策略训练。

> 📖 **MyRobot 初学者使用手册与开发指南**：  
> 如果您是第一次使用本项目，想要快速启动仿真、使用键盘遥控机器人、体验 Phase 15 闭环驻车基准模型 (`model_130`) 或启动新训练，请直接查阅面向初学者的完整手册：👉 **[README_MYROBOT.md](README_MYROBOT.md)** （包含命令速查表、快速启动、全流程图与常见问题解答）。

---

## 目录结构概览

```text
IsaacTu/
├── isaacgym/       # NVIDIA Isaac Gym 底层 GPU 并行物理仿真引擎
├── legged_gym/     # 足式/轮足机器人环境定义、任务配置与训练/演示入口
│   └── legged_gym/
│       ├── envs/myrobot/   # 自定义双足轮足机器人环境与配置
│       └── scripts/
│           ├── train.py    # 强化学习训练入口
│           ├── play.py     # 策略回放与性能评测
│           └── teleop.py   # 实时键盘交互遥控系统 (WASD + P/C/E/Z)
├── rsl_rl/         # 高性能纯 GPU 强化学习算法库 (PPO 等)
├── scripts/        # 自动化评测与验证脚本集 (21工况、驻车PI验证等)
└── README.md       # 工作空间目录与架构说明文档
```

---

## 核心架构与协同关系

三大核心组件相互配合，构成了高效的 GPU 闭环训练系统：

```mermaid
flowchart LR
    A["isaacgym<br/>(底层 GPU 并行物理引擎)"] -->|"物理状态 (关节/速度/接触)"| B["legged_gym<br/>(环境/任务/奖励/观测)"]
    B -->|"Observation / Reward"| C["rsl_rl<br/>(PPO 策略与价值网络)"]
    C -->|"Action (关节位置/力矩/轮速)"| B
    B -->|"驱动指令"| A
```

1. **`isaacgym`**：负责在单张 GPU 上并行模拟成千上万个机器人刚体与关节动力学，提供张量级别的状态读取与力矩/位置控制接口。
2. **`legged_gym`**：负责机器人任务建模，包括环境定义、观测空间设计、奖励函数计算、域随机化（Domain Randomization）以及复杂地形生成。
3. **`rsl_rl`**：负责强化学习算法的更新，收集 GPU 显存中的 Rollout 数据，执行 PPO 损失计算与网络反向传播。

---

## 常用操作与运行指南

### 1. 启动双足轮足机器人训练
```bash
# 启动 512 并发环境训练
python legged_gym/legged_gym/scripts/train.py --task=myrobot --num_envs=512

# 在现有模型检查点基础上继续微调 (--resume)
python legged_gym/legged_gym/scripts/train.py --task=myrobot --resume --num_envs=512
```

### 2. 策略演示与评估回放
```bash
# 自动加载最新检查点并在 3D 窗口中回放 (Phase 15 model_130)
python legged_gym/legged_gym/scripts/play.py --task=myrobot --load_run Oct14_00-28-55_phase15_height_034_060_exploration --checkpoint 130
```

### 3. 🎮 实时键盘交互遥控 (Teleoperation)
运行专属遥控脚本，可以在 3D 仿真窗口中通过键盘实时驾驶机器人（默认加载 Phase 15 固化基准）：
```bash
python legged_gym/legged_gym/scripts/teleop.py --task=myrobot --model_path evaluation_assets/phase15/models/phase15_m130.pt
```

#### 按键操作映射：
- **`W` / `↑`**：前进加速（单次 `+0.04 m/s`，平滑上限 `+0.50 m/s`）
- **`S` / `↓`**：减速 / 倒车行驶（单次 `-0.04 m/s`，平滑下限 `-0.30 m/s`）
- **`A` / `←`**：向左旋转掉头
- **`D` / `→`**：向右旋转掉头
- **`C` / `J`**：**机身下蹲**（安全下限 $0.34\text{ m}$，避免物理干涉）
- **`E` / `U`**：**机身升高**（安全上限 $0.52\text{ m}$）
- **`Z` / `N`**：**一键恢复标称站立高度 ($0.398\text{ m}$)**
- **`P` 键**：**热切换驻车速度闭环 PI 补偿**（HUD 实时显示 `[ON]` / `[OFF]`，默认开启，抑制驻车慢速漂移）
- **`Space`（空格键）**：**紧急刹车 / 驻车静止自平衡（速度清零）**
- **`R` 键**：重置机器人回到初始原点（平滑指令归零，避免冲击）
- **`F` 键**：第三人称视角跟踪锁定当前机器人

---

## 自定义双足轮足机器人架构特点 (`MyRobot`)

- **混合驱动控制**：
  - 4 个腿部转动关节（髋/膝）采用刚性 PD 位置控制，维持机身站立高度（$0.34\text{m} \sim 0.52\text{m}$）与左右横滚水平。
  - 2 个驱动轮采用高响应力矩前馈 + CACM 接触感知降额 + 闭环偏航补偿驱动，提供充沛平衡与行驶动力。
- **物理空间对称映射**：针对 URDF 左右关节轴异号（$+X$ 与 $-X$），在控制器底层实现物理空间镜像统一，左右腿天然同向伸缩、双轮同向平稳滚动。
- **外环驻车速度闭环 PI 补偿器**：在零速驻车阶段实时补偿底层策略的网络动力学偏置，消除定向后退爬行，20秒驻车位移抑制率达 89%～95%，同时保持零力矩激化与 100% 连续任务通过率。
- **工业级 EMA 动作低通滤波**：过滤 50Hz 网络高频动作锯齿毛刺，彻底消除机械腿抽搐与震颤，实现平稳巡航。
