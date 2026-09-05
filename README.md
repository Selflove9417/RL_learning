# 足式与轮足机器人强化学习仿真平台 (Isaac Gym + Legged Gym + RSL-RL)

本项目工作空间整合了基于 **NVIDIA Isaac Gym**、**ETH Zurich Legged Gym** 以及 **RSL-RL** 的机器人强化学习端到端训练框架，支持四足机器人、双足机器人及轮足机器人的大规模 GPU 并行仿真与运动控制策略训练。

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
│           └── teleop.py   # 实时键盘交互遥控系统 (WASD)
├── rsl_rl/         # 高性能纯 GPU 强化学习算法库 (PPO 等)
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
# 自动加载最新检查点并在 3D 窗口中回放
python legged_gym/legged_gym/scripts/play.py --task=myrobot
```

### 3. 🎮 实时键盘交互遥控 (Teleoperation)
运行专属遥控脚本，可以在 3D 仿真窗口中通过键盘实时驾驶机器人：
```bash
python legged_gym/legged_gym/scripts/teleop.py --task=myrobot
```

#### 按键操作映射：
- **`W` / `↑`**：前进加速（每次 `+0.25 m/s`，最高 `1.5 m/s`）
- **`S` / `↓`**：减速 / 倒车行驶（每次 `-0.25 m/s`，最低 `-1.0 m/s`）
- **`A` / `←`**：向左旋转掉头（每次 `+0.50 rad/s`）
- **`D` / `→`**：向右旋转掉头（每次 `-0.50 rad/s`）
- **`Space`（空格键）**：**紧急刹车 / 驻车静止自平衡（速度清零）**
- **`R` 键**：重置机器人回到初始原点
- **`F` 键**：第三人称视角跟踪锁定当前机器人

---

## 自定义双足轮足机器人架构特点 (`MyRobot`)

- **混合驱动控制**：
  - 4 个腿部转动关节（髋/膝）采用刚性 PD 位置控制，维持机身站立高度（$0.38\,\text{m}$）与左右横滚水平。
  - 2 个驱动轮采用高响应速度闭环驱动，提供充沛平衡与行驶动力。
- **物理空间对称映射**：针对 URDF 左右关节轴异号（$+X$ 与 $-X$），在控制器底层实现物理空间镜像统一，左右腿天然同向伸缩、双轮同向平稳滚动。
- **工业级 EMA 动作低通滤波**：过滤 50Hz 网络高频动作锯齿毛刺，彻底消除机械腿抽搐与震颤，实现如丝般顺滑的平稳巡航。
