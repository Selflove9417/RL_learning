# 跨障任务核心修复与物理重评测 —— 技术验收报告

- 仓库：`Selflove9417/RL_learning`
- 分支：`feat/dynamic-obstacle-clearing-jump`
- 基线提交（本轮开始前）：`d245fef980e2cf7fd0b40ada1b924e24fb6940c4`
- 工作目录：`/home/xy/IsaacTu_jump`
- 评测对象：冻结策略 `legged_gym/logs/myrobot_running_jump/Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt`
- 约束遵守情况：**未启动任何 PPO 训练**；**未改动 URDF / 执行器物理限幅 / 冻结模型权重**；**未覆盖任何历史基准 JSON**（旧脚本保留原样，仅加 `SUPERSEDED` 说明，结果写入全新文件名）。

> **本报告分两个阶段。**
> 第 0–8 节是 **Phase A**（基线 `d245fef` → `2eccf1c`）：建立可信判据与评测器，结论「0/8 成功」。
> **第 9 节是 Phase B**（基线 `2eccf1c` → `1253960`）：按后续任务书修复五项训练语义缺陷。
> Phase B 修正了 Phase A 若干数字的口径（采样动作 → 确定性动作、`lambda` 单位、几何误差界），
> **两者不得混算均值**；以第 9 节为准，第 0–8 节保留作为历史记录。

---

## 0. 结论先行

**跨障训练准入：不通过（NOT ADMITTED）。**

理由不是「本轮代码仍不可信」，而是**目标模型自身能力不足**，且这一结论在修正后的判据下更硬：

| 指标 | 无障碍（free-run，n=8） | 有障碍（n=8） |
|---|---|---|
| 完整成功 | 0/8（无跨障成功定义） | **0/8** |
| 障碍碰撞 | — | **8/8** |
| 干净跨越（`cleared`） | — | **0/8** |
| 真正离地并落地 | 8/8 | **2/8** |
| 平均飞行距离 | 0.1465 m (n=8) | 0.1916 m (n=2) |
| 跨障所需前向位移 | — | 0.2284 m (n=6) |
| 平均缺口 | — | **+0.039 m**（seed 20 缺 6.5 cm，seed 27 缺 1.3 cm） |
| 恢复速度达 70% 门槛 | **0/8** | **0/8** |

连续多障碍（8 种子 × 600 步 = 87 个 cycle）：碰撞 79、`cleared` 9、**`cleared_and_clean` 0**、跌倒 0、episode reset 79 次且**原因全为 `OBSTACLE_COLLISION`**。

因此本轮交付的是一套**可信的判据与评测器**，它给出的裁决是：该冻结策略不具备跨障能力。若用它作为 PPO 热启动，会被一个「跳跃高度足够（峰值净空 21.7 cm）但前向位移只有 9 cm」的行为带偏；本轮修复正是为了让下一轮 PPO 的奖励、触发与终止不再奖励这种错误行为。

---

## 1. 最小必要源码 diff

`git diff --stat`（相对 `d245fef`）：

```
 legged_gym/legged_gym/envs/myrobot/myrobot_dual_env.py          |  84 +-
 legged_gym/legged_gym/envs/myrobot/myrobot_obstacle_jump_env.py | 898 +++++++++++++++++----
 legged_gym/legged_gym/envs/myrobot/myrobot_running_jump_env.py  |  47 ++
 legged_gym/legged_gym/scripts/play_obstacle_jump.py             |  18 +-
 tests/benchmark_obstacle_jump.py                                |  17 +
 tests/paired_obstacle_comparison.py                             |  18 +
 6 files changed, 936 insertions(+), 146 deletions(-)
```

新增文件（未跟踪）：

```
 legged_gym/legged_gym/envs/myrobot/obstacle_geometry.py   547 行   URDF 几何包络模块
 tests/evaluate_obstacle_jump.py                           581 行   单回合评测器（含相位标记）
 tests/evaluate_obstacle_jump_continuous.py                503 行   连续多障碍评测器
 tests/analyze_first_contact.py                            351 行   首次接触归因取证
 tests/test_obstacle_jump_semantics.py                     404 行   反例单测套件（29 项）
```

### 1.1 任务一：奖励符号与事件时序

`myrobot_obstacle_jump_env.py`

- **`_reward_obstacle_collision()`**（双重负号）：原 `return -(collision_pen + force_pen)` 配 `cfg.rewards.scales.obstacle_collision = -5.0` ⇒ 乘积为正 ⇒ **碰撞反而得奖励**。现改为 `return collision_pen + force_pen`（正值 ≥ 0），符号完全由 config scale 拥有。docstring 写明该契约。
  实测：干净 0、软碰 `-10.0`、硬碰 `-15.0`，0→300 N 扫描单调不增 `-5.00 -7.50 -10.00 -12.50 -15.00 -15.00 -15.00`。
- **`_reward_obstacle_cleared_success()`**（重复发放）：原 `return where(self.obstacle_cleared, 1, 0)`，而 `obstacle_cleared` 是锁存 bool ⇒ 整局剩余每步 +5.0。现改为 `return self.clear_success_event.to(rew_buf.dtype)`，只在边沿事件发一次。
  实测：仅锁存无事件 = 0.0；边沿 = 1.0；下一步 = 0.0。
- **指标时序**：原 `compute_obstacle_metrics()` 在 `super().step()` **之后**调用，而 `super().step()` 内部已完成 `check_termination → compute_reward → reset_idx → compute_observations`，即奖励/终止用的是上一步的障碍指标。现改为钩子：在 `myrobot_running_jump_env.step()` 的终止判定**之前**插入 `self._update_task_metrics()`，序号校验 `hook(3738) < termination(4704) < reward(5170)`。
- **终止判据**：`myrobot_running_jump_env.step()` 新增 `self.fall_buf = fall.clone()`（区分物理跌倒与超时/任务中止），并 `fall = fall | self._task_failure_mask()`。障碍 env 覆写 `_task_failure_mask() = has_obstacle_collision | task_failed`。失败原因分两类：`OBSTACLE_COLLISION`、`CROSSED_WITHOUT_LIFTOFF`（后包络越过盒后缘却从未离地）。
- **重置清零**：原 `if "episode" in self.extras:` 守卫依赖基类 `reset_idx` 创建该键，而本类覆盖了 `reset_idx` 且不调 `super()` ⇒ **首轮永不清零**。新增 `_finalize_episode_stats(env_ids)` 自带 `extras['episode']` 创建与 `episode_sums` 清零。单测断言清零后 `max=0`。
- **PPO 三不变量未破坏**（任务一第 6 点）：`tests/regression_test_world_rewards_and_loss_mask.py` 在 `--task myrobot_running_jump` 与 `--task myrobot_obstacle_jump` 上**结果完全一致**，全 PASS：

```
[PASS] Actor Loss Mask Correctness: Balance 78 步全 0.0 / Jump+Recover 42 步全 1.0
[PASS] World-Frame Velocity Rewards at Liftoff Window (Steps 44~46)
[PASS] RolloutStorage Actor Mask Storage & Retrieval Validated.
[PASS] mini_batch_generator yields actor_masks_batch with shape: torch.Size([20, 1])
ALL PRE-TRAINING REGRESSION TESTS PASSED SUCCESSFULLY!
```

### 1.2 任务二：真实空间触发与碰撞包络

新建 `obstacle_geometry.py`，文件头写明**认知契约**：两类量永不混淆。

- **EXACT**：两个轮子的解析圆柱/胶囊（URDF `radius=0.070, length=0.0885`，轴沿局部 X，局部 X 偏置 `link_004=+0.03825001` / `link_007=+0.01925`），以及障碍解析盒（identity 姿态，Y 区间精确）。
- **CONSERVATIVE OUTER BOUND**：其余 5 个 link 全是三角 mesh（`base_link` 269 个、`link_002/003/005/006` 共 20 个）。Isaac Gym **没有** per-shape AABB 或 shape-pair 查询接口（已实测枚举 gymapi 符号，`RigidShapeP` 只有 `compliance/contact_offset/filter/friction/rest_offset/restitution/rolling_friction/thickness/torsion_friction`，无 `type`/`radius`）。故用 URDF 顶点云的支撑点集做外包围，**仅用于触发/包含裕度，绝不当穿透深度上报**。

核心 API：`urdf_link_vertex_cloud`（含二进制 STL 解析）、`wheel_forward_interval`、`wheel_bottom_z`、`wheel_box_signed_gap`（capsule↔AABB 有符号距离，段上 65 点采样，间距 1.4 mm）、`wheel_clearance_above_box_top`（footprint 受限的精确垂直净空）、`link_y_extents(env, z_ceiling)`、`robot_forward_interval(env, z_ceiling)`、`obstacle_y_interval`。

**两个必须报告的几何事实：**

1. **历史触发公式低估了 0.2213 m**（不只此前估计的 9–11 cm）。`root_states[:,1] + wheel_radius` 漏掉了轮心 +0.09~0.114 m 的纵向偏置**和**腿部前伸。单测断言最小低估量 `0.1119 m`，标称站姿实测 `front_y - (base_y + R) = 0.2213 m`。
2. **z-ceiling 是必需的**。未过滤时 reset 瞬间前缘 = `0.2895`（`base_link` 的 mesh 顶点，位于障碍顶面 0.05 m **以上**），落在障碍区间 `[0.28, 0.32]` 内 ⇒ **第 1 步就误触发**。过滤后 `front_y = 0.1872`（`link_004`）、`dist = 0.0928`，正好在 `spatial_trigger_dist = 0.09` 窗口外缘。单测：无 ceiling `0.2847` → 有 ceiling `0.1834`。

**speculative contact 辨伪（关键修正）**。Isaac Gym 的接触列表是「推测性近邻」而非「撞击」列表：碰撞体进入 20 mm `contact_offset` 就会产生记录。实测一个真实擦身而过（解析 gap **+9.885 mm**）产生了：

```
body0=7(obstacle) body1=3(wheel) lambda=0.0  minDist=20.000mm  initialOverlap=True
```

原代码只要看到 `body0|body1 == num_bodies` 就判碰撞 ⇒ 干净通过被扣满 `OBSTACLE_COLLISION`。现改为三条独立证据源：`contact_pair`（`lambda > 1e-3` N·s）、`analytic_geometry`（capsule/box 精确相交，不依赖接触列表存在）、`force_attribution`（障碍净力 > 2 N，**无法给出接触对，不得当接触对上报告**）。

**净空口径修正**。原 `wheel_bottom_min_z - obstacle_h` 把「轮最低点」与「盒顶面」两个**不同位置**的量相减，导致站在盒子旁边的机器人报 `-0.0477 m`（声称在障碍内部），而解析 gap 是 `+0.0979 m`。新增 `wheel_clearance_above_box_top`：对轴线采样点求「球面可达 footprint 的最低 z」，仅在 `d <= R` 时有效。6 例验证全 PASS：

```
base_y base_z | gap_L    gap_R    | new_clr      | OLD      | verdict
+0.000 +0.400 | +0.0979 +0.0979 | +inf         | -0.0477  | OK 标称站姿
+0.130 +0.400 | -0.0274 -0.0274 | -0.0375      | -0.0477  | OK 抵达盒前缘
+0.230 +0.455 | +0.0108 +0.0108 | +0.0114      | +0.0073  | OK 轮刚过顶面
+0.300 +0.560 | +0.1349 +0.1349 | +inf         | +0.1123  | OK 飞越 +10cm
```

单测另断言「analytic gap 与 footprint clearance 在相交时符号一致」：`gap=-0.02736 clr=-0.03755`。

### 1.3 任务三：并行环境障碍坐标

- `self.obstacle_py`（Python float 共享标量）→ `self.obstacle_py_t`（每环境张量）。
- `obstacle_py` 保留为**只读 property**（报 env0 偏移），使 `tests/benchmark_obstacle_jump.py`、`tests/paired_obstacle_comparison.py`、`tests/verify_obstacle_jump_continuous.py`、`tests/diagnose_contact_and_penetration.py`、`play_obstacle_jump.py` 等历史消费者不破。
- `_reset_root_states` 原 `self.obstacle_py = obs_py` 会抹掉其它 env 的重定位 ⇒ 改 `self.obstacle_py_t[env_ids] = obs_py`。
- 连续模式重定位原逐 env 循环却写同一个共享标量 ⇒ 新 `_reposition_obstacles(env_mask)` 全矢量化，只写 `obs_actor_ids = idxs*2 + 1`。
- 所有障碍几何量改为从 `obstacle_root_states` 逐环境读取（`obstacle_y_interval`），不再依赖任何共享标量。

多环境回归（单测 E 组，全 PASS）：目标 env delta `0.3000`、其它 env 仍 `0.3000`、其它 env 障碍 actor 位姿未动、reset 恢复 cfg 值 `0.3000`、`obstacle_y_interval` 跟随各 env 自身坐标、**静态障碍真实参与 PhysX 碰撞（下落峰值 10808 N）**。

### 1.4 任务四：评测与恢复判据

`myrobot_dual_env.py`：

- 导入补 `quat_rotate`（原只有 `get_euler_xyz`）。新增 `_world_forward_velocity()`。
- **RECOVER 出口三事件改为互斥**。原代码 `if normal_handoff or timeout_handoff or hard_timeout_abort:` 后用 `elif` 链写事件名，`TIMEOUT_HANDOFF` 与 `RECOVER_ABORT` 会被 `normal_handoff` 掩盖。现改为每个判据自带 `and not <prior>`。
- **70% 速度门槛**：`RECOVERY_SPEED_FRACTION = 0.70`；`ratio = v_fwd / cmd_vy`；`recovery_sustained = (good_steps >= 2) and (regression_steps <= 1)`；`normal_handoff` 现在额外要求 `recovery_sustained`。原 `vxy_calm < 1.20` 完全没有速度门槛。

  **运行时证据（16 个 run 的 `last_transition_event` 分布）**：

  ```
  TIMEOUT_HANDOFF          8     ← 全部 8 个 free-run 恢复都是超时交还，无一是 NORMAL_HANDOFF
  LIFTOFF                  4
  TOUCHDOWN_ENTER_RECOVER  2
  ENTER_RUNNING_JUMP       2
  NORMAL_HANDOFF           0     ← 70% 门槛生效：没有任何恢复被判为正常交还
  RECOVER_ABORT            0
  ```

  这正是判据生效的直接证据：在旧代码下这 8 次恢复都会走 `normal_handoff`（因为 `settle_stable_steps >= 3` 就够），事件名会被误报为 `NORMAL_HANDOFF`，即「恢复成功」。现在它们诚实地标为 `TIMEOUT_HANDOFF`——**恢复是超时兜底，不是恢复成功**。
- `reset_idx` 清零 5 个 recovery 缓冲。
- **每次真实碰撞周期独立峰值**：`_begin_new_obstacle_cycle` 在每次触发时清零 `peak_obstacle_contact_force`，不复用历史峰值。
- **reset 后状态陈旧根治**（本轮最隐蔽的 bug）：`reset_idx` 用 `set_actor_root_state_tensor_indexed` 写入新位姿，但 **Isaac Gym 不会自动刷新 state 张量**，故 reset 之后的 `_update_task_metrics(post_reset=True)` 读到上一回合末位姿（`front_y≈0.2825` 在障碍区间内、轮底 0、盒顶 0.05 ⇒ `clearance = -0.05`、`overlap_below_top=True` 且 `has_taken_off=False` ⇒ 幻影 `CROSSED_WITHOUT_LIFTOFF`）。修法：`reset_idx` 末尾补 `refresh_actor_root_state_tensor` / `refresh_rigid_body_state_tensor` / `refresh_dof_state_tensor`。修复前后对比 `minclr=-0.0500 reason=CROSSED_WITHOUT_LIFTOFF` → `minclr=1.0000 reason=NONE`。
- **pre-reset 快照**（`_capture_episode_snapshot`，快照字典 38 个字段，其中 5 个终止帧几何量由 `compute_obstacle_metrics` 的 `_last_geometry` 提供）：在 `reset_idx` 之前冻结上一回合的全部判据。没有它，「at first contact」读到的是新回合的干净站姿（`y=0.0 gap=0.0979`）—— 历史脚本的 `touchdown_pos_y = 0.0`、`flight_dist = -0.106 m` 正是此因。

### 1.5 任务五：评测器

三个新评测器（`tests/evaluate_obstacle_jump.py`、`evaluate_obstacle_jump_continuous.py`、`analyze_first_contact.py`），均按「真几何 + pre-reset 快照 + 正证据碰撞 + 70% 门槛 + 真无障碍对照臂」设计。无障碍臂把盒子移到 `y=60`、关闭触发，**但仍在 worker 内复现同一 spatial 判据打到虚拟平面**，使两臂只差「盒子是否物理存在」。

---

## 2. 运行命令

```bash
cd /home/xy/IsaacTu_jump
source ./activate_jump.sh          # 必须：清除被 AppImage 污染的 PYTHONHOME
```

```bash
# 反例单测（29 项）
python -u tests/test_obstacle_jump_semantics.py --pipeline cpu

# PPO 训练契约回归（必须全 PASS）
python -u tests/regression_test_world_rewards_and_loss_mask.py \
    --task myrobot_obstacle_jump --headless

# 单回合评测：0.35 m/s × 8 种子 × 有/无障碍
python -u tests/evaluate_obstacle_jump.py --pipeline cpu

# 连续多障碍：8 种子 × 600 步
python -u tests/evaluate_obstacle_jump_continuous.py --pipeline cpu

# 首次接触刚体 + 世界位姿归因
python -u tests/analyze_first_contact.py --pipeline cpu
```

**为什么必须 `--pipeline cpu`**：`use_gpu_pipeline=True` 时 `gym.get_env_rigid_contacts()` 被 Carb 插件拒绝——
`Function GymGetEnvRigidContactCount cannot be used with the GPU pipeline after simulation starts. Please use the tensor API if possible.`
CPU pipeline 下 `physx.use_gpu` 仍为 True（GPU 仍做物理），只是接触对可读。**GPU pipeline 下没有任何 shape-pair 查询接口**，此时只能拿到障碍侧净力 + 各刚体力，`first_contact_source` 会落为 `force_attribution`，报告中不得称其为接触对。

---

## 3. 测试结果与原始 JSON

三个 JSON 均以最终代码产出，放在 `evaluation_assets/`，全部为**新文件名**：

| 文件 | 内容 |
|---|---|
| `obstacle_jump_reeval_results.json` (960 KB) | 单回合 8 种子 × 2 臂，含逐 step 轨迹与相位标记 |
| `obstacle_jump_continuous_reeval_results.json` (129 KB) | 连续 8 种子 × 600 步，87 个 cycle 明细 |
| `first_contact_attribution.json` (162 KB) | 8 种子的首次接触刚体、世界位姿、证据源、时序 |

### 3.1 反例单测：29/29 PASS

```
A. collision penalty sign     干净 0 / 软碰 -10.000 / 硬碰 -15.000 / 扫描单调不增
B. success edge event         仅锁存 0 / 边沿 1.000 / 下一步 0
C. timing                     hook@3738 < termination@4704 < reward@5170
D. termination                collision 进 mask / ploughing→CROSSED_WITHOUT_LIFTOFF /
                              proximity 不算失败 / clean clearance 不算失败 / episode_sums 清零
F. geometry                   真实前缘超前 base_y+R 至少 0.1119 m / z-ceiling 0.2847→0.1834 /
                              标称距离 0.0964 > 窗 0.090 / clearance 与 gap 符号一致
E. per-env obstacle           delta 0.3000 / 其它 env 0.3000 / actor 未动 / reset 恢复 /
                              区间跟随 / 静态障碍物理峰值 10808 N
PASSED 29 / 29   RESULT: PASS
```

### 3.2 单回合评测：8 种子 × 2 臂

```
obstacle   n=8  success=0  collision=8  cleared=0  took_off=6  touched_down=2
           flight=0.192 m (n=2)  required=0.228 m (n=6)  shortfall=0.039 m (n=2)
           min_clearance=+0.000 m  peak_force=283.5 N  recovery=4% (0/8 sustained)
           reasons={'OBSTACLE_COLLISION': 8}
free_run   n=8  success=0  collision=0  cleared=0  took_off=8  touched_down=8
           flight=0.146 m (n=8)  required=n/a  shortfall=n/a
           min_clearance=+1.000 m  peak_force=0 N  recovery=-31% (0/8 sustained)
           reasons={'NONE': 8}
```

逐种子（有障臂）：

```
seed trigger liftoff first_contact pass touchdown flight_m required_m shortfall_m min_clr_m peak_N
  11     34      42         59        -        -     n/a     0.2048      n/a    -0.0001   204.2
  13     33      41         60        -        -     n/a     0.25094     n/a    +0.0002   168.5
  15     33      41         61        -        -     n/a     0.22341     n/a    +0.0004   307.9
  20     33      40         62        -       61  0.15664   0.22205   0.06541   +0.0007    96.2
  25     33      -1         38        -        -     n/a       n/a        n/a   -0.0003   459.4
  27     33      40         66        -       61  0.22662   0.23923   0.01261   -0.0001   651.1
  31     34      -1         40        -        -     n/a       n/a        n/a   -0.0001   100.2
  35     33      41         61        -        -     n/a     0.22970      n/a    -0.0001   280.4
```

逐种子（无障臂）：

```
seed fell  liftoff touchdown flight_m  recovery_ratio  reset_step
  11 True      42        60   0.0818      -1.8868          88
  13 False     41        61   0.13794      0.3477         200
  15 False     41        61   0.14896      0.0859         200
  20 True      40        61   0.15664      0.0041          89
  25 False     40        62   0.10086     -0.3307         200
  27 False     40        61   0.22737     -0.2617         200
  31 False     42        63   0.17551     -0.3232         200
  35 False     41        62   0.14281     -0.1418         200
```

### 3.3 连续多障碍：8 种子 × 600 步

```
seeds=8  cycles=87  took_off=82  collision=79  cleared=9  cleared_and_clean=0
mean_peak_force=391.87 N   mean_recovery_ratio=0.3028 (n=10)
episode_resets=79  falls=0  reasons={'OBSTACLE_COLLISION': 79}
cycle_outcomes={'EPISODE_RESET:OBSTACLE_COLLISION': 79, 'STEP_LIMIT': 8}
```

逐种子：

```
seed cycles collision cleared cleared_and_clean took_off mean_peak_N max_peak_N
  11    11       10       0                 0       10       204.46    1013.11
  13    11       10       0                 0       10        69.40     323.07
  15    11       10       4                 0       10       136.83     448.18
  20    11       10       0                 0       10        72.67     100.51
  25    10        9       5                 0       10       234.71     882.03
  27    11       10       0                 0       11       795.61     912.00
  31    11       10       0                 0       10       820.01     987.25
  35    11       10       0                 0       11       786.99     921.40
```

**`cleared` 与 `collision` 可以并存，且已逐帧证实为物理真实**（不是指标 bug）。seed 15 铁证：

```
step 118  cleared 0->1 | base_y=+0.2787 front=+0.4815 rear=+0.3340 box_rear=+0.320 rmargin=+0.0140
step 120  RESET reason=OBSTACLE_COLLISION coll=True cleared=True
碰撞帧接触对: b0=7 b1=3 lambda=403.18 minDist=0.02 normal=[0,-0.959,-0.284]
解析 gap = [-6e-06, +0.007965]   ← 左轮真实相交 6 µm
```

即：先干净越过（后包络超过盒后缘 14 mm），两帧后轮子擦到盒**后上角**。因此在汇总里 `cleared_and_clean = cleared and not collision` 是唯一权威口径。

### 3.4 首次接触归因（任务四专项）

```
runs=8  with_contact=8  contact_before_liftoff=2  liftoff=6
first-contact bodies : {'link_007': 4, 'link_004': 4}
evidence sources     : {'contact_pair': 8}
collision=8  cleared=0
```

```
seed  contact_step  body       world_pos                    force_N  before_liftoff  timing
  11       59       link_007   [ 0.01563, 0.32000, 0.05]     204.2   False          mid-air
  13       60       link_004   [ 0.43529, 0.32000, 0.05]     168.5   False          mid-air
  15       61       link_004   [ 0.44616, 0.32000, 0.05]      49.2   False          mid-air
  20       62       link_007   [-0.04321, 0.32000, 0.05]      96.2   False          POST-touchdown
  25       38       link_007   [ 0.04934, 0.28000, 0.05]     459.4   True           NO-LIFTOFF
  27       66       link_004   [ 0.35626, 0.28000, 0.05]     545.5   False          POST-touchdown
  31       40       link_007   [-0.03339, 0.28000, 0.05]     100.2   True           NO-LIFTOFF
  35       61       link_004   [ 0.38778, 0.29128, 0.05]       0.3   False          mid-air
```

世界坐标严格落在盒棱上：`y = 0.32`（后缘）或 `0.28`（前缘）、`z = 0.05`（顶面），全部 `contact_pair` 证据（CPU pipeline + `lambda > 1e-3` 冲量门）。

> **对历史结论的更正（任务四明确要求核查）**：`paired_obstacle_comparison_results.json` 声称「8 个种子的首次障碍接触均早于真实离地」。在修正几何下**该结论不成立**：实际为 **2/8**（seed 25 @step 38、seed 31 @step 40，二者都是从未锁存离地就撞上）。其余 6 个为 4 个空中碰撞 + 2 个落地后碰撞。历史那个 8/8 是旧几何（少算 0.22 m 前缘）+ 旧触发窗口的产物，不是同一判据下的可比数据。旧脚本已加 `SUPERSEDED` 头说明此事。

**接触对 API 的诚实边界**：Isaac Gym 提供 `gym.get_env_rigid_contacts()`，返回 `RigidContact` 结构化数组，含 `body0/body1/env0/env1/normal/lambda/minDist/localPos0/localPos1` 等字段，**但不提供 shape 索引对**。`localPos0` 是 `body0` 局部坐标系内的点 —— 第一版直接把它当世界坐标上报，导致每次接触的 `y` 恒为 `±0.02`（盒子半宽的本地值），看起来像世界坐标其实不是。已修为 `p_world = p_obstacle + R_obstacle @ localPos`。

### 3.5 任务五.3 相位对比（首次接触前 / 触发时 / 真实离地时 / 通过时 / 落地后）

由 `phase_markers` 提供，全部取自 pre-reset 轨迹（终止帧的几何量由 episode snapshot 修复，否则会显示新回合的干净站姿）。`common_step` 是 env 自己的时钟（与所有事件步戳一致），**不等于循环下标**。

```
seed 11: trigger=34 liftoff=42 first_contact=59
  at_trigger            cs=34  BALANCE  y=+0.0418 z=0.4066 front=+0.1917 gap=+0.0896  dist_to_box=+0.0883
  at_liftoff            cs=42  JUMP     y=+0.0894 z=0.5864 front=+0.2969 gap=+0.0315  dist_to_box=-0.0169
  before_first_contact  cs=58  JUMP     y=+0.1614 z=0.4264 front=+0.1614 gap=+0.0177  clr=+0.0177
  at_first_contact      cs=59  RESET    y=+0.1644 z=0.3899 front=+0.6649 gap=-0.0001  clr=-0.0001

seed 25: trigger=33 liftoff=-1 first_contact=38   ← 从未离地即撞
  at_trigger            cs=33  BALANCE  y=+0.0436 z=0.3997 front=+0.1962 gap=+0.0851
  before_first_contact  cs=37  JUMP     y=+0.0761 z=0.4241 front=+0.2685 gap=+0.0140
  at_first_contact      cs=38  RESET    y=+0.0800 z=0.4653 front=+0.2830 gap=-0.0001

seed 27: trigger=33 liftoff=40 touchdown=61 first_contact=66   ← 落地后才撞
  at_liftoff            cs=40  JUMP     y=+0.0903 z=0.5797 front=+0.2860 gap=+0.0018
  at_touchdown          cs=61  JUMP     y=+0.3170 z=0.3825 front=+0.5129 gap=+0.0030
  at_first_contact      cs=66  RESET    y=+0.3525 z=0.2511 front=+0.5498 gap=-0.00003
```

---

## 4. 失败原因汇总

### 4.1 冻结模型的物理行为（本轮最核心的发现）

峰值飞行净空 **21.7 cm**（`max_flight_clearance = 0.2174`），远高于 5 cm 障碍。**高度从来不是问题**。问题是水平位移：

- base_y 从触发时的 `0.104` 只前进到落地/碰撞时的 `0.195~0.204`，**约 9 cm**；
- 要跨越，后包络必须越过盒后缘 0.32，所需根节点位移 **0.205–0.251 m**；
- 实测飞行距离 **0.147–0.227 m**，缺口 **1.3–6.5 cm**（seed 20 缺 6.5 cm、seed 27 缺 1.3 cm）。

即该策略学到的是**近乎原地垂直起落**。这与历史 `paired_obstacle_comparison_results.json` 的「飞行距 8.7 cm / 所需 25.47 cm / 缺口 12 cm」方向一致，但历史数字因几何错误而量级偏大；修正后的缺口更小（1.3–6.5 cm），结论（0% 成功）不变。

> **⚠ 上方「近乎原地垂直起落」的诊断已被 §9.8 推翻，保留原文以记录 Phase A 当时的判读。**
> Phase B 改用确定性动作（`act_inference`）后测得空中前向速度 **0.361–0.405 m/s（均值 0.385），高于
> 0.35 m/s 指令**，滞空 21–22 步、位移 0.152–0.178 m。本节的「约 9 cm」是采样动作下单个种子的观测，
> 被误推成了普遍结论；真实缺陷是**滞空比所需少约 5.7 步**，不是"没有向前飞"。
> 同样的读法也适用于本节「0.147–0.227 m / 缺口 1.3–6.5 cm」这一行 —— 它是采样模式（n 很小）的数字。

### 4.2 分类失败清单

| 失败模式 | 单回合（n=8） | 连续（n=87 cycles） |
|---|---|---|
| `OBSTACLE_COLLISION` | 8 | 79 |
| 2 个为离地前撞（seed 25、31） | — | — |
| 4 个为空中撞（11、13、15、35） | — | — |
| 2 个为落地后撞（20、27） | — | — |
| `CROSSED_WITHOUT_LIFTOFF` | 0 | 0 |
| 物理跌倒（free-run） | 2（seed 11、20） | 0 |
| 恢复速度未达 70% | 8/8 | 有障 0/10 sustained |

### 4.3 恢复速度：必须并列报告两个口径

- 本报告口径（`touchdown .. touchdown+40` 全窗）：有障臂 **+4.0%**、无障臂 **−31.3%**，`recovery_70pct = 0/8`。
- 旧脚本口径（`td+15 .. td+40`）：`recovery_vw = 0.1983`、`vel_retention_pct = 56.65`。

**差异根因是窗口选择偏差，不是度量错误**。冻结 model_3.pt 触地后**前 15 步**整体后移（Phase A 采样动作、seed 11 free-run：`s60 y=+0.1732` → `s79 y=-0.0248`，即 **19.8 cm**），旧窗口恰好跳过这段后滑期，只采到接近零的一段，所以看起来「恢复了 56%」。两个数字必须并列披露，不能只报一个。

> **⚠ 量级更正（Phase B 复算后）**：本节是 **Phase A（采样动作）**的单种子观测，不能当作普遍量级。
> 用已提交的两份 JSON 的 `traces` 逐帧差分（`dt=0.02`，窗口 `td..td+40`）复算落地后最大后滑：
> **Phase A = 0–27.03 cm，均值 5.77 cm（4/8 种子为 0）**；**Phase B（确定性）= 1.16–3.70 cm，均值 2.57 cm**。
> 另：原文引的 `s86 y=-0.0964` 有误，实际 `s86 y=-0.0607`。详见 §9.8.1 —— 确定性模式下后滑只有 1.2–3.7 cm，
> 且约 0.9 s 后策略自行恢复巡航，**因此「后滑」不是跨障失败的主因，滞空不足才是**。

### 4.4 已知的指标口径注意事项（供验收核对）

- `mean_flight_m` 的分母只有落地的 2 个 run，`mean_required_m` 的分母是起跳的 6 个 run。并排列出会让人读出「单次 run 从未出现过的差值」，因此 JSON 里每个均值都配 `n_*_measured`。
- 无障臂 `min_clearance = +1.000` 是哨兵值（盒子在 60 m 外，净空无意义），不是测量值。
- `pass_step` 在 8 个种子里全为 `-1`：**没有任何一个 run 让后包络越过盒后缘**，与 `cleared = 0/8` 一致。

---

## 5. 无障碍 vs 有障碍对照

| 指标 | 无障碍 | 有障碍 | 差异解读 |
|---|---|---|---|
| 起跳 | 8/8 | 6/8 | 2 个在离地前就撞上 |
| 落地 | 8/8 | 2/8 | 6 个在空中被障碍终止 |
| 飞行距离 | 0.1465 m (n=8) | 0.1916 m (n=2) | 有障臂均值偏高是因为只有 2 个「幸运」样本落地 |
| 障碍碰撞 | 0 | 8 | 障碍物理真实参与 |
| 峰值接触力 | 0 N | 283.5 N（单回合）/ 391.9 N（连续） | 静态盒在重压下可达 10808 N |
| 恢复达 70% | 0/8 | 0/8 | 无障臂也是 0，说明是策略问题不是障碍问题 |
| 跌倒 | 2/8 | 0/8 | 有障臂更早被终止，来不及跌倒 |

**关键对照结论**：恢复速度在**无障碍时也是 0/8 达标**（均值 −31%）。这说明恢复失败不是障碍造成的，而是冻结策略本身的落地后行为缺陷。

---

## 6. 是否满足跨障训练准入

**不满足。**

准入需要同时满足：真几何触发、真实碰撞证据、一次性事件奖励、每环境独立状态、70% 恢复门槛 —— 这些**本轮已全部实现并由 29 项反例单测 + 8 种子实测锁定**。但准入的另一半是**目标策略具备基础跨障能力**，而冻结 `model_3.pt` 在修正后的判据下：

- 单回合 0/8 成功、8/8 碰撞；
- 连续 87 cycles 中 `cleared_and_clean = 0`；
- 只覆盖了所需前向位移的 70–96%（缺 1.3–6.5 cm）；
- 恢复速度 0/8 达标，且无障碍时同样 0/8。

**建议的下一步（不在本轮范围）**：以本轮修复后的奖励与触发为起点重新训练，重点解决两件事 —— (a) 起跳后的滞空时长（当前策略空中速度已达标 0.385 m/s，但滞空比所需少约 5.7 步），(b) 落地瞬态的恢复速度（当前落地后 1.2–3.7 cm 后滑、约 0.9 s 自行恢复巡航，但超出环境 0.40 s 的 RECOVER 窗口）。在此之前不应把 `model_3.pt` 当作已具备跨障能力的热启动基线。

> **本节的诊断已被 §9.8 / §9.8.1 取代（更精确）**：Phase B 用确定性动作重跑后发现，空中前向速度其实**达标**
> （0.385 m/s > 0.35 指令），真正的缺陷是**滞空短约 5.7 步**，而不是"垂直起落"；落地后的后滑只有
> **1.2–3.7 cm**（不是此前的 17 cm），且约 0.9 s 后自行恢复巡航。上面 (a)(b) 的表述按 §9.8 系列为准。

---

## 7. 提交信息

- 基线：`d245fef980e2cf7fd0b40ada1b924e24fb6940c4`
- 分支：`feat/dynamic-obstacle-clearing-jump`
- **代码/测试/数据交付提交 SHA：`2eccf1c303b477068c8562421c1ac9bf88ec5805`**
  该提交内含全部源码改动、4 个新测试脚本、3 个原始 JSON 与本报告初版。
- 其上可能有若干 **仅改本报告文字** 的后续提交（不触碰代码、测试或 JSON），
  因此验收代码时请以 `2eccf1c` 为准，用 `git diff d245fef 2eccf1c` 复核全部实质改动。
- 已推送至 `origin/feat/dynamic-obstacle-clearing-jump`。
- 远端校验命令：`git ls-remote --heads origin feat/dynamic-obstacle-clearing-jump`

---

## 8. 复现注意（踩坑清单）

1. **必须 `source ./activate_jump.sh`**：裸 `python` 不在 PATH；直接调 venv python 会因 `PYTHONHOME=/tmp/.mount_*/usr/` 污染报 `ModuleNotFoundError: No module named 'encodings'`。
2. **必须从 `/home/xy/IsaacTu_jump` 根目录启动**：`myrobot_dual_env.py` 的 `balance_model_path` 是相对路径。
3. **`task_registry.make_env()` 之后必须调 `env.reset()`**：否则 root state 张量仍是 PhysX 创建 actor 时的 `z=0` 位姿（`base_init_state` 的 `z=0.4` 只有 `reset_idx` 才写入），第 0 步就会判 fall。
4. **读 `failure_reason` / `has_obstacle_collision` 必须在 `reset_idx` 之前**：`env.step()` 返回后它们已被清零。正确做法是读 `env.episode_snapshot`，或 monkey-patch `compute_obstacle_metrics` / `_update_task_outcome` 在内部取证（注意后者运行时 latch 已置位，看不到 0→1 翻转）。
5. **每个进程只能建一个 Isaac Gym Foundation**：第二次 `make_env` 会 `Foundation object exists already` 并核心转储。故单测只建一个 3-env 场景供全部用例复用。
6. **`common_step_counter ≠ 循环下标`**：它在 `env.step()` 内递增，比循环下标大 1，且每回合归零。
7. **`env.bilateral_clearance` 是 `(N,)` 而非 `(N,2)`**；`env.wheel_clearance` 才是 `(N,2)`。
8. **`set_actor_root_state_tensor` 必须传 `gymtorch.unwrap_tensor(t.contiguous())`**，普通 torch tensor 会 `TypeError`。
9. **`env.cfg.rewards.scales` 不支持下标**，须 `getattr(...)`。
10. **`gymapi.SimParams()` 默认重力为零**：手写探针必须 `sp.gravity = gymapi.Vec3(0, 0, -9.81)`。
11. **本机 headless only**：无 `/dev/dri`、无 X11，不能用交互 viewer 录屏。

---

# Phase B：五项训练语义修复

- 基线：`2eccf1c303b477068c8562421c1ac9bf88ec5805`
- 交付提交：**`1253960796a89ab6719c20fedfe51e7ca6a54bcd`**（11 文件，+47239 / −135）
- 试验方案提交：`a4c3fe1`（`PPO_TRIAL_PLAN.md`，只描述方案，未训练）
- 约束：**未启动 PPO 训练**；**未改 URDF / 执行器限幅 / 冻结模型**；**未改历史基准 JSON**

## 9.0 Phase B 结论

**判据侧：通过。策略侧：仍然 0/8，但结论比 Phase A 更硬、更可信。**

| 指标 | Phase A（采样动作） | **Phase B（确定性 `act_inference`）** |
|---|---|---|
| 完整成功 | 0/8 | **0/8** |
| 障碍碰撞 | 8/8 | **8/8** |
| 干净跨越 `cleared` | 0/8 | 1/8 |
| 真正离地 | 6/8 | **8/8** |
| 落地 | 2/8 | **2/8** |
| 平均飞行距 | 0.1916 m (n=2) | 0.1744 m (n=2) |
| 跨障所需位移 | 0.2284 m (n=6) | 0.2091 m (n=8) |
| 恢复达 70% | 0/8 | **0/8** |

> **「恢复达 70%」一行必须配合 §9.8.1 阅读**：0/8 不是「策略不会巡航」（系统落地约 0.9 s 后确实以
> 0.33 m/s 恢复巡航并保持），但**也不是「判据窗口太短」** —— 那次恢复发生在 `RECOVER` 0.40 s 超时
> 交接给 `BALANCE` **之后**（seed 11：step 62 触地 → 81 超时 → 108 达速）。`RECOVER` 阶段本身确实失败，
> 所以判据拒绝 `TIMEOUT_HANDOFF` 是对的。三层含义见 §9.8.1，任务语义决策见 §9 第 5 项。

确定性模式让 `took_off` 从 6/8 升到 8/8 —— **采样噪声此前掩盖了真实的离地行为**。这是 Phase B 必须重跑的核心理由。

> **对「平均飞行距」一行的读法提醒**：`0.1744 m (n=2)` 与 `0.2091 m (n=8)` 的**分母不同**（有障臂只有 2/8
> 回合真正落地，其余 6 回空中就撞上）。因此 `0.2091 − 0.1744 = 0.0347 m` 这个差**不是**同一批回合的
> 缺口，只能当方向参考。同一批回合的严格缺口见 §9.8 的逐种子表（用无障臂 8/8 全样本算，
> 平均缺 0.0435 m / 约 5.7 步滞空）。

## 9.1 任务一：成功奖励分档（先通过后碰撞不得提前领奖）

**改动**：`obstacle_cleared_success` 从「几何通过即发满额 5.0」拆成两档。

```python
        pass_clean = (self.has_taken_off & self.has_touched_down
                      & (rear_margin > 0.005)
                      & (~self.has_obstacle_collision) & (~self.obstacle_cleared))
        if pass_clean.any():
            self.obstacle_cleared |= pass_clean
            self.geometric_pass_event |= pass_clean      # TIER 1: 0.6
            self.success_armed |= pass_clean

        recovered_sustained = self._recovery_succeeded()
        mission_success = (self.success_armed
                           & self.obstacle_cleared
                           & (~self.has_obstacle_collision)
                           & (~self.task_failed)
                           & (self.takeoff_step >= 0)
                           & (self.touchdown_step >= 0)
                           & recovered_sustained
                           & (~self.final_success_paid))       # TIER 2: 5.0
```

- **TIER 1**（`obstacle_geometric_pass = 0.6`，约满额 1/8）：只证明几何通过。
- **TIER 2**（`obstacle_cleared_success = 5.0`）：还要求真实离地、落地、持续巡航、`NORMAL_HANDOFF`，每回合至多一次。
- **永久解除**：`self.success_armed &= ~self.has_obstacle_collision`，并在首次碰撞时写入
  `success_disarmed_reason ∈ {'COLLISION_AFTER_PASSAGE', 'COLLISION_BEFORE_PASSAGE'}`。

**反例测试（E 组 11 项全 PASS）**：`pass-then-collide is NOT a success`、`re-arming cannot resurrect a collided episode`、`pass without sustained recovery is NOT a success`、`tier-1 passage pays on its own event tier1=1.000 tier2=0.000`。

## 9.2 任务二：连续恢复判据（`>= 2` 步不能证明持续）

**改动**（`myrobot_dual_env.py`）：从「窗口内累计 2 步」改为**连续 run 计数器**。

```python
        RECOVERY_SPEED_FRACTION = 0.70
        RECOVERY_HOLD_STEPS = 10          # 0.20 s 不间断
        RECOVERY_CURVE_LEN = 50           # 原始速度环形缓冲
        qualifying = (ratio >= 0.70 and att_calm and yaw_calm
                      and joint_calm and grounded)
```

任何一步不达标 → `recovery_hold_steps` 归零。三个出口写入
`recovery_outcome ∈ {'NORMAL_HANDOFF', 'TIMEOUT_HANDOFF', 'RECOVER_ABORT'}`，
`_recovery_succeeded()` 只认 `NORMAL_HANDOFF`，即 **TIMEOUT / ABORT 明确不算成功**。

**原始速度曲线**已进入评测 JSON（`derived.recovery_curve` + `recovery_curve_samples`，来自 env 侧环形缓冲，非评测器重构）。
判据来源标注为 `recovery_verdict_source = 'env._recovery_succeeded'`，旧的均值启发式降级为并列参考项 `recovery_mean_window_heuristic`。

**实测效果**（无障臂 seed 11）：`recovery_outcome = TIMEOUT_HANDOFF`、`best_hold_steps = 0`、`grounded_steps = 20`，
曲线末段持续为负 `[-0.0853, -0.1015, -0.1166, -0.1305]` m/s —— **旧判据会把这类回合全部算作恢复成功。**

> **补充（§9.8.1）**：上面的曲线只有 20 个采样，因为它来自 `RECOVER` 状态（0.40 s 后即退出）。
> 把窗口延长到整个回合后可见系统在**落地后 45–46 步（0.90–0.92 s）** 恢复 ≥70% 指令速度并保持 91–92 步 ——
> 但该恢复发生在 `RECOVER` 超时交接给 `BALANCE` **之后**（seed 11：step 62 触地 → 81 超时 → 108 达速）。
> 所以「`best_hold_steps = 0`」既不是"策略从未站稳"，也不能简单归因于窗口太短：**`RECOVER` 阶段本身确实失败**。
> 判据的真假三层含义见 §9.8.1；判据改动本身（连续 run 计数、只认 `NORMAL_HANDOFF`）依然正确。

## 9.3 任务三：几何包络精度（三项诚实化）

### (a) 零填充幻影点（真实 bug，已修）

`register_link_points` 用 `torch.zeros` 把点云 pad 到 K，旧 `link_y_extents` 只按 body 有效性过滤，
**pad 的零点是活点**，会落回 `rb_states[b, 0:3]`（body 原点）。对 `link_003`/`link_006`（髋部安装、原点在碰撞体外）会静默拉偏 min/max。
修法：新增 `env.link_point_mask`（`(num_bodies, K)` bool），逐点过滤。

### (b) 支撑点削减被证伪 → 改用凸包

96 方向 Fibonacci 采样在 4000 次随机旋转下的**最大低估**实测：

```
base_link 8.896 mm   link_005 4.505 mm   link_002 4.308 mm
link_003 1.730 mm    link_006 1.166 mm   wheels 0 mm
```

`base_link` 低估 8.9 mm 与障碍高度 50 mm 同量级，不可接受。
改用 `scipy.spatial.ConvexHull` 顶点（394452 → **443 点**）：线性泛函在凸包顶点取极值，故**任意旋转精确**。
旧误差值保留为常量 `REDUCTION_MEASURED_UNDERESTIMATE_M` 备查。

### (c) 采样误差界更正（旧声称 0.2 mm，实际 0.691 mm）

轴线 65 点采样步长 `h = 0.0885/64 = 1.383 mm`；距离沿线段凸 ⟹ 采样最小值**高估**真实最小值最多 `h/2 = 0.691 mm`。
旧 docstring 声称 ~0.2 mm，**错 3.5 倍**。

> **这条更正有实质影响**：Phase A 报告的首接触净空（−0.00033 ~ +0.00065 m）**全部低于该分辨率**，
> 因此只能报**符号**，不能报量值。`wheel_clearance_above_box_top` 已显式标注为 NOT "EXACT"。

### 三类反例（H 组，全 PASS）

- **姿态变化**：400 组随机四元数 × 7 link，凸包 vs 未缩减原始点云，`worst_under < 1e-6`；旧 Fibonacci 点集回归锚 `old_under > 1e-3`。
- **障碍边角接近**：左/右角各 6 mm 净空与 4 mm 重叠，解析 gap 与 **4001 点轴线扫描**和**独立表面点云**（17×72×37）三方一致。
- **非车轮连杆**：`link_y_extents` 对全部 7 link 保持**外包围**性质（`violations == 0`），并断言源码含
  `EXACT SUPPORT FUNCTION OVER A CONSERVATIVE OUTER BOUND` 与 `never used as a penetration depth`。

## 9.4 任务四：接触单位与坐标系（一处重大自我更正）

### `lambda` 是**力 N**，不是冲量 N·s

我在 Phase B 中途曾按任务书措辞把 `lambda` 改标为冲量并加「1/dt = 200 倍单位错误」的注释。
**这个判断是错的**，用实验推翻了它（`tests/test_contact_force_units.py`，已固化为回归测试）：

隔离场景：0.2 m 立方体、density 1000 → m = 8.0 kg、`m·g = 78.4800 N`，自由落体至静止后读接触记录：

```
        dt     lambda sum        m*g [N]      lam/(m*g)   lam/(m*g*dt)
  0.005000       78.47963       78.48000       0.999995        199.999
  0.010000       78.47991       78.48000       0.999999        100.000
  0.002500       78.48085       78.48000       1.000011        400.004
spread of lambda across dt (relative) = 1.552e-05
VERDICT: lambda is a FORCE in N
```

`lambda` 与 dt 无关且等于 `m·g` ⟹ **它是接触力，单位牛顿**。旧代码把 `lambda` 命名 `force` 是正确的。
当前代码保留两个字段并澄清语义：`force_n`（该接触对）与 `net_force_n`（刚体净力）。

### 接触点坐标系（真实 bug，已修）

`localPos0` 是 `body0` 局部系的点，第一版直接当世界坐标上报，导致每次接触 `y` 恒为盒半宽 ±0.02。
修法：新增模块级

```python
def contact_point_world(contact, obstacle_root_state, obstacle_is_body0,
                        obstacle_body_index, device=None):
    # 按障碍实际所在 slot 选 localPos0/localPos1
    # p_world = p_obs + R_obs @ local
    # 返回 (None, 'flag_mismatch') / (None, 'obstacle_not_in_pair') 做交叉校验
```

**顺序反转测试（F 组）**：同一物理点，障碍在 `body0` 与在 `body1` 两种排列下解出**同一世界坐标**（< 1e-6），
`point_frame` 分别为 `'obstacle_local0'` / `'obstacle_local1'`；错 flag → `'flag_mismatch'`；无障对 → `'obstacle_not_in_pair'`。

### 首次接触归因（8 种子确定性，全部发生在离地之后）

按 `touchdown_step` 与 `first_contact_step` 的先后再细分：**2 个在真实落地后**擦到盒后上角，
**6 个在空中就撞上**（`touchdown_step = -1`，从未触地）。

```
seed step body      world_pos                    pair_force_N  net_force_N  before_liftoff  phase
11   61   link_004  [0.4377, 0.32, 0.05]          1013.1        1013.1       False           MID_AIR
13   65   link_004  [0.4364, 0.32, 0.05]            68.4         328.8       False           AFTER_TOUCHDOWN
15   61   link_004  [0.4350, 0.32, 0.05]           361.2         361.2       False           MID_AIR
20   61   link_004  [0.4412, 0.32, 0.05]           100.5         100.5       False           MID_AIR
25   70   link_004  [0.4344, 0.32, 0.05]             7.6          22.1       False           AFTER_TOUCHDOWN
27   60   link_004  [0.4395, 0.32, 0.05]           582.3         598.7       False           MID_AIR
31   61   link_004  [0.4375, 0.32, 0.05]           959.6         987.3       False           MID_AIR
35   60   link_004  [0.4393, 0.32, 0.05]           409.4         409.4       False           MID_AIR
```

**与 Phase A 的分歧必须如实披露**：Phase A（采样动作）报「2/8 早于离地」；Phase B（确定性）报 **0/8 早于离地**，
全部 `before_liftoff = False`。两者是不同动作模式的轨迹，不可混算。

### 冲突辨伪：speculative contact ≠ collision

Phase A 遗留的关键缺陷是 `_collect_obstacle_contacts` 把**任何**含障碍的 `RigidContact` 都当碰撞。
实测反例（真实策略 step 59，轮子与盒仍有 **9.885 mm 解析间隙**）：

```
  b0=7 b1=3 lambda=0.000 minDist=0.02000 initialOverlap=True
```

`lambda=0` + `minDist == contact_offset` + `initialOverlap=True` 是 PhysX 的 **speculative（推测性近邻）记录**。
现要求 `lambda > IMPULSE_EPS (1e-3)` 才算接触正证据；近邻记录只驱动 `is_crossing_obstacle`，**不驱动终止与惩罚**。
另有独立几何证据源 `_geometric_wheel_intersection()`（解析胶囊-盒，不依赖接触列表，GPU 下也可用）。

## 9.5 任务五：统一动作模式

`tests/evaluate_obstacle_jump.py` 与 `tests/analyze_first_contact.py` 改为 `policy.act_inference(obs)`（确定性）。
随机 `policy.act(obs)` 保留为独立测试模式（`--action_mode stochastic`）。

- 模式写入 JSON（`action_mode` + `action_mode_note`）**与文件名**（默认 `..._<mode>_results.json`）；
- `--overwrite` 不显式给出时**拒绝覆盖**已存在文件；
- **新旧采样模式不得混合算均值** —— Phase A 的 `obstacle_jump_reeval_results.json` 零改动。

## 9.6 本轮修复的一个 GPU 训练致命缺陷（Phase A 遗留）

`reset_idx` 后 `rb_states` 在 **GPU pipeline 上仍是上一回合末的位姿**（CPU 上会立即发布）。实测 2 env @ `base_y = 0.62`：

```
######## GPU ########
  post_reset: task_failed=[True, True] reason=['CROSSED_WITHOUT_LIFTOFF', 'CROSSED_WITHOUT_LIFTOFF']
              rear_margin=[0.0924, 0.0923]
######## CPU ########
  post_reset: task_failed=[False, False] reason=['NONE', 'NONE']
```

旧代码在 `post_reset=True` 时重新评估几何 ⟹ **每次成功穿过障碍，下一回合会在第一步被判失败并终止**。
修法：`_update_task_metrics(post_reset=True)` 直接 return，不评估任何几何（`reset_idx` 已清零全部缓冲）。
`step()` 的第一次真实调用会在终止与奖励之前重算全部指标。

## 9.7 测试与数据

| 测试 | Phase A | **Phase B** |
|---|---|---|
| `tests/test_obstacle_jump_semantics.py` | 29 项 | **75 项全 PASS** |
| `tests/test_ppo_smoke.py --pipeline gpu` | 无 | **41/41 PASS** |
| `tests/test_ppo_smoke.py --pipeline cpu --rl_device cpu` | 无 | **38/38 PASS** |
| `tests/test_contact_force_units.py` | 无 | **PASS**（3 个 dt） |

**PPO 数据链路 smoke test（不更新权重）**：真实障碍环境、8 env × 48 step 的收缩 rollout（`runner.num_steps_per_env` 与 `alg.init_storage` 同步缩小，`make_alg_runner(..., log_root=None)` 保证不建 run 目录、不存 checkpoint），Actor 35 / Critic 35 / Action 6，
验证 `RolloutStorage` 形状 `(T, N, 1)`、GAE 递推（`lhs = rhs`）、done 步 return == 存储 reward（bootstrap 已折入 reward）、
`actor_loss_mask` 二值进 storage、`time_outs`/`terminal_observation` 生效、奖励含正负号、update 返回 `(value_loss, surrogate_loss)`
与 39 个诊断 key 全有限、梯度有限且参数确实移动。
**GPU pipeline 专项**：确认 `GymGetEnvRigidContactCount` 不可用（预期）时，解析几何仍能独立检出重叠
（`n_env_intersecting=8 deepest gap=-0.07000 m`），**不依赖 CPU-only 接口**。

**8 种子确定性评测**（`evaluation_assets/obstacle_jump_reeval_deterministic_results.json`，新文件名，1.11 MB）：

```
obstacle   n=8  success=0  collision=8  cleared=1  took_off=8  touched_down=2
           mean_flight_m=0.1744 (n=2)  mean_required_m=0.2091 (n=8)  mean_shortfall_m=0.0341 (n=2)
           mean_min_clearance_m=0.0002  mean_peak_force_N=477.6
           mean_recovery_ratio=+0.0602  recovery_70pct=0
           reasons={'OBSTACLE_COLLISION': 8}
free_run   n=8  success=0  collision=0  cleared=0  took_off=8  touched_down=8
           mean_flight_m=0.1656 (n=8)  mean_recovery_ratio=-0.1389  recovery_70pct=0
           reasons={'NONE': 8}
```

逐种子（有障臂）：

```
seed  reason               cleared  min_clr_m  flight_m  required_m  recovery_ratio
 11   OBSTACLE_COLLISION   False    +0.0001    n/a       0.2088      0.000
 13   OBSTACLE_COLLISION   False    +0.0000    0.1703    0.2069      0.286
 15   OBSTACLE_COLLISION   False    +0.0012    n/a       0.2088      0.000
 20   OBSTACLE_COLLISION   False    +0.0003    n/a       0.2099      0.000
 25   OBSTACLE_COLLISION   True     +0.0000    0.1786    0.2101      0.196
 27   OBSTACLE_COLLISION   False    -0.0001    n/a       0.2118      0.000
 31   OBSTACLE_COLLISION   False    +0.0001    n/a       0.2047      0.000
 35   OBSTACLE_COLLISION   False    -0.0000    n/a       0.2114      0.000
```

> `min_clr_m` 的绝对值全部落在 §9.3(c) 的 0.691 mm 分辨率以内 ⟹ **只能读符号，不能读量值**。7/8 为非负、1/8 略负。

## 9.8 冻结模型的真实能力边界（Phase B 最终判读）

**高度够，空中速度够，滞空不够；落地后有一段小后滑但系统约 0.9 s 后自行恢复巡航。**

- 峰值飞行净空 **21.7 cm** vs 障碍 **5 cm** —— 高度从来不是瓶颈。
- `flight_dist` 0.174 m vs `required` 0.209 m（有障臂，n=2/n=8）—— 缺 3.4 cm；
  无障碍臂 0.166 m（n=8），说明这就是该策略的固有飞行距，与障碍无关。
- **无障碍臂 8/8 恢复窗口内的平均前向速度与指令反号**（占指令速度 **−8.4% ~ −18.1%**，均值 **−13.9%**），
  换算成实测位移是**落地后累计后滑 1.16–3.70 cm（均值 2.57 cm）**，窗口内前向速度最低 **−0.097 ~ −0.131 m/s**
  —— 但这是**瞬态**，约 0.9 s 后系统以 0.33 m/s 恢复巡航并保持到回合结束（详见 §9.8.1）；
- **8/8 `TIMEOUT_HANDOFF`、`recovery_best_hold_steps = 0`** —— `RECOVER` 状态内从未连续 10 步维持 70% 速度。
  注意这只是「在 `RECOVER` 内没有」，不等于系统不会巡航：巡航是在 `RECOVER` 超时交接给 `BALANCE` 之后
  才恢复的，三层含义见 §9.8.1；
- 8/8 有障臂的首次接触全部在**离地之后**，其中 **6/8 空中就撞上**（从未触地）——落点不在障碍之后。

> **⚠ 更正（本表前一版把速度比例当成了距离）**：早期版本写「落地后整体后滑 13–18 cm」，那是把
> `recovery_ratio` 的百分比（−13.9% / −18.1%）误抄成了厘米数 —— **量纲错了 4 倍**。用已提交的
> `obstacle_jump_reeval_deterministic_results.json` 的 `traces` 逐帧差分（`dt = 0.02 s`）复算是
> **1.16–3.70 cm**：`seed11 3.70 / seed13 2.22 / seed15 2.01 / seed20 2.96 / seed25 1.16 / seed27 2.84 / seed31 2.98 / seed35 2.68`（cm）。
> 对照组 Phase A（采样动作、`obstacle_jump_reeval_results.json`）同法复算是 **0–27.03 cm（均值 5.77 cm）**，
> 4/8 种子完全为 0。**后滑是真实存在的，但量级远小于此前写的 13–18 cm，且它不是跨障失败的成因**（见下条）。

**结论**：冻结 `model_3.pt` 的**空中前向速度其实是达标的** —— 无障碍臂 8/8 的 `flight_dist / flight_steps / dt`
落在 **0.361–0.405 m/s**，均值 **0.3848 m/s**，**高于**指令的 0.35 m/s。问题出在**滞空时长比所需少约 5.7 步**：

```
seed  dur  flight_m  required_m  short_m  speed_mps  extra_steps_needed
 11    21   0.1516     0.2088     0.0572    0.3610        7.9
 13    22   0.1703     0.2069     0.0367    0.3869        4.7
 15    22   0.1751     0.2088     0.0337    0.3980        4.2
 20    22   0.1710     0.2099     0.0389    0.3886        5.0
 25    22   0.1783     0.2101     0.0319    0.4052        3.9
 27    21   0.1611     0.2118     0.0507    0.3835        6.6
 31    21   0.1547     0.2047     0.0500    0.3684        6.8
 35    21   0.1624     0.2114     0.0490    0.3866        6.3
--------------------------------------------------------------
mean        0.1656     0.2091     0.0435    0.3848        5.7   (range 3.9–7.9)
```

`required_m = 0.320 − takeoff_rear_y`（后包络要越过盒后缘所需的根节点位移），
`extra_steps_needed = short_m / (speed * dt)`。注意有障臂与无障臂在**同一种子下 flight 完全相同**
（如 seed 13 两侧都是 0.1703 m）—— 障碍没有影响起跳与飞行，只在落点处发生作用。

**结论**：冻结 `model_3.pt` 学到的是「前向速度达标、但滞空短约 5.7 步（0.114 s）的跳跃」；落地后还有一段
**1.2–3.7 cm 的后滑瞬态**（不是此前写的 13–18 cm），后包络因此停在 0.263–0.288
（`takeoff_rear_y + flight`）、距 0.320 的盒后缘还差 **3.2–5.7 cm**。
作为 PPO 热启动是有价值的研究起点（空中速度跟踪已具备，只需延长滞空），
但**不能把它当作已具备跨障能力**。试验方案见 `PPO_TRIAL_PLAN.md`，其中 §4 的两项奖励改动正是针对这两个缺口
（主项是直指滞空的 `obstacle_rear_clearance_progress`，后滑项因只是 1.2–3.7 cm 的瞬态而降为次要）。

### 9.8.1 「0/8 恢复成功」的正确定读：不是「不会巡航」，也不是「判据错」

上表里 8/8 `TIMEOUT_HANDOFF`、`best_hold = 0` 容易被两种方式误读：一种读成「该策略落地后彻底废了」，
另一种读成「判据窗口太短、策略其实没问题」。**两种读法都不对**，真相需要精确分层。
用同一份已提交的确定性 `traces` 逐帧差分（`dt = 0.02 s`，阈值 `0.7 × 0.35 = 0.245 m/s`）：

```
seed  y@td     后滑幅度    首次≥70%指令速度   连续保持   保持段平均速度
 11   0.2419   3.70 cm        td+46 步        91 步      0.3293 m/s
 13   0.2642   2.22 cm        td+45 步        92 步      0.3441 m/s
 15   0.2670   2.01 cm        td+45 步        92 步      0.3457 m/s
 20   0.2631   2.96 cm        td+46 步        91 步      0.3388 m/s
 25   0.2688   1.16 cm        td+45 步        92 步      0.3465 m/s
 27   0.2567   2.84 cm        td+46 步        92 步      0.3346 m/s
 31   0.2540   2.98 cm        td+46 步        91 步      0.3304 m/s
 35   0.2587   2.68 cm        td+46 步        92 步      0.3353 m/s
```

`dt = 0.02 s` ⟹ 恢复时刻 **0.90–0.92 s**，保持段 **1.82–1.84 s**，速度 **0.330–0.347 m/s**（指令 0.35）。

**【关键的机制细节】这个巡航不是在 `RECOVER` 里恢复的，而是在 `RECOVER` 超时交接给 `BALANCE` 之后恢复的。**
逐帧核对（seed 11 free-run，`touchdown_step = 62`）：

| 事件 | 步 | FSM | 前向速度 |
|---|---|---|---|
| 落地触地 | 62 | `RECOVER`(4) | 瞬时 −0.03 |
| `RECOVER` 0.40 s 超时 → `timeout_handoff` → `BALANCE` | **81** | `RECOVER`→`BALANCE` | −0.09 |
| 后滑谷值 | 84 | `BALANCE`(0) | −0.16 |
| 首次持续 ≥70% 指令速度 | **108** | `BALANCE`(0) | +0.25 → 稳定 0.33 |

即：**`RECOVER` 阶段本身从未恢复巡航；巡航是由超时交接后的 `BALANCE` 策略重建的。**

**所以三句话必须分开：**

1. 「8/8 未通过恢复判据」——**为真，且判据是对的**。任务书明确要求 `TIMEOUT_HANDOFF` 不得冒充成功
   （E 组反例已锁定），而现状 8/8 都是 `TIMEOUT_HANDOFF`。判据在这里**正确地**拒绝了它。
2. 「策略不具备落地后巡航能力」——**为假**。落地约 0.9 s 后系统确实以 0.33 m/s 恢复巡航并保持到回合结束。
3. 「问题只是判据窗口太短」——**也不完全对**。恢复不是靠延长 `RECOVER` 窗口得到的，
   而是靠**放弃 `RECOVER`、交给 `BALANCE`**。若只把窗口放宽到 0.9 s，`RECOVER` 里仍然是失败轨迹，
   只是有了更多时间；真正起作用的是状态切换。

**这对训练的直接含义**（已同步到 `PPO_TRIAL_PLAN.md` §2）：需要决策的是**任务语义**，而不是窗口数值 ——
「落地后由 `BALANCE` 重建巡航」算不算「完成跨障任务」？

- 若**算**：则应把成功判据从「`RECOVER` 内 sustained」改为「在落地后的固定时间窗内（如 1.0 s）
  达到 sustained 巡航，不论由哪个 FSM 状态提供」，并相应允许 `TIMEOUT_HANDOFF` 作为"中性"而非"失败"。
  这**修改了任务书对 `TIMEOUT_HANDOFF` 的禁止条款**，必须由用户批准。
- 若**不算**（保持现状，推荐）：则应把训练目标定为「让 `RECOVER` 在 0.40 s 内自行恢复」，
  这是一项真实的能力要求，而现状 8/8 不达标。

本轮**未改动任何代码或判据**，只如实记录这一分层事实。

## 9.9 Phase B 踩坑补充（承接 §8）

12. **`--pipeline cpu` 时须同时 `--rl_device cpu`**：否则 `env.device='cpu'` 而 policy 在 cuda，env 会把 cuda 动作写进 CPU 缓冲并崩在 `filtered_actions[jump_active]`（仓库既有行为，非本轮引入）。
13. **GPU 下 `rb_states` 需要 `simulate()` 才重新发布**：`set_actor_root_state_tensor` + 两次 refresh 不够；CPU 下两次 refresh 即可。
14. **每进程只能建一个 Foundation**，故跨 dt 的实验必须用子进程（`tests/test_contact_force_units.py` 的 `--single <dt>` 模式）。
15. **`PPO.update()` 返回 tuple `(value_loss, surrogate_loss)`**，丰富诊断在 `self.last_update_diagnostics`（39 key）。
16. **`RolloutStorage` 张量形状是 `(num_steps, num_envs, 1)`**，不是 `(T, N)`。
17. **`advantages` 在 `compute_returns` 末尾被原地归一化**，要验 GAE 递推必须用 `raw_adv = returns - values` 复原。
18. **`task_registry.make_alg_runner(..., log_root=None)` 是无日志入口**：可保证 smoke test 不会建 run 目录或存 checkpoint。
19. **两轮是 bicycle 纵向布局**（y/z 相同、x 不同），`wheel_box_signed_gap` 取两轮 min，单轮断言必须读列 0/1 而非 `.min()`。
20. **位姿四元数约定是 `(x, y, z, w)`**，写在 `root_states[3:7]`；yaw 的分量在 `qz/qw`。
