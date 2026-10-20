# 跨障任务核心修复与物理重评测 —— 技术验收报告

- 仓库：`Selflove9417/RL_learning`
- 分支：`feat/dynamic-obstacle-clearing-jump`
- 基线提交（本轮开始前）：`d245fef980e2cf7fd0b40ada1b924e24fb6940c4`
- 工作目录：`/home/xy/IsaacTu_jump`
- 评测对象：冻结策略 `legged_gym/logs/myrobot_running_jump/Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt`
- 约束遵守情况：**未启动任何 PPO 训练**；**未改动 URDF / 执行器物理限幅 / 冻结模型权重**；**未覆盖任何历史基准 JSON**（旧脚本保留原样，仅加 `SUPERSEDED` 说明，结果写入全新文件名）。

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
- **pre-reset 快照**（`_capture_episode_snapshot`，43 字段）：在 `reset_idx` 之前冻结上一回合的全部判据。没有它，「at first contact」读到的是新回合的干净站姿（`y=0.0 gap=0.0979`）—— 历史脚本的 `touchdown_pos_y = 0.0`、`flight_dist = -0.106 m` 正是此因。

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

**差异根因是窗口选择偏差，不是度量错误**。冻结 model_3.pt 触地后**前 15 步整体后滑约 17 cm**（seed 11 free-run：`s60 y=+0.1732` → `s79 y=-0.0248` → `s86 y=-0.0964`），旧窗口恰好跳过这段后滑期，只采到接近零的一段，所以看起来「恢复了 56%」。两个数字必须并列披露，不能只报一个。

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

**建议的下一步（不在本轮范围）**：以本轮修复后的奖励与触发为起点重新训练，重点解决两件事 —— (a) 起跳后的水平推进（当前策略把跳跃做成了垂直起落），(b) 落地后的前向速度保持（当前落地后整体后滑 17 cm）。在此之前不应把 `model_3.pt` 当作跨障训练的热启动基线。

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
