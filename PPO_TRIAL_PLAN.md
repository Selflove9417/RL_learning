# 短期 PPO 试验方案（待批准后执行）

本文件只描述**方案**，不含任何训练执行结果。本轮已按任务书要求停止，等待批准。

- 代码基线：`1253960796a89ab6719c20fedfe51e7ca6a54bcd`（分支 `feat/dynamic-obstacle-clearing-jump`）
- 参照数据：`evaluation_assets/obstacle_jump_reeval_deterministic_results.json`（确定性 `act_inference`，8 种子）
- 前提事实见 §1，方案本身见 §3–§7。

---

## 1. 为什么现在可以开始试验（以及为什么不能指望它一次成功）

### 1.1 已修好的训练语义（本轮交付）

| 项目 | 修复前 | 修复后 | 证据 |
|---|---|---|---|
| 成功奖励 | 几何通过即发满额 5.0，先通过后碰撞照样领奖 | 分两档：几何通过 0.6；满额需 真离地+完整通过+零接触+落地+持续巡航 且 `NORMAL_HANDOFF`，每回合至多一次，任何碰撞永久解除 | 75 项单测 E 组 11 项 |
| 恢复判据 | `good_steps >= 2`（窗口内任意两帧） | 连续 10 步（0.20 s）不间断达标；姿态/角速度/关节稳定 + 双轮承载；`TIMEOUT_HANDOFF`/`RECOVER_ABORT` 明确不算 | `myrobot_dual_env.py` `_recovery_succeeded` |
| 触发包络 | `root_y + wheel_radius`，低估前缘 0.22 m | URDF 凸包顶点支撑函数，任意旋转精确；逐点 mask 修掉零填充幻影点 | 单测 F/H 组 |
| 接触单位 | `lambda` 被当 N·s，错误除以 dt | 实测为力 N（m·g 比值 0.999995，跨 dt 不变） | `tests/test_contact_force_units.py` |
| 评测动作 | `policy.act` 采样 | `policy.act_inference` 确定性；模式写入 JSON 与文件名 | 8 种子重跑 |

### 1.2 冻结模型的真实能力边界（决定试验设计的约束）

`model_3.pt` 在修正判据下 8 种子 × 有障臂：**0/8 成功、8/8 碰撞**。逐项分解：

- **高度够**：峰值飞行净空 21.7 cm，障碍仅 5 cm。
- **水平距不足**：`flight_dist` 均值 0.174 m，`required` 均值 0.209 m，缺 3.4 cm（有障臂，n=2/n=8 —— 分母不同，只能作方向参考）。更有说服力的是用**无障臂 8/8 全样本**算：飞行速度 0.361–0.405 m/s（均值 0.385）**已经超过 0.35 m/s 指令**，但滞空只有 21–22 步，**比所需少约 5.7 步（0.114 s）**，导致后包络停在 0.263–0.288、距盒后缘 0.320 还差 3.2–5.7 cm。也就是说**速度对、方向对，只是飞得不够久**。
- **落地后倒退**：无障碍臂 8/8 恢复速度为负（均值 −14%，最差 −18%），策略触地后倒了 13–18 cm，把仅有的前进量全部吃掉。
- **起跳时序错**：8/8 障碍臂的首次接触**全部发生在离地之后**（`before_liftoff=False`），其中 **6/8 在空中就撞上**（`touchdown_step=-1`，从未触地），只有 2/8 真正落地后才擦到盒后上角。也就是说当前策略不是"跳得不够远"，而是**跳跃的落点根本不在障碍之后**。
- **恢复从未成功**：无障碍臂 8/8 都是 `TIMEOUT_HANDOFF`，`recovery_best_hold_steps=0`。旧判据会把这 8 次全部记为恢复成功。

> 因此试验的第一目标不是"提高成功率"，而是**延长滞空约 5–6 步** 与 **让落地后不再倒退** 这两件事。
> 前者已具备正确的速度跟踪，只需把跳跃拉长；后者是完全反方向的行为，必须靠奖励压下去。

### 1.3 两个已知的、会影响训练的陷阱（本轮已修 / 需注意）

1. **GPU pipeline 的 post-reset 幻影失败（本轮已修）**：`reset_idx` 后 `rb_states` 在 GPU 上仍是上一回合末的位姿，旧代码会在新回合第一步把它判成 `CROSSED_WITHOUT_LIFTOFF` 并终止——每次成功穿过障碍都会静默销毁下一回合。实测 2 env @ `base_y=0.62`：GPU 两个都给 `CROSSED_WITHOUT_LIFTOFF`，CPU 都干净。修复是 `post_reset=True` 时不再评估几何。
2. **CPU pipeline 才能拿到接触对**：`GymGetEnvRigidContactCount` 在 GPU pipeline 下不可调用。训练路径已确认不依赖它（smoke test 在 GPU 下 41/41 PASS，靠解析几何独立检出重叠）。但**若要用接触对做日志分析，必须跑 `--pipeline cpu`**，此时需同时 `--rl_device cpu`，否则 env 会把 cuda 上的动作写进 CPU 缓冲（仓库既有行为，非本轮引入）。

---

## 2. 试验要回答的问题

| # | 问题 | 判据 |
|---|---|---|
| Q1 | 从 `model_3.pt` 热启动，能否在有限步数内提高无障臂的**落地后速度保持**？ | 无障碍臂 `recovery_sustained` 从 0/8 上升；`mean_recovery_ratio` 由 −14% 转正 |
| Q2 | 跳跃的**滞空时长**能否增加（不是速度，速度已达标）？ | 有障臂 `mean_flight_m` 从 0.174 m 上升、`mean_shortfall_m` 下降；等价地看 `flight_steps` 从 21–22 步向 27 步靠拢 |
| Q3 | 重新初始化 Jump Actor（不热启动）是否比热启动更快获得这两项能力？ | 两臂在同一预算下对比 Q1/Q2 指标 |
| Q4 | 新奖励分档是否按设计生效（几何通过不再等于成功）？ | 训练日志中 `obstacle_geometric_pass` 与 `obstacle_cleared_success` 的出现次数与累计值可区分 |

---

## 3. 实验臂设计（2 臂 × 2 目标，共 4 个 run）

**统一基线**：`--task myrobot_obstacle_jump`（0.35 m/s，`episode_length_s=4.0`，`num_envs=64`）。

| run | 初始化 | 目标 |
|---|---|---|
| A1 | `warm_start=True`, `warm_start_path=model_3.pt`（当前 cfg 默认） | 热启动 + 现有奖励 |
| A2 | 同上 | 热启动 + 强化落地后退/水平距的两项奖励（见 §4） |
| B1 | **不热启动**（`warm_start=False`，Jump Actor 随机初始化，Balance 用冻结 `e15_m130.pt`） | 重新初始化 + 现有奖励 |
| B2 | 同上 | 重新初始化 + 强化奖励 |

> **为什么保留 B 臂**：`model_3.pt` 的落地后滑是系统性缺陷（无障臂 8/8 负速度），热启动可能让策略困在这个局部行为里。B 臂是判断"是否该从零训 Jump Actor"的唯一手段，成本只是一次训练。

**输出目录必须独立**（不得写入历史 run）：
```
legged_gym/logs/myrobot_obstacle_jump/trial_a1_warm_baseline/
legged_gym/logs/myrobot_obstacle_jump/trial_a2_warm_tuned/
legged_gym/logs/myrobot_obstacle_jump/trial_b1_scratch_baseline/
legged_gym/logs/myrobot_obstacle_jump/trial_b2_scratch_tuned/
```
实现方式（**已在源码中核实，比初稿更简单**）：`--run_name` 本来就是 CLI 覆盖项
（参数表在 `legged_gym/legged_gym/utils/helpers.py:202` 的 `get_args()` 里，
应用在 `helpers.py:155-156`：`cfg_train.runner.run_name = args.run_name`），
`--warm_start` / `--warm_start_optimizer` / `--warm_start_path` 也都是现成的 CLI 开关
（`helpers.py:196-199` 定义，`helpers.py:161-167` 应用，实际加载在
`legged_gym/legged_gym/utils/task_registry.py:168-176`）。

因此：
- **A1 不需要任何新配置类** —— 现有 `MyRobotObstacleJumpCfgPPO` 默认就是 `warm_start=True` + `model_3.pt`；
- **B1/B2 需要子类** —— 因为 `--warm_start` 只有 `store_true` 语义（`helpers.py:197`），
  `helpers.py:161-162` 只会在真值时把它设为 `True`，**没有把默认 True 改回 False 的命令行路径**；
- **A2/B2 需要子类** —— 奖励 scale 不在 CLI 覆盖范围内。

合计需要 **3 个** `CfgPPO` 子类（A2 一个、B1 一个、B2 一个），不是初稿写的 4 个（A1 白算了一个），
也不是我在上一段临时改口的 1 个（漏掉了 B 臂关不掉 `warm_start` 这件事）。

---

## 4. §A2/§B2 的奖励调整（唯一允许的奖励改动）

只调两项，且都直接对应 §1.2 的两个缺陷，不做大改：

1. **落地后退惩罚**（新增）：触地后 `v_fwd < 0` 的步给负奖励。
   ```python
   # scales 里新增
   post_touchdown_backslide = -2.0
   ```
   ```python
   def _reward_post_touchdown_backslide(self):
       # 0 before touchdown; penalise backwards ground speed afterwards.
       # `_get_world_forward_velocity` is the running-jump helper the obstacle env
       # already uses (myrobot_obstacle_jump_env.py:1179, :1706), i.e. forward
       # speed projected onto the body's heading -- not raw world Y.
       backward = torch.clamp(-self._get_world_forward_velocity(), min=0.0)
       return backward * (self.has_touched_down & (~self.task_failed)).float()
   ```
   理由（**这是量化过的，不是猜测**）：现状 `_reward_post_touchdown_forward = 2.5 × exp(-err²/(2σ²))`，`σ = tracking_sigma = 0.08`、`cmd_vy = 0.35`。把实测的后滑速度代进去：

   ```
   v_fwd    err      reward    d(reward)/dv
    0.000  -0.350   0.00007     0.0038
   -0.050  -0.400   0.00000     0.0002
   -0.130  -0.480   0.00000     0.0000   ← 实测的典型后滑速度
   -0.180  -0.530   0.00000     0.0000   ← 实测最差后滑
    0.200  -0.150   0.17242     4.0411
    0.350   0.000   1.00000    -0.0000
   ```

   即：**策略真正后滑时（−0.13 至 −0.18 m/s），这一项的奖励值和梯度都已经是 0**，它在运行点上完全平坦，
   对"停下来"没有任何拉力。8/8 后滑不是巧合，是这个奖励形状的必然结果。
   新增项提供的是**线性**梯度（`backward` 对 `v_fwd` 的导数恒为 −1），在后滑区间内持续有效。

2. **越障净距奖励**（新增），把奖励从「速度」改为「距离」：
   ```python
   def _reward_obstacle_rear_clearance_progress(self):
       # Progress of the REAR envelope toward the box rear face, normalised by
       # the root travel still owed at liftoff. This is the quantity that decides
       # success; the existing terms reward a per-step velocity ratio instead.
       #
       # `takeoff_rear_y`  : rear envelope at liftoff (myrobot_obstacle_jump_env.py:482, written :1130)
       # `robot_rear_margin`: rear envelope minus box rear face (:403, written :1058)
       # Both already exist; no new env buffer is needed.
       #
       # At liftoff: robot_rear_margin == takeoff_rear_y - obs_rear_y == -owed,
       # so `travelled` starts at exactly 0 and reaches `owed` when the rear
       # envelope passes the rear face.
       airborne = self.has_taken_off & (~self.has_touched_down)
       obs_rear_y = self.obstacle_root_states[:, 1] + 0.5 * self.obstacle_w
       owed = torch.clamp(obs_rear_y - self.takeoff_rear_y, min=0.05)
       travelled = torch.clamp(self.robot_rear_margin + owed, min=0.0)
       return torch.clamp(travelled / owed, 0.0, 2.0) * airborne.float()
   ```
   ```python
   obstacle_rear_clearance_progress = 1.0
   ```

   **理由（必须先说清现有奖励已经做了什么，否则会重复计奖）**：

   现有两项都是**逐速度**形式的：
   ```
   forward_progress_across_obstacle = 2.5 * clamp(v_fwd / cmd, 0, 1.25)   空中
   forward_velocity_retention       = 2.5 * exp(-(v_fwd-cmd)^2/(2*0.08^2))  全周期
   ```
   - 它们是**逐步**奖励，所以滞空越久累计越多 —— **现有奖励并非没有延长滞空的激励**。我最初写的"空中水平推进没有独立项"是不准确的。
   - 真正的问题是**在运行点已经饱和**：实测空中速度 0.385 m/s（> 0.35 指令），
     `clamp(0.385/0.35, 0, 1.25) = 1.10`，距上限 1.25 只剩 12% 余量。也就是说策略**已经拿满了这一项**，
     再往前飞几乎没有额外奖励，剩下的行为差异（少 5.7 步滞空）落在这一项的盲区里。
   - 距离型奖励没有这个上限：它以「还欠多少位移」为尺度，直接度量成败判据本身。

   > **诚实标注**：这一项是**待验证的假设**，不是已证实的必要修复。
   > 它之所以值得试，是因为现有两项在 0.385 m/s 处几乎饱和；但它也可能与现有项高度相关而无效。
   > 这正是 A1/B1（现有奖励）作为**对照臂**存在的原因 —— 若 A2/B2 相对 A1/B1 没有改善，
   > 就说明问题不在奖励形状，而在别处（例如探索不足或 FSM 时序），届时应当停下来重新诊断，而不是加更多项。

   **用现有 8 种子数据离线验算过这个形状**（不是纸面推导）：

   ```
   seed  takeoff_rear  owed   margin@liftoff  travelled@liftoff  margin@end  frac@end
    11     0.1112     0.2088     -0.2088            0.0000        -0.0572    0.7261
    13     0.1131     0.2069     -0.2069            0.0000        -0.0367    0.8227
    15     0.1112     0.2088     -0.2088            0.0000        -0.0337    0.8388
    20     0.1101     0.2099     -0.2099            0.0000        -0.0389    0.8145
    25     0.1099     0.2101     -0.2101            0.0000        -0.0319    0.8484
    27     0.1082     0.2118     -0.2118            0.0000        -0.0507    0.7605
    31     0.1153     0.2047     -0.2047            0.0000        -0.0500    0.7558
    35     0.1086     0.2114     -0.2114            0.0000        -0.0490    0.7680
   ```

   两条性质都成立：
   - `travelled@liftoff` **恒为 0.0000**（起跳时刻没有白送奖励，不会奖励"跳起来"本身）；
   - 现有策略落地时只走到 **0.73–0.85**（不是 1.0，更没有撞上 `clamp` 上限 2.0），
     即**该项在真实运行点上未饱和、仍有 15–27% 的可提升梯度** —— 这正是它相对速度型的优势。
     速度型在 0.385 m/s 处已到 `1.10 / 1.25`（88%），只余 12%。
   （`frac@end` 是在**无障臂**的飞行距离上算的，因为它在 8/8 回合都有真实触地；
   有障臂 6/8 空中就撞上、`touchdown_step=-1`，没有可信的落地时刻可用。）

**不做的事**：不动 `obstacle_collision` 符号、不动 `obstacle_cleared_success` 分档、不动触发几何、不动 URDF/执行器限幅、不改冻结模型。

---

## 5. 训练预算与停止准则

- **每 run 预算**：先给 **150 iterations**（`num_steps_per_env=200`、64 envs ⇒ ≈ 1.92 M 环境步/run）。4 个 run 合计 ≈ 7.7 M 步，3060 Ti 上约 4–6 小时（按现有 `myrobot_running_jump` 吞吐估算）。
- **早停（任一满足即停）**：
  - `kl_mean` 连续 10 iteration > `early_stop_kl = 0.03`（配置已开 `enable_early_stopping`）；
  - 奖励均值连续 20 iteration 不升且 `clip_fraction > 0.4`；
  - 出现 NaN/Inf（应当直接中止并保留现场）。
- **绝不**为了"跑到收敛"而延长预算；150 iteration 只够判断**方向**，不是最终训练。

---

## 6. 监控指标（每 iteration 都看，写入 TensorBoard）

必看 9 项，缺一不可：

1. `reward` 总量与**分项**（尤其 `obstacle_geometric_pass` vs `obstacle_cleared_success` 的累计占比 —— 验证 Q4）
2. `kl_mean` / `kl_p95`（阈值 0.015 目标、0.03 早停）
3. `clip_fraction`（持续 >0.3 说明步长过大）
4. `actor_grad_norm_mean` / `critic_grad_norm_mean`（`max_grad_norm=1.0`；若长期被裁剪说明损失尺度异常）
5. `surrogate_loss` 与 `value_loss`
6. `entropy` / `action_std_*`（崩塌到 0 意味着过早收敛）
7. `episode_length`（若骤降到 <40 步说明大量早终止 —— 可能是幻影失败复现）
8. **`extras['episode']` 里的 `rew_obstacle_collision` / `rew_obstacle_cleared_success` / `rew_obstacle_geometric_pass`**
9. 自定义：每 10 iteration 跑一次 8 种子确定性评测（用本轮评测器），跟踪 `mean_flight_m` / `mean_recovery_ratio` / `recovery_70pct` 的真实走向

> 第 9 项是关键：训练奖励涨不等于任务能力涨（本轮已证明旧奖励会奖励"先通过后碰撞"）。**能力判据只认确定性评测的真实几何指标，不认训练 rewards。**

---

## 7. 独立测试种子（不得用于训练）

训练用种子里程碑 `11`（与现有评测一致）之外，**验收只用**：

```python
TEST_SEEDS = [101, 103, 107, 109, 113, 127, 131, 137]
```

这 8 个种子**不参与任何训练/调参**，只用于最后验收。本轮 8 种子（11,13,15,20,25,27,31,35）保留为其历史对照基线，不得混入新均值。

---

## 8. 执行命令（批准后）

```bash
cd /home/xy/IsaacTu_jump
source ./activate_jump.sh

# A1 热启动 + 现有奖励（现有 CfgPPO 默认就是 warm_start=True + model_3.pt，无需额外开关）
python -u legged_gym/legged_gym/scripts/train.py \
  --task myrobot_obstacle_jump \
  --headless --num_envs 64 --seed 11 --max_iterations 150 \
  --experiment_name myrobot_obstacle_jump \
  --run_name trial_a1_warm_baseline

# B1 不热启动 + 现有奖励
#  ⚠ 这条命令现在跑不了：必须先按 §3 落地 b1 的 CfgPPO 子类（warm_start = False），
#    然后才有 --task myrobot_obstacle_jump_b1。保留在此仅为说明目标命令形状。
python -u legged_gym/legged_gym/scripts/train.py \
  --task myrobot_obstacle_jump_b1 \
  --headless --num_envs 64 --seed 11 --max_iterations 150 \
  --experiment_name myrobot_obstacle_jump \
  --run_name trial_b1_scratch_baseline

# ⚠ 为什么不能直接用 --task myrobot_obstacle_jump 跑 B1：
#   warm_start 默认是 True（myrobot_obstacle_jump_config.py:113-114）
#   且 --warm_start 只有 store_true 语义（helpers.py:197），
#   helpers.py:161-162 只在真值时把它设为 True，没有把默认 True 改回 False 的命令行路径。
#   所以 B1/B2 必须用 CfgPPO 子类置 warm_start = False。
#   这修正了 §3 的结论：A1 免配置，B1/B2 与 A2 都需要子类（共 3 个，而非初稿的 4 个，也不是 1 个）。

# A2 热启动 + §4 两项奖励（需先落地 §4 的 rewards 子类）
python -u legged_gym/legged_gym/scripts/train.py \
  --task myrobot_obstacle_jump_tuned \
  --headless --num_envs 64 --seed 11 --max_iterations 150 \
  --run_name trial_a2_warm_tuned

# 每 10 iteration 的确定性评测（不覆盖历史结果）
python -u tests/evaluate_obstacle_jump.py \
  --pipeline cpu --action_mode deterministic \
  --model_path legged_gym/logs/myrobot_obstacle_jump/trial_a1_warm_baseline/model_<N>.pt \
  --out evaluation_assets/trial_a1_iter<N>_deterministic_results.json
```

> `--pipeline cpu` 是为了拿到接触对归因；若只看性能可用 `--pipeline gpu`（更快），但两者数据**不可混合**。

---

## 9. 需要批准的事项

1. 是否按 §3 的 **4 臂设计**执行（或先只跑 A1+A2 两臂，省一半时间）。
2. §4 的两项奖励改动是否可接受（这是本轮"不改奖励语义"之外唯一的放宽）。
3. §5 的 **150 iteration / run** 预算是否合适。
4. 训练期间是否允许占用 GPU（与评测互斥，3060 Ti 8 GB 无法同时跑 64 env 训练 + 评测）。
