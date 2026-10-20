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
| Q2 | 跳跃的**水平位移**能否增加？ | `mean_flight_m` 从 0.174 m 上升，`mean_shortfall_m` 下降 |
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
实现方式：新增 4 个 `CfgPPO` 子类（各设 `run_name` 与 `warm_start`），或在 `train.py` 加一个 `--run_name` 覆盖。**推荐前者**，因为 `run_name` 已由 `task_registry` 拼接进 `log_dir`，不需要改训练脚本。

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
   理由：现状 `post_touchdown_forward = 2.5` 只奖励正速度，对负速度没有梯度压力；实测 8/8 后滑说明这个缺口是真实的。

2. **水平飞行距奖励**（新增，或提高现有 `forward_progress_across_obstacle` 权重）：
   ```python
   def _reward_flight_horizontal_progress(self):
       # Reward forward travel measured from liftoff, while airborne only.
       # `takeoff_pos_y` is the pre-existing liftoff snapshot
       # (myrobot_dual_env.py:142, written at :469).
       airborne = self.has_taken_off & (~self.has_touched_down)
       travelled = self.root_states[:, 1] - self.takeoff_pos_y
       return torch.clamp(travelled, min=0.0) * airborne.float()
   ```
   ```python
   flight_horizontal_progress = 1.5
   ```
   理由：现状奖励在离地/触地两端都有项，但空中水平推进没有独立项；实测飞行距 0.174 m 对 5 cm 障碍够用、对所需 0.209 m 不够，需要给这一段单独梯度。

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

# A1 热启动 + 现有奖励
python -u legged_gym/legged_gym/scripts/train.py \
  --task myrobot_obstacle_jump \
  --headless --num_envs 64 --seed 11 --max_iterations 150 \
  --experiment_name myrobot_obstacle_jump \
  --run_name trial_a1_warm_baseline

# A2 热启动 + §4 两项奖励
#  (需先落地 §4 的 rewards 子类，例如 myrobot_obstacle_jump_tuned_cfg)
python -u legged_gym/legged_gym/scripts/train.py \
  --task myrobot_obstacle_jump_tuned \
  --headless --num_envs 64 --seed 11 --max_iterations 150 \
  --run_name trial_a2_warm_tuned

# B1 / B2 同上，但 CfgPPO 里设 warm_start = False

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
