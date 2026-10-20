# MyRobot Phase C — A1 Pilot 受控训练与冻结前后对照

本轮授权范围（任务书 m04870）只有四项：核对并推送 Phase B、重新确认 8 项训练准入、跑 **最多 10 iterations** 的 A1 热启动 Pilot、用相同确定性模式做冻结前后对照。
**未执行 A2 / B1 / B2，未跑 150 iterations，未改动 URDF、力矩限制、控制周期、物理引擎配置，未覆盖任何历史模型或历史基准 JSON。**

结论摘要：

1. Phase B 代码已推送到远程独立分支，提交可查（见 §1）。
2. 8 项训练准入 **36/36 PASS**，其中 GPU pipeline 下 PPO 数据链路完整可用，且训练路径不依赖只在 CPU pipeline 可用的接触对接口。
3. Pilot 跑完 10 iterations，无 NaN/Inf，**但 2 次触发内建 KL 早停**（iteration **5 与 9**，各在 8/20 次更新后停止。Phase D §5 更正了此处原写的「4 与 8」：`diagnostics.json` 用的是 1-based iteration，训练日志表头是 0-based，两者相差 1）。
4. 冻结前后对照：**碰撞率 8/8 → 3/8，离地前向速度 0.258 → 0.390 m/s（0.74 → 1.11 × 指令速度），腾空位移反而略降 0.176 → 0.169 m**。完整通过成功率**两臂都是 0/8**。
5. 查出**两个真实缺陷**，它们使本轮的两个「改善」指标不可直接采信、也使其中的一个「失败」指标完全是假阳性：
   - **缺陷 B**：`CROSSED_WITHOUT_LIFTOFF` 在机器人**已经完整完成一次跳跃之后**误触发（AFTER 的 5/8「失败」全是这个假阳性）。
   - **缺陷 A**：触发用几何包络带 z-ceiling 后，其「后缘」实际只描述低于 5 cm 的部件，比整机后缘靠前约 0.25 m，使 TIER-1 通过判据比严格判据宽松约 0.25 m，并使机载段的 `takeoff_front_y`/`takeoff_rear_y` 退化为 root_y。
6. **建议：不要现在执行 A2。**先修缺陷 B（一条最小补丁），否则任何后续训练的终止/奖励信号都是错的。

---

## 1. 第一步：Phase B 提交核实与推送

远程查询不到 `1253960` 的根因：该提交**从未推送**。远程 `feat/dynamic-obstacle-clearing-jump` 停在 Phase A 的 `dcbe58c`。

```
$ git branch --show-current
feat/dynamic-obstacle-clearing-jump
$ git status --porcelain          # 空（推送时刻）
$ git log -5 --oneline
8c562df docs(obstacle_jump): correct the backslide magnitude and split the recovery verdict
8fb4cdc docs(obstacle_jump): note the mismatched denominators in the flight line
44e0df3 docs(trial-plan): fix the reward rationale and the experiment arms
81f67a5 docs(obstacle_jump): correct the first-contact phase and the flight diagnosis
a4c3fe1 docs(trial-plan): add the short PPO trial plan
```

本地核实 `1253960796a89ab6719c20fedfe51e7ca6a54bcd` 存在（`git cat-file -t` → `commit`，作者 `Selflove9417`，`Tue Oct 20 16:56:02 2026 +0800`），且 `dcbe58c` 是 HEAD 祖先，可 fast-forward。未推送的 7 个提交是 `1253960`、`a4c3fe1`、`a6d6647`、`81f67a5`、`44e0df3`、`8fb4cdc`、`8c562df`。

**推送方式：非强制推送、推到独立分支，Phase A 的 ref 一字未动。**

```
$ git push origin HEAD:refs/heads/feat/obstacle-jump-phaseb
 * [new branch]      HEAD -> feat/obstacle-jump-phaseb
```

推送后远程状态：

```
dcbe58c8ee7ac25cd66d39935b20ff77d65833c0  refs/heads/feat/dynamic-obstacle-clearing-jump   (Phase A，未改动)
8c562df633566d3f297cb007d053100f21876167  refs/heads/feat/obstacle-jump-phaseb            (Phase B HEAD)
```

本地 HEAD == 远程新分支 == `8c562df633566d3f297cb007d053100f21876167`。`1253960796a89ab6719c20fedfe51e7ca6a54bcd` 从该分支可达，可在 GitHub 上直接打开。

---

## 2. 第二步：8 项训练准入复核（全部通过）

| # | 项目 | 证据 | 结果 |
|---|---|---|---|
| 1 | 75 项训练语义反例测试 | `tests/test_obstacle_jump_semantics.py --pipeline cpu` → `PASSED 75 / 75  RESULT: PASS` | PASS |
| 2 | GPU pipeline PPO smoke test | `tests/test_ppo_smoke.py --pipeline gpu` → `PASSED 41 / 41`；`--pipeline cpu` → `PASSED 38 / 38` | PASS |
| 3 | Actor 35 / Critic 35 / Action 6 | `num_obs=35 num_privileged_obs=35 num_actions=6`；actor 首层 35、末层 6；critic 首层 35；checkpoint 17 个权重键 **17/17 逐位相同** | PASS |
| 4 | GPU reset 后无幻影失败 | 把 env 摆到 `base_y = 0.6195`（已越过障碍）后 reset：`n_task_failed=0`、`reasons=['NONE']` | PASS |
| 5 | `model_3.pt` 冻结并作热启动 | sha256 `669da439…4f85`，`warm_start=True`，`warm_start_path` 指向该文件，`resume=False` | PASS |
| 6 | actor_loss_mask / time_outs / terminal_observation | 每步都发布；mask 逐环境二值且两种状态都出现；强制 timeout 后 `n_timeout=4`；bootstrap 奖励 `stored=10.41098 == r + γV(terminal_observation)`，且**不等于**回退值 `10.22297` | PASS |
| 7 | 碰撞奖励为负、成功奖励不提前发 | 无接触恰为 `-0.00000`；`mild=-7.500 severe=-15.000` 单调；清零 latch 单付 `0.00000`、仅 on-event 付 `5.000`；`obstacle_geometric_pass=0.6 < 5.0` | PASS |
| 8 | 运行目录与历史 checkpoint 隔离 | `logs/myrobot_obstacle_jump/trial_a1_pilot10` 启动前不存在；该实验名下历史 checkpoint 数 `n=0`；历史 99 个 checkpoint 未触碰 | PASS |

准入脚本：`tests/phasec_admission.py`（本轮新建）。关键实测输出：

```
[PASS] bootstrap reward equals r + gamma*V(terminal_observation)   stored=10.41098 expected=10.41098
[PASS] bootstrap did NOT fall back to the step's own value         stored=10.41098 own_value_would_be=10.22297
[PASS] identical=17/17
contact_pairs_available = False      (GPU)
[PASS] collision detected without the contact-pair API  n_env_intersecting=8 deepest gap=-0.07000 m
```

最后一行是准入 2 的核心：在 **GPU pipeline（`contact_pairs_available=False`）** 下，训练路径靠 `obstacle_geometry` 的解析几何独立检出重叠，**不依赖 `gym.get_env_rigid_contacts` 这个只在 CPU pipeline 可用的接口**。

---

## 3. 第三步：A1 Pilot 运行记录

### 3.1 配置与命令

```
python legged_gym/legged_gym/scripts/train.py \
  --task myrobot_obstacle_jump --headless \
  --num_envs 64 --max_iterations 10 --run_name trial_a1_pilot10
```

| 项 | 值 |
|---|---|
| task | `myrobot_obstacle_jump`（原始奖励，未增加任何新奖励项） |
| 初始模型 | 冻结 `model_3.pt`（`warm_start=True`，`warm_start_path` 指该文件，`resume=False`） |
| `num_envs` / `num_steps_per_env` | 64 / 200 → 12800 步/iteration |
| `max_iterations` | 10（`--max_iterations 10` 覆盖默认 5） |
| `save_interval` | 1 → `model_1.pt` … `model_10.pt` |
| 超参（原封未动） | `clip_param=0.2`、`entropy_coef=0.01`、`num_learning_epochs=5`、`num_mini_batches=4`、`lr=5.0e-5`（`fixed`）、`gamma=0.99`、`lam=0.95`、`desired_kl=0.015`、`early_stop_kl=0.03`、`max_grad_norm=1.0` |
| 运行目录 | `legged_gym/logs/myrobot_obstacle_jump/Oct20_18-27-49_trial_a1_pilot10/` |
| 原始日志 | 同上目录内 `pilot_a1_10iter.log`（21114 B）+ `diagnostics.json`（17697 B）+ `events.out.tfevents.*` |
| 退出码 / 耗时 | `TRAIN_EXIT=0`，约 65 s（6.44 s/iteration） |

### 3.2 逐 iteration 原始指标（`diagnostics.json`，迭代号为 1..10）

| it | reward | ep_len | KL mean | KL max | clip | entropy | grad_tot | grad_actor | grad_critic | updates |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.870 | 51.23 | 0.01095 | 0.09107 | 0.1447 | 0.4434 | 44.52 | 4.96 | 42.22 | 20/20 |
| 2 | 1.903 | 51.86 | 0.00466 | 0.05536 | 0.0546 | 0.4581 | 75.04 | 4.59 | 73.52 | 20/20 |
| 3 | 2.142 | 55.61 | 0.00813 | 0.08402 | 0.1033 | 0.4668 | 74.75 | 5.38 | 72.79 | 20/20 |
| 4 | 2.046 | 53.74 | 0.01133 | 0.10127 | 0.1464 | 0.4754 | 60.11 | 5.86 | 58.03 | 20/20 |
| 5 | 2.380 | 57.05 | 0.01351 | 0.09248 | 0.1354 | 0.4772 | 123.60 | 6.78 | 123.40 | **7/20 EARLY STOP** |
| 6 | 2.394 | 57.21 | 0.00447 | 0.05014 | 0.0526 | 0.4776 | 38.43 | 4.49 | 36.60 | 20/20 |
| 7 | 2.182 | 54.49 | 0.00651 | 0.17306 | 0.0624 | 0.4825 | 57.02 | 5.19 | 55.31 | 20/20 |
| 8 | 2.527 | 58.18 | 0.01101 | 0.11654 | 0.1328 | 0.4883 | 48.52 | 6.01 | 46.40 | 20/20 |
| 9 | 2.583 | 58.91 | 0.01569 | 0.27387 | 0.1360 | 0.4911 | 89.84 | 7.06 | 89.53 | **7/20 EARLY STOP** |
| 10 | 2.641 | 57.86 | 0.00542 | 0.06714 | 0.0679 | 0.4942 | 27.92 | 4.83 | 25.80 | 20/20 |

`early_stop_reason`：
- iteration 5：`KL mean (0.030711) exceeded threshold (0.030000) at update 8/20`
- iteration 9：`KL mean (0.033657) exceeded threshold (0.030000) at update 8/20`

参数位移极小且稳（`actor_param_delta_rel` = 0.003647 / 0.002297 / 0.002308 / 0.002263 / 0.001295 / 0.002405 / 0.002228 / 0.002209 / 0.001278 / 0.002399），`action_std_mean` 0.3698 → 0.3707 基本不动 ⟹ **策略没有崩、也没有被 KL 保护打歪**。

### 3.3 训练期奖励分量（`Mean episode rew_*`）

- **`rew_obstacle_cleared_success` 10/10 恒为 `0.0000`** —— 完整使命成功一次都没发放，与门控设计一致。
- `rew_obstacle_geometric_pass` 仅 iteration 0–2 非零（0.0010 / 0.0015 / 0.0030），iteration 3 起归零。
- `rew_obstacle_collision` 全程为负（−0.0300 ~ −0.0762），符号正确。

### 3.4 判读

- **无 NaN / Inf，无断言崩溃，无形状错误。** Value loss 单调下降 14.9347 → 4.8325；entropy 上升 0.4434 → 0.4942（未坍缩）。
- Mean reward 1.870 → 2.641（+41%），Mean episode length 51.23 → 57.86。**这个上升不能当作跨障能力提升**，原因见 §4.3 与 §5。
- **Critic 梯度范数显著高于 Actor**（25.8–123.4 vs 4.5–7.1），且峰值恰好在两次早停的 iteration。Value loss 还在 4.83 量级，critic 尚未收敛，是下一阶段要盯的量。
- **异常 KL 是否发生：是。** 两次 iteration 的 `kl_mean` 越过 `desired_kl=0.015`，并在第 8 次更新时越过 `early_stop_kl=0.03`，由仓库内建的早停逻辑截断（只执行 7/20 次更新），`learning_rate` 保持 `fixed` 5.0e-5、未做任何自适应。**我没有放宽 KL 上限、没有改 `desired_kl`/`early_stop_kl`、没有改学习率。**

### 3.5 本轮自身的一个执行偏差（必须披露）

训练的 NaN/Inf 与早停看门狗**没有真正生效**。原因是我的 shell 写法：

```bash
cd X && source ./activate_jump.sh && LOG=/tmp/... && python train.py ... &
```

`&` 把整条 `&&` 链放进子 shell，父 shell 的 `$LOG` 是空串，于是后续 `grep "$LOG"` 静默失败（`grep: : 没有那个文件或目录`），看门狗从未真正检查日志。
**后果**：iteration 5 与 9 的早停没有被看门狗捕捉并中断训练。**但这两次早停是 PPO 内建 KL 保护触发的，按设计只会截断该 iteration 的参数更新次数，不会让训练崩坏**；训练本身 10 iterations 干净收尾（`TRAIN_EXIT=0`），所以我**没有重跑**，以免浪费一个已经合规完成的 run。正确写法是把 `LOG=` 放在 `&` 之前单独赋值，或用字面路径。

> **Phase D 跟进**：该类 shell 引用失败已被 `scripts/train_watchdog.py` 结构性消除（训练命令作为 argv 尾部传入，不经过 `source`/`&&`；日志路径启动前解析为绝对路径并验证存在且在增长）。回归测试 `tests/test_train_watchdog.py` **41/41 PASS**，并用 Phase C 的真实训练日志复核：正确解析全部 10 次 iteration 并识别出这 2 次早停。

---

## 4. 第四步：冻结前后 8 项对照

### 4.1 方法

```
python tests/evaluate_obstacle_jump.py --task myrobot_obstacle_jump --headless --pipeline cpu \
  --model_path <model> --action_mode deterministic \
  --seeds 11 13 15 20 25 27 31 35 \
  --out evaluation_assets/obstacle_jump_pilot_a1_{before,after}_results.json
```

- 两臂**同一份代码、同一份物理配置、同一确定性模式**（`policy.act_inference(obs)`），种子为任务书指定的既有基准 `[11,13,15,20,25,27,31,35]`，**独立测试种子 `[101,103,107,109,113,127,131,137]` 未动用**。
- BEFORE = 冻结 `model_3.pt`；AFTER = Pilot 最新 `model_10.pt`。
- 每 seed 两臂（有障 / 无障对照），共 16 run × 2。
- 分析脚本：`tests/compare_pilot_a1.py`（本轮新建），输出 `evaluation_assets/obstacle_jump_pilot_a1_comparison.json`。
- 历史文件 `obstacle_jump_reeval_deterministic_results.json` / `obstacle_jump_reeval_results.json` **零改动**，新数据落在新文件名下。

### 4.2 八项指标

**有障臂（8 seed）**

| # | 指标 | BEFORE `model_3.pt` | AFTER `model_10.pt` |
|---|---|---|---|
| 1 | 前向离地速度 | 0.2580 m/s（0.737 × 指令） | **0.3897 m/s（1.113 × 指令）** |
| 2 | 腾空时间 / 水平位移 | 0.460 s / 23.0 步（n=2）；位移 **0.1761 m**；空中均速 0.3829 m/s | 0.420 s / 21.0 步（n=6）；位移 **0.1690 m**；空中均速 0.4024 m/s |
| 3a | 完整几何通过（**严格**：整机过盒后缘） | 0/8 | **0/8** |
| 3b | 完整几何通过（从 trace 复算的宽松口径） | 0/8 | 0/8（失真，见 §5 缺陷 A） |
| 3c | 完整几何通过（**env 自身记录的宽松口径** = `outcome.cleared`：仅低于 5 cm 的部件过盒后缘） | 1/8 = 12.5% | **5/8 = 62.5%** |
| 4 | 障碍接触率 | **8/8 = 100%** | **3/8 = 37.5%** |
| 5 | 真实落地率 | 2/8 = 25% | **6/8 = 75%** |
| 6 | `NORMAL_HANDOFF` / `TIMEOUT_HANDOFF` / `RECOVER_ABORT` | 0 / 0 / 0（全部 `NONE`，因碰撞在 RECOVER 之前终止） | 0 / **5** / 0（另有 3 个 `NONE`） |
| 7 | 落地后实际稳定巡航 | trace 可分析的落地 run 2 个，**0/2** 达到 10 步持续窗口 | trace 可分析的落地 run 5 个，**0/5** 达到 |
| 8 | **完整跨障成功率** | **0/8** | **0/8** |

**无障对照臂（8 seed）** —— 用于确认「变化来自跨障行为，不是策略整体退化」

| # | 指标 | BEFORE | AFTER |
|---|---|---|---|
| 1 | 前向离地速度 | 0.2580 m/s | 0.3897 m/s |
| 2 | 腾空 / 位移 | 0.450 s / 22.5 步；**0.1683 m**；0.3740 m/s | 0.420 s / 21.0 步；**0.1640 m**；0.3905 m/s |
| 3 | 通过 | 0/8（无障碍） | 0/8 |
| 4 | 障碍接触 | 0/8 | 0/8 |
| 5 | 真实落地率 | 8/8 | 8/8 |
| 6 | 恢复事件 | `TIMEOUT_HANDOFF` 8/8 | `TIMEOUT_HANDOFF` 8/8 |
| 7 | 落地后持续巡航（10 步窗口） | **8/8**，落地→巡航 0.912 s，巡航持续 1.813 s | **8/8**，落地→巡航 0.840 s，巡航持续 1.895 s |
| 8 | 完整成功 | 0/8 | 0/8 |

### 4.3 怎么读这张表（三条独立的结论）

**(a)「碰撞率 100% → 37.5%」是真实变化，但不能直接当作跨障能力提升。**
无障臂的腾空位移**同样**从 0.1683 掉到 0.1640 m，说明训练**没有**让策略跳得更远；有障臂的碰撞减少主要来自**离地速度提高（0.258 → 0.390 m/s）**，而空中位移没变 ⟹ 它在同样的滞空时间内没有获得额外水平行程，只是起跳更猛。空中前向速度仍在 0.39–0.40 m/s，高于 0.35 m/s 的指令值，与 Phase B 的发现一致。**真正的缺口依然是滞空时长：21 步 vs 所需约 27 步。**

**(b)「3c 通过 1/8 → 5/8」有一半是测量定义问题，不能当作任务达标。**
`3c` 用的是 env 自己的 `robot_rear_margin`，而该量在加了 z-ceiling 之后只描述**低于 5 cm 障碍顶面的部件**（见 §5 缺陷 A）。整机后缘在 root 之后约 0.209 m，低于 5 cm 的部件后缘却在 root 之后约 +0.047 m ⟹ 两者相差约 **0.25 m**。按严格判据（整机过盒后缘）**两臂都是 0/8**：从起跳到落地 `base_y` 需前进约 **0.42 m**，实测只有 0.169 m，缺口 2.5 倍。`3b` 是从 trace 复算的同口径量，但它对 AFTER 臂不可用，因为 env 在机载时把 `takeoff_rear_y` 记成了 root_y。

**(c)「5/8 以 `CROSSED_WITHOUT_LIFTOFF` 结束」是假阳性，不是失败。**
这 5 个 run 全部**完整完成了跳跃**（离地、落地、宽松通过判据成立），是在**跳完之后**被这个判据误杀的。详见 §5 缺陷 B。

---

## 5. 本轮查出的两个真实缺陷

### 缺陷 B（最严重，影响终止信号与训练语义）

**现象。** AFTER 臂 8 个 run 里有 5 个以 `CROSSED_WITHOUT_LIFTOFF` 结束，而它们的 trace 显示机器人已经完整跳过障碍（`taken_off` 在 step 39 置 1、`touchdown` 在 step 60、`cleared` 在 step 60）。

**取证。** 仪器化探针 `tests/probe_crossed_without_liftoff.py`（复刻 worker 设置：`cfg.seed=13`、`cfg.trigger.mode='spatial'`、`cfg.episode_length_buf` 从 0 起、1 env）包裹 `_update_task_outcome` 打印判据的两个操作数：

```
  79    4   0.3259    0.0485     1     1     0     81 NONE
  80    0   0.3267    0.0504     1     1     0     82 NONE
     ctx ep_len=83 fsm=0 root_y=+0.3276 rb_base_y=+0.3276 obs_rear=+0.3200
  >> LATCH step=83 rear_margin=+0.0535 taken_off=False collision=False
```

`ctx` 行是决定性的：闩锁发生时**没有任何 reset 迹象**（`ep_len=83` 在增长、`root_y` 与 `rb_states` 一致、fsm 仍在运行）。机器人的位姿是真的、`rear_margin=+0.0535` 也是真的，**唯一的假量是 `has_taken_off=False`**。

**根因。** `has_taken_off` 是一个**被两套机制共用的布尔量，而两者对它的生命周期定义不同**：

- 障碍任务的写法把它当作「**本次障碍周期内是否离过地**」：`legged_gym/legged_gym/envs/myrobot/myrobot_obstacle_jump_env.py:1453`
  ```python
        ploughed = (self.robot_rear_margin > 0.005) & (~self.has_taken_off)
  ```
- 而 `legged_gym/legged_gym/envs/myrobot/myrobot_dual_env.py` 把它当作**双足 FSM 的本次跳跃周期标志**，在每次 FSM 迁移时直接清零：`:262`、`:335`、`:396`、`:403`、`:446`（其中 `:403` 是回到 `STATE_JUMP` 时 `self.has_taken_off[i] = False`）。

当 FSM 在机器人**仍物理位于障碍后缘之后**的时候把该标志清零，`ploughed` 立刻变成 True，一次完整的成功跳跃被记成「未离地横穿」。

**影响。**
1. 评测：AFTER 的 5/8「失败」是假的；`no_task_failure` 与 `success` 统计被污染。
2. **训练：`_task_failure_mask()` 把 `task_failed` 并入 `fall`，所以这类 episode 会被提前终止并获得失败信号 —— 也就是说 Pilot 的奖励/终止信号本身就是错的。**这是「不要现在执行 A2」的直接理由。
3. 与 Phase B 修过的 GPU post-reset 幻影失败**不是同一个 bug**（那个是 reset 后位姿陈旧；这个是运行中共享标志被清）。同一句 `CROSSED_WITHOUT_LIFTOFF`、两种成因。

**建议最小补丁（本轮未实施，待批准）。** 让障碍任务持有自己的周期闩锁，不再读 FSM 的标志：

```python
# _init_buffers
self.task_took_off = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
# compute_obstacle_metrics，在更新 has_taken_off 之后
self.task_took_off |= self.has_taken_off
# _update_task_outcome
ploughed = (self.robot_rear_margin > 0.005) & (~self.task_took_off)
# reset_idx / _begin_new_obstacle_cycle
self.task_took_off[...] = False
```

同时应把 `takeoff_step`/`takeoff_front_y`/`takeoff_rear_y` 的来源改为该闩锁的上升沿，以避开 §5 缺陷 A 中「机载时包络退化」的污染。

### 缺陷 A（几何包络的语义退化）

**现象。** 加了 z-ceiling 过滤（Phase B 为修「reset 即触发」而引入）之后，`robot_forward_interval` 的「后缘」不再是整机后缘。reset 时刻实测（box front/rear/top = 0.2800 / 0.3200 / 0.0500，`base_y=-0.0018`）：

```
NO ceiling   : front=0.2895 rear=-0.2107 span=0.5003 body=base_link
with ceiling : front=0.1872 rear=+0.0448 span=0.1424 body=link_004
rear 相对 base_y: 无 ceiling -0.2090   有 ceiling +0.0466
```

机体与髋部 link 的支撑点世界高度都高于 5 cm 障碍顶面，被 `world[...,2] <= z_ceiling` 全部滤掉，只剩轮子；机载更高时连轮子也被滤掉，走 `nothing_below_ceiling` 回退到 `rb_states[:,0,1]`（root_y）。

**后果。**
1. `robot_rear_margin > 0.005` 实际只要求 `base_y > 0.278`，而整机越过后缘要求 `base_y > 0.529` ⟹ **TIER-1 通过判据比严格判据宽松约 0.25 m**（两臂同受影响，故对照仍可比，但绝对值不可当「通过」）。
2. 机载段 `front_y == rear_y == root_y`。实测 seed13 AFTER：step 38 `front=0.2461 rear=0.1042` → step 39 起 `front=rear=0.1084=root_y`，直到 step 57。⟹ 起飞瞬间快照 `takeoff_front_y`/`takeoff_rear_y` 被写成 root_y，**AFTER 全部 8/8 三个起飞量恒等**，`required_dist_m` / `shortfall_m` 对 AFTER 失真（该行在 BEFORE 与 AFTER 之间不可直接相减）。

**本轮未改。** 修法有两条路，需要设计决策：(i) 把「用于触发」的包络与「用于通过判据」的包络拆成两个不同的量；(ii) 用障碍盒的解析 AABB 与整机凸包做真正的三维分离轴/最近距离，不再靠单一 Y 区间。两者都会改变训练语义，故留待批准。

---

## 6. 是否执行 A2？是否把 A1 扩展到 30 iterations？

**两个都建议暂缓。**

| 问题 | 判断 | 理由 |
|---|---|---|
| 工程稳定性 | **通过** | 10 iterations 无 NaN/Inf、无断言、无形状错误；参数位移 rel ≈ 2×10⁻³；entropy 未坍缩；GPU 数据链路完整（准入 36/36） |
| 训练方向 | **有信号但被污染** | 离地速度 0.737 → 1.113 × 指令是真实的；但终止判据有 5/8 假阳性（缺陷 B），奖励信号与 episode 长度都不可信 |
| 执行 A2（原奖励 + 强化奖励四臂对照） | **不要现在做** | 缺陷 B 未修时，任何奖励改动带来的差异都会被同一个假阳性终止噪声掩盖；`rew_obstacle_rear_clearance_progress` 恰恰依赖 `takeoff_rear_y`，而该量在机载时退化（缺陷 A） |
| 把 A1 扩展到 30 iterations | **不要现在做** | 目的不是「跑够步数」，而是「信号可信」。在缺陷 B 修好前多跑 20 个 iteration 只是放大噪声 |
| 是否值得做 A2 | **修完缺陷 B 后值得** | A1 已证明策略可被推动（离地速度 +51%），且碰撞率确实下降；真正要做的是把主奖励对准滞空时长，而不是重跑 |

**建议顺序**：(1) 批准并修缺陷 B → (2) 重新跑一次同规格 10-iteration A1（作为修复后的干净基线，确认 `CROSSED_WITHOUT_LIFTOFF` 归零、episode 长度不再被假终止截断）→ (3) 再决定 A2。缺陷 A 可以并进 (2)，但要作为独立的设计决策。

---

## 7. 本轮交付物清单

| 类别 | 内容 |
|---|---|
| 远程提交 | 分支 `feat/obstacle-jump-phaseb` @ `8c562df633566d3f297cb007d053100f21876167`（Phase B 全部 7 个提交，含 `1253960796a89ab6719c20fedfe51e7ca6a54bcd`） |
| 训练产物 | `legged_gym/logs/myrobot_obstacle_jump/Oct20_18-27-49_trial_a1_pilot10/`（`model_1..10.pt`、`diagnostics.json`、`training_snapshot.json`、`events.out.tfevents.*`、`pilot_a1_10iter.log`）—— 在 `logs/` 下，**未被 git 跟踪**，符合隔离要求 |
| 新评测数据 | `evaluation_assets/obstacle_jump_pilot_a1_before_results.json`、`..._after_results.json`、`..._comparison.json` |
| 新脚本 | `tests/phasec_admission.py`（准入 36 项）、`tests/compare_pilot_a1.py`（8 项对照）、`tests/probe_crossed_without_liftoff.py`（缺陷 B 取证探针） |
| 未改动 | `model_3.pt`（sha256 `669da439…4f85` 与准入时一致）、URDF、力矩限制、控制周期、物理引擎配置、四份历史基准 JSON |

---

## 8. 需要批准的事项

1. **修缺陷 B**（`has_taken_off` 共享标志）。本轮未改代码。这是一条约 6 行的补丁，但会改变终止判据的行为，按任务书「不要擅自…」的要求留待批准。
2. **是否同时修缺陷 A**（把触发包络与通过判据包络拆开，或改用真正的三维几何）。这会改变 TIER-1 的语义。
3. 修完后是否重跑 10-iteration A1 作为干净基线。
4. 是否进入 A2（奖励强化四臂对照）。
5. 悬而未决的任务语义问题（Phase B 遗留）：落地后由 `BALANCE`（而非 `RECOVER`）重建巡航算不算完成跨障任务。

**训练已停止，未启动任何后续 run，等待下一次技术审查。**
