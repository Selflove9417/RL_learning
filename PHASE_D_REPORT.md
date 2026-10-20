# MyRobot Phase D — 任务闩锁、几何分层与训练初始化链路核验

基线：`ba9c8075590c8cd6b46b6f89d2a036b58c5d5f9b`（Phase C A1 pilot）
分支：`feat/obstacle-jump-phased`
**本轮未启动任何 PPO 训练**；未改动 URDF、执行器力矩限制、控制周期、物理引擎配置；未覆盖任何历史模型或历史基准 JSON。

---

## 0. 结论摘要（先看这一节）

### 0.1 我上一轮的结论错了，这里更正

Phase C 我写过「`learn()` 不调用 `env.reset()`，所以训练不受 GPU 首轮 reset 伪影影响」。**这个结论是错的。** 用户核实的位置正确：

```
rsl_rl/rsl_rl/runners/on_policy_runner.py:94
    ，  _ = self.env.reset()          # 在 OnPolicyRunner.__init__ 里
```

而 `legged_gym/legged_gym/scripts/train.py:126-133` 的顺序是 `make_env` → `make_alg_runner`（构造 `OnPolicyRunner`，**内含 reset**）→ `learn()`。所以伪影**就在训练初始化路径上**，不是只在评测器里。

### 0.2 一次针对真实训练初始化链路的验证结果

用 `tests/verify_training_init_chain.py` 精确复现 `train.py` 的构造顺序（`make_env` → `make_alg_runner`，后者构造 `OnPolicyRunner` 并在 `__init__` 内调 `reset()`），然后在 `runner.alg.act()` + `env.step()` 的真实 PPO rollout 路径上跑前 25 步（64 envs，不训练、不存盘）：

| 观测项 | GPU pipeline | CPU pipeline |
|---|---|---|
| runner init reset 后 **env0** `root_z`（标称 0.400） | **0.415517** | 0.398674 |
| 同帧 env1–env15 的 `root_z` | 0.39863–0.39868（舍入后 ≈0.3986x） | 0.39862–0.39868（同区间） |
| env0 相对同批环境的偏离 | **+16.9 mm**（离群） | **−0.0 mm**（与同批无差别） |
| env0 第 0–3 步接触 | legC=0, footC=0（**悬空**） | legC=0, footC=0（悬空） |
| env0 第 4–12 步接触 | **legC≥1, footC=0（腿撑地，轮不着地）** | legC=0, footC=2（轮撑地） |
| env0 首次终止 | **第 13 步**（`ep_len` 归零、`reset_buf=True`） | 25 步内无终止（`ep_len` 连续 26） |
| 全 64 环境中 25 步内终止数 | **1（仅 env0）** | **0** |
| reset 后 env0 恢复 | 第 14 步起 footC=2、legC=0（正常），`root_z` 回到 0.385→0.397 | — |

**关于「off-nominal 计数」的口径说明**：JSON 里 `n_envs_off_nominal_after_init_reset = 64`，但阈值是 `|root_z − 0.400| > 1e-4`，而这个机器人写入的出生高度本身经一个受控周期后就沉降到 0.3986x，**任何环境都会超过 1e-4**。所以那个 64 **不表示 64 个环境都坏了**，它只是说明该阈值区分不出异常。真正有判别力的证据是 **env0 与同批其余环境之间的 16.9 mm 落差**，以及**唯一一个 env0 在 13 步内终止**。报告正文按这个口径写，不引用那个会误导的 64。

**已验证事实**：该伪影是 **GPU pipeline 专有、仅 env0、且确实落在真实训练初始化链路上**。表现为 env0 被抬到 0.4155 m（比同批高 16.9 mm）、初始数步悬空后腿部着地而非轮部着地、13 步内触发跌倒终止；env0 一经 reset 即恢复正常。


**尚未解释的现象**：只定位到「GPU 上第一次 `reset()` 之后的第一个受控周期，env0 的求解器状态不从写入值开始」。**没有追到 Isaac Gym 内部哪一行**。这不影响本轮其余结论，但也不能声称已根因定位。

### 0.3 「双重 Reset」只能算临时规避，本轮不采纳为修复

用户要求必须说明它与单次 Reset 的差异。实测（`tests/probe_double_reset_difference.py`，GPU 与 CPU 各一次）：

**起始状态：完全相同。** `dof_pos`、`root_xyz`、`root_quat`、`rb_base_z` 的最大偏差**全部为 0**。

原因是可实测的，不是巧合：本机器人 `default_joint_angles` 六轴**全为 0.0**，而 `_reset_dofs` 写的是 `default_dof_pos * torch_rand_float(0.5, 1.5, ...)`，零向量乘任何随机数仍是零。**但 RNG 确实被消耗**（GPU 0.291368 → 0.739759；CPU 0.060339 → 0.090925），所以「第二次 reset 抽取中性」**只在标称位姿是零向量时成立**，换一个机器人或换一组默认关节角就会立刻变成非中性。

**物理历史：不同，而且可能是负面的。**

| 首个受控周期后 `root_z`（标称 0.400） | GPU | CPU |
|---|---|---|
| 单次 reset | `[0.41552, 0.39868, 0.39864, 0.39863]`，最大偏离 **0.01552 m** | `[0.39867, 0.39868, 0.39864, 0.39863]`，最大偏离 0.00137 m |
| 双次 reset | `[0.39662, 0.40220, 0.40173, 0.40147]`，最大偏离 0.00338 m | `[0.40203, 0.40220, 0.40173, 0.40147]`，最大偏离 **0.00220 m** |

- 在 GPU 上双次 reset 让偏离从 0.01552 降到 0.00338 m（**这就是它「看起来能用」的原因**）。
- 在 CPU 上双次 reset 反而把偏离从 0.00137 放大到 0.00220 m。**它不是免费的**：每回合多消耗一个完整受控周期（`decimation` 个物理子步）把坏周期吸收掉，并且这一周期的接触会留在求解器历史里。
- `reset_idx` **不**推进 `common_step_counter`（只有 `step()` 推进），所以额外一步的时钟代价来自那次吸收用的 `step()`，不是 reset 本身。

**结论**：双次 reset **改变了物理历史**（多一个受控周期），在 CPU 上还会让结果更差，且其「抽取中性」依赖于本机器人默认关节角为零这一偶然条件。**不采纳为正式修复**；若要采用，必须连同上述差异一起写进评测协议，并且只在 GPU 评测器里用，绝不能混进 CPU 基准数字。

### 0.4 Phase D 核心任务已完成

| 项目 | 状态 | 证据 |
|---|---|---|
| 缺陷 B：任务级独立离地闩锁 | **完成** | 反例测试 8 项全 PASS，含 Phase C 原始假阳性帧的回归锚 |
| 缺陷 A：几何语义三层拆分 | **完成** | 三层独立 latch + 反例量化（凸包对任意旋转精确，仅顶点裁剪前缘最多低估 32.0 mm / 后缘 300.8 mm） |
| 三层账目分离（严格/安全/端到端） | **完成** | 评测器 `accounts` 三键，互不合并 |
| 训练看门狗修复 | **完成** | `tests/test_train_watchdog.py` 41/41，并用 Phase C 真实日志复核 |
| Phase C KL 计数更正 | **完成** | 1-based iteration 5/9；日志表头 0-based 4/8 |

### 0.5 冻结模型重评测：修复解除了假象，暴露了更糟的真实情况

> **Phase D.1 更正**：`NORMAL_HANDOFF` 在 32 个 run 中出现 **0 次**，说明当前冻结策略在这些样本中未达到判据，不能证明状态机逻辑不可达。约 4 步余量也不是不可达证明，因为速度保持可以与缓冲/稳定计数同时累积。有障两臂均有 8/8 真实碰撞、整机通过 0/8；严格成功为零不能仅归因于恢复门槛。以下历史表格沿用 Phase D 原定义，Phase D.1 新定义与小样本实测另见 `PHASE_D1_REPORT.md`。


8 种子（11,13,15,20,25,27,31,35）确定性 `act_inference`，0.35 m/s，CPU pipeline 接触对归因：

| 指标 | BEFORE (`model_3.pt`, `669da439…`) | AFTER (`model_10.pt`, `b6afad50…`) |
|---|---|---|
| 障碍碰撞 | **8/8** | **8/8** |
| 真实离地 | 8/8 | 8/8 |
| 真实落地 | 2/8 | **6/8** |
| 双轮干净通过 | 0/8 | **3/8** |
| 落地无遮挡 | 2/8 | **5/8** |
| 整机离开障碍区 | **0/8** | **0/8** |
| 腾空位移（可测样本） | 0.1744 m (n=2) | 0.1601 m (n=6) |
| 所需前向位移 | 0.4245 m | 0.4126 m |
| 缺口 | 0.2475 m | 0.2520 m |
| 腾空速度 | 0.3964 m/s | 0.4002 m/s |
| 腾空时长 | 0.44 s | 0.40 s |
| 障碍峰值接触力 | 475.2 N | 253.7 N |
| **严格跨障成功率** | **0/8** | **0/8** |
| **无碰撞安全通过率** | **0/8** | **0/8** |
| **端到端重新巡航率（有障）** | **0/8** | **0/8** |

**Phase C 那 5 个假阳性种子的去向（13、15、25、27、35）**——这是本轮最关键的逐种子证据。

先说清两件事，避免误读这张表：
- 左列的「第 65 / 61 / 70 / 60 步、`CROSSED_WITHOUT_LIFTOFF`」来自 **Phase C 当时的代码**（记录在 `PHASE_C_A1_PILOT_REPORT.md` §208–217），**不能也不应**用在修复后代码上重跑 BEFORE 臂复现——修复后 BEFORE 臂 8 个种子也已全部变成 `OBSTACLE_COLLISION`（见下表下方说明），这正是缺陷 B 被修掉的表现。
- 右列是**本轮**同一批种子在**修复后代码**上的实测值。

| seed | Phase C（旧代码）：终止步 / 原因 | Phase D（修复后）：终止步 / 原因 | 说明 |
|---|---|---|---|
| 13 | 65 / `CROSSED_WITHOUT_LIFTOFF`（假阳性） | **88 / `OBSTACLE_COLLISION`** | 解除假终止后又跑了 23 步才真正撞上 |
| 15 | 61 / `CROSSED_WITHOUT_LIFTOFF`（假阳性） | **90 / `OBSTACLE_COLLISION`** | 同上，+29 步 |
| 25 | 70 / `CROSSED_WITHOUT_LIFTOFF`（假阳性） | **90 / `OBSTACLE_COLLISION`** | 同上，+20 步 |
| 27 | 60 / `CROSSED_WITHOUT_LIFTOFF`（假阳性） | **87 / `OBSTACLE_COLLISION`** | 同上，+27 步 |
| 35 | 60 / `CROSSED_WITHOUT_LIFTOFF`（假阳性） | **87 / `OBSTACLE_COLLISION`** | 同上，+27 步 |

**更强的一条证据（覆盖到 BEFORE 臂）**：在修复后的代码上重跑，**BEFORE 臂（`model_3.pt`）的 8 个种子也全部是 `OBSTACLE_COLLISION`**，没有一个是 `CROSSED_WITHOUT_LIFTOFF`。也就是说，假阳性在两个模型上同时消失——它从来不是 AFTER 模型的属性，而是判据代码的属性。这一点比「AFTER 的 5 个种子退出假阳性」更有说服力。

逐种子明细（本轮实测，`runs[].outcome`）：

| seed | BEFORE 步 / 落地 / 双轮干净 / 落地安全 | AFTER 步 / 落地 / 双轮干净 / 落地安全 |
|---|---|---|
| 11 | 61 / ✗ / ✗ / ✗ | 62 / ✗ / ✗ / ✗ |
| 13 | 65 / ✓ / ✗ / ✓ | **88** / ✓ / **✓** / ✓ |
| 15 | 61 / ✗ / ✗ / ✗ | **90** / ✓ / **✓** / ✓ |
| 20 | 61 / ✗ / ✗ / ✗ | 63 / ✓ / ✗ / ✗ |
| 25 | 70 / ✓ / ✗ / ✓ | **90** / ✓ / **✓** / ✓ |
| 27 | 60 / ✗ / ✗ / ✗ | **87** / ✓ / ✗ / ✓ |
| 31 | 61 / ✗ / ✗ / ✗ | 62 / ✗ / ✗ / ✗ |
| 35 | 60 / ✗ / ✗ / ✗ | **87** / ✓ / ✗ / ✓ |

`whole_body_passed` 在**全部 16 个 run** 中都是 `False`。


**必须说清楚的三层判读**（Phase C 沿用的纪律）：

1. 「严格成功率 0/8」——**为真**，且现在是在真实碰撞上得出的，不再建立在假终止上。
2. 「After 比 Before 强」——**方向性为真但不足以跨障**：落地率 2→6、双轮通过 0→3、碰撞力 475→254 N，但腾空位移**没有变长**（0.1744→0.1601），缺口 0.2475→0.2520 m **基本没动甚至略增**。
3. 「整机离开障碍区 0/8」——**两臂都是 0，且这是所有失败种子的共同短板**。碰撞发生在机身尚未完全驶离时。

**不能宣布成功**：按任务书要求，任何障碍接触、穿模、跌倒、提前着陆都不判为完整成功。本轮两臂的严格、安全、端到端三账在有障臂上**全部为 0**。

### 0.6 GPU 评测臂作废（必须披露）

`obstacle_jump_phaseD_before_gpu_deterministic_results.json` 中 8/8 全部在**第 15 或 17 步**以 `fell` 终止，`collision=0`、`took_off=0`。这与 0.2 节的 GPU 首轮 reset 伪影完全一致（env0 首轮即被抬起、腿撑地、13–17 步内跌倒）。

**该 GPU 臂的数据无效，不用于任何结论**，也不与 CPU 数字混算。任务书要求「CPU/GPU 判据一致性验证」——本轮的诚实答案是：**在 GPU 评测器的起始协议未修好之前，做不到一致的 GPU 评测**；能做的只有 GPU **训练数据链路**回归（0.7 节，通过）。

---

## 1. 缺陷 B：任务级真实离地闩锁

### 1.1 根因（Phase C 探针原始证据）

```
ctx ep_len=83 fsm=0 root_y=+0.3276 rb_base_y=+0.3276
    taken_off=False rear_margin=+0.0535
```

位姿是真的、`rear_margin` 是真的，**只有 `taken_off` 是假的**。`has_taken_off` / `has_touched_down` / `takeoff_step` / `touchdown_step` 全部由 `myrobot_dual_env` 拥有，它的 FSM 在每次状态迁移时清零（`:262 / :335 / :396 / :403 / :446`），因为对 dual 任务它们表示「本次跳跃周期」。

### 1.2 修复

新增**每环境、每障碍周期独立**的任务闩锁（`myrobot_obstacle_jump_env.py`）：

```python
self.task_took_off      = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
self.task_takeoff_step  = torch.full((self.num_envs,), -1, ..., dtype=torch.float)
self.task_touchdown_step= torch.full((self.num_envs,), -1, ..., dtype=torch.float)
```

- **置位条件**：只有真实离地上升沿 `took_off_now = self.has_taken_off & (~self.task_took_off)`。
- **清零位置**：仅 `reset_idx`（第 815 行）与 `_begin_new_obstacle_cycle`（第 1785 行）。**FSM 的 JUMP/RECOVER/BALANCE 迁移不触碰**。
- **`CROSSED_WITHOUT_LIFTOFF` 改读任务闩锁**：
  ```python
  ploughed = (self.robot_rear_margin > 0.005) & (~self.task_took_off)
  ```
- **连带复核**（任务书 1.5）：`takeoff_step` / `touchdown_step` 在 `extras` 与快照中**改为发布任务闩锁**（原先发布 FSM 的，会被 `request_teleop_jump` 重置为 -1）；`obstacle_cleared` / `success_armed` 复核后**不受 FSM 影响**（`request_teleop_jump` 不动它们），并有显式断言测试。

### 1.3 五个必需反例测试（全部 PASS）

| 要求 | 测试结果 |
|---|---|
| 从未离地却穿过障碍，仍正确失败 | `ploughing without liftoff is a failure  reason=CROSSED_WITHOUT_LIFTOFF` |
| 正常跳跃、落地、转入 BALANCE，不产生假失败 | `a landed jump is not a plough-through after the FSM moves on  reason=NONE rear_margin=+0.0535 task_took_off=True fsm_taken_off=False` |
| 正常跳跃后发生障碍接触，仍正确终止 | 通过（`pass-then-collide is NOT a success`） |
| 多环境交错 Reset 不污染闩锁 | `env2 (never jumped) is the only one failed`；`resetting one env preserves the others' task latch  env0 took_off=True takeoff=39 touchdown=60`；`the reset env's own latch is cleared  env2 took_off=False takeoff=-1` |
| 连续多周期跳跃闩锁正常复位 | `a new cycle re-arms the liftoff latch  took_off=False takeoff=-1 touchdown=-1` |

补充两条「权威归属」断言：
- `request_teleop_jump does not clear the task liftoff latch  takeoff=39 touchdown=60`
- `the FSM's own counters are cleared by that same call  fsm takeoff_step=-1 (task latch still 39)`

---

## 2. 缺陷 A：几何指标三层拆分

### 2.1 为什么要拆

旧实现用**同一个 z-ceiling 包络**同时回答三个不同问题。该包络的后缘只描述低于 0.05 m 盒顶的部件，实测比整机后缘**靠前约 0.25 m**；且机器人在空中时会退化成 `root_y`。用一个量回答三个问题，任何两个答案冲突时都无法判断谁错。

### 2.2 三层定义（`obstacle_geometry.py`）

| 层 | 函数 | 语义 | 关键约束 |
|---|---|---|---|
| **L1 接近触发** | `approach_envelope(env)` | 只考虑**可能触及当前障碍**的碰撞几何（`link_y_extents_clipped` + 精确边交点裁剪） | **不允许**回退到 `root_y`；够不到时返回 `could_reach=False`，触发就保持不动 |
| **L2 轮部通过与安全落地** | `wheel_crossing_state(env)` | 每轮独立的 `interval_front/rear`、`bottom_z`、`overlaps_footprint`、`above_top`、`danger`、`passed_rear` | `danger` = 水平投影重叠 **且** 最低点低于盒顶。**重叠本身只是邻近，不是碰撞** |
| **L3 整机最终通过** | `whole_body_interval(env)` | 整机全 hull、无高度裁剪的后缘 | 用于「机器人已完整驶离」；**不要求在落地瞬间就满足** |

三个 latch 分别锁存（`wheel_pass_clean` / `landing_safe` / `whole_body_passed`），每步把 22 个键写入 `env.obstacle_layers`，包含**独立于所有包络的** PhysX 接触证据（`has_obstacle_collision`、`peak_obstacle_contact_force_N`）和**明确标注为运动学交叉核对而非包络**的 `root_estimated_rear_y` / `root_estimated_margin`。

### 2.3 量化证据（缺陷 A 的反例测试）

| 主张 | 实测 |
|---|---|
| 凸包顶点法对任意旋转精确 | 400 随机姿态 × 7 link，最大低估 **0.000e+00 m** |
| 被取代的 96 方向支撑点法确实低估 | 最大低估 **0.008896 m** |
| 仅顶点裁剪低估前缘（缺陷 A 本体） | 相对精确边交点裁剪最多低估 **+32.009 mm**（pitch=20, ceil=0.20），14 个可比帧内边裁剪从不低于顶点裁剪 |
| 仅顶点裁剪低估后缘 | 最多 **+300.844 mm**（pitch=10, ceil=0.20）——机器人看起来比实际**短**了 30 cm |

**关于「精确」的措辞纪律**（任务书 2.6）：圆柱包络是解析精确的；其余 5 个 mesh link 用凸包顶点支撑是**对任意旋转精确的外界**，但不是穿透深度的精确值，源码与报告都不得称其为「精确穿透测量」。

**不允许做的事**（本轮全部遵守）：未改变障碍尺寸/位置/高度；未把高于盒顶的机身水平投影重叠解释为碰撞；未在腾空期间无依据回退到 `root_y` 并当作真实后缘；未把任何新的距离奖励候选加入正式训练奖励。

---

## 3. 恢复成功语义：三账分离，不合并

任务书 §4 要求分别给出，不得合并成一个数字。评测器 `derived.accounts` 三个键：

| 账目 | 定义 | BEFORE | AFTER |
|---|---|---|---|
| `strict_obstacle_clearance` | 离地 + 落地 + 双轮干净 + 落地无遮挡 + 整机离开 + 零障碍接触 + 70% 持续巡航且以 `NORMAL_HANDOFF` 收尾 + 未跌倒 | **0/8** | **0/8** |
| `collision_free_safe_pass` | 双轮干净 + 落地无遮挡 + 整机离开 + 零障碍接触（**不要求**速度、恢复、姿态） | **0/8** | **0/8** |
| `end_to_end_recruise` | 触地后任意时刻达到 ≥70% 指令速度并保持 10 连续步，**不论哪个 handoff 关闭了 RECOVER** | **0/8** | **0/8** |

无障臂（free-run）的 `end_to_end_recruise` **两臂都是 8/8**——这是关键对照：**策略本身具备落地后重建巡航的能力**，有障臂的 0/8 不是恢复能力缺失，而是从未通过障碍那一关。

**恢复是如何收尾的（这一点必须单独说）。** 有障臂的 `handoff_counts`：

| 臂 | `NORMAL_HANDOFF` | `TIMEOUT_HANDOFF` | `RECOVER_ABORT` | `NONE` | BALANCE 接管延迟（n） |
|---|---|---|---|---|---|
| BEFORE | 0 | 0 | 0 | **8** | 无（n=0） |
| AFTER | **0** | **5** | 0 | 3 | **0.38 s（n=5）** |

BEFORE 臂 8 个 run **一次触地都没有发生**（`touched_down=2` 但那些 run 在 RECOVER 建立前就撞了），所以 handoff 全是 `NONE`。AFTER 臂有 5 个 run 真的进入 RECOVER 并以 **`TIMEOUT_HANDOFF`** 收尾、转由 `BALANCE` 接管（平均 0.38 s 后接管）。

**按任务书的严格判据，`TIMEOUT_HANDOFF` 不算成功**，所以这 5 个 run 在 `strict_obstacle_clearance` 和 `recovery_70pct` 上仍然计 0。这与「AFTER 落地率从 2 提升到 6、但严格成功率仍是 0」并不矛盾——**落地变多了，但落地后是超时被 BALANCE 接管，不是自主恢复巡航**。`recruise_within_1s` 两臂都是 0/8。

恢复常数口径（已核实，三个数各自独立、不要混用）：`RECOVERY_HOLD_STEPS = 10`（`myrobot_dual_env.py:187`，连续 10 步达标才算持续）、`RECOVERY_WINDOW = 40`（`tests/evaluate_obstacle_jump.py:57`，触地后评测观察窗）、`RECOVERY_CURVE_LEN = 50`（`myrobot_dual_env.py:191`，原始速度环形缓冲长度）。

**`NORMAL_HANDOFF` 在全部 32 个 run 中出现 0 次。** 这是本轮新暴露、且必须上报的现象：无障臂 8/8 都以 `TIMEOUT_HANDOFF` 收尾，有障臂 AFTER 5 个也是 `TIMEOUT_HANDOFF`。也就是说 **RECOVER 状态从来没能在超时前靠自身条件正常退出**，每次都是超时后被 BALANCE 接管。

其后果正是三账分离的价值所在：
- 若把「恢复成功」定义成 `NORMAL_HANDOFF`，**本次样本中两臂的恢复账为 0**，无障臂也未达到此快速恢复判据。
- `end_to_end_recruise`（不看 handoff 来源、只看原始速度曲线）无障臂 8/8，说明**巡航能力确实存在，只是从未在 RECOVER 内按时达成**。

**机制（已定位，`myrobot_dual_env.py:588-597`）**：`normal_handoff` 要求 `settle_stable_steps >= 3` **且** `recovery_sustained`，而 `recovery_sustained` 要求 `recovery_hold_steps >= RECOVERY_HOLD_STEPS(10)` 且 `recovery_regression_steps <= 1`；`timeout_handoff` 在 `fsm_state_timer >= 0.40 s`（即 20 步）时无条件触发（只要 grounded 且尚未 normal）。源码注释自己算的时间预算是「handoff 约在第 16 步、timer 约 0.32 s，超时 0.40 s，约 4 步余量」。

实测的 handoff 时刻：AFTER 臂 5 个 run 全部是**触地第 62 步 → handoff 第 81 步**，即**触地后第 19 步、timer≈0.38 s**，紧贴 0.40 s 超时线（差 1 步）。这给出了当前样本未出现 `NORMAL_HANDOFF` 的时间约束背景：`all_settle_calm` 需要 `cushion_time`（0.25 s = 13 步）之后才可能为真，再累积 3 步 → 最早第 16 步；而 `recovery_sustained` 需要 10 步连续达标，两者同时满足的窗口只有约 4 步，实测全部落在窗口之外，于是先撞上 20 步超时。

**Phase D.1 更正**：这些观测不能证明 `NORMAL_HANDOFF` 逻辑不可达；源码允许速度保持与缓冲并行累积。32 次为零只支持当前策略未达标。有障两臂 8/8 真实碰撞、整机通过 0/8 本身也足以否决完整跨障，不应把严格成功为零全归因于恢复条件。




`TIMEOUT_HANDOFF` 不判为成功（保持任务书严格标准），但额外记录 BALANCE 接管后是否实际恢复巡航及时延：BEFORE free-run 触地→达速均值 **1.113 s**、AFTER **1.035 s**；`recruise_within_1s` 两臂都是 0/8。

后滑幅度（`mean_backslide_m`）：BEFORE free-run 0.06175 m、AFTER 0.01126 m；有障臂 BEFORE 0.0 / AFTER 0.00474 m。**这是真实的米，不是从百分比误抄来的**（Phase B 那次量纲错误已更正）。

---

## 4. 训练执行保护整改

### 4.1 Phase C 的失败模式

```bash
cd X && source ./activate_jump.sh && LOG=/tmp/x.log && python train.py ... &
```

末尾 `&` 把整条 `&&` 链放进子 shell，父 shell 的 `$LOG` 是空串，`grep "$LOG"` 静默变成 `grep ""`——失败无声。

### 4.2 结构性消除（`scripts/train_watchdog.py`）

不靠 shell 引用正确性，而是**取消引用本身**：训练命令作为 argv 尾部传入，不经过 `source`/`&&`；日志路径**启动前解析为绝对路径**并校验文件存在且在增长，坏路径是硬错误而非静默空操作。

### 4.3 六项要求与验证

| 要求 | 实现 | 验证 |
|---|---|---|
| 日志路径在正确 shell 环境生效 | 绝对路径 + 存在性 + 增长检查 | PASS |
| 能发现 NaN/Inf | 逐行扫描 | PASS |
| 能识别 PPO KL 早停事件 | 解析 `KL mean (...) exceeded threshold (...)` | PASS，**用 Phase C 真实日志复核：10 次 iteration 全部解析，识别出 2 次早停** |
| 能按规则停止整个训练进程 | 规则触发即终止进程树 | PASS |
| 停止后保留日志与已写入 checkpoint | 先 flush 再退出 | PASS |
| 退出码正确传递，不能静默失败 | 显式返回码 | PASS |

`tests/test_train_watchdog.py`：**41/41 PASS**。

### 4.4 Phase C KL 计数不一致的更正

根因：`diagnostics.json` 的 `iteration` 字段是 **1-based**，训练日志表头是 **0-based**，两者相差 1。已实测确认为 iteration **5 与 9**（`kl_mean` 0.030711 / 0.033657，`early_stop_reason` 明确写 `at update 8/20`）。

已更正 `PHASE_C_A1_PILOT_REPORT.md`：
- 摘要第 3 条：「iteration 4 与 8」→「iteration **5 与 9**」，并注明口径差异。
- §3.5 追加 Phase D 追注：该类 shell 引用失败已被看门狗结构性消除，并附 41/41 与真实日志复核结果。

---

## 5. 验证汇总

```
py_compile                                    COMPILE_OK
tests/test_obstacle_jump_semantics.py --pipeline cpu    114 / 114 PASS
tests/test_ppo_smoke.py               --pipeline gpu     41 / 41 PASS
tests/test_train_watchdog.py                             41 / 41 PASS
```

GPU 训练数据链路回归（`test_ppo_smoke.py --pipeline gpu` 41/41）覆盖任务书第 5 项要求的「GPU pipeline 下必要的终止、奖励与初始化回归」：奖励正负号、终止标志、`actor_loss_mask`、`time_outs`、`terminal_observation`、以及「不依赖只在 CPU 可用的接触对接口也能检出几何重叠」。

**未执行**（按用户第 5 条：未明确必要性前不重跑）：完整的 32 单元 GPU×CPU 对照矩阵。**作废**：GPU 冻结模型评测臂（0.6 节）。

---

## 6. 事实分级（任务书 §6 的强制要求）

### 6.1 已验证事实

1. `rsl_rl/rsl_rl/runners/on_policy_runner.py:94` 的 `OnPolicyRunner.__init__` 调用 `_, _ = self.env.reset()`；`legged_gym/legged_gym/scripts/train.py:124-133` 的 `train()` 顺序为 `make_env` → `make_alg_runner`（构造 runner，内含 reset）→ `learn()`。**训练初始化路径包含 reset**，我 Phase C 的「不受影响」结论错误，此处更正。
2. GPU pipeline 下该 reset 之后 env0 的 `root_z = 0.415517`，比同批其余环境的 0.39863–0.39868 **高约 16.9 mm**（离群）；env0 前 3 步悬空、第 4–12 步腿着地轮不着地、**第 13 步终止**、reset 后恢复。CPU 下 env0 为 0.398674，与同批无差别，25 步内 0 终止。
3. 缺陷 B 假阳性已消除：修复后代码上，**BEFORE 与 AFTER 两臂共 16 个有障 run 的 `reason` 全部为 `OBSTACLE_COLLISION`，`CROSSED_WITHOUT_LIFTOFF` 出现 0 次**（Phase C 为 5 次）。
4. 双次 reset **不改变**出生位姿（`dof_pos`/`root_xyz`/`root_quat`/`rb_base_z` 偏差全 0），因为 `default_joint_angles` 六轴全为 0.0，`default_dof_pos * torch_rand_float(...)` 恒为 0；但 RNG **确实被消耗**（GPU 0.291368→0.739759，CPU 0.060339→0.090925），一般情形下非抽取中性。
5. 双次 reset **改变**物理历史：GPU 首个受控周期偏离从 0.01552 降至 0.00338 m，而 **CPU 上反而从 0.00137 增至 0.00220 m**——不是免费规避。
6. 三账在有障臂上**全部 0/8**（两臂相同）；无障臂 `end_to_end_recruise` **两臂都 8/8**。
7. `NORMAL_HANDOFF` 在 **32 个 run 中出现 0 次**；有障 AFTER 的 5 个 handoff 全靠 0.40 s 超时触发，实测 handoff 落在触地后第 19 步（timer≈0.38 s），源码自述时间预算仅约 4 步余量。
8. 无障臂触地→达速：BEFORE 1.113 s、AFTER 1.035 s（`n_seconds_to_cruise_measured` 需非零才有效，见 §3 口径说明）。

### 6.2 尚未解释的现象

1. **GPU 首轮 reset 伪影的根因未追到**。只确认了现象、范围（仅 env0，+16.9 mm 离群）与管线依赖（GPU 专有），没有定位到 Isaac Gym 内部或本仓库 GPU 路径缺了哪一步。**不得表述为「已根因定位」。**
2. **Phase D.1 裁决已明确**：保留 0.40 s 与 10 步，`NORMAL_HANDOFF` 独立作为快速恢复指标；正式任务采用落地后 2.0 s 内持续、安全的巡航恢复。旧 `end_to_end_recruise` 只检查速度，不直接作为正式成功。32 次未达标不构成逻辑不可达证明。
3. **整机 `whole_body_passed` 两臂全为 0**（16 个有障 run 全 False）。这与「碰撞发生在机身完全驶离之前」一致，但本轮未逐帧确认每个种子的碰撞时刻机身还差多少，因此**不能断言**「碰撞一定发生在整机驶离前」。
4. 腾空位移在 AFTER 反而略降（0.1744→0.1601 m，但 n 从 2 变 6，样本集不同）。**这两个均值不可直接比较**——分母不同，已在 JSON 中分别给出 `n_flight_measured`。`mean_shortfall_m` 同理（n 为 2 vs 6）。
5. BEFORE 臂 `touched_down=2` 却 `handoff` 全为 `NONE`：这两个 run 在触地后、RECOVER 建立前就碰撞终止了（td=63，reset=65/70）。**这是因果顺序问题，不是判据失效**，但意味着 BEFORE 臂的恢复数据几乎为空，两臂的恢复指标本就不可比。



### 6.3 临时规避方法（本轮**不采纳**为修复）

- **双重 Reset**：能消除 GPU env0 首轮崩溃，但改变物理历史、在 CPU 上更差、且「抽取中性」依赖本机器人默认关节角为零。**不采纳**。若将来要用，必须：只在 GPU 评测器内使用、写入评测协议、绝不与 CPU 基准混算、并记录每回合额外消耗的一个受控周期。
- 本轮**没有**为了让任何旧模型成功而移动障碍、放宽 KL 上限、放宽恢复判据或改物理力矩限制。

---

## 7. 交付物与复现命令

### 7.1 新增/修改文件

```
legged_gym/legged_gym/envs/myrobot/myrobot_obstacle_jump_env.py   +333/-  缺陷 B 任务闩锁、缺陷 A 三层 latch、快照与 extras
legged_gym/legged_gym/envs/myrobot/obstacle_geometry.py           +274     register_link_edges / link_y_extents_clipped /
                                                                          approach_envelope / whole_body_interval / wheel_crossing_state
tests/evaluate_obstacle_jump.py                                   +941/-  三层几何、三账分离、十项指标、相位标记、预重置快照
tests/test_obstacle_jump_semantics.py                             +523     I 组（缺陷 B 8 项）、J 组（缺陷 A）、几何反例
scripts/train_watchdog.py                                         新增     argv 尾部传命令、绝对日志路径、NaN/KL/停滞规则
tests/test_train_watchdog.py                                      新增     41 项
tests/verify_training_init_chain.py                               新增     真实训练初始化链路（runner.__init__ 的 reset + 25 步）
tests/probe_double_reset_difference.py                            新增     双次 vs 单次 reset 差异实测
PHASE_C_A1_PILOT_REPORT.md                                        修改     KL 计数 4/8 → 5/9，§3.5 追注
```

### 7.2 原始 JSON（新增，未覆盖任何历史文件）

```
evaluation_assets/obstacle_jump_phaseD_before_deterministic_results.json       3.8 MB
evaluation_assets/obstacle_jump_phaseD_after_deterministic_results.json        4.0 MB
evaluation_assets/obstacle_jump_phaseD_before_gpu_deterministic_results.json   0.6 MB  （作废，见 0.6）
evaluation_assets/training_init_chain_gpu.json
evaluation_assets/training_init_chain_cpu.json
evaluation_assets/double_reset_difference_gpu.json
evaluation_assets/double_reset_difference_cpu.json
```

### 7.3 复现命令

```bash
cd /home/xy/IsaacTu_jump
source ./activate_jump.sh

# 训练初始化链路（本轮唯一一次，GPU 与 CPU 各一次）
python tests/verify_training_init_chain.py --pipeline gpu --num_envs 64
python tests/verify_training_init_chain.py --pipeline cpu --num_envs 64

# 双重 Reset 差异实测
python tests/probe_double_reset_difference.py --pipeline gpu
python tests/probe_double_reset_difference.py --pipeline cpu

# 反例与回归
python tests/test_obstacle_jump_semantics.py --pipeline cpu
python tests/test_ppo_smoke.py --pipeline gpu
python tests/test_train_watchdog.py
```

---

## 8. 训练准入裁决（供技术负责人）

**本轮代码层面的训练语义阻塞项已修复**：缺陷 B 的假阳性终止不再出现于任何种子、三层几何指标互相独立、看门狗可用、GPU 训练数据链路 41/41。

**但 GPU 评测器仍不可用于冻结模型对照**：起始协议未修好之前，GPU 臂 8/8 首轮跌倒。这不影响**训练**（训练不经过评测器的起始协议），但意味着**任何未来的前后对照必须明确声明使用哪条 pipeline**，且不得跨 pipeline 混合。

**不建议**在 GPU 评测器起始协议修好之前重跑 A1 Pilot：那会让「Pilot 前后对照」只能拿到 CPU 臂，而 Pilot 训练本身跑在 GPU 上，对照的两端不在同一 pipeline。

**建议的下一步（待批准，本轮不执行）**：
1. **先裁决 `NORMAL_HANDOFF` 时间预算**（见 §3、§6.2-2）。这是当前最紧的语义阻塞：它让 `strict_obstacle_clearance` 与 `recovery_70pct` 恒为 0，任何后续对照都会在这两个指标上读出「没有差异」，从而得出错误结论。三条路线择一：放宽 `timeout_handoff` 的 0.40 s / 下调 `RECOVERY_HOLD_STEPS` / 承认 RECOVER 只卸残余并把恢复判据改判给 `end_to_end_recruise`。
2. 是否授权修复 GPU 评测器起始协议（在评测器内显式做一次 solver priming，并把「额外消耗一个受控周期」写进协议；**不采用** double-reset 这一具体形式）。
3. 修好 1 与 2 后再决定是否重跑干净的 A1 Pilot、是否进 A2。**在此之前重跑 Pilot 意义有限**。
4. 任务语义遗留问题仍待决：**落地后由 `BALANCE`（而非 `RECOVER`）重建巡航，算不算完成跨障任务？**（Phase B 提出，本轮严格判据下按「不算」计，故端到端账单列。）这与第 1 条是同一个语义问题的两面。

本轮到此停止，未启动任何 PPO 训练。
