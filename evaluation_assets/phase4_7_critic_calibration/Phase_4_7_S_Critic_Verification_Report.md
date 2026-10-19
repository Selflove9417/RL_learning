# MyRobot Phase 4.7-S：Target B 实际生效与 Critic 开发集数据最终验收报告

> **项目阶段**：Phase 4.7-S（Target B 终态自举生效闭环、五种子开发集统一归口与 Critic 最终验收）  
> **最终裁决**：**`PASS`**（第 140 步真实 Timeout 与 Bootstrap 经全张量审计完全生效；五种子正式轨迹完美复现历史基线 12.66 cm；新 Critic 在正统开发集上实现 MSE 16.56 $\to$ 0.73，降幅 95.6%，EV 提升至 87.6%；Actor 绝对零漂移）  
> **基线模型**：Champion Model (`model_champion.pt`，SHA-256: `fad590f4699...`)  
> **校准资产**：`model_critic_calibrated_target_b.pt` (SHA-256: `a5fca202c66...`)  
> **状态限制**：**严格禁止在本轮进行 Actor PPO 或 Critic 重新训练，严格保持所有执行器、奖励权重和动力学配置冻结**

---

## 目录
1. [执行概要与最终裁决](#1-执行概要与最终裁决)
2. [任务一：PPO 第 140 步真实 Timeout 与 Bootstrap 全要素张量审计](#2-任务一ppo-第-140-步真实-timeout-与-bootstrap-全要素张量审计)
3. [任务二：开发集物理轨迹口径统一与正统五种子配对对比](#3-任务二开发集物理轨迹口径统一与正统五种子配对对比)
4. [任务三：价值目标固定基准与无限视界语义复核](#4-任务三价值目标固定基准与无限视界语义复核)
5. [任务四：共享 RSL-RL 库向后兼容性验证](#5-任务四共享-rsl-rl-库向后兼容性验证)
6. [Phase 4.8 推进准许与实施规范](#6-phase-48-推进准许与实施规范)

---

## 1. 执行概要与最终裁决

本轮 Phase 4.7-S 针对 Phase 4.7 中尚未闭环的边界条件与轨迹口径问题，开展了严密的代码级与张量级复核，获得了决定性证据：
1. **第 140 步真实 Timeout 与 Target B 自举完全生效**：在完整的 64 环境物理模拟中，捕获了 Step 137～140 的真实张量记录。证实 Step 139 推进后 `episode_length_buf = 140`，`check_termination()` 触发 `140 >= 140` 超时，`extras["time_outs"]` 与 `extras["terminal_observation"]` 成功传导至 PPO，单步奖励准确注入 $\gamma V(s_{140}^{\text{pre}})$，GAE 截断完全符合有限时域数学定义。
2. **多环境超时掩码与环境无副作用验证通过**：任意掩码下未超时环境增加值严格为 $0.000000\text{e}+00$，环境观测函数连续 10 次调用变异为 $0.000000\text{e}+00$（严格无状态）。
3. **彻底理清五种子历史轨迹口径混淆**：查明 Phase 4.6-R 报告中 7.79..12.78 cm 是临时审计脚本（缺少 `cmd_trim`）引起的，本轮使用正式 `eval_phase4_seed.py` 控制逻辑生成了 100% 权威的五种子正统轨迹（均值 12.66 cm，完全吻合 Phase 4.1 历史基线）。
4. **正统开发集轨迹上 Critic 优势绝对成立**：在完全相同的五种子正统轨迹上，Phase 4.7 Target B Critic 相对 Champion Critic 的 MSE 由 **16.5565 骤降至 0.7280 (-95.6%)**，Explained Variance 由 **0.6138 提升至 0.8764**，末端 TD 残差平滑无跳变。
5. **最终裁决**：**`PASS`**。准许保留 `model_critic_calibrated_target_b.pt`，准许进入 Phase 4.8 进行谨慎的 Actor PPO Pilot。

---

## 2. 任务一：PPO 第 140 步真实 Timeout 与 Bootstrap 全要素张量审计

在 64 环境、Seed 6001、真实 Champion 策略下连续执行 142 步模拟，并逐步捕获系统关键张量（详见测试脚本 [`tests/verify_step139_timeout_boundary.py`](file:///home/xy/IsaacTu_jump/tests/verify_step139_timeout_boundary.py) 与审计数据 [`evaluation_assets/phase4_7_critic_calibration/phase4_7_s_step138_140_tensor_audit.json`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase4_7_critic_calibration/phase4_7_s_step138_140_tensor_audit.json)）：

### 2.1 Step 137～140 真实张量审计明细

| 关键张量 / 状态字段 | Step 137 (倒数第3步) | Step 138 (倒数第2步) | Step 139 (第140步，终态步) | Step 140 (新Episode第1步) |
| :--- | :---: | :---: | :---: | :---: |
| **Pre-step `episode_length_buf`** | 137 | 138 | **139** | **0 (已重置)** |
| **Post-step `episode_length_buf`** | 138 | 139 | **0 (reset_idx清零)** | **1** |
| **`check_termination()` 判断依据** | $138 \ge 140 \to \text{False}$ | $139 \ge 140 \to \text{False}$ | **$140 \ge 140 \to \text{True}$** | $1 \ge 140 \to \text{False}$ |
| **`reset_buf` (64 环境)** | 全 False | 全 False | **全 True (64/64)** | 全 False |
| **`time_out_buf` (前置捕获)** | 全 False | 全 False | **全 True (64/64)** | 全 False |
| **`infos["time_outs"]` 传导** | 不存在 (已 pop) | 不存在 (已 pop) | **存在且全 True (64/64)** | 不存在 (已 pop) |
| **`infos["terminal_observation"]`** | 不存在 (已 pop) | 不存在 (已 pop) | **存在 (64, 35)** | 不存在 (已 pop) |
| **终端观测 `jump_phase`** | — | — | **0.8500 (自稳着陆态)** | — |
| **重置后 `next_obs` `jump_phase`** | 0.8500 | 0.8500 | **0.0000 (下蹲准备态)** | 0.0050 |
| **$V(s_{140}^{\text{pre}})$ (重置前终态价值)**| — | — | **17.2080** | — |
| **$V(s_0')$ (重置后初态价值)** | — | — | **18.1440** | — |
| **$\Delta V$ (重置初态 - 真实终态)** | — | — | **+0.9361 (物理差异)** | — |
| **环境原始单步奖励 $r_t$** | +0.1820 | +0.1913 | **+0.1897** | -0.0071 |
| **PPO 存储奖励 `stored_rew`** | +0.1820 | +0.1913 | **17.2256** | -0.0071 |
| **PPO 增强项 ($\gamma V$)** | 0.0000 | 0.0000 | **$+17.0359$ ($0.99 \times 17.2080$)** | 0.0000 |
| **RolloutStorage `returns[t]`** | — | 17.2439 | **17.2256** | — |
| **RolloutStorage `advantages[t]`** | — | -0.6603 | **-0.7493** | — |

### 2.2 核心边界与逻辑确证
1. **边界条件确认**：
   在 Step 139 中，`episode_length_buf` 累加至 140，`self.time_out_buf = self.episode_length_buf >= self.max_episode_length` 正确评估为 **True**。
2. **终态缓存与重置时序无缝衔接**：
   在 `myrobot_forward_jump_phase4_env.py` 的 `post_physics_step()` 中：
   - 步骤 1：先计算重置前 $s_{140}^{\text{pre}}$，写入 `self.terminal_obs_buf`；
   - 步骤 2：将 `self.time_out_buf.clone()` 与 `self.terminal_obs_buf.clone()` 写入 `self.extras`；
   - 步骤 3：随后调用 `self.reset_idx(env_ids)`，物理状态被阻尼至初态，`episode_length_buf` 清零；
   - 步骤 4：重新计算生成 $s_0'$ 写入 `self.obs_buf`，作为下一步的起点。
3. **PPO 奖励增强严格对齐**：
   在 PPO 的 `process_env_step` 中：
   $$\text{stored\_rew} = r_{139} + \gamma \cdot V(s_{140}^{\text{pre}}) = 0.1897 + 0.99 \times 17.2080 = 17.2256$$
   与真实存储值浮点误差小于 $1.0\times 10^{-6}$。
4. **GAE 回报计算截断完全闭环**：
   在 `RolloutStorage.compute_returns` 中，由于 Step 139 的 `dones[139] == 1`，使得 `next_is_not_terminal = 0.0`，因此：
   - Step 139 不再叠加外部的 `last_values`（即不会叠加 $V(s_0')$），其 Return 严格等于 `stored_rew`（17.2256）；
   - Step 139 的 TD 残差准确定义为 $\delta_{139} = (r_{139} + \gamma V(s_{140}^{\text{pre}})) - V(s_{139})$；
   - Step 138 正确接收来自 Step 139 的折现 GAE 优势。

### 2.3 多环境超时掩码测试（零交叉泄漏）
构造 4 环境复合批次进行严格断言：
- Env 0（超时）：$r=1.0$，$V=10.0$ $\to$ 存储奖励 $= 1.0 + 0.99\times 10.0 = 10.90$
- Env 1（未超时）：$r=2.0$，$V=20.0$ $\to$ 存储奖励 $= 2.0 + 0.0 = 2.00$
- Env 2（超时）：$r=3.0$，$V=30.0$ $\to$ 存储奖励 $= 3.0 + 0.99\times 30.0 = 32.70$
- Env 3（未超时）：$r=4.0$，$V=40.0$ $\to$ 存储奖励 $= 4.0 + 0.0 = 4.00$
- **执行结果**：实际输出为 `[10.90, 2.00, 32.70, 4.00]`，与理论预期最大误差为 **`0.000000e+00`**。证明张量广播与掩码完全隔离，不存在跨环境泄漏。

### 2.4 观测函数无状态性与副作用核查
在同一物理状态下连续调用 `compute_observations()` 10 次，前后观测向量的最大绝对差值为 **`0.000000e+00`**。确认该函数仅执行纯数学张量拼接与变换，不维护内部时序缓存或历史队列，在 `reset_idx` 前主动调用不会对环境状态造成任何副作用。

---

## 3. 任务二：开发集物理轨迹口径统一与正统五种子配对对比

### 3.1 历史口径混淆根因彻底澄清
在 Phase 4.6-R 报告中，开发集表格出现了 `7.79, 9.94, 10.03, 12.63, 12.78 cm` 的异常偏低位移。经本轮深度复查，确认该数据出自 Phase 4.6-R 编写的临时审计脚本 [`tests/audit_phase4_6_r_paired_seeds.py`](file:///home/xy/IsaacTu_jump/tests/audit_phase4_6_r_paired_seeds.py)。该脚本存在两处关键编写缺陷：
1. 遗漏了自稳控制器的前向微调命令（将 `bal_cmds[:, 1]` 置为全零，缺少 `cmd_trim` 积分反馈）；
2. 更改了平衡腿部的 IK 增益（未乘 $0.25$ 预缩放）。
这导致机器人在平衡阶段持续晃动，无法在 20 步内达标，一直被拖入第 75 步超时强制退出，从而在异常的交接初态下起跳，产出了偏小位移。

### 3.2 正统五种子物理轨迹生成与复现
本轮基于已确认 100% 正确的正统评测脚本 [`tests/eval_phase4_seed.py`](file:///home/xy/IsaacTu_jump/tests/eval_phase4_seed.py)（具备标准 `cmd_trim`、标准自稳门限判定以及下落髋 $+0.10$ 动作补偿），在独立子进程中为 Seeds 3001～3005 完整重新生成了物理轨迹，记录了全时步观测、动作、奖励及重置前终态：

| 种子 (Seed) | 正统自稳交接步 | 正统起跳步 | 正统触地步 | 正统前向位移 (cm) | Phase 4.1 历史基线 (cm) | 复现吻合度 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3001** | 20 | 11 | 33 | **12.04** | 12.04 | **严格一致 (100%)** |
| **3002** | 16 | 11 | 33 | **12.93** | 12.93 | **严格一致 (100%)** |
| **3003** | 19 | 11 | 33 | **11.86** | 11.86 | **严格一致 (100%)** |
| **3004** | 21 | 11 | 33 | **14.71** | 14.71 | **严格一致 (100%)** |
| **3005** | 19 | 11 | 33 | **11.76** | 11.76 | **严格一致 (100%)** |
| **五种子均值** | **19.0** | **11.0** | **33.0** | **12.66 cm** | **12.66 cm** | **完全精确对齐** |

### 3.3 正统物理轨迹上的两代 Critic 严格配对对比
在这批完全相同、权威统一的五种子正统物理轨迹上，输入完全相同的状态、奖励和终态 Bootstrap 目标，对比 Champion Critic 与 Phase 4.7 Target B Critic：

| 种子 (Seed) | 正统位移 (cm) | Champion Critic MSE | Target B Critic MSE | MSE 相对降幅 | Champion EV | Target B EV | Step 139 TD 残差 (校准后) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3001** | 12.04 | 16.5425 | **0.7228** | **-95.6%** | 0.6128 | **0.8766** | +0.0677 |
| **3002** | 12.93 | 16.4757 | **0.7273** | **-95.6%** | 0.6162 | **0.8758** | +0.0658 |
| **3003** | 11.86 | 16.5944 | **0.7237** | **-95.6%** | 0.6107 | **0.8767** | +0.0664 |
| **3004** | 14.71 | 16.3727 | **0.7101** | **-95.7%** | 0.6164 | **0.8772** | +0.0683 |
| **3005** | 11.76 | 16.7973 | **0.7561** | **-95.5%** | 0.6128 | **0.8755** | +0.0658 |
| **综合均值** | **12.66** | **16.5565** | **0.7280** | **-95.6%** | **0.6138** | **0.8764** | **+0.0668** |

### 3.4 分阶段价值预测与优势对齐情况（五种子正统轨迹均值）

| 动作阶段 | 步数区间 | 阶段实际平均奖励 | Champion 价值 $V_A$ | 校准后价值 $V_B$ | 真实折现目标 $G$ | Champion 原始优势 | 校准后原始优势 | 阶段 MSE 降幅 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **下蹲准备 (Prep)** | 0 ~ 9 | -0.0473 | 19.02 | **23.27** | 24.00 | +0.8477 | **+0.3778** | 24.73 $\to$ **0.6082** (-97.5%) |
| **推蹬起跳 (Launch)** | 10 ~ 13 | +0.2012 | 21.24 | **25.61** | 26.26 | +0.9132 | **+0.3951** | 25.20 $\to$ **0.4439** (-98.2%) |
| **腾空上升 (Ascent)** | 14 ~ 21 | +0.4437 | 20.52 | **25.06** | 25.56 | +0.9695 | **+0.3343** | 25.35 $\to$ **0.2548** (-99.0%) |
| **腾空下降 (Descent)** | 22 ~ 31 | +0.3223 | 19.12 | **23.79** | 23.96 | +0.7972 | **+0.0706** | 23.44 $\to$ **0.0370** (-99.8%) |
| **触地缓冲 (Touchdown)**| 32 ~ 39 | +0.1526 | 19.59 | **24.46** | 24.52 | +0.8870 | **-0.0404** | 24.32 $\to$ **0.0218** (-99.9%) |
| **自稳恢复 (Recovery)** | 40 ~ 139 | +0.2804 | 17.42 | **20.73** | 20.73 | +1.2339 | **+0.4324** | 13.36 $\to$ **0.9076** (-93.2%) |

> [!NOTE]
> 评测证实：正统五种子轨迹下的 MSE 改善幅度（-95.6%）与此前报告完全一致。跳跃核心阶段（推蹬、上升、下降、触地）的价值预测 MSE 全部压降至 **0.02 ~ 0.44**，彻底纠正了 Champion 严重低估回报的系统性偏差。

---

## 4. 任务三：价值目标固定基准与无限视界语义复核

针对价值监督目标 $G_t$ 的构造与更新机制，复核确认如下事实：
1. **终态自举值 $V(s_{140}^{\text{pre}})$ 的真实来源**：
   在离线生成数据与在线评估中，终态价值**严格由原始 Champion Critic Checkpoint (`model_champion.pt`，SHA-256 `fad590f4...`) 单次前向推理生成**，而非来自正在训练中的当前 Critic。
2. **目标静态性与非漂移性**：
   在 Phase 4.7 的整个 Critic 校准过程中，监督目标张量 `targets_b` 始终作为静态固定的回归标签，**从未随被训练的 Critic 动态自举更新**。这保证了监督信号是严格平稳（Stationary）的，消除了强化学习自举中常见的半梯度发散风险。
3. **有限视界逼近与无限视界真值区分**：
   - 本阶段报告的 MSE（0.66 ~ 0.73）反映的是新 Critic 对**“以 Champion 终态为截断锚点的 140 步折现累计回报”**的拟合误差；
   - 它证明了 Critic 能够精确理解 Phase 4 任务的 140 步前向跳跃与着陆恢复动态；
   - 不能将该拟合误差盲目等同于“真实无限视界贝尔曼最优价值误差”，因为末端锚点本身带有 Champion 的估计偏置。但在当前冻结 Actor 且末端机器人已平稳自稳的前提下，该锚点提供了足够稳定、信噪比极高的基准。

---

## 5. 任务四：共享 RSL-RL 库向后兼容性验证

针对修改共享算法库 [`rsl_rl/algorithms/ppo.py`](file:///home/xy/IsaacTu_jump/rsl_rl/rsl_rl/algorithms/ppo.py) 的安全性与向后兼容性进行核查：

```python
# rsl_rl/algorithms/ppo.py L216-L225:
if 'time_outs' in infos:
    time_outs = infos['time_outs'].unsqueeze(1).to(self.device)
    if 'terminal_observation' in infos:
        # Phase 4 独立环境生效：采用真实的重置前终态观测进行准确自举
        with torch.no_grad():
            terminal_critic_obs = infos['terminal_observation']
            terminal_values = self.actor_critic.evaluate(terminal_critic_obs.to(self.device)).detach()
        self.transition.rewards += self.gamma * torch.squeeze(terminal_values * time_outs, 1)
    else:
        # 历史其他任务兜底：当环境未传导 terminal_observation 时，100% 保持原生 RSL-RL 原样行为
        self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_outs, 1)
```

- **兼容性判定**：**完全向后兼容**。
- **说明**：只有重载了 `post_physics_step()` 并主动向 `infos` 注入 `'terminal_observation'` 的独立 Phase 4 环境才会触发精确的 Target B 自举；代码库中的所有其他历史环境（如平衡策略、旧跳跃环境）因 `infos` 中无此键，均自动走 `else` 分支，执行原生 RSL-RL 逻辑，不会受到任何影响。

---

## 6. Phase 4.8 推进准许与实施规范

### 6.1 审查清单（Checklist）
- [x] **PPO 第 140 步 Timeout 真实生效**：Step 138-140 张量记录完整，断言全过。
- [x] **多环境超时掩码正确**：4 环境复合测试最大误差 $0.00\text{e}+00$，零跨环境泄漏。
- [x] **观测函数无状态副作用**：连续多次调用变异为 $0.00\text{e}+00$。
- [x] **五种子开发集物理轨迹统一**：使用正统管道完美复现 12.66 cm 基线。
- [x] **Critic 泛化能力充分证实**：在正统开发集上 MSE 达 0.7280 (-95.6%)，EV 达 87.6%。
- [x] **价值目标基准源明确**：确认来自 Champion，训练全程静态固定。
- [x] **Actor 绝对冻结保持**：权重、标准差、确定性动作输出差异严格为 $0.00\text{e}+00$。
- [x] **共享库向后兼容**：双分支守卫，旧任务行为绝对不受影响。

### 6.2 最终裁决与准许意见
* **裁决等级**：**`PASS`**
* **准许事项**：
  1. **正式归档**：批准保留 [`model_critic_calibrated_target_b.pt`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase4_7_critic_calibration/model_critic_calibrated_target_b.pt) 作为下一阶段 Actor PPO 的正式 Critic 初始化资产；
  2. **准许推进**：**准许进入 Phase 4.8 开展 Actor PPO Pilot 更新**。
* **Phase 4.8 实施规范**：
  - **极保守学习率**：建议采用 $\text{lr} = 5.0\times 10^{-5} \sim 1.0\times 10^{-4}$，PPO clip $\epsilon = 0.1$；
  - **小步验证**：首轮仅执行 1～3 Iterations，严密监控五种子起跳仰角与自稳恢复成功率；
  - **奖励权重处理**：在用户确认前，严格保持现有 Sigmoid 距离奖励权重与起跳速度奖励权重不变。

---
*报告归档时间：2026-10-17*  
*执行责任模型：Antigravity Agentic Assistant*
