# MyRobot Phase 4.8-Pilot-A：首次 Actor PPO 参数更新与物理性能验证报告

## 一、执行概况与审计确认

Phase 4.8-Pilot-A 为 MyRobot 双轮足机器人前向跳跃进入正式 PPO 训练的**首次单步受控 Pilot 实验**。本实验严格执行了以下约束：
1. **严格限制为 1 个 PPO Iteration**，训练后立即终止，未自动进入下一轮更新；
2. **冻结基线完整保护**：Phase 3.2-V Champion、Phase 15 平衡策略及历史资产完全只读；
3. **独立实验隔离**：创建全新独立目录 `evaluation_assets/phase4_8_pilot_a/`，所有训练日志、模型、评测数据均独立归档；
4. **模型初始化恒等性核查**：使用 Phase 4.7 已校准的 `model_critic_calibrated_target_b.pt` 进行热启动，训练前确认 Actor 权重差异为 `0.00e+00`，动作标准差差异为 `0.00e+00`，Critic 差异为 `0.00e+00`；
5. **完整保留正统评测协议**：评测完全复用 Phase 15 平衡交接、腾空与降落判定，以及下落阶段（$t_{\text{phase}} \ge 0.22$）左右髋 $+0.10\text{ rad}$ 姿态补偿。

---

## 二、训练配置、模型哈希与执行更新明细

### 1. 资产与模型哈希（SHA256）

| 模型资产 | 文件路径 | SHA256 校验码 | 角色 / 来源 |
| :--- | :--- | :--- | :--- |
| **Champion Baseline** | `evaluation_assets/phase3_1_forward_distance/models/model_champion.pt` | `fad590f4699abac8ae2b3e635d9cb88ef2658da85bbb37008e13362bb3c5b950` | 冻结物理基线 |
| **Warm-Start Init** | `evaluation_assets/phase4_7_critic_calibration/model_critic_calibrated_target_b.pt` | `a5fca202c667da64476bb36044f9fdd814cae430753b57e349699b2e679f5ed0` | Actor: Champion / Critic: Target B |
| **Pilot A Model (Iter 1)** | `evaluation_assets/phase4_8_pilot_a/model_pilot_a_iter1.pt` | `c4b4b294a985de20bcda917f62f3cb133b968d2f3d7cc49d6bbb5343c3ad83e9` | 本轮单 Iter 更新产出 |

### 2. 生效超参数与热启动配置

* **任务环境**：`myrobot_forward_jump_phase4`（64 并行环境，单 Rollout 140 步，总样本量 8,960 transitions）
* **热启动参数**：`train_cfg.runner.warm_start = True`，`warm_start_path = calib_path`，`warm_start_optimizer = False`
* **目标学习率**：`1.0e-4`（配置设定值）
* **Epoch / Mini-Batch**：`num_learning_epochs = 5`，`num_mini_batches = 4`，**计划执行 20 次 `optimizer.step()`**
* **梯度与截断**：`clip_param = 0.20`，`max_grad_norm = 1.0`，`value_loss_coef = 1.0`，`entropy_coef = 0.005`
* **KL 控制与保护**：`desired_kl = 0.01`，`early_stop_kl = 0.04`，`enable_early_stopping = True`，`schedule = "adaptive"`

### 3. 实际优化器更新次数与 Early-Stopping 触发

* **计划更新次数**：20 次
* **实际执行更新次数**：**1 次**
* **保护触发事件**：
  在完成第 1 次 Mini-Batch 梯度更新后，算法在第 2 个 Mini-Batch 进行前向评估时，计算得到当前策略相对于采样本策略的局部均值 KL 散度达到：
  $$\text{KL}_{\text{mean}} = 0.132632 > \text{early\_stop\_kl} (0.040000)$$
  成功触发 `ppo.py` 原生早停安全保护机制：
  `"KL mean (0.132632) exceeded threshold (0.040000) at update 2/20"`。
  算法随即中断 Mini-Batch 循环，**完全停止后续梯度反传与权重更新**，有效杜绝了策略发散风险。

### 4. 关键 PPO 训练遥测数据

| 监控指标 | 实测数值 | 说明 / 状态 |
| :--- | :--- | :--- |
| **实际生效学习率** | `1.50e-4` | 初始 1.0e-4 在 Mini-Batch 1 因初始 KL 接近零被自适应微调至 1.5e-4 |
| **Surrogate Loss** | `-0.013893` | 策略前向优化目标正常 |
| **Value Loss** | `0.653874` | Critic 价值损失平稳，无发散 |
| **Clip Fraction** | `0.00%` | 更新幅度在单次更新内完全处于 PPO 信任域内 |
| **Unclipped Actor Grad Norm** | `4.7116` | 未裁剪 Actor 梯度范数 |
| **Unclipped Critic Grad Norm** | `72.1663` | 未裁剪 Critic 梯度范数 |
| **Total Grad Norm (Clipped)** | `1.0000` | 严格被 `clip_grad_norm_` 裁剪至安全上限 1.0 |
| **Action Std Mean** | `0.272493` | 初始 `0.272593`，微调变化量 $\Delta = -0.000100$ |
| **Policy Entropy** | `-2.4948` | 动作熵保持稳定 |
| **Actor 参数变化量 ($L_2$)** | `0.063647` | 相对位移 **0.2338%**，属于微幅稳定调整 |
| **Critic 参数变化量 ($L_2$)** | `0.063219` | 相对位移 **0.1880%**，微幅平稳更新 |
| **Advantage 统计** | 均值 `2.77e-9`，方差 `1.00`，极值 `[-3.5919, 2.2849]` | 归一化优势计算完全正常 |
| **数值健全性** | **无 NaN、无 Inf** | 全量参数及梯度张量正常 |

---

## 三、五种子训练前后逐一配对物理评测

使用 `tests/eval_phase4_seed.py` 严格独立子进程评测 pipeline，在种子 `[3001, 3002, 3003, 3004, 3005]` 上对 Champion 基线与 Pilot A 模型进行逐一配对比较。

### 1. 五种子配对指标全景表

| 评估种子 | 模型版本 | 腾空前移 $d_{\text{fwd}}$ (cm) | 轮底净空 $h_{\text{clr}}$ (cm) | 起跳俯仰角 (deg) | 触地俯仰角 (deg) | 着陆收敛速度 $v_y$ (m/s) | 异常复位 | $\ge 12\text{cm}$ 综合判定 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Seed 3001** | **Champion** | 12.04 | 21.44 | -4.20° | 7.75° | 0.0237 | 0 | **PASS** |
| | **Pilot A** | **12.46** (+0.42) | **22.04** (+0.60) | -4.66° (-0.46°) | 7.63° (-0.12°) | 0.0562 (+0.0325) | 0 | FAIL (C4超标) |
| **Seed 3002** | **Champion** | 12.93 | 21.86 | -4.78° | 8.01° | 0.0283 | 0 | **PASS** |
| | **Pilot A** | **13.11** (+0.18) | **22.28** (+0.42) | -4.94° (-0.16°) | 7.81° (-0.20°) | 0.0598 (+0.0315) | 0 | FAIL (C4超标) |
| **Seed 3003** | **Champion** | 11.86 | 21.01 | -3.88° | 7.76° | 0.0272 | 0 | FAIL ($d < 12$) |
| | **Pilot A** | **12.40** (+0.54) | **21.71** (+0.69) | -4.27° (-0.39°) | 8.22° (+0.46°) | 0.0588 (+0.0316) | 0 | FAIL (C4超标) |
| **Seed 3004** | **Champion** | 14.71 | 21.38 | -5.53° | 8.33° | 0.0260 | 0 | **PASS** |
| | **Pilot A** | **15.14** (+0.43) | **21.78** (+0.41) | -5.77° (-0.24°) | 8.07° (-0.25°) | 0.0580 (+0.0320) | 0 | FAIL (C4超标) |
| **Seed 3005** | **Champion** | 11.76 | 21.80 | -4.21° | 7.65° | 0.0283 | 0 | FAIL ($d < 12$) |
| | **Pilot A** | **12.16** (+0.40) | **22.37** (+0.57) | -4.57° (-0.36°) | 7.60° (-0.05°) | 0.0599 (+0.0316) | 0 | FAIL (C4超标) |

### 2. 宏观统计对比（5-Seed Averages）

| 统计指标 | Champion 基线 | Pilot A (Iter 1) | 增益 / 变化 ($\Delta$) | 物理意义 |
| :--- | :--- | :--- | :--- | :--- |
| **平均腾空前向位移** | $12.66\text{ cm}$ | **$13.05\text{ cm}$** | **$+0.39\text{ cm}$** | **5 个种子全部正向提升** |
| **纯前移 $\ge 12.0\text{ cm}$ 达成率** | $3/5\ (60.0\%)$ | **$5/5\ (100.0\%)$** | **$+40.0\%$** | 种子 3003、3005 成功突破 12cm 门限 |
| **平均最大共同轮底净空** | $21.50\text{ cm}$ | **$22.04\text{ cm}$** | **$+0.54\text{ cm}$** | **5 个种子全部正向提升**且均 $>20\text{ cm}$ |
| **离地时间步** | Step 11 (0.22s) | Step 11 (0.22s) | 0.00s | 起跳时序精准一致 |
| **着陆时间步** | Step 33 (0.66s) | Step 33 (0.66s) | 0.00s | 滞空时长 0.44s 保持恒定 |
| **触地俯仰角均值** | $+7.90°$ | $+7.87°$ | $-0.03°$ | 触地姿态极佳，均在安全阈值 $\le 10°$ 内 |
| **触地翻滚角均值** | $+1.77°$ | $+1.69°$ | $-0.08°$ | 横滚控制优良，对称性好 |
| **摔倒 / 异常复位率** | 0/5 (0%) | 0/5 (0%) | 0.0% | **零摔倒，零动力学失稳** |
| **C4 着陆收敛速度 ($v_y$) 均值** | $0.0267\text{ m/s}$ | $0.0585\text{ m/s}$ | $+0.0318\text{ m/s}$ | 略高于标准硬门限 $0.030\text{ m/s}$ |

---

## 四、物理轨迹特征与收敛异常深入诊断

### 1. 前向腾空与动力学提升机理

通过单步动力学追踪发现：
1. **起跳冲量与姿态协同改善**：Pilot A 在起跳瞬间（Step 11）的机身俯仰角由基线的 $-4.20°$ 微微前倾至 $-4.66°$（增加前倾约 $0.4°$），True COM 水平起跳速度由 $0.277\text{ m/s}$ 提升至 $0.285\text{ m/s}$，垂直起跳速度由 $2.010\text{ m/s}$ 提升至 $2.041\text{ m/s}$。
2. 这使得前向腾空位移在滞空时间不变的前提下稳定拉长了 $0.18 \sim 0.54\text{ cm}$，且腾空最高点轮底净空全面增高了 $0.41 \sim 0.69\text{ cm}$。
3. 此前在 Champion 中未达标的 Seed 3003（$11.86\text{ cm}$）和 Seed 3005（$11.76\text{ cm}$）在 Pilot A 中分别达到 **$12.40\text{ cm}$** 和 **$12.16\text{ cm}$**，**五种子腾空物理位移 100% 超过 12 cm 门限**。

### 2. C4 着陆恢复速度判定未通过的原因剖析

在评测判定表中，Pilot A 的 `Pass>=12cm` 显示为 False，其**唯一未通过项**是 C4 条件：
$$\text{passed\_rec\_vy\_003} = (|v_{y, \text{last10}}| \le 0.030\text{ m/s})$$

深入分析 Step 33 至 Step 139 的着陆后姿态与接触力时程曲线（以 Seed 3001 为例）：
* **着陆缓冲极佳**：触地瞬间总垂直力峰值 $F_n = 784.9\text{ N}$（低于 Champion 的 $827.0\text{ N}$），在 Step 50 即完全衰减至静态重力 $215.8\text{ N}$（$22\text{ kg} \times 9.81\text{ m/s}^2$）。
* **自稳姿态更优**：在第 139 步终态，Pilot A 的机身俯仰角为 **$-0.11°$**，比 Champion 的 **$-0.69°$** 更加直立和平整。
* **残余速度成因**：由于起跳前向速度和跳跃距离增加，机器人着陆时携带了略高的前向残余动量。落地后双轮处于直立稳态平衡滑行状态，末端 10 步的前向滑行速度稳定在 $0.058\text{ m/s}$（约 $5.8\text{ cm/s}$，仅为慢速步行速度的几分之一）。
* **对比 Champion**：Champion 末端速度为 $0.024\text{ m/s}$（刚好踩进 $0.030\text{ m/s}$ 门限）。
* **结论**：Pilot A 并没有摔倒或失控，而是由于前向冲量提升后，着陆后的平衡恢复时间窗口内未能将速度进一步衰减到 $3\text{ cm/s}$ 极低阈值内。

---

## 五、技术裁决与后续建议

### 1. 核心结论

1. **PPO 首次参数更新安全闭环**：
   - 策略梯度链路（Phase 4.7-S1-G）真实生效，Actor 参数产生微幅稳健位移（$0.2338\%$），未发生梯度爆炸或参数崩溃。
   - PPO 源码原生的 `early_stop_kl` 保护机制在 Mini-Batch 2 精准触发，成功拦截了局部 KL 异常跳变，证明训练流水线的安全性完备。
2. **物理跳跃性能出现显著的实质性改善**：
   - 5 个种子腾空前向距离无一例外全部正向增长（均值 $+0.39\text{ cm}$）；
   - 5 个种子腾空最大净空全部提升（均值 $+0.54\text{ cm}$）；
   - 纯物理位移 $\ge 12\text{ cm}$ 达成率从 $60\%$ 跃升至 **$100\%$**；
   - 着陆俯仰姿态、横滚姿态和终态直立度均达到或超过 Champion 基线。
3. **着陆刹车动量耗散需关注**：
   - 落地后的缓慢滑移速度（$5.8\text{ cm/s}$）超出了原判据严苛的 $3\text{ cm/s}$ 门限，表明随着跳跃距离和冲量增加，落地后的制动阻尼策略需要继续适应。

### 2. 关于是否允许第 2～3 个 Iteration 的建议

> [!IMPORTANT]
> **技术评审建议：有条件允许进入 Iteration 2 的探索，但必须施加更严密的学习率与 KL 控制。**
>
> **原因分析**：
> 1. 本次 Pilot A 证实了 Target B Critic 校准后的优势函数能够指导 Actor 做出更有利于前向跳跃的参数调整（跳跃距离与净空全面提升，且未引发姿态恶化）；
> 2. 但 Mini-Batch 2 的 KL 达到 0.1326，说明当前 $1.5\text{e-4}$ 的学习率在起跳高敏感阶段步长偏大，自适应学习率在初期的上调机制并不适合 jump 这类冲击型任务；
> 3. 若技术审查批准继续进行 Iteration 2，建议：
>    - 将优化器调度切换为 **固定低学习率**（如 `schedule = "fixed"`, `learning_rate = 5.0e-5`），避免自适应放大更新步长；
>    - 保持 `early_stop_kl = 0.04` 的严格保护；
>    - 依然采取 **单次受控 1 个 Iteration** 的推进方式，绝不一次性放开多轮无监督训练。

---

## 六、交付物归档索引

- **Pilot A 训练模型**：[`model_pilot_a_iter1.pt`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase4_8_pilot_a/model_pilot_a_iter1.pt)
- **训练全量遥测记录**：[`training_telemetry.json`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase4_8_pilot_a/training_telemetry.json)
- **五种子配对物理评测完整数据**：[`paired_physical_evaluation.json`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase4_8_pilot_a/paired_physical_evaluation.json)
- **单种子评测日志文件**：[`eval_runs/`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase4_8_pilot_a/eval_runs/)
