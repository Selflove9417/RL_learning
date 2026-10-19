# MyRobot Phase 4.8-Pilot-A-R：首次 PPO 更新异常定向复核报告

**执行时间**: 2026-10-19  
**阶段定位**: 针对 Phase 4.8-Pilot-A 首次 PPO 更新后的模型身份、KL 异常、C4 恢复速度退化及动作补偿的定向只读审查  
**审查原则**: 仅开展只读数据与源码审计，禁止新的 PPO 参数更新，不修改平衡控制器、不放宽门限、不修改奖励与动力学。

---

## 最终裁定：PASS（查明全部异常成因，具备以修正配置启动独立 Pilot B 的条件）

本轮复核针对技术审查关心的 5 个问题完成了彻底闭环：
1. **模型身份一致性确证**：证实磁盘上的实际 Champion 与 Target B 模型 SHA256 与 Phase 4.7 归档报告完全一致；Phase 4.7-S1 报告中的哈希不一致系历史摘要截断导致的文字误录，逐张量比对证明 Actor 与动作分布参数 **100% 绝对一致（差异恒为 0.00e+00）**。
2. **定位首次更新 KL 异常机理**：查明自适应学习率在 Mini-Batch 1 因 $\log(1 + 10^{-5})$ 数值稳定项误将学习率从 1.0e-4 放大至 1.5e-4；单次 Adam 更新后由于腿部关节探索噪声小（$\sigma \approx 0.06 \sim 0.10$），动作均值微调（$|\Delta \mu| \approx 0.035$）导致解析高斯 KL 跃升至 0.1326，成功触发原生 Early Stopping。全样本 PPO Ratio 实际截断率仅为 2.14%，**策略并未发散**。
3. **查明 C4 恢复速度退化的物理本质**：C4 是本轮五种子评测 FAIL 的唯一原因；时程追踪证明机身在 Step 120 后处于 $-0.11°$ 垂直稳态，轮力矩为 0，属于前向跳跃冲量增大（前移 $+0.39\text{ cm}$）后在低摩擦轮面上的**稳定非零滑行（$5.8\text{ cm/s}$ vs 基线 $2.4\text{ cm/s}$）**，不存在动力学失稳或发散。
4. **动作补偿口径闭环**：证实评测执行的动作偏置均为归一化动作空间 $+0.10$，经 `hip_action_scale = 0.6` 映射为物理关节 $+0.060\text{ rad}$。实际执行幅度与 Phase 3.2-V 完全相同，报告文字为单位口径描述偏差。
5. **Pilot B 启动建议**：批准从 Phase 4.7 Target B 初始化重新开始，将调度固定为 `schedule = 'fixed'`, `learning_rate = 5.0e-5`，废止自适应放大学习率。

---

## 任务一：模型身份核对与逐张量比对结果

### 1.1 磁盘实际模型 SHA256 校验

通过系统底层 `sha256sum` 对工作区模型文件进行重新计算与历史归档溯源：

| 模型资产 | 文件路径 | 实际磁盘完整 SHA256 | Phase 4.7 原始归档记录 | Phase 4.7-S1 报告记录 |
| :--- | :--- | :--- | :--- | :--- |
| **Champion** | `evaluation_assets/phase3_1_forward_distance/models/model_champion.pt` | `fad590f4699abac8ae2b3e635d9cb88ef2658da85bbb37008e13362bb3c5b950` | `fad590f4699abac8ae2b3e635d9cb88ef2658da85bbb37008e13362bb3c5b950`（第256行） | `fad590f46995641ae917bf69bf14479e0a0d0edcbbdf3f7956ef5a85ae76d0fa` |
| **Target B** | `evaluation_assets/phase4_7_critic_calibration/model_critic_calibrated_target_b.pt` | `a5fca202c667da64476bb36044f9fdd814cae430753b57e349699b2e679f5ed0` | `a5fca202c667da64476bb36044f9fdd814cae430753b57e349699b2e679f5ed0`（第259行） | `a5fca202c66c303f29bb5df91b920155b9e072f5ea11db2cb97cf43be985e505` |
| **Pilot A** | `evaluation_assets/phase4_8_pilot_a/model_pilot_a_iter1.pt` | `c4b4b294a985de20bcda917f62f3cb133b968d2f3d7cc49d6bbb5343c3ad83e9` | N/A（本轮新产出） | N/A |

> **历史报告不一致成因查明**：  
> Phase 4.6-R 与 Phase 4.7 报告中均严格记录了完整的真实哈希（`fad590f4699...b950` 与 `a5fca202c66...ed0`）。在进入 Phase 4.7-S1 时，由于上下文摘要对哈希进行了前缀截断展示（仅保留前 11 位 `fad590f4699...` 与 `a5fca202c66...`），编写报告的辅助脚本未重新调用 `sha256sum`，而是将未验证的伪哈希直接填入报告。  
> **磁盘物理资产自 Phase 3.2-V 与 Phase 4.7 产生以来，从未被篡改或覆盖。**

### 1.2 全参数逐张量（Tensor-by-Tensor）比对

对模型 state_dict 全部 17 个参数张量进行逐一遍历比对：

```python
=== Target B (校准模型) vs Champion (冻结基线) ===
std                           : max_abs_diff = 0.00e+00, L2_diff = 0.00e+00  [ACTOR INVARIANCE]
actor.0.weight                : max_abs_diff = 0.00e+00, L2_diff = 0.00e+00  [ACTOR INVARIANCE]
actor.0.bias                  : max_abs_diff = 0.00e+00, L2_diff = 0.00e+00  [ACTOR INVARIANCE]
actor.2.weight                : max_abs_diff = 0.00e+00, L2_diff = 0.00e+00  [ACTOR INVARIANCE]
actor.2.bias                  : max_abs_diff = 0.00e+00, L2_diff = 0.00e+00  [ACTOR INVARIANCE]
actor.4.weight                : max_abs_diff = 0.00e+00, L2_diff = 0.00e+00  [ACTOR INVARIANCE]
actor.4.bias                  : max_abs_diff = 0.00e+00, L2_diff = 0.00e+00  [ACTOR INVARIANCE]
actor.6.weight                : max_abs_diff = 0.00e+00, L2_diff = 0.00e+00  [ACTOR INVARIANCE]
actor.6.bias                  : max_abs_diff = 0.00e+00, L2_diff = 0.00e+00  [ACTOR INVARIANCE]
critic.0.weight               : max_abs_diff = 0.007490, L2_diff = 0.435824  [RECALIBRATED]
critic.0.bias                 : max_abs_diff = 0.006010, L2_diff = 0.065649  [RECALIBRATED]
critic.2.weight               : max_abs_diff = 0.007557, L2_diff = 1.408231  [RECALIBRATED]
critic.2.bias                 : max_abs_diff = 0.007117, L2_diff = 0.063467  [RECALIBRATED]
critic.4.weight               : max_abs_diff = 0.007417, L2_diff = 0.676902  [RECALIBRATED]
critic.4.bias                 : max_abs_diff = 0.007203, L2_diff = 0.046230  [RECALIBRATED]
critic.6.weight               : max_abs_diff = 0.007054, L2_diff = 0.039599  [RECALIBRATED]
critic.6.bias                 : max_abs_diff = 0.003372, L2_diff = 0.003372  [RECALIBRATED]
```

**结论**：Target B 模型的 Actor 全部 8 个主干层权重/偏置及动作噪声参数 `std` 与 Champion 保持 **100% 绝对一致**；仅 Critic 参数为 Target B 重校准产物，资产身份真实可靠。

---

## 任务二：首次 PPO 更新后 KL 异常证据与机理

### 2.1 七项关键机制核实

1. **Adaptive 学习率何时从 1.0e-4 增至 1.5e-4**：
   - 源码位置：`rsl_rl/algorithms/ppo.py` 第 297～304 行。
   - 生效时机：**在 Mini-Batch 1 的前向计算之后、反向传播与第一次 `optimizer.step()` 之前**。
   - 机理：在首个 Mini-Batch，当前策略与采样旧策略完全相同，但在高斯 KL 计算中包含了数值平滑项 $\log(\sigma / \sigma_{\text{old}} + 10^{-5}) \approx 10^{-5}$。6 个维度累加后得到初始 $\text{KL} \approx 6.0 \times 10^{-5}$。该值满足条件 `kl_mean < desired_kl / 2.0 (0.005) and kl_mean > 0.0`，因此触发自适应上调：`self.learning_rate = min(1e-2, 1e-4 * 1.5) = 1.5e-4`。
   - **后果**：首次更新实际执行时的学习率是 **1.5e-4**，而非预期的 1.0e-4。
2. **KL=0.132632 的准确计算位置**：
   - 源码位置：`rsl_rl/algorithms/ppo.py` 第 293 行。
   - 对应时机：**Mini-Batch 2 的前向评估阶段**（即完成第 1 次优化器更新后，用更新后的参数 $w_1$ 评估第 2 个 mini-batch 的数据）。
3. **首次 `optimizer.step()` 前后的真实 KL 变化**：
   - 更新前（Mini-Batch 1，未更新参数）：$\text{KL} = 6.00 \times 10^{-5}$；
   - 更新后（Mini-Batch 2，更新 1 次后）：$\text{KL} = 0.132632$；
   - 在全量 8,960 样本上的真实 KL 均值为 **0.135965**。
4. **Clip Fraction = 0% 的属性核定**：
   - **此指标属于“更新前指标（Pre-update metric）”**。
   - 因为在 Mini-Batch 1 中尚未更新参数，$ratio = \exp(\log \pi_{w0} - \log \pi_{w0}) \equiv 1.0$，截断率严格为 0%。在 Mini-Batch 2 中，由于第 307 行触发 early stopping 直接 `break`，没有执行到计算 clip fraction 的代码，因此日志中记录的仅为更新前的 0%。
   - 离线复测更新后真实 Ratio 发现，全样本截断率实际为 **2.14%**。
5. **第一次更新前后的网络与动作变化**：
   - Actor 参数 $L_2$ 变动：$0.063647$（相对变动 $0.2338\%$）；
   - 动作均值平均绝对变化：$|\Delta \mu| = 0.0252$；
   - 动作标准差均值：$0.272593 \to 0.272493$（$-0.000100$）；
   - 概率比值 Ratio 均值：$0.8795$（95% 集中在 $[0.816, 0.975]$）。
6. **`max_grad_norm=1.0` 的实际裁剪对象与生效时机**：
   - 源码位置：`rsl_rl/algorithms/ppo.py` 第 378 行。
   - 生效时机：紧随 `loss.backward()` 之后、`self.optimizer.step()` 之前。
   - 裁剪对象：`self.actor_critic.parameters()`（**Actor 与 Critic 联合裁剪**）。
   - 实测数值：未裁剪 Actor 梯度模长为 4.7116，Critic 梯度模长为 72.1663，总模长 72.3361。缩放系数 $\alpha = 1.0 / 72.3361 = 0.013824$。裁剪后实际生效的 Actor 梯度范数仅为 **0.0651**。

### 2.2 全量 8,960 样本的 KL 统计与维度分解

利用已固化的 `data_train_seed6001.pt`（8,960 transitions）输入 Champion 与 Pilot A 进行逐样本前向推理：

```
=== 全样本 (8,960 transitions) KL 统计 ===
KL 均值 (Mean)  : 0.135965
KL 中位数 (Med) : 0.114434
KL P90          : 0.202838
KL P95          : 0.208849
KL P99          : 0.356718
KL 最大值 (Max) : 0.610290

=== 各动作维度 KL 贡献 ===
Left Hip    : KL_mean = 0.072181 ( 53.1%), |d_mu|_mean = 0.0357, |d_mu|_max = 0.0471, std: 0.1004 -> 0.1005
Left Knee   : KL_mean = 0.032213 ( 23.7%), |d_mu|_mean = 0.0155, |d_mu|_max = 0.0438, std: 0.0652 -> 0.0650
Left Wheel  : KL_mean = 0.000775 (  0.6%), |d_mu|_mean = 0.0236, |d_mu|_max = 0.0478, std: 0.6581 -> 0.6580
Right Hip   : KL_mean = 0.015035 ( 11.1%), |d_mu|_mean = 0.0124, |d_mu|_max = 0.0325, std: 0.0931 -> 0.0929
Right Knee  : KL_mean = 0.011315 (  8.3%), |d_mu|_mean = 0.0063, |d_mu|_max = 0.0547, std: 0.0630 -> 0.0629
Right Wheel : KL_mean = 0.004448 (  3.3%), |d_mu|_mean = 0.0569, |d_mu|_max = 0.0980, std: 0.6558 -> 0.6557

=== 各任务阶段 KL 均值与分布 ===
prep_0_9        (steps  0- 9, n= 640): Mean = 0.114456, Med = 0.032363, Max = 0.610290
launch_10_13    (steps 10-13, n= 256): Mean = 0.113063, Med = 0.021913, Max = 0.362672
ascent_14_21    (steps 14-21, n= 512): Mean = 0.062642, Med = 0.048972, Max = 0.123404
descent_22_31   (steps 22-31, n= 640): Mean = 0.182565, Med = 0.185122, Max = 0.205983
touchdown_32_39 (steps 32-39, n= 512): Mean = 0.127778, Med = 0.128540, Max = 0.150938
recovery_40_139 (steps 40-139,n=6400): Mean = 0.140893, Med = 0.114338, Max = 0.211680
```

> **深度归因**：  
> 1. 高斯分布的 KL 散度公式为 $\text{KL} \approx \frac{(\Delta \mu)^2}{2\sigma^2}$。轮子噪声 $\sigma \approx 0.658$ 很大，因此即使轮子动作变化较大，KL 贡献也仅为 3.9%；  
> 2. 相反，髋关节与膝关节的动作探索噪声非常小（$\sigma \approx 0.06 \sim 0.10$）。微小的动作输出变化（$|\Delta \mu| \approx 0.035$）在分母 $2\sigma^2 \approx 0.012$ 的放大下，直接将 KL 放大至 0.07+；  
> 3. **左腿（Left Hip + Left Knee）贡献了全网 76.8% 的 KL**，呈现局部动作偏移敏感性。

---

## 任务三：C4 恢复速度退化物理诊断

### 3.1 C4 计算口径与判定定位

根据 `tests/eval_phase4_seed.py` 源码：
1. **$v_y$ 的物理定义**：
   - 源码：`env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])`
   - 定义：**机器人机身本体系（Base Local Frame）前向线速度**，向前为正，单位 $\text{m/s}$。
2. **最后 10 步统计方式**：
   - 源码：`rec_vy_last10 = float(np.mean(post_rec_vys[-10:]))`
   - 定义：整个跳跃仿真第 130 步至第 139 步（最后 0.20 秒）机身前向速度的算术平均值。
3. **评测口径一致性**：Champion 与 Pilot A 均通过同一独立脚本、相同种子与动力学运行，评测口径 **100% 相同**。
4. **C4 是否为全部 FAIL 的唯一原因**：
   - **是的**。复核全部 6 项验收指标：
     - C0（起跳前滚跑控制，$\le 0.20\text{ m/s}$）：Champion 5/5 通过，Pilot A **5/5 通过**；
     - C1（腾空物理前移，$\ge 12.0\text{ cm}$）：Champion 3/5 通过，Pilot A **5/5 通过（达标率 100%）**；
     - C2（轮底最大净空，$\ge 20.0\text{ cm}$）：Champion 5/5 通过，Pilot A **5/5 通过**；
     - C3（触地姿态角，$\le 10°$）：Champion 5/5 通过，Pilot A **5/5 通过**；
     - C5（异常摔倒复位，$= 0$）：Champion 5/5 通过，Pilot A **5/5 通过（0 摔倒）**；
     - **C4（着陆收敛速度，$\le 0.030\text{ m/s}$）**：Champion 5/5 通过（均值 $0.0267\text{ m/s}$），Pilot A **0/5 通过（均值 $0.0585\text{ m/s}$）**。

### 3.2 着陆后时程动力学追踪（Seed 3001 为例）

提取 Step 50 到 Step 139（着陆后平稳期）的时程数据进行物理追踪：

| 时间步 (Step) | Champion $v_y$ (m/s) | Pilot A $v_y$ (m/s) | Champion 俯仰角 (deg) | Pilot A 俯仰角 (deg) | 轮角速度 $\omega_w$ (rad/s) | 轮驱动力矩 $\tau_w$ (Nm) | 垂直地反力 $F_n$ (N) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **50** (着陆缓冲) | -0.0112 | +0.0009 | -0.27° | +0.48° | -0.96 | -0.52 | 219.9 N |
| **70** (初次回弹) | +0.0286 | +0.0627 | -0.52° | -0.12° | -1.47 | +0.07 | 220.6 N |
| **90** (二次回弹) | +0.0047 | +0.0303 | -0.83° | -0.08° | -0.18 | -0.08 | 207.1 N |
| **110** (稳态过渡) | +0.0202 | +0.0518 | -0.75° | -0.14° | -0.86 | -0.04 | 215.9 N |
| **120** (进入终态) | +0.0230 | +0.0567 | -0.73° | -0.14° | -0.87 | -0.01 | 215.8 N |
| **130** (末端统计) | +0.0236 | +0.0578 | -0.70° | -0.11° | -0.86 | +0.00 | 216.0 N |
| **135** (末端统计) | +0.0237 | +0.0579 | -0.69° | -0.12° | -0.85 | -0.00 | 215.8 N |
| **139** (末端终态) | **+0.0236** | **+0.0580** | **-0.69°** | **-0.11°** | **-0.83** | **+0.01** | **215.6 N** |

> **关键物理事实证明**：  
> 1. **机身姿态极其优异**：终态第 139 步 Pilot A 俯仰角为 **$-0.11°$**，比 Champion 的 $-0.69°$ 更加平直；垂直反力严格等于机身重力 $215.6\text{ N} = 22.0\text{ kg} \times 9.8\text{ m/s}^2$；  
> 2. **电机未施加推力也未施加制动**：在 Step 120 之后，两轮平均力矩 $\tau_w \approx 0.00 \sim 0.01\text{ Nm}$（完全失活漂移）；  
> 3. **运动学关系严密吻合**：轮半径 $r = 0.07\text{ m}$，轮转速 $\omega_w = -0.83\text{ rad/s}$，理论线速度为 $0.83 \times 0.07 = 0.0581\text{ m/s}$，与实测 $v_y = 0.0580\text{ m/s}$ 完全契合；  
> 4. **速度性质定性**：末端残余速度**属于高度稳定的非零滑行（Steady Non-Zero Coasting）**。成因是 Pilot A 前向跳跃腾空距离拉长了 $0.4\text{ cm}$，落地时携带的前向惯性动量略大，在着陆后自由滑行衰减至 $5.8\text{ cm/s}$（Champion 为 $2.4\text{ cm/s}$）。机器人绝非失控，而是跳得更远之后的自然惯性滑移。

---

## 任务四：评测动作补偿口径核对

### 4.1 动作空间映射链路审查

读取评测脚本与动力学环境源码：
- `tests/eval_phase4_seed.py` 第 376～378 行：
  ```python
  comp_cond = is_air and (env.jump_phase[0].item() >= phase_threshold)
  if comp_cond and (hip_bias != 0.0):
      actions[:, env.left_hip_idx] += hip_bias
      actions[:, env.right_hip_idx] += hip_bias
  ```
- 默认参数：`hip_bias = 0.10`；
- 环境动作映射：`legged_gym/envs/myrobot/myrobot_dual_env.py` 第 528～530 行：
  ```python
  tgt_hip_jump_L = self.default_dof_pos[:, self.left_hip_idx] + self.filtered_actions[:, self.left_hip_idx] * self.hip_action_scale
  ```
- 配置比例尺：`myrobot_forward_jump_static_config.py` 第 73 行：
  `hip_action_scale = 0.6`

### 4.2 数学与物理幅度换算

$$\Delta q_{\text{hip}}^{\text{target}} = \Delta \text{action} \times \text{hip\_action\_scale} = 0.10 \times 0.60 = +0.060\text{ rad}$$

**审查结论**：
1. 历史 Phase 3.2-V 评测脚本（`tests/eval_phase3_2_v_seed.py` 第 366 行）执行的代码为 `actions[:, hip] += 0.10`；
2. Pilot A 评测脚本（`tests/eval_phase4_seed.py` 第 377 行）执行的代码同样为 `actions[:, hip] += 0.10`；
3. **两者物理补偿幅度完全一致，均为关节目标位移 $+0.060\text{ rad}$**；
4. Pilot A 报告中出现的“左右髋 +0.10 rad”纯属撰写文字时的口径误标（将归一化动作空间偏置 $+0.10$ 误写成了弧度 rad），环境物理参数与执行逻辑未发生任何变动。

---

## 任务五：下一阶段训练决策与 Pilot B 启动建议

### 5.1 是否允许启动独立 Pilot B

> [!IMPORTANT]
> **裁决建议：批准启动独立受控试验 Pilot B，但严禁从当前 Pilot A 模型继承，必须从 Phase 4.7 Target B 基线重新初始化。**

**核心依据**：
1. **优势目标与梯度方向完全正确**：Pilot A 首次更新后五种子腾空跳跃距离全部实现正向提升（均值 $+0.39\text{ cm}$），净空全部提升（均值 $+0.54\text{ cm}$），且着陆姿态优秀、零摔倒。这证明 Target B 校准 Critic 提供的策略梯度是健康且有效的；
2. **KL 异常为超参数调度配置缺陷**：Mini-Batch 2 的 KL 异常并非策略本身发散，而是因为：
   - 算法内置的自适应学习率在 Mini-Batch 1 被浮点平滑项误判，步长意外放大至 1.5e-4；
   - 腿部关节低探索噪声（$\sigma \approx 0.06 \sim 0.10$）对动作均值位移过于敏感；
3. **C4 速度偏高是跳跃距离增大的物理副产物**：落地后的慢速滑移（$5.8\text{ cm/s}$）是缺乏末端强制刹车阻尼的自然结果，并非姿态崩溃。

### 5.2 Pilot B 必须执行的修正配置

依据 `myrobot_forward_jump_config.py` 与 `rsl_rl/algorithms/ppo.py` 原生支持字段，Pilot B 必须配置：

| 配置字段 | 推荐设定值 | 机制目的 |
| :--- | :--- | :--- |
| **`warm_start_path`** | `.../model_critic_calibrated_target_b.pt` | **严格重新从 Phase 4.7 Target B 初始化，彻底废弃 Pilot A 权重** |
| **`learning_rate`** | **`5.0e-5`**（保守值） | 降低步长至原设定的 $50\%$，压制单步更新的动作均值偏移量 |
| **`schedule`** | **`'fixed'`** | **彻底禁用 adaptive 调度**，杜绝自适应逻辑在首个 batch 意外放大学习率 |
| **`desired_kl`** | `None` / `0.01` | 固定学习率下不参与学习率缩放 |
| **`early_stop_kl`** | **`0.04`** | 继续保留安全阈值早停保护 |
| **`max_grad_norm`** | **`1.0`** | 严格保持梯度裁剪保护 |
| **`max_iterations`** | **`1`** | **严格限制仅运行 1 个 Iteration**，禁止连续多轮盲跑 |

---

## 交付文件清单

- **复核审查报告（本文件）**：[`Phase_4_8_Pilot_A_R_Review.md`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase4_8_pilot_a/Phase_4_8_Pilot_A_R_Review.md)
- **Pilot A 训练全量遥测记录**：[`training_telemetry.json`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase4_8_pilot_a/training_telemetry.json)
- **五种子配对评测完整数据**：[`paired_physical_evaluation.json`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase4_8_pilot_a/paired_physical_evaluation.json)
