# Phase 3.1 车轮刹车物理诊断实验全量审计报告

**实验代号**: `Phase 3.1 Wheel Brake Diagnostic Audit`  
**测试对象**: Phase 3.1 Champion Checkpoint (`model_champion.pt`，MD5: `45ead50b801b5a322ea4f107e4922f75`)  
**任务性质**: 纯物理动力学因果诊断实验（**不训练、不微调、不改奖励、不改观测/动作空间、不改 URDF/物理参数**）  
**时间戳**: 2026-10-16  

---

## 一、诊断实验目的与核心假说

在上一阶段的 `Phase 3.1 Failure Root-Cause Audit` 中，20 种子鲁棒性评估暴露出明显的双峰行为：Class B 短跳组在跳跃准备期末（Step 5，$t = 0.10\text{ s}$）因车轮向前自由旋转累积了约 $-0.35 \sim -0.42\text{ m/s}$ 的后向水平速度赤字，导致其即使在推蹬期产生了高达 $+0.45\text{ m/s}$ 的前向推蹬冲量，起跳瞬间的 $V_{y,\text{takeoff}}$ 仍然仅有微弱的 $+0.03\text{ m/s}$，最终腾空位移仅 $1 \sim 3\text{ cm}$。

本轮纯诊断实验的**唯一目的**：
> **检验假说**：如果通过真实的轮关节执行器路径，仅在准备期（$t \le 0.10\text{ s}$，Step 0 ～ 5）对左右轮施加与轮速反向的受控阻尼刹车，能否抑制后向速度赤字，使 Champion 恢复稳定的前向起跳？

### 严格冻结边界
- **网络与权重**：100% 冻结 Champion checkpoint；
- **算法与训练**：禁止 PPO 训练、禁止 fine-tuning、禁止修改任何奖励权重与观测接口；
- **物理与配置**：保持 URDF 质量、惯量、接触摩擦系数、PD 增益（`stiffness: 0.0, damping: 0.5`）、关节扭矩上限（轮 $\pm 12\text{ Nm}$，膝 $\pm 120\text{ Nm}$，髋 $\pm 140\text{ Nm}$）绝对不变；
- **执行器合规性**：严禁修改 root state、严禁瞬移（teleport）、严禁人为硬编码 $V_y$ 或位置，必须通过实际车轮动作接口 `action[wheel]` 产生对抗力矩；
- **时间窗口严格性**：干预仅在 $t \le 0.10\text{ s}$（Step 0 ～ 5）有效，从 Step 6（$t \ge 0.12\text{ s}$）推蹬开始瞬间，**100% 恢复为原始网络动作输出**。

---

## 二、评测模式与实验设置

实验采用三档对比设计，覆盖全量 20 个固定随机种子，单次评测运行 140 仿真步（2.8 秒）：

| 模式 | 标识符 | 刹车系数 $k_{\text{brake}}$ | 附加轮端力矩特性 | 适用时间窗口 | 说明 |
|:---|:---|:---:|:---|:---:|:---|
| **Mode A** | `Mode_A_control` | $0.00$ | $\Delta \tau_w = 0$ | 全程无干预 | 原始 Champion 基准（对照组） |
| **Mode B** | `Mode_B_weak_brake` | $0.02$ | $\Delta \tau_w = -0.20 \cdot \omega_w\text{ Nm}$ | 仅 Step 0 ～ 5 | 弱车轮刹车，轻度阻尼 |
| **Mode C** | `Mode_C_strong_brake` | $0.05$ | $\Delta \tau_w = -0.50 \cdot \omega_w\text{ Nm}$ | 仅 Step 0 ～ 5 | 强车轮刹车，中度阻尼 |

> **20 个确定性种子序列**：  
> `[42, 100, 200, 300, 400, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 50, 150, 250, 350, 450]`（共 60 次独立物理评估）

---

## 三、20 种子全量对照统计结果（Section XI 规范表）

下表为 20 个独立随机种子在三档模式下的全量物理统计与行为分类汇总：

| 指标 | 原始 Champion (Mode A) | 弱刹车 (Mode B) | 强刹车 (Mode C) |
|:---|---:|---:|---:|
| **准备期最大 $\|V_y\|$** | $+0.140 \pm 0.087\text{ m/s}$ `[+0.05, +0.36]` | $+0.140 \pm 0.087\text{ m/s}$ `[+0.05, +0.36]` | $+0.140 \pm 0.087\text{ m/s}$ `[+0.05, +0.36]` |
| **Step 5 $V_y$** | $-0.198 \pm 0.116\text{ m/s}$ `[-0.43, -0.08]` | $-0.190 \pm 0.108\text{ m/s}$ `[-0.41, -0.08]` | $-0.183 \pm 0.100\text{ m/s}$ `[-0.39, -0.07]` |
| **Takeoff $V_y$** | $+0.144 \pm 0.155\text{ m/s}$ `[-0.37, +0.27]` | $+0.113 \pm 0.161\text{ m/s}$ `[-0.40, +0.25]` | $+0.075 \pm 0.170\text{ m/s}$ `[-0.43, +0.22]` |
| **Takeoff $V_z$** | $+1.976 \pm 0.143\text{ m/s}$ `[+1.68, +2.11]` | $+1.981 \pm 0.144\text{ m/s}$ `[+1.67, +2.12]` | $+1.985 \pm 0.147\text{ m/s}$ `[+1.66, +2.13]` |
| **推蹬 $\Delta V_y$ ($V_{y,\text{TO}} - V_{y,\text{S5}}$)** | $+0.342 \pm 0.145\text{ m/s}$ `[-0.24, +0.47]` | $+0.303 \pm 0.138\text{ m/s}$ `[-0.26, +0.40]` | $+0.258 \pm 0.138\text{ m/s}$ `[-0.30, +0.34]` |
| **前向腾空位移** | $+6.21 \pm 6.31\text{ cm}$ `[-14.1, +11.6]` | $+4.89 \pm 6.56\text{ cm}$ `[-15.1, +10.8]` | $+3.27 \pm 6.96\text{ cm}$ `[-16.4, +9.7]` |
| **最低共同净空** | $+20.77 \pm 1.76\text{ cm}$ `[+16.9, +23.3]` | $+20.82 \pm 1.76\text{ cm}$ `[+17.2, +23.3]` | $+20.82 \pm 1.82\text{ cm}$ `[+16.9, +23.3]` |
| **Touchdown $\|Roll\|$** | $8.15 \pm 1.79^\circ$ `[4.8, 10.6]` | $7.84 \pm 1.82^\circ$ `[4.2, 10.2]` | $7.67 \pm 1.74^\circ$ `[4.1, 9.9]` |
| **Touchdown $\|Pitch\|$** | $4.72 \pm 3.61^\circ$ `[0.9, 14.1]` | $4.69 \pm 3.68^\circ$ `[0.6, 13.8]` | $4.79 \pm 3.85^\circ$ `[0.4, 14.2]` |
| **恢复后 $V_y$** | $+0.028 \pm 0.011\text{ m/s}$ `[+0.02, +0.06]` | $+0.028 \pm 0.012\text{ m/s}$ `[+0.02, +0.06]` | $+0.029 \pm 0.012\text{ m/s}$ `[+0.01, +0.06]` |
| **综合通过率 (Joint Pass)** | **25.0%** (5/20) | **25.0%** (5/20) | **20.0%** (4/20) |
| **异常 reset** | 0/20 | 0/20 | 0/20 |
| **正常前跃数量 ($\ge 8\text{cm} \ \& \ \ge 20\text{cm}$)** | **7/20** | **7/20** | **4/20** |
| **短跳数量 ($0 \sim 8\text{cm}$)** | **7/20** | **7/20** | **7/20** |
| **倒跳数量 ($< 0\text{cm}$)** | **2/20** | **3/20** | **6/20** |
| **净空不足数量 ($< 20\text{cm}$)** | 8/20 | 8/20 | 8/20 |
| **准备期滑移数量 ($\text{slip}\ge 2\text{cm} \text{ or } \omega\ge 15\text{rad/s}$)** | 8/20 | 8/20 | 9/20 |

---

## 四、核心因果问题实证回答（Section IX）

### 问题 1：弱/强刹车是否成功消除了准备期 $V_y \approx -0.35 \sim -0.42\text{ m/s}$ 这种异常状态？
> **实证结论：否，未能消除。**  
- **数据证据**：  
  - Step 5 $V_y$ 均值从 Mode A 的 $-0.198\text{ m/s}$ 仅微幅变为 Mode B 的 $-0.190\text{ m/s}$ 和 Mode C 的 $-0.183\text{ m/s}$。  
  - 在典型短跳种子中，Seed 2 的 Step 5 $V_y$ 分别为：Mode A: $-0.419\text{ m/s}$，Mode B: $-0.405\text{ m/s}$，Mode C: $-0.390\text{ m/s}$；Seed 13 分别为：$-0.366 \to -0.361 \to -0.352\text{ m/s}$。  
  - 即使施加 $k=0.05$（产生高达 $9.6\text{ Nm}$ 的制动力矩），车轮与地面在准备期的相对运动仅被轻度吸收，Class B 仍存在高达 $-0.35 \sim -0.39\text{ m/s}$ 的后退动量。

### 问题 2：如果准备期后向速度被轻微抑制，$V_{y,\text{takeoff}}$ 是否随之恢复到正值？
> **实证结论：否，起跳水平速度不仅没有恢复，反而全面下降。**  
- **数据证据**：  
  - 全 20 种子起跳 $V_{y,\text{takeoff}}$ 均值呈现阶梯式下滑：Mode A ($+0.144\text{ m/s}$) $\to$ Mode B ($+0.113\text{ m/s}$) $\to$ Mode C ($+0.075\text{ m/s}$)。  
  - 在 Seed 2 中，$V_{y,\text{takeoff}}$ 从 Mode A 的 $+0.034\text{ m/s}$ 恶化为 Mode B 的 $-0.023\text{ m/s}$，Mode C 进一步跌至 $-0.095\text{ m/s}$，直接将 Class B 短跳转变为倒跳。  
  - 倒跳种子数量从 Mode A 的 2 个增加到 Mode B 的 3 个，Mode C 更激增至 6 个。

### 问题 3：当前 Champion 原有推蹬 $\Delta V_{y,\text{push}}$ 是否在施加刹车后仍然保持？
> **实证结论：否，原有推蹬冲量受到显著削弱。**  
- **数据证据**：  
  - 推蹬期水平速度增量 $\Delta V_y$ 均值：Mode A ($+0.342\text{ m/s}$) $\to$ Mode B ($+0.303\text{ m/s}$) $\to$ Mode C ($+0.258\text{ m/s}$)，衰减幅度达 **24.6%**。  
  - **动力学根因**：  
    原有 Champion 策略是在“准备期允许轮子前滚/机身后退”的动力学构型下训练出来的。在 Mode A 中，机身在 Step 5 的后仰与下蹲形成了一定的机械拉伸蓄能；当准备期被外加阻尼限制后，Step 5 的下蹲几何构型发生微变。进入 Step 6 恢复神经网络控制时，观测到的关节速度与机身姿态脱离了策略原有的“引爆点”，导致 Step 7～8 策略给出的反向推蹬轮力矩明显降低（如 Seed 2 中 Step 7 轮力矩从 $-5.63\text{ Nm}$ 萎缩至 $-0.80\text{ Nm}$），前向地反力推力损失严重。

### 问题 4：净空是否受到明显影响？特别是 $V_{z,\text{takeoff}}$ 是否下降？
> **实证结论：否，竖向推蹬与净空完全未受损害。**  
- **数据证据**：  
  - $V_{z,\text{takeoff}}$ 均值：Mode A ($1.976\text{ m/s}$) $\to$ Mode B ($1.981\text{ m/s}$) $\to$ Mode C ($1.985\text{ m/s}$)，完全保持不变（方差仅 0.005 m/s）。  
  - 最低共同净空均值：Mode A ($20.77\text{ cm}$) $\to$ Mode B ($20.82\text{ cm}$) $\to$ Mode C ($20.82\text{ cm}$)。  
  - 证明腿部髋、膝伸展爆发的垂直冲量生成机制与准备期车轮旋转阻尼物理上高度解耦。

### 问题 5：Seed 17 / 23 的倒跳是否也得到改善？
> **实证结论：否，完全没有改善，且倒跳距离进一步加大。**  
- **数据证据**：  
  - Seed 17 位移：Mode A ($-14.07\text{ cm}$) $\to$ Mode B ($-15.06\text{ cm}$) $\to$ Mode C ($-16.41\text{ cm}$)；  
  - Seed 23 位移：Mode A ($-4.06\text{ cm}$) $\to$ Mode B ($-5.70\text{ cm}$) $\to$ Mode C ($-7.65\text{ cm}$)。  
  - **动力学机理**：逐帧遥测显示，Seed 17 与 23 的倒跳主要发生在 **Step 6 ～ 9（推蹬期）**。机器人在触地下蹲时前倾角过大（Pitch $> 2^\circ$），触发了平衡控制器的正向车轮滚转补偿（Step 7 轮力矩打满正向饱和 $+12.0\text{ Nm}$），反向加速机身后退。由于干预窗口严格受限于 Step 0 ～ 5，准备期刹车完全无法干预推蹬期的平衡反馈，证实**倒跳与准备期滑移是两个彼此独立的动力学病因**。

---

## 五、关键种子逐帧轨迹动力学对比

通过对比四个典型种子的逐帧时间序列（数据归档于 `evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/key_trajectories/`）：

```
[Seed 100: 正常前跃]
  Mode A: S5_Vy = -0.120 m/s -> TO_Vy = +0.259 m/s -> 位移 = +11.62 cm, 净空 = 22.35 cm
  Mode B: S5_Vy = -0.120 m/s -> TO_Vy = +0.239 m/s -> 位移 = +10.52 cm, 净空 = 22.42 cm
  Mode C: S5_Vy = -0.124 m/s -> TO_Vy = +0.212 m/s -> 位移 =  +9.31 cm, 净空 = 22.49 cm
  * 物理效应：由于本身滑移较小，刹车轻微削弱了水平推蹬速度，但位移完美收敛在 8~10 cm 目标区间内。

[Seed 2: Class B 短跳原型]
  Mode A: S5_Vy = -0.419 m/s -> TO_Vy = +0.034 m/s -> 位移 =  +1.65 cm, 净空 = 22.97 cm (短跳)
  Mode B: S5_Vy = -0.405 m/s -> TO_Vy = -0.023 m/s -> 位移 =  -1.04 cm, 净空 = 23.11 cm (倒跳)
  Mode C: S5_Vy = -0.390 m/s -> TO_Vy = -0.095 m/s -> 位移 =  -4.50 cm, 净空 = 23.10 cm (重度倒跳)
  * 物理效应：刹车仅消除 0.029 m/s 后退速度，但导致 Step 7 轮端推蹬地反力从 -5.63 Nm 塌缩至 -0.80 Nm，推蹬损失远远超过准备期减速收益。

[Seed 17: 姿态补偿倒跳原型]
  Mode A: S5_Vy = -0.133 m/s -> TO_Vy = -0.369 m/s -> 位移 = -14.07 cm, 净空 = 18.53 cm
  Mode B: S5_Vy = -0.133 m/s -> TO_Vy = -0.396 m/s -> 位移 = -15.06 cm, 净空 = 18.32 cm
  Mode C: S5_Vy = -0.133 m/s -> TO_Vy = -0.431 m/s -> 位移 = -16.41 cm, 净空 = 18.06 cm
  * 物理效应：准备期轮速与速度完全无异；在 Step 7~8 轮端因前倾触发 +12.0 Nm 滚平衡推力，倒跳发生在刹车释放之后。
```

---

## 六、最终裁决与判定规则（Section XII）

根据用户给定的三项排他性裁决规则：

> - **结论 A**：准备期车轮刹车显著改善 Step 5 后向速度、Takeoff $V_y$、前向腾空位移，且无明显损害。  
> - **结论 B**：准备期速度改善（或未改善），但最终前跃能力没有明显改善。说明**“准备期后退不是唯一瓶颈”**。停止，不修改 reward。  
> - **结论 C**：刹车同时严重损害 $V_z$、净空或落地稳定性。说明当前轮端制动与跳跃推蹬存在动力学耦合。

### 裁决确认：【结论 B】（兼具对推蹬耦合的警示）

1. **排除结论 A**：数据证实，准备期车轮刹车**完全没有改善最终前跃能力**。前向腾空位移从 $+6.21\text{ cm}$ 进一步下降至 $+3.27\text{ cm}$，正常前跃种子从 7 个减少到 4 个，倒跳种子增加至 6 个。
2. **排除结论 C 的充分条件**：实验中垂直起跳速度 $V_z$（$1.976 \to 1.985\text{ m/s}$）、共同净空（$20.77 \to 20.82\text{ cm}$）以及着陆姿态稳定性（Roll $\sim 7.8^\circ$、Pitch $\sim 4.7^\circ$）**均未受到损害**，因此竖向跳跃并没有被制动破坏。
3. **符合结论 B 核心结论**：
   > **“准备期车轮运动并不是导致跳跃距离不足的唯一瓶颈。”**
   > 单纯在准备期加入轮端刹车或阻尼，不仅不能阻止倒跳（Seed 17/23 机制在推蹬期），反而会改变准备期下蹲构型，破坏已有策略在推蹬期依赖的初始动量匹配，削弱推蹬 $\Delta V_y$。
   > **裁决执行：停止准备期外加轮端物理刹车方向，不要盲目在环境奖励中单纯惩罚轮速。**

---

## 七、归档资产与交付清单（Section XIII）

所有代码、评测记录与对照数据集均已完整归档，未修改任何已有 Phase 3.1 资产与 Champion 权重：

1. **诊断实验代码**：
   - 评测核心：[`tests/eval_single_diagnostic.py`](file:///home/xy/IsaacTu_jump/tests/eval_single_diagnostic.py)
   - 批处理与统计引擎：[`tests/run_wheel_brake_diagnostic.py`](file:///home/xy/IsaacTu_jump/tests/run_wheel_brake_diagnostic.py)
2. **20 种子原始模式结果**：
   - 目录：[`evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/Mode_A_control/`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/Mode_A_control/)
   - 遥测总表：[`evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/Mode_A_control_telemetry.csv`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/Mode_A_control_telemetry.csv)
3. **20 种子弱刹车结果**：
   - 目录：[`evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/Mode_B_weak_brake/`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/Mode_B_weak_brake/)
   - 遥测总表：[`evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/Mode_B_weak_brake_telemetry.csv`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/Mode_B_weak_brake_telemetry.csv)
4. **20 种子强刹车结果**：
   - 目录：[`evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/Mode_C_strong_brake/`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/Mode_C_strong_brake/)
   - 遥测总表：[`evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/Mode_C_strong_brake_telemetry.csv`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/Mode_C_strong_brake_telemetry.csv)
5. **逐帧关键轨迹数据**：
   - [`trajectory_comparison_seed_100.csv`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/key_trajectories/trajectory_comparison_seed_100.csv) (正常前跃对照)
   - [`trajectory_comparison_seed_2.csv`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/key_trajectories/trajectory_comparison_seed_2.csv) (短跳原型对照)
   - [`trajectory_comparison_seed_17.csv`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/key_trajectories/trajectory_comparison_seed_17.csv) (姿态补偿倒跳对照)
   - [`trajectory_comparison_seed_23.csv`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/key_trajectories/trajectory_comparison_seed_23.csv) (复合倒跳对照)
6. **统计汇总 JSON**：
   - [`evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/wheel_brake_diagnostic_summary.json`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/wheel_brake_diagnostic/wheel_brake_diagnostic_summary.json)
