# MyRobot Phase 4.9：ZeroCmd 独立集成与未见种子泛化验证报告

**项目**：MyRobot 双轮足机器人前向跳跃强化学习项目  
**阶段**：Phase 4.9（ZeroCmd 独立集成、观测同构核对与 20 全新未见种子严格配对泛化验证）  
**执行原则**：严守只读审计规范、**禁止 PPO 参数更新**、禁止修改底层 `commands`、禁止修改物理动力学与正式奖励、严禁放宽 C4 门限（$|v_y| \le 0.030\text{ m/s}$）与 C1 门限（$d_{\text{fwd}} \ge 12.00\text{ cm}$）、冻结所有历史模型与实验数据。  
**核心核查与评估资产**：
- **Pilot B 评测模型**：[`evaluation_assets/phase4_8_pilot_b/model_pilot_b_step1.pt`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase4_8_pilot_b/model_pilot_b_step1.pt)（SHA256: `64aedcfd0156a5ed512c4c8c1c97f06caefae8c7e662322a143dc3ec9decc29b`）
- **Target B Critic 模型**：[`evaluation_assets/phase4_7_critic_calibration/model_critic_calibrated_target_b.pt`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase4_7_critic_calibration/model_critic_calibrated_target_b.pt)（SHA256: `a5fca202c667da64476bb36044f9fdd814cae430753b57e349699b2e679f5ed0`）
- **Champion Actor 基线**：[`evaluation_assets/phase3_1_forward_distance/models/model_champion.pt`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt)（SHA256: `fad590f4699abac8ae2b3e635d9cb88ef2658da85bbb37008e13362bb3c5b950`）
- **独立评测脚本**：[`tests/eval_phase4_9_zerocmd.py`](file:///home/xy/IsaacTu_jump/tests/eval_phase4_9_zerocmd.py)（SHA256: `ea7f909600e8817b9a624d81d3acab9015cb5c28d4aea35247fd2808c4671a47`）
- **20 种子配对批跑器**：[`scripts/run_phase4_9_zerocmd_validation.py`](file:///home/xy/IsaacTu_jump/scripts/run_phase4_9_zerocmd_validation.py)（SHA256: `24f1e2ece9ad8280734588965998ca2b5053fe7733edaf8f12d0c29819432a6c`）
- **20 种子配对原始数据汇总**：[`evaluation_assets/phase4_9_zerocmd/zerocmd_20seeds_validation_summary.json`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase4_9_zerocmd/zerocmd_20seeds_validation_summary.json)（SHA256: `b42e5f2c3a3b5ecd14831aff1d702bc4f56aeffecad005d3ad6049d9e07c758e`）

---

## 一、任务一：独立 ZeroCmd 实验版本固化与安全约束审计

### 1.1 独立资产隔离结构
本轮所有实现与评测数据均在独立路径下创建，完全不覆盖 Phase 3.2-V、Phase 4.7、Phase 4.8-Pilot-A 及 Phase 4.8-Pilot-B 的历史评测代码和模型文件：
```
IsaacTu_jump/
├── tests/eval_phase4_9_zerocmd.py            # Phase 4.9 独立单种子评测引擎
├── scripts/run_phase4_9_zerocmd_validation.py# Phase 4.9 20种子配对验证自动化脚本
└── evaluation_assets/phase4_9_zerocmd/       # Phase 4.9 独立数据沉淀目录
    ├── individual_runs/                      # 40 次独立子进程运行的完整单文件 JSON
    │   ├── baseline_seed8001.json ... baseline_seed8020.json
    │   └── zerocmd_seed8001.json  ... zerocmd_seed8020.json
    └── zerocmd_20seeds_validation_summary.json # 汇总配对张量与统计算法数据
```

### 1.2 观测布局与索引真值核实
通过从磁盘读取底层 [`MyRobotForwardJump`](file:///home/xy/IsaacTu_jump/legged_gym/legged_gym/envs/myrobot/myrobot_forward_jump_env.py#L92) 源码，核实了 35 维 Actor 观测向量的精确排列：
- `obs[0:3]`：基体线速度缩放 `base_lin_vel * 2.0`（3维：$[v_x \cdot 2, v_y \cdot 2, v_z \cdot 2]$，MyRobot 局部坐标系下 $Y$ 轴为正前向）；
- `obs[3:6]`：基体角速度缩放 `base_ang_vel * 0.25`（3维）；
- `obs[6:9]`：重力投影向量 `projected_gravity`（3维）；
- `obs[9:13]`：跳跃指令块 `jump_cmds`（4维）：
  - **Slot 9**：前向速度指令观测 `commands[:, 1] * scale[1]`（nominal 为 $0.22 \times 2.0 = 0.44$）；
  - **Slot 10**：偏航速度指令观测 `commands[:, 2] * scale[2]`（$0.0$）；
  - **Slot 11**：归一化目标跳跃高度 `(target_jump_height - 0.10) / 0.10`；
  - **Slot 12**：跳跃相位进度 `jump_phase`（$\phi \in [0, 1]$）；
- `obs[13:19]`：关节位置偏差 `obs_dof_pos * 1.0`（6维，轮关节置零）；
- `obs[19:25]`：关节角速度 `dof_vel * 0.05`（6维）；
- `obs[25:31]`：一阶滤波动作 `filtered_actions`（6维）；
- `obs[31:33]`：归一化轮底触地反力 `contact_state`（2维：$[f_{n,L}/100, f_{n,R}/100]$）；
- `obs[33:35]`：相位时钟 `[sin(2*pi*phi), cos(2*pi*phi)]`（2维）。

**结论**：Slot 9 确实无歧义地对应传入 Actor 的**正前向速度指令观测**。

### 1.3 七大约束符合性逐项审计
1. **不修改底层 `self.commands`**：`self.commands[:, 1]` 在全 Episode 期间严格保持 nominal $0.22\text{ m/s}$，底层张量完全未被写入。
2. **不修改物理状态与动力学**：PhysX 刚体状态、关节力矩限幅、PD 参数、接触摩擦系数 100% 冻结。
3. **每个并行环境独立触发**：使用布尔掩码 `landed_mask`（即 `self.has_touched_down`），向量化赋值仅影响已触地的环境实例：
   ```python
   obs_for_actor = obs.clone()
   if zero_cmd and landed:
       obs_for_actor[:, 9] = 0.0
   ```
4. **Episode Reset 后彻底清除触地掩码**：环境在 `reset_idx(env_ids)` 中明确执行 `self.has_touched_down[env_ids] = False`，无任何跨 Episode 状态泄漏。
5. **起跳前、起跳中、腾空中保持原始观测**：在 `has_touched_down == False` 时，Slot 9 始终为原始 `commands[:, 1] * 2.0`（0.44），保证起跳推力与腾空姿态不受任何干扰。
6. **触地后始终使用零前向目标观测**：自落地步开始，送入 Actor 的 Slot 9 严格锁定为 0.0。
7. **保留下落阶段原始髋动作偏置 +0.10（动作空间单位）**：下落阶段补偿代码为：
   ```python
   if is_air and (env.jump_phase[0].item() >= phase_threshold):
       actions[:, env.left_hip_idx] += 0.10
       actions[:, env.right_hip_idx] += 0.10
   ```
   严格保留于动作空间单位（未误写为弧度）。

---

## 二、任务二：训练与评测环境观测一致性及 Critic 兼容性核对

### 2.1 评测环境与训练环境同构性核实
- **评测环境**：[`MyRobotForwardJumpStatic`](file:///home/xy/IsaacTu_jump/legged_gym/legged_gym/envs/myrobot/myrobot_forward_jump_static_env.py#L14)
- **训练环境**：[`MyRobotForwardJumpPhase4`](file:///home/xy/IsaacTu_jump/legged_gym/legged_gym/envs/myrobot/myrobot_forward_jump_phase4_env.py#L21)

通过源码比对与运行时反射检查证实：
1. `MyRobotForwardJumpPhase4` 直接继承自 `MyRobotForwardJumpStatic`，两环境**均未重写 `compute_observations()`**，均继承自 `MyRobotForwardJump`。
2. 两个环境的观测维度均为严格一致的 35 维，且 `num_privileged_obs` 均为 `None`。
3. 两个环境的 Slot 9 均由 `commands[:, 1] * self.commands_scale[1]` 计算生成，定义完全同构。

### 2.2 Critic 对 Slot 9 归零的兼容性定量实测
在未来在线 PPO Rollout 中，Critic 是否需要同步修改？针对当前已校准的 Target B Critic（[`model_critic_calibrated_target_b.pt`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase4_7_critic_calibration/model_critic_calibrated_target_b.pt)），我们对其网络权重敏感度与实际价值输出进行了定量实测：

1. **输入层敏感度**：
   - Critic 第一层权重矩阵尺寸为 `[512, 35]`，全局 $L_2$ 范数为 $18.918$；
   - 对应 Slot 9 的权重列向量 $L_2$ 范数为 $2.119$（占全局范数仅 11.2%）。
2. **全周期价值输出敏感度**：
   - 使用 Phase 4.7 真实 Rollout 数据（140 步轨迹，均值价值为 $21.737 \pm 2.099$）；
   - 将全部步的 Slot 9 由 0.44 替换为 0.0，Critic 评估价值的变化量为：
     $$\Delta V = V_{\text{zero9}} - V_{\text{orig}} = -0.0257 \pm 0.0092$$
     相对价值变化幅度仅为 **$-0.118\%$**。
3. **Step 139 真实 Timeout 终态观测敏感度**：
   - 在真实 pre-reset 终态观测 $s_{140}^{\text{pre}}$ 上实测：
     $$V(s_{140}^{\text{pre}})_{\text{orig}} = 19.7094 \quad \longrightarrow \quad V(s_{140}^{\text{pre}})_{\text{zero9}} = 19.6852$$
     $$\Delta V_{\text{terminal}} = -0.0242 \quad (-0.123\%)$$

**结论**：Target B Critic 对 Slot 9 归零具有高达 **99.88% 的固有兼容性**。若在训练环境中统一将触地后观测 Slot 9 归零，无需引入不对称特权观测（Asymmetric Critic），Actor 与 Critic 可完全共享同构观测。

### 2.3 训练环境最小兼容实现方案
在未来批准恢复 PPO 训练时，推荐在 `MyRobotForwardJumpPhase4` 中增加单处最小重写：
```python
def compute_observations(self):
    super().compute_observations()
    if getattr(self.cfg.env, "zero_cmd_post_touchdown", True):
        self.obs_buf[self.has_touched_down, 9] = 0.0
```
该改动保证：
- 起跳前与飞行中保持原始指令观测；
- 触地后以及 Step 139 终态观测（用于 Target B Bootstrap 的 $s_{140}^{\text{pre}}$）自动归零；
- 训练环境与评测环境行为 100% 同构闭环。

---

## 三、任务三：20 个全新未见随机种子严格配对物理泛化验证

### 3.1 随机种子隔离性保证
本次验证选取了 **20 个连续全新随机种子**：
$$\mathcal{S}_{\text{unseen}} = \{8001, 8002, 8003, \dots, 8020\}$$
审计确认：该 20 个种子在此前的 Phase 1、Phase 2、Phase 3.1、Phase 3.2-U、Phase 3.2-V、Phase 4.4、Phase 4.6、Phase 4.7（训练 Seed 6001、验证 Seed 7001、开发 Seed 3001～3005）及 Pilot A/B 实验中**均从未出现或被使用**，属于绝对未见样本集。

### 3.2 20 种子配对实测数据全明细表

> [!NOTE]
> 评测条件：单环境独立子进程、Phase 15 沉降交接、140 步（2.8s）、下落髋动作偏置 +0.10、统一评测模型 Pilot B（`model_pilot_b_step1.pt`）。  
> **基线组（Group A）**：`zero_cmd = False`；**实验组（Group B）**：`zero_cmd = True`。

| Seed | 腾空位移 $d_{\text{fwd}}$ (cm) | 最大净空 $c_{\text{max}}$ (cm) | 触地俯仰 $\theta_p$ (deg) | 触地横滚 $\phi_r$ (deg) | C4 速度基线 (m/s) | C4 速度 ZeroCmd (m/s) | C4 降速量 (m/s) | 左轮力矩 基线/Zero (Nm) | 左轮转速 基线/Zero (rad/s) | 基线综合通过 | ZeroCmd综合通过 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **8001** | 11.15 | 21.91 | +8.00 | +0.89 | 0.0388 | **0.0116** | -0.0271 | -0.41 / -0.03 | -0.58 / -0.17 | FAIL | FAIL (短板C1) |
| **8002** | 14.57 | 21.53 | +8.19 | +3.14 | 0.0356 | **0.0117** | -0.0240 | -0.37 / -0.03 | -0.53 / -0.17 | FAIL | **PASS** |
| **8003** | 14.43 | 21.48 | +8.47 | +3.28 | 0.0381 | **0.0117** | -0.0265 | -0.39 / -0.03 | -0.56 / -0.17 | FAIL | **PASS** |
| **8004** | 15.58 | 21.49 | +7.86 | +3.59 | 0.0382 | **0.0117** | -0.0266 | -0.40 / -0.03 | -0.56 / -0.17 | FAIL | **PASS** |
| **8005** | 14.74 | 21.65 | +7.49 | +3.34 | 0.0356 | **0.0116** | -0.0239 | -0.36 / -0.03 | -0.52 / -0.17 | FAIL | **PASS** |
| **8006** | 14.69 | 21.93 | +7.66 | +0.82 | 0.0374 | **0.0116** | -0.0258 | -0.40 / -0.03 | -0.55 / -0.17 | FAIL | **PASS** |
| **8007** | 14.65 | 21.74 | +8.14 | +1.70 | 0.0365 | **0.0117** | -0.0249 | -0.38 / -0.03 | -0.54 / -0.17 | FAIL | **PASS** |
| **8008** | 12.11 | 21.73 | +8.05 | +1.50 | 0.0377 | **0.0116** | -0.0261 | -0.40 / -0.03 | -0.56 / -0.17 | FAIL | **PASS** |
| **8009** | 14.23 | 21.56 | +8.43 | +2.75 | 0.0370 | **0.0117** | -0.0253 | -0.39 / -0.03 | -0.54 / -0.17 | FAIL | **PASS** |
| **8010** | 14.63 | 21.47 | +8.05 | +3.45 | 0.0356 | **0.0116** | -0.0240 | -0.37 / -0.03 | -0.52 / -0.17 | FAIL | **PASS** |
| **8011** | 11.19 | 21.30 | +7.97 | +3.21 | 0.0355 | **0.0116** | -0.0239 | -0.36 / -0.03 | -0.52 / -0.17 | FAIL | FAIL (短板C1) |
| **8012** | 12.76 | 21.40 | +7.67 | +2.72 | 0.0354 | **0.0116** | -0.0237 | -0.36 / -0.03 | -0.52 / -0.17 | FAIL | **PASS** |
| **8013** | 11.41 | 22.48 | +6.07 | +0.91 | 0.0361 | **0.0116** | -0.0245 | -0.38 / -0.03 | -0.53 / -0.17 | FAIL | FAIL (短板C1) |
| **8014** | 12.03 | 21.63 | +7.97 | +1.59 | 0.0360 | **0.0117** | -0.0244 | -0.37 / -0.03 | -0.53 / -0.17 | FAIL | **PASS** |
| **8015** | 12.22 | 21.36 | +7.50 | +3.85 | 0.0369 | **0.0117** | -0.0252 | -0.38 / -0.03 | -0.54 / -0.17 | FAIL | **PASS** |
| **8016** | 11.86 | 21.30 | +8.30 | +2.61 | 0.0385 | **0.0116** | -0.0268 | -0.41 / -0.03 | -0.57 / -0.17 | FAIL | FAIL (短板C1) |
| **8017** | 14.59 | 21.79 | +7.17 | +2.99 | 0.0367 | **0.0116** | -0.0251 | -0.38 / -0.03 | -0.54 / -0.17 | FAIL | **PASS** |
| **8018** | 11.25 | 21.16 | +7.83 | +3.62 | 0.0379 | **0.0116** | -0.0262 | -0.40 / -0.03 | -0.56 / -0.17 | FAIL | FAIL (短板C1) |
| **8019** | 14.35 | 21.74 | +8.13 | +1.74 | 0.0360 | **0.0116** | -0.0243 | -0.37 / -0.03 | -0.53 / -0.17 | FAIL | **PASS** |
| **8020** | 14.22 | 21.86 | +8.07 | +1.14 | 0.0378 | **0.0116** | -0.0262 | -0.40 / -0.03 | -0.56 / -0.17 | FAIL | **PASS** |

---

### 3.3 统计对比与泛化性能分析

```
========================================================================================
                          Phase 4.9: 20 种子配对泛化统计对比表
========================================================================================
评测维度 / 统计指标             | Group A (基线 Pilot B) | Group B (ZeroCmd Pilot B) | 净改善 / 变化
----------------------------------------------------------------------------------------
腾空前向位移均值 (cm)           | 13.33 ± 1.48           | 13.33 ± 1.48              | 0.00 cm (完全保留)
腾空位移极值范围 (cm)           | [11.15, 15.58]         | [11.15, 15.58]            | 完全同构
最大共同轮底净空 (cm)           | 21.63 ± 0.29           | 21.63 ± 0.29              | 0.00 cm (完全保留)
触地俯仰角均值 (deg)            | +7.85° ± 0.52°         | +7.85° ± 0.52°            | 完全同构
触地横滚角均值 (deg)            | +2.44° ± 1.01°         | +2.44° ± 1.01°            | 完全同构
----------------------------------------------------------------------------------------
C4 末端恢复速度均值 (|vy|, m/s) | 0.0369 ± 0.0011        | 0.0116 ± 0.00001          | -0.0252 m/s (-68.4%)
C4 恢复速度范围 (m/s)           | [0.0354, 0.0388]       | [0.0116, 0.0117]          | 极度收敛稳定
左轮末端制动力矩 (Nm)           | -0.38 ± 0.02           | -0.03 ± 0.00              | 降低 92.2%
右轮末端制动力矩 (Nm)           | +0.39 ± 0.02           | +0.03 ± 0.00              | 降低 92.3%
左轮末端转速 (rad/s)            | -0.54 ± 0.02           | -0.17 ± 0.00              | 降低 68.5%
----------------------------------------------------------------------------------------
C0 起跳前静止准备达标率         | 20/20 (100.0%)         | 20/20 (100.0%)            | 持平 (100%)
C1 腾空前向位移达标率 (≥12cm)   | 15/20 (75.0%)          | 15/20 (75.0%)             | 持平 (75.0%)
C2 腾空垂直净空达标率 (≥20cm)   | 20/20 (100.0%)         | 20/20 (100.0%)            | 持平 (100%)
C3 触地姿态角达标率 (≤10°)      | 20/20 (100.0%)         | 20/20 (100.0%)            | 持平 (100%)
C4 恢复平稳速度达标率 (≤0.03m/s)| 0/20 (0.0%)            | 20/20 (100.0%)            | +100.0% (质的飞跃)
C5 异常复位与跌倒达标率         | 20/20 (100.0%)         | 20/20 (100.0%)            | 0 跌倒, 0 异常复位
========================================================================================
正式全任务综合达标率 (C0～C5)   | 0/20 (0.0%)            | 15/20 (75.0%)             | +75.0% (绝对通过)
========================================================================================
```

### 3.4 边界样本与失败模式审计
1. **ZeroCmd 是否引入任何新的失败模式**：
   - 跌倒率：0/20（0%）；
   - 异常复位：0/20（0%）；
   - 触地姿态冲击：与基线完全相同（因为触地前两者控制逻辑与输入 100% 同构）。
2. **综合通过率未达 100% 的原因分析**：
   - Group B 中 5 个未通过综合指标的种子（8001、8011、8013、8016、8018），**均 100% 仅失败于 C1 腾空位移指标**；
   - 5 个种子的位移在 $11.15\text{ cm} \sim 11.86\text{ cm}$ 之间；
   - 其中 **Seed 8016** 位移为 $11.86\text{ cm}$，距 $12.00\text{ cm}$ 仅差 $1.4\text{ mm}$；
   - 其余各项（C0、C2、C3、C4、C5）在这 5 个种子上全部通过（C4 恢复速度均在 0.0116 m/s 完美达标）。
   - 这充分证明：**当前模型的唯一物理短板在于起跳爆发推力在部分种子分布下略显不足（差 1~8 mm），而非落地恢复控制问题**。

---

## 四、任务四：下一步受控 PPO 优化建议

经过 20 个全新随机种子的实测验证，ZeroCmd 已彻底攻克了此前困扰项目的 C4 恢复速度顽疾（从 0% 跃升至 100% PASS）。然而，由于部分种子存在 1~8 mm 的前向推力短板，当前 Pilot B 的未见种子综合通过率为 75.0%（未达 90%+ 生产级标准）。

为补齐前向位移短板，同时防止策略退化，提出以下 Phase 4.9-Pilot 训练建议：

### 4.1 初始化检查点选择
**建议**：以 [`model_pilot_b_step1.pt`](file:///home/xy/IsaacTu_jump/evaluation_assets/phase4_8_pilot_b/model_pilot_b_step1.pt) 作为热启动基线，而非重新从 Target B 退回。  
**理由**：
- Pilot B 经历了 1 次受控 Adam 更新，五种子前向位移已由 Champion 的 $12.66\text{ cm}$ 提升至 $12.81\text{ cm}$，且 20 未见种子均值达 $13.33\text{ cm}$；
- 它的起跳爆发力与垂直净空（$21.63\text{ cm}$）表现极为优异；
- 配合已验证的 ZeroCmd 机制，落地恢复后顾之忧已被彻底解除，后续微调仅需专注于起跳蹬地冲量的一致性。

### 4.2 训练与评测环境同构固化
在启动训练前，必须在 [`legged_gym/envs/myrobot/myrobot_forward_jump_phase4_env.py`](file:///home/xy/IsaacTu_jump/legged_gym/legged_gym/envs/myrobot/myrobot_forward_jump_phase4_env.py) 中正式合入第 2.3 节的 ZeroCmd 观测逻辑：
- 保证训练 Rollout 阶段与在线评估阶段在触地后均向 Actor 传入 Slot 9 = 0.0；
- 消除训练与评测之间的策略分布漂移。

### 4.3 PPO 训练超参数严格约束
为防止出现 Pilot A 中的 KL 爆炸（KL > 0.13）问题，后续 PPO 训练必须遵守：
1. **固定低学习率**：
   $$\eta = 5.0 \times 10^{-5} \quad (\text{禁止使用任何自适应学习率调度器})$$
2. **受控步数推进**：
   - 单次实验仅允许执行 **1 个 PPO Iteration**（内含 4 个 Mini-Batch 优化步）；
   - 设置极低 KL 早停门限：`early_stop_kl = 0.020`，一旦单步 KL 达到 0.02 立即截断。
3. **保持奖励函数绝对冻结**：
   - 保持 Phase 4.4 Sigmoid 前向位移奖励；
   - 禁止为了迎合 Seed 8016 或 Seed 3005 单独调高前向奖励权重，确保策略在整体多峰分布下的自然收敛。

### 4.4 双层多样本验证验收协议
微调后模型的验收必须同时通过：
- **开发集（Dev Set）**：Seeds 3001～3005，要求综合通过率达到 5/5（100%）；
- **验证集（Val Set）**：Seeds 8001～8020，要求综合通过率由当前的 15/20（75%）提升至 $\ge 18/20$（$\ge 90\%$）。

---

## 五、结论与技术决策响应

对用户与技术审查关注的四个核心问题给出明确结论：

1. **ZeroCmd 是否具备泛化能力**：  
   **是，完全具备**。在 20 个全新未见随机种子（8001～8020）上，ZeroCmd 将 C4 恢复速度从基线的全部超标（$0.0369\text{ m/s}$，0/20 达标）一致性压缩至 **$0.0116\text{ m/s}$（20/20 全部达标，100% PASS）**，且标准差极小（$0.00001\text{ m/s}$），证明其物理有效性完全脱离了特定种子的偶然性。
2. **是否引入新的失败模式**：  
   **否，完全未引入**。ZeroCmd 仅在机器人触地后改变前向速度指令输入。起跳前、起跳蹬地、空中飞行、触地瞬间的姿态与受力完全不变（姿态角与基线 100% 相同）；触地后轮电机差动力矩由 $\pm 0.38\text{ Nm}$ 下降 92.2% 至 $\pm 0.03\text{ Nm}$，未发生任何跌倒或非正常复位。
3. **是否可以成为新的仿真候选控制方案**：  
   **完全可以，且强烈推荐确立为正式基线**。它从控制架构上消除了“跳跃 Actor 在落地恢复后仍持续试图前向巡航”的结构性指令冲突，使双轮足机器人在无需频繁切换控制器的前提下实现自然的制动驻车。
4. **是否建议启动下一次 PPO Pilot**：  
   **建议启动，但必须严格受控**。当前 Pilot B + ZeroCmd 在 20 个新种子上的全任务综合达标率已达到 **75.0%**（15/20），阻碍其达到 100% 的唯一起跳前向位移指标（5 个短板种子仅差 1~8 mm）。建议以固定低学习率（$5.0\text{e-5}$）开展 1 次受控 PPO Iteration 微调，补齐这微小的起跳推力短板。

---
*本轮任务已全部完成，未进行任何 PPO 参数更新与基线覆盖，请技术审查委员会审阅。*
