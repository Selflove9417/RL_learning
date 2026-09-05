# 阶段 25：巡航节流软裕度（Cruise Throttle Margin）与差动纠偏空间释放实施计划

## 目标简述

阶段 24 的时序物理诊断查明：**轮速差闭环纠偏本身有效，但策略在巡航时长期把两侧共模动作打满 `a_c = -1.0`（7.0 Nm 硬件满推）**。
这导致电机关节在硬件限幅处发生**单侧不对称截断**：
1. 打滑轮被差动反馈刹车减扭有效（例如降至 4.5 Nm）；
2. 抓地轮本应补偿增扭（例如增至 8.5 Nm），但被 URDF `±7.0 Nm` 硬件极限硬性截断；
3. 两轮因此产生瞬时单侧推力脉冲差，引发约 $2.5\text{ rad/s}$ 的瞬态偏航尖峰，导致车速在 $0.20 \sim 0.50\text{ m/s}$ 间震荡。

本阶段核心方案：**完全冻结已验证控制层（速度权重 3.0、偏航闭环 1.5、轮速差闭环 0.10、动作对称 -1.0、弱阻尼 0.3 与腿部 PD），仅新增“巡航节流软裕度”单项奖励**：
\[
a_c = \frac{a_L + a_R}{2}
\]
当车速接近目标速度或超速时，惩罚 $|a_c| > 0.7$ 的共模满油门行为；当明显低于目标速度（起步加速、爬坡受阻）以及静止驻车状态（`cmd_v = 0`）时，严格不施加惩罚。
这促使策略在巡航时主动将油门收拢在 $|a_c| \le 0.7$（约 $4.9\text{ Nm}$），为抓地侧电机预留约 **$2.1\text{ Nm}$** 的软裕度，**显著降低闭环差动被硬件限幅截断的概率**。

---

## 关键修正与确认项 (User Review Required)

> [!IMPORTANT]
> **1. 奖励符号规范（非负 cost 返回）**：
> 在 `legged_gym` 框架中，奖励函数内部返回正的 cost / 误差度量，由配置权重 `wheel_throttle_margin = -2.0` 乘成负惩罚：
> \[
> \text{reward\_func} = \operatorname{cruise\_gate} \cdot \max(0, |a_c| - 0.7)^2 \ge 0
> \]
> 函数内部绝无负号，杜绝符号翻转。

> [!IMPORTANT]
> **2. 静止指令门控（杜绝误伤驻车策略）**：
> 在计算巡航门控时，引入显式非零速度指令门控 `is_nonzero_cmd = (|cmd_v| > 0.05).float()`：
> \[
> \operatorname{cruise\_gate} = \text{is\_nonzero\_cmd} \cdot \mathbf{1}(v_{\text{deficit}} \le 0.10)
> \]
> 当 $v_{\text{cmd}} = 0$（驻车/静止）时，$\operatorname{cruise\_gate} \equiv 0$。**完全不干预驻车状态下的轮端动作，彻底保护已验证通过的驻车平衡策略**。

> [!IMPORTANT]
> **3. 真正冻结速度权重与课程**：
> 确保阶段 25 单变量隔离：
> ```python
> tracking_lin_vel = 3.0
> tracking_lin_vel_curriculum = False
> tracking_lin_vel_start = 3.0
> tracking_sigma = 0.18  # 保持 0.18 冻结，严禁使用 0.10
> ```

> [!NOTE]
> **4. 软裕度物理边界定位**：
> $|a_c| \le 0.7$ 对应基准转矩 $\le 4.9\text{ Nm}$，预留约 $2.1\text{ Nm}$ 软裕度。由于偏航闭环（上限 $1.5\text{ Nm}$）与轮速差闭环（上限 $1.5\text{ Nm}$）同向叠加理论最大需 $3.0\text{ Nm}$，该阈值提供的是**软裕度**，表述为**“显著降低不对称截断概率”**。

> [!TIP]
> **5. 差动力矩新增裁剪专用诊断逻辑**：
> `diff_torque_clipped_fraction` 严格仅统计**“差动反馈导致的新增裁剪”**：
> 分别记录 `base_tau`（前馈 + 单轮弱阻尼后）与 `base_tau + diff_correction`。
> **仅当 `|base_tau| < 7.0 - 1e-3`（基准力矩原本未饱和），而叠加差动后 `|total_tau| >= 7.0 - 1e-3` 时**，才计入差动裁剪！杜绝将共模自身满推造成的原本饱和错误计入。

---

## 阶段 24 检查点 (18600~18900) 筛查基线

| Checkpoint | 跌倒数 | 控速 RMSE | 速度均值 | 偏航 P95 | 轮速差 P95 | 动作差 P95 | 转矩差 P95 | 双轮着地率 | 轮端饱和率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **model_18600.pt** (入选起点) | **0** | **0.146 m/s** | **0.318 m/s** | 2.706 rad/s | **22.81 rad/s**| **0.000** | **2.93 Nm** | 78.3% | 28.8% |
| **model_18650.pt** | **0** | 0.156 m/s | 0.297 m/s | **2.491 rad/s** | 27.94 rad/s | **0.000** | 3.11 Nm | **82.1%** | **26.6%** |
| **model_18700.pt** | 1 | 0.191 m/s | 0.292 m/s | 2.646 rad/s | 20.34 rad/s | 0.000 | 3.41 Nm | 75.2% | 31.7% |
| **model_18750.pt** | 1 | 0.198 m/s | 0.315 m/s | 2.823 rad/s | 23.66 rad/s | 0.000 | 3.59 Nm | 70.6% | 28.0% |
| **model_18800.pt** | **0** | 0.174 m/s | 0.277 m/s | 2.530 rad/s | 22.90 rad/s | **0.000** | 3.38 Nm | 79.4% | 33.1% |
| **model_18850.pt** | **0** | 0.170 m/s | 0.281 m/s | 2.773 rad/s | 20.43 rad/s | **0.000** | 2.93 Nm | 79.2% | 29.3% |
| **model_18900.pt** | **0** | 0.166 m/s | 0.308 m/s | 2.848 rad/s | 25.35 rad/s | **0.000** | 3.20 Nm | 77.6% | 27.4% |

**确定**：以 `model_18600.pt` 为起点。

---

## 拟议代码变更 (Proposed Changes)

### 1. 配置层：冻结速度课程并配置节流裕度参数

#### [MODIFY] [myrobot_config.py](file:///home/admin/IsaacTu/legged_gym/legged_gym/envs/myrobot/myrobot_config.py)
- 强制速度权重配置：
  ```python
  tracking_lin_vel = 3.0
  tracking_lin_vel_start = 3.0
  tracking_lin_vel_curriculum = False
  # tracking_sigma 保持 0.18
  ```
- 轮速差阻尼冻结：
  ```python
  wheel_diff_damping = 0.10
  wheel_diff_damping_start = 0.10
  wheel_diff_damping_curriculum = False
  ```
- 新增节流阈值与欠速门限：
  ```python
  wheel_throttle_threshold = 0.70          # 共模油门软上限，保留 30% (2.1 Nm) 裕度
  wheel_throttle_deficit_margin = 0.10     # 欠速容忍门限 (m/s)
  ```
- 在 `rewards.scales` 中增加：
  ```python
  wheel_throttle_margin = -2.0             # 负权重，惩罚超额油门
  ```

---

### 2. 环境与奖励层：实现非负惩罚度量与差动裁剪诊断

#### [MODIFY] [myrobot_env.py](file:///home/admin/IsaacTu/legged_gym/legged_gym/envs/myrobot/myrobot_env.py)
- 在 `_init_buffers` 与 `reset_idx` 中初始化 `self.wheel_diff_clipped` 布尔缓冲区；
- 在 `_compute_torques` 中分别提取 `base_tau` 与 `diff_correction`，仅当 `|base_tau| < limit - 1e-3` 且 `|base_tau + diff| >= limit - 1e-3` 时置 `self.wheel_diff_clipped = True`；
- 实现正定且保护驻车的奖励函数 `_reward_wheel_throttle_margin(self)`：
  ```python
  def _reward_wheel_throttle_margin(self):
      a_comm = 0.5 * (self.filtered_actions[:, self.left_wheel_idx] +
                      self.filtered_actions[:, self.right_wheel_idx])
      cmd_v = self.commands[:, 1]
      act_v = self.base_lin_vel[:, 1]
      
      cmd_sign = torch.sign(cmd_v)
      is_nonzero_cmd = (torch.abs(cmd_v) > 0.05).float()
      v_deficit = (cmd_v - act_v) * cmd_sign
      
      # 静止指令 (cmd_v=0) 严格置 0，不误伤驻车策略
      # 运动工况下仅接近目标或超速 (v_deficit <= 0.10 m/s) 时门控为 1
      cruise_gate = is_nonzero_cmd * (v_deficit <= self.wheel_throttle_deficit_margin).float()
      
      excess = torch.clamp(torch.abs(a_comm) - self.wheel_throttle_threshold, min=0.0)
      return cruise_gate * torch.square(excess)
  ```

---

### 3. 评测工具升级：增加三项专用诊断指标

#### [MODIFY] [evaluate_checkpoint.py](file:///home/admin/IsaacTu/legged_gym/legged_gym/scripts/evaluate_checkpoint.py)
- 记录巡航阶段 $|a_c| > 0.7$ 标记、采样值以及 `env.wheel_diff_clipped` 标志；
- 输出 JSON 增加：
  - `cruise_throttle_exceed_fraction`
  - `cruise_action_comm_p95`
  - `diff_torque_clipped_fraction`

---

## 验证计划 (Verification Plan)

### 1. 离线自动化测试 (全部通过)
- **数学、门控与裁剪逻辑测试** (`scratch/test_phase25_math.py`)：
  - 8 项单测全部通过：
    1. 返回值严格 $\ge 0$；
    2. 起步加速（$v_{\text{act}}=0, v_{\text{cmd}}=0.39$）满油门 0 cost；
    3. 爬坡受阻（$v_{\text{act}}=0.25, v_{\text{cmd}}=0.40$）满油门 0 cost；
    4. 稳定巡航（$v_{\text{act}}=0.35, v_{\text{cmd}}=0.39$）$|a_c| \le 0.70$ 时 0 cost，满油门触发惩罚；
    5. 超速巡航（$v_{\text{act}}=0.48, v_{\text{cmd}}=0.39$）触发惩罚；
    6. 倒车工况对称性验证；
    7. **驻车静止保护验证**（$v_{\text{cmd}}=0$ 时满油门 0 cost，严格杜绝误伤）；
    8. **差动裁剪诊断逻辑验证**（排除共模原本饱和，仅记录差动新增截断）。
- **真实 GPU 环境契约测试** (`scratch/test_phase25_contract.py`)：
  - 运行通过：参数完全冻结（`tracking_lin_vel=3.0`, `diff_damping=0.10`），步进 24 步无 NaN。

### 2. 训练与全工况评测（待用户批准后执行）
- 增量训练 300 轮（18600 $\to$ 18900）；
- 运行 21 组基准工况评测（7 场景 $\times$ 3 Seeds）；
- 核心门禁判定：直行控速 RMSE `< 0.10 m/s`、偏航 P95 `< 0.30 rad/s`、差动裁剪率显著下降、0 跌倒。
