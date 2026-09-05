# 阶段 27：差动优先力矩分配（Differential-Priority Torque Allocation）实施计划

## 目标简述

阶段 25 与 26 的时序物理诊断与实验数据证明了至关重要的一点：
**问题根源不在于节流奖励的量纲或高斯尺度，而是“奖励软约束”根本无法克服已固化的满油门开环本能**。一旦发生姿态或速度扰动导致轻微欠速，门控衰减使策略重获满推动力，重新形成“加速 $\to$ 前倾侧倾 $\to$ 单轮失载 $\to$ 差动被 7Nm 硬件不对称截断 $\to$ 偏航尖峰与剧烈掉速”的极限环振荡。

本阶段放弃依赖策略“自觉收油门”的奖励软约束思路，**从阶段 24 入选的最优基准检查点 `model_18600.pt` 分叉，彻底撤销阶段 25/26 的节流奖励，转而在执行器低层引入“差动优先力矩分配”（Differential-Priority Torque Allocation）硬性控制律**：

### 核心动力学与控制分配公式
设两轮基于动作前馈与弱速度阻尼的基准力矩为 $\tau_L^{base}, \tau_R^{base}$，偏航与防滑差动纠偏需求为 $\Delta\tau_{yaw}, \Delta\tau_\omega$。
定义共模推进力矩与期望差动力矩：
\[
\tau_c = \frac{\tau_L^{base} + \tau_R^{base}}{2}
\]
\[
\tau_d = \frac{\tau_R^{base} - \tau_L^{base}}{2} + \Delta\tau_{yaw} + \Delta\tau_\omega
\]
在电机硬件极限 $\tau_{\max} = 7.0\text{ Nm}$ 下，当差动纠偏激活（$|\tau_d| > 0$）时，优先保障差动纠偏空间，**自适应动态收拢共模推进力矩的硬上限**：
\[
\tau_c \leftarrow \operatorname{clip}\left(\tau_c,\, -(\tau_{\max} - |\tau_d|),\, \tau_{\max} - |\tau_d|\right)
\]
最终下发给左右轮的实际力矩：
\[
\tau_L = \tau_c - \tau_d, \qquad \tau_R = \tau_c + \tau_d
\]

### 物理与数学保证
1. **严格杜绝不对称硬件截断**：
   由三角不等式，$|\tau_L| = |\tau_c - \tau_d| \le |\tau_c| + |\tau_d| \le (\tau_{\max} - |\tau_d|) + |\tau_d| = \tau_{\max} = 7.0\text{ Nm}$，同理 $|\tau_R| \le 7.0\text{ Nm}$。**两轮实际力矩天然严格位于 $[-7.0, +7.0]\text{ Nm}$ 内，差动力矩 $\tau_R - \tau_L = 2\tau_d$ 获得 100% 完整执行，绝不发生逐侧截断诱发的额外推力脉冲**！
2. **按需削减推进，无偏航扰动时全额推进**：
   当无需纠偏（$\tau_d = 0$）时，$\tau_c$ 上限恢复为 $7.0\text{ Nm}$，机器人保持全额加速推进能力；仅在单轮打滑或偏航需要纠偏时，自动压低共模基准，确保纠偏瞬间不破功。

---

## 关键确认项 (User Review Required)

> [!IMPORTANT]
> **1. 分叉起点与奖励撤销**：
> - 坚决废弃 `model_19200.pt`，从阶段 24 的入选最优高巡航检查点 `model_18600.pt` 恢复分叉；
> - 彻底撤销阶段 25/26 的 `wheel_throttle_margin` 奖励项（权重置为 `0.0`），消除任何对抗性软惩罚。

> [!IMPORTANT]
> **2. 控制层 100% 严格冻结**：
> - 保持 `tracking_lin_vel = 3.0`（课程关闭，`tracking_sigma = 0.18` 严格冻结）；
> - 保持偏航差动闭环（$K_{\text{yaw}} = 1.5, \Delta\tau_{\max} = 1.5\text{ Nm}$）；
> - 保持非转向轮速差阻尼（$K_\omega = 0.10, \Delta\tau_{\max} = 1.5\text{ Nm}$）；
> - 保持动作对称惩罚（$r = -1.0$）、单轮弱速度阻尼（$K_d = 0.3$）及全部腿部 PD 控制器完全冻结。

> [!IMPORTANT]
> **3. 阶段 27 放行门禁**：
> - 直行差动新增裁剪率 `diff_torque_clipped_fraction` 必须降至 **接近 0.0%**；
> - 直行偏航 P95 `< 0.30 rad/s`；
> - 速度 RMSE 达到 `< 0.10 m/s`（或显著改善且不恶化）；
> - 21/21 零跌倒，腿部饱和率 `< 2%`；
> - **在全部门禁通过前，严格维持物理真机部署禁令**。

---

## 拟议代码变更 (Proposed Changes)

### 1. 配置层

#### [MODIFY] [myrobot_config.py](file:///home/admin/IsaacTu/legged_gym/legged_gym/envs/myrobot/myrobot_config.py)
- 撤销 `wheel_throttle_margin` 惩罚：
  ```python
  # 阶段 27: 撤销节流软裕度奖励，转由低层差动优先分配器硬保障
  wheel_throttle_margin = 0.0
  ```
- 移除 `wheel_throttle_threshold` 和 `wheel_throttle_deficit_sigma`。

---

### 2. 环境执行器层

#### [MODIFY] [myrobot_env.py](file:///home/admin/IsaacTu/legged_gym/legged_gym/envs/myrobot/myrobot_env.py)
- 在 `_compute_torques` 中实现差动优先力矩分配：
  ```python
  # 保存施加差动校正前的基准力矩 (前馈 + 单轮速度阻尼)
  base_tau_L = torques[:, self.left_wheel_idx].clone()
  base_tau_R = torques[:, self.right_wheel_idx].clone()

  # 阶段 22: 有界偏航率闭环差动转矩反馈
  yaw_rate_error = self.commands[:, 2] - self.base_ang_vel[:, 2]
  delta_tau_yaw = torch.clamp(
      self.current_wheel_yaw_damping * yaw_rate_error,
      -self.wheel_yaw_torque_limit,
      self.wheel_yaw_torque_limit,
  )

  # 阶段 24: 非转向工况下的轮速差闭环抑制
  is_straight_cmd = (torch.abs(self.commands[:, 2]) < 0.1).float()
  wheel_speed_error = self.dof_vel[:, self.left_wheel_idx] - self.dof_vel[:, self.right_wheel_idx]
  delta_tau_omega = torch.clamp(
      self.current_wheel_diff_damping * wheel_speed_error,
      -self.wheel_diff_torque_limit,
      self.wheel_diff_torque_limit,
  ) * is_straight_cmd

  # 阶段 27: 差动优先力矩分配 (Differential-Priority Torque Allocation)
  # tau_c: 共模推进力矩; tau_d: 期望差动纠偏力矩
  tau_c = 0.5 * (base_tau_L + base_tau_R)
  tau_d = 0.5 * (base_tau_R - base_tau_L) + delta_tau_yaw + delta_tau_omega

  # 电机物理极限与差动优先共模动态裁剪
  tau_limit = self.torque_limits[self.left_wheel_idx]
  tau_d_clamped = torch.clamp(tau_d, -tau_limit, tau_limit)
  tau_c_limit = torch.clamp(tau_limit - torch.abs(tau_d_clamped), min=0.0)
  tau_c_allocated = torch.clamp(tau_c, -tau_c_limit, tau_c_limit)

  # 分配至左右轮
  torques[:, self.left_wheel_idx] = tau_c_allocated - tau_d_clamped
  torques[:, self.right_wheel_idx] = tau_c_allocated + tau_d_clamped
  ```
- 简化 `_reward_wheel_throttle_margin` 返回 `0.0`。

---

## 验证计划 (Verification Plan)

### 1. 离线数学与属性单测
- `scratch/test_phase27_math.py`：
  - 10,000 组随机 $\tau_L^{base}, \tau_R^{base}, \Delta\tau_{yaw}, \Delta\tau_\omega$ 采样测试；
  - 验证 $|\tau_L| \le 7.0$ 且 $|\tau_R| \le 7.0$ 严格 100% 成立；
  - 验证当 $|\tau_d| \le 7.0$ 时，实际输出力矩差 $\tau_R - \tau_L \equiv 2\tau_d$ 严格无截断；
  - 验证无差动需求时 $\tau_c$ 允许打满 $7.0\text{ Nm}$；
  - 验证满推加速（$\tau_c = -7.0, \tau_d = 1.5$）下，输出严格为 $\tau_L = -7.0, \tau_R = -4.0$，无多余力矩脉冲。

### 2. Zero-Shot 快速离线验证 (`model_18600.pt`)
- 在不训练的情况下，直接在 `model_18600.pt` 上运行直行诊断：
  - 检验 `diff_torque_clipped_fraction` 是否直接清零；
  - 检验瞬态偏航尖峰是否被有效平抑。

### 3. 增量微调训练与 21 工况全量评测
- 从 `model_18600.pt` 续训 300 轮（至 18900 轮，命名为 `diff_priority_allocation_from_18600_to_18900`）；
- 运行 7 场景 $\times$ 3 Seeds（21 工况）全量三表评测；
- 严格对照直行偏航 `< 0.30 rad/s`、控速 RMSE `< 0.10 m/s` 门禁进行裁决。
