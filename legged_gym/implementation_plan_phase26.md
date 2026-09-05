# 阶段 26：归一化与连续高斯门控的巡航节流软裕度实施计划

## 目标简述

阶段 25 的实验与时序物理诊断查明：
1. 节流惩罚项没有伤害驻车（驻车 0 摔倒、RMSE 0.012 m/s、超额率 0.0%），且实现了全场景 21/21 零摔倒，直行共模动作均值从 `-0.7725` 收拢至 `-0.4896`；
2. 但巡航 P95 油门仍有 `95.7%` 时间打满 `1.0`，控速 RMSE（`0.175 m/s`）与偏航 P95（`2.745 rad/s`）仍未达标。

核心瓶颈在于：
- **未归一化的二次方导致惩罚最大值仅为 0.09**：$(|a_c| - 0.7)^2 \le 0.09$，配合权重 `-2.0` 时单步惩罚仅 `-0.18`（乘以 dt 后为 `-0.0036`），天然远小于最大 `3.0`（单步 `+0.060`）的速度奖励，形成约 16.7 : 1 的失配，PPO 梯度毫不犹豫选择牺牲节流以压榨速度；
- **$0.10\text{ m/s}$ 硬欠速门限制造了“免罚振荡死区”**：偏航尖峰触发侧滑跌速后，门控瞬间关断为 0，策略在起步与脱困时全力满推，刚入巡航区受微弱惩罚又发生打滑跌速，形成极限环。

**本阶段核心方案**：
将节流惩罚重构为**“归一化 + 连续高斯门控”**结构，并提升权重至 `-3.0`：
\[
c_{\text{throttle}} =
\mathbf{1}(|v_{cmd}| > 0.05)
\left[\frac{\max(0, |a_c| - 0.70)}{0.30}\right]^2
\exp\left(-\frac{\max(v_{\text{deficit}}, 0)^2}{2(0.05)^2}\right)
\]
配置权重：`wheel_throttle_margin = -3.0`。

- 在目标速度及超速巡航时，满油门（$|a_c| = 1.0$）cost 严格归一化为 **`1.0`**，单步惩罚为 **`-3.0`**，与最大线速度跟踪奖励（`+3.0`）处于**同等绝对量级**，形成强有力的负梯度迫使油门退出 1.0 饱和区；
- 欠速时（$v_{\text{deficit}} > 0$）惩罚沿高斯曲线平滑连续衰减，消除硬门限切换导致的极限环振荡死区，同时起步加速不受抑制；
- 静止驻车工况（$|v_{cmd}| \le 0.05$）通过指示函数严格置零，100% 隔离保护已通过的驻车平衡策略；
- 从 `model_18900.pt` 恢复续训，继续**完全冻结**速度权重 `3.0`、`tracking_sigma=0.18`、偏航闭环（$K_{\text{yaw}}=1.5, \Delta\tau_{\max}=1.5$）、轮速差闭环（$K_\omega=0.10, \Delta\tau_{\max}=1.5$）、动作对称（$r=-1.0$）以及全部腿部 PD 控制器。

---

## 关键确认项 (User Review Required)

> [!IMPORTANT]
> **1. 归一化与量级匹配**：
> 超额项除以 $0.30$ 后平方，使得 $|a_c| \in [0.70, 1.0]$ 时 cost 线性映射至 $[0.0, 1.0]$。配合权重 `-3.0`，巡航满油门惩罚绝对值达到 $3.0$，彻底扭转上一轮 $0.18$ 对 $3.0$ 的梯度失配。

> [!IMPORTANT]
> **2. 连续高斯门控衰减特性**：
> 取 $\sigma_{\text{deficit}} = 0.05\text{ m/s}$：
> - 巡航与超速（$v_{\text{deficit}} \le 0$）：$\exp(0) = 1.0$；
> - 微小欠速（$v_{\text{deficit}} = 0.05\text{ m/s}$，如 $v = 0.34\text{ m/s}$）：$\exp(-0.5) \approx 0.606$；
> - 中度欠速（$v_{\text{deficit}} = 0.10\text{ m/s}$，如 $v = 0.29\text{ m/s}$）：$\exp(-2.0) \approx 0.135$；
> - 起步加速（$v_{\text{deficit}} = 0.39\text{ m/s}$，如 $v = 0.0\text{ m/s}$）：$\exp(-30.4) \approx 0.0$。
> 既杜绝了硬门限断崖切断，又完全不阻碍起步冲量。

> [!IMPORTANT]
> **3. 物理真机部署禁令**：
> 当前直行控速与偏航未达门禁标准，**严格禁止物理真机部署**。

---

## 拟议代码变更 (Proposed Changes)

### 1. 配置层

#### [MODIFY] [myrobot_config.py](file:///home/admin/IsaacTu/legged_gym/legged_gym/envs/myrobot/myrobot_config.py)
- 配置更新：
  ```python
  wheel_throttle_threshold = 0.70          # 共模油门软上限，保留 30% (2.1 Nm) 裕度
  wheel_throttle_deficit_sigma = 0.05      # 连续高斯门控尺度 [m/s]
  ```
- 奖励权重更新：
  ```python
  wheel_throttle_margin = -3.0             # 归一化后与速度跟踪奖励同量级 (-3.0 vs +3.0)
  ```

---

### 2. 环境与奖励层

#### [MODIFY] [myrobot_env.py](file:///home/admin/IsaacTu/legged_gym/legged_gym/envs/myrobot/myrobot_env.py)
- 初始化新增读取：
  ```python
  self.wheel_throttle_deficit_sigma = getattr(self.cfg.rewards, 'wheel_throttle_deficit_sigma', 0.05)
  ```
- 重写 `_reward_wheel_throttle_margin`：
  ```python
  def _reward_wheel_throttle_margin(self):
      a_comm = 0.5 * (self.filtered_actions[:, self.left_wheel_idx] +
                      self.filtered_actions[:, self.right_wheel_idx])
      cmd_v = self.commands[:, 1]
      act_v = self.base_lin_vel[:, 1]

      cmd_sign = torch.sign(cmd_v)
      is_nonzero_cmd = (torch.abs(cmd_v) > 0.05).float()
      v_deficit = (cmd_v - act_v) * cmd_sign

      # 连续高斯门控: 欠速时连续平滑衰减，巡航或超速时为 1.0
      deficit_pos = torch.clamp(v_deficit, min=0.0)
      continuous_gate = torch.exp(-torch.square(deficit_pos) / (2.0 * (self.wheel_throttle_deficit_sigma ** 2)))

      # 归一化油门超额项: [0, 1] 范围
      excess_span = max(1e-4, 1.0 - self.wheel_throttle_threshold)
      normalized_excess = torch.clamp(torch.abs(a_comm) - self.wheel_throttle_threshold, min=0.0) / excess_span
      return is_nonzero_cmd * continuous_gate * torch.square(normalized_excess)
  ```

---

## 验证计划 (Verification Plan)

### 1. 离线自动化测试
- **数学单测** (`scratch/test_phase26_math.py`)：
  - 测试 1：返回值严格非负；
  - 测试 2：$|a_c| \le 0.70$ 时 cost 严格为 0；
  - 测试 3：巡航 $|a_c| = 1.0$ 时 cost 严格为 1.0；
  - 测试 4：起步加速（$v=0, v_{cmd}=0.39$）时 cost 严格为 0；
  - 测试 5：连续衰减平滑性（$v_{\text{deficit}} = 0.05 \to \sim 0.606, 0.10 \to \sim 0.135$）；
  - 测试 6：超速巡航（$v > v_{cmd}$）满油门 cost 为 1.0；
  - 测试 7：倒车对称性；
  - 测试 8：静止驻车保护（$v_{cmd} = 0$ 时满油门 cost 严格为 0）。
- **GPU PhysX 契约测试** (`scratch/test_phase26_contract.py`)：
  - 加载环境与 `model_18900.pt`，验证单步与 24 步前向，无 NaN、无力矩越界。

### 2. 审查与执行流
- 代码修改完成后，执行测试并向用户输出 Diff 及单测结果；
- 用户审查通过后，方可启动续训与评测流程。
