# MyRobot Phase 6A-R：PPO 异常归因与更新机制深入审计报告

**报告版本**：v1.0-Formal  
**执行角色**：MyRobot 强化学习执行开发 Agent  
**代码基线分支**：`phase6a-kl-early-stopping` (`6b3beb658b9eb210423906292b6e94a3c2a3abcd`)  
**历史冻结基准**：`backup/phase5.3-freeze` (`d70f164e52acd5e2c9376c5a455d6897f73f5a78`)  
**主基线检查点**：`legged_gym/logs/myrobot/Oct10_15-48-32_turning_height_stage5_3_formal/model_25.pt`  
**基线 SHA256**：`d67474d0e47f908a8c863e605b8d5de7ea1bd9f89f41d2c303fbef33d963536c`  
**审计日期**：2026-10-11  

---

## 摘要与执行概述

在 Phase 6A 实验中，引入 KL Early Stopping（B 组）后，策略在 21 工况基准下的联合达标率从 Stage 1 model_25 的 **44.07%** 骤降至 **9.96%**，甚至大幅落后于原始 PPO（A 组，**37.24%**）。初版技术报告将该严重退化初步归因于“Critic 饥饿”。

本阶段（Phase 6A-R）通过源码级静态审查、单步 Rollout/Update 确定性回放、网络权重漂移轨迹分析及优化器状态溯源，完成了对异常现象的全链路归因审计。

### 核心审计结论：
1. **KL 计算公式本身数学正确，但物理尺度严重不对称**：
   轮部关节由于在 Stage 1 中收敛到了极小的探索方差（$\sigma_{wheel} \approx 0.10$ vs $\sigma_{knee} \approx 0.93$），在对角高斯 KL 散度计算中，车轮维度的误差权重被放大了 **$86.5$ 倍**。单个 mini-batch 更新后，轮部两个驱动维度贡献了高达 **84.57%** 的总 KL 散度。
2. **算术平均受极端离群样本主导（Fallen State Outliers）**：
   在单步更新后，mini-batch 内 KL 中位数仅为 **0.0188**（未超 0.02 阈值），但均值却被拉高至 **0.1504**，最大单样本 KL 达到 **454.82**。其根因是未归一化的大观测值（摔倒/高度越界机器人，观测模长高达 126.66，Advantage 高达 +56.4）通过未受限线性层直接放大了输出均值漂移 $\Delta\mu$。
3. **Warm Start 破坏了 Adam 动量并引发了“伪极小 KL 假象”**：
   Warm Start 清空了 Adam 的一阶与二阶矩缓存（重置为 0），并将学习率从 model_25 收敛时的 $7.59 \times 10^{-5}$ 强行提升至标称 $2.5 \times 10^{-4}$（$3.3\times$ 跃升）。在首个 mini-batch 开始前，策略与旧样本同源，KL 理论为 0（数值计算为 $6.0 \times 10^{-5}$），Adaptive LR 调度误判为“KL 极小”，在更新前**再次将学习率提升 1.5 倍至 $3.75 \times 10^{-4}$**（相当于收敛 LR 的近 5 倍），导致首步梯度范数爆发（Actor 梯度范数高达 53.79），单步即破坏了策略。
4. **原“Critic 饥饿”结论缺乏因果隔离**：
   PPO 实现中 Actor 与 Critic 共享 mini-batch 循环与同一个优化器。Early Stopping 触发时同时斩断了 Actor 与 Critic 的更新。B 组每轮仅更新 1 个 mini-batch（只遍历了全量采样的 25%），Actor 同样遭受了严重的数据饥饿与离群样本偏置。性能崩溃是**高 LR 单步离群冲击**与 **Critic/Actor 双重训练饥饿**共同作用的恶果。
5. **Phase 6B 准入判定**：
   当前**不具备**直接进入 Phase 6B 的条件。必须先修复 Warm Start 优化器继承/LR Warmup 与 KL 离群截断问题。

---

## 一、源码位置与调用链审计

本审计涉及的核心调用链路及关键代码位置如下表所示：

| 功能模块 | 文件路径 | 函数 / 方法 | 关键代码行 | 功能与审计描述 |
| :--- | :--- | :--- | :--- | :--- |
| **训练入口与配置加载** | [`legged_gym/scripts/train.py`](file:///home/xy/IsaacTu/legged_gym/legged_gym/scripts/train.py) | `train()` | L27-L32 | 初始化环境与 Runner，调用 `task_registry.make_alg_runner` |
| **Warm Start 加载逻辑** | [`legged_gym/utils/task_registry.py`](file:///home/xy/IsaacTu/legged_gym/legged_gym/utils/task_registry.py) | `make_alg_runner()` | L168-L173 | 加载 `model_25.pt`，显式设置 `load_optimizer=False`，导致 Adam 动量清空 |
| **训练主循环控制** | [`rsl_rl/runners/on_policy_runner.py`](file:///home/xy/IsaacTu/rsl_rl/rsl_rl/runners/on_policy_runner.py) | `learn()` | L105-L151 | 执行 Rollout 采样、调用 `alg.compute_returns()` 与 `alg.update()` |
| **检查点存取接口** | [`rsl_rl/runners/on_policy_runner.py`](file:///home/xy/IsaacTu/rsl_rl/rsl_rl/runners/on_policy_runner.py) | `load()`, `save()` | L253-L273 | `load` 方法接收 `load_optimizer` 参数，决定是否恢复优化器状态 |
| **PPO 优化器与参数初始化** | [`rsl_rl/algorithms/ppo.py`](file:///home/xy/IsaacTu/rsl_rl/rsl_rl/algorithms/ppo.py) | `__init__()` | L78 | 创建 `optim.Adam(self.actor_critic.parameters(), lr=learning_rate)`，未继承历史状态 |
| **Mini-batch 生成器** | [`rsl_rl/storage/rollout_storage.py`](file:///home/xy/IsaacTu/rsl_rl/rsl_rl/storage/rollout_storage.py) | `mini_batch_generator()` | L156-L198 | 将 49,152 条采样划分为 4 个 batch，每个 batch 12,288 条样本 |
| **策略前向与动作分布** | [`rsl_rl/algorithms/ppo.py`](file:///home/xy/IsaacTu/rsl_rl/rsl_rl/algorithms/ppo.py) | `update()` | L157-L163 | 计算当前 `mu_batch`、`sigma_batch`、`value_batch` |
| **数值稳定性检查** | [`rsl_rl/algorithms/ppo.py`](file:///home/xy/IsaacTu/rsl_rl/rsl_rl/algorithms/ppo.py) | `update()` | L164-L168 | 检查 `mu` 与 `sigma` 中是否存在 NaN/Inf 或非正数值 |
| **KL 散度计算实现** | [`rsl_rl/algorithms/ppo.py`](file:///home/xy/IsaacTu/rsl_rl/rsl_rl/algorithms/ppo.py) | `update()` | L170-L175 | 对角高斯解析 KL 计算，并追加至 `kl_all` 列表 |
| **自适应 LR 调度更新** | [`rsl_rl/algorithms/ppo.py`](file:///home/xy/IsaacTu/rsl_rl/rsl_rl/algorithms/ppo.py) | `update()` | L176-L185 | 根据 `kl_mean` 动态缩放 `self.learning_rate` 并同步到优化器 |
| **KL Early Stopping 判定** | [`rsl_rl/algorithms/ppo.py`](file:///home/xy/IsaacTu/rsl_rl/rsl_rl/algorithms/ppo.py) | `update()` | L186-L191 | 若 `kl_mean > early_stop_kl`，设置标志并执行 `break` 跳出 mini-batch 循环 |
| **损失计算与反向传播** | [`rsl_rl/algorithms/ppo.py`](file:///home/xy/IsaacTu/rsl_rl/rsl_rl/algorithms/ppo.py) | `update()` | L192-L225 | 计算 Surrogate、Value Loss，梯度裁剪并执行 `optimizer.step()` |
| **迭代汇总指标统计** | [`rsl_rl/algorithms/ppo.py`](file:///home/xy/IsaacTu/rsl_rl/rsl_rl/algorithms/ppo.py) | `update()` | L235-L282 | 聚合 `kl_cat`、损失均值、参数变化量，存入 `last_update_diagnostics` |

---

## 二、KL 计算与统计异常分析

### 2.1 数学公式与实现映射

在连续动作空间下，策略假定为多维对角高斯分布 $\pi(\cdot|s) = \mathcal{N}(\mu(s), \text{diag}(\sigma^2))$。采样旧策略为 $\pi_{old} = \mathcal{N}(\mu_{old}, \text{diag}(\sigma_{old}^2))$，当前更新策略为 $\pi_{curr} = \mathcal{N}(\mu_{curr}, \text{diag}(\sigma_{curr}^2))$。

从 $\pi_{old}$ 到 $\pi_{curr}$ 的对角多元高斯 KL 散度理论公式为：
$$D_{\mathrm{KL}}(\pi_{old} \parallel \pi_{curr}) = \int \pi_{old}(a|s) \log \frac{\pi_{old}(a|s)}{\pi_{curr}(a|s)} da$$
其解析闭式解为：
$$D_{\mathrm{KL}}(\pi_{old} \parallel \pi_{curr}) = \sum_{i=1}^{d} \left[ \log \frac{\sigma_{curr, i}}{\sigma_{old, i}} + \frac{\sigma_{old, i}^2 + (\mu_{old, i} - \mu_{curr, i})^2}{2 \sigma_{curr, i}^2} - \frac{1}{2} \right]$$

代码中对应实现（[`ppo.py:171-173`](file:///home/xy/IsaacTu/rsl_rl/rsl_rl/algorithms/ppo.py#L171-L173)）：
```python
kl = torch.sum(
    torch.log(sigma_batch / old_sigma_batch + 1.e-5) 
    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) 
    - 0.5, 
    axis=-1
)
```
**数学审计判定**：该公式与符号方向完全一致，无代数错误。

### 2.2 均值变化 vs 标准差变化对 KL 的贡献分解

在单步更新回放测试中，我们在第 1 个 mini-batch 步进后，对第 2 个 mini-batch（12,288 样本）的各分项进行了数值分离：
- **标准差变化贡献项**：
  $$\Delta_{\sigma} = \sum_{i=1}^6 \left[ \log \frac{\sigma_{curr, i}}{\sigma_{old, i}} + \frac{\sigma_{old, i}^2}{2\sigma_{curr, i}^2} - 0.5 \right]$$
  实测值：**$0.000021$**（占总 KL 比例 **$< 0.015\%$**）。
- **动作均值变化贡献项**：
  $$\Delta_{\mu} = \sum_{i=1}^6 \frac{(\mu_{old, i} - \mu_{curr, i})^2}{2\sigma_{curr, i}^2}$$
  实测值：**$0.150374$**（占总 KL 比例 **$99.985\%$**）。

**结论**：在 10 轮微调的早期阶段，可学习参数 $\sigma$ 变化极其微弱（基线 $\sigma$ 漂移 L2 仅 0.0009），**KL 散度的数值爆炸 99.99% 来自于 Actor MLP 输出均值 $\mu$ 的漂移**。

### 2.3 轮部小方差导致的物理维度不对称性量化

基线 `model_25.pt` 在 Stage 1 收敛后的 6 维动作噪声标准差如下：
- `L_hip`：$0.3675$
- `L_knee`：$0.9299$
- `L_wheel`：**$0.1008$**
- `R_hip`：$0.3764$
- `R_knee`：$0.9292$
- `R_wheel`：**$0.1052$**

在 KL 均值项公式中，分母为 $2\sigma_i^2$：
- 膝关节：$2 \times (0.93)^2 \approx 1.73$
- 轮部关节：$2 \times (0.10)^2 \approx 0.020$
- **敏感度倍数比**：$\frac{1.73}{0.020} = \mathbf{86.5 \times}$！

这意味着：**轮部速度指令产生同样的数值偏离（例如 $\Delta\mu = 0.2\ \text{rad/s}$），所产生的 KL 惩罚是膝关节角度同样偏离的 86.5 倍！**

通过回放实验 Mini-batch 2 逐维度统计证实：

| 关节名称 | 物理意义 | 收敛标准差 $\sigma_i$ | 单步后分维度平均 KL | 占总 KL 比例 |
| :--- | :--- | :--- | :--- | :--- |
| `L_hip` | 左髋摆动角 | $0.3673$ | $0.014031$ | $9.33\%$ |
| `L_knee` | 左膝伸屈角 | $0.9297$ | $0.001212$ | $0.81\%$ |
| **`L_wheel`** | **左驱动轮速度** | **$0.1006$** | **$0.071939$** | **$47.83\%$** |
| `R_hip` | 右髋摆动角 | $0.3766$ | $0.002544$ | $1.69\%$ |
| `R_knee` | 右膝伸屈角 | $0.9289$ | $0.005421$ | $3.60\%$ |
| **`R_wheel`** | **右驱动轮速度** | **$0.1049$** | **$0.055249$** | **$36.74\%$** |
| **合计** | - | - | **$0.150395$** | **$100.00\%$** |

> **关键证据**：仅 `L_wheel` 与 `R_wheel` 两个轮部维度，就贡献了 **$84.57\%$** 的总 KL 散度！KL 提前停止几乎完全由车轮小方差维度触发。

### 2.4 离群样本主导算术平均（以 #399 摔倒样本为例）

审计进一步揭示，Mini-batch 2 的样本 KL 呈现极端厚尾分布：
- 样本中位数（Median）：**$0.0188$**（**低于 0.02 阈值！**）
- 90分位数（p90）：$0.1400$
- 95分位数（p95）：$0.2457$
- 99分位数（p99）：$0.5457$
- **最大值（Max）：$454.8171$**
- 算术平均值（Mean）：**$0.1504$**

从分布可见，**超过 50% 的正常平稳样本其实际更新幅度完全处于 0.02 的安全信赖域内**。但少数离群样本将整体均值暴力拉高了 8 倍！

**离群样本 #399 深度剖析**：
- **物理状态**：机器人已摔倒处于重置边缘（机身高度 `root_height = -5.7656`，全身观测向量二范数达到 **$126.66$**，而正常直立样本仅为 $5 \sim 15$）。
- **优势函数**：GAE Advantage 高达 **$+56.41$**。
- **输出漂移**：
  $$\mu_{old} = [-22.81,\ 71.60,\ -8.65,\ 46.47,\ 188.93,\ -4.27]$$
  $$\mu_{curr} = [-26.21,\ 74.15,\ -6.38,\ 45.44,\ 195.26,\ -2.60]$$
  $$\Delta\mu = [-3.40,\ +2.54,\ +2.27,\ -1.03,\ +6.33,\ +1.67]$$
- **该样本单点 KL 分解**：
  - 左轮 KL 贡献：**$254.64$**
  - 右轮 KL 贡献：**$126.66$**
  - 样本总 KL：**$454.82$**

**机理解释**：Actor 网络输出层为未经有界激活函数截断的线性层（`nn.Linear(128, 6)`）。当处于摔倒或碰撞边缘的极端观测输入时，未归一化的大特征通过高权值直接放大输出均值变动，叠加轮部 $1/(2\sigma^2) \approx 50$ 的放大系数，单样本瞬时爆出成百上千的极端 KL。

---

## 三、Warm Start 冲击与优化器瞬态分析

### 3.1 Adam 优化器状态丢失与学习率断层

基线检查点 `model_25.pt` 是 Stage 1 经过 500 个梯度更新步（50 迭代）后保存的模型：
- `model_25.pt` 中保存的 Adam 优化器状态包含完整的 17 组参数一阶动量 $m_t$ 和二阶动量 $v_t$（`step = 500`）。
- 其收敛时的自适应学习率为：**$7.59375 \times 10^{-5}$**。

然而在 Phase 6A 中，[`task_registry.py:172`](file:///home/xy/IsaacTu/legged_gym/legged_gym/utils/task_registry.py#L172) 采用了：
```python
runner.load(warm_start_path, load_optimizer=False)
```
这一操作导致：
1. **Adam 历史动量被彻底抹除**：新建的 Adam 优化器内部状态为空（`state={}`）。
2. **初始学习率突增 3.3 倍**：强制重置为配置中的标称学习率 **$2.5 \times 10^{-4}$**。

当 Adam 在 $m_0 = 0, v_0 = 0$ 的初始状态下执行第一步时：
$$\hat{m}_1 = g_1, \quad \hat{v}_1 = g_1^2 \implies \Delta\theta_1 = -\alpha \frac{\hat{m}_1}{\sqrt{\hat{v}_1} + \epsilon} \approx -\alpha \cdot \text{sign}(g_1)$$
优化器失去了平滑动量阻尼，对所有非零梯度的网络参数执行了接近最大标称步长（$\approx 2.5 \times 10^{-4}$）的全量跨越！

### 3.2 初始梯度范数爆发

在 Iteration 1 的 Mini-batch 1 执行反向传播时，测量得到的梯度范数如下：
- **Actor 参数梯度范数**：**$53.7854$**
- **Critic 参数梯度范数**：**$6.3653$**
- **Std 参数梯度范数**：**$0.1944$**
- **全网络总梯度范数**：**$54.1611$**（被 `clip_grad_norm_` 强行截断至 $1.0$）

> **关键对比**：Actor 梯度范数是 Critic 的 **8.45 倍**！Actor 网络承受了极大的更新压力。

### 3.3 自适应调度逻辑在第 1 步的严重误判

审计发现了原有 RSL-RL Adaptive LR 与 Early Stopping 组合下的致命时序耦合：

在 Iteration 1 的 **Mini-batch 1** 中：
1. 此时参数尚未发生任何梯度更新，$\mu_{batch}$ 与 $\mu_{old\_batch}$ 完全相同。
2. 计算得到的 KL 散度为纯数值浮点残差：$D_{\mathrm{KL}} \approx 6.008 \times 10^{-5}$。
3. 检查自适应逻辑（[`ppo.py:180`](file:///home/xy/IsaacTu/rsl_rl/rsl_rl/algorithms/ppo.py#L180)）：
   ```python
   elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
       self.learning_rate = min(1e-2, self.learning_rate * 1.5)
   ```
   因为 $0 < 6.008 \times 10^{-5} < 0.005$（`desired_kl / 2.0`），该分支判定为**真**！
4. **致命加速**：PPO 误以为策略更新幅度严重不足，在执行第 1 次优化器 step 前，将学习率**乘以 1.5 倍提升到了 $3.75 \times 10^{-4}$**！
5. 结果：**Mini-batch 1 实际上是以 $3.75 \times 10^{-4}$（相当于 model_25 收敛值的 $4.94\times$）的高昂学习率执行了无动量的 Adam 更新！**

随后进入 **Mini-batch 2**：
1. 由于 Step 1 的超大步幅，计算出的 KL 均值暴涨至 $0.3811$（$> 0.02$）。
2. 自适应逻辑将学习率除以 1.5，退回 $0.25 \times 10^{-4}$。
3. **KL Early Stopping 触发，直接 `break` 终止了本次迭代！**
4. 剩余 18 个 mini-batch 全部跳过，本轮总更新次数固定为 1 次。

> **因果闭环**：
> 在整个 10 轮训练中，B 组陷入了死循环：每轮开始时在 Mini-batch 1 被误判提速至 $3.75 \times 10^{-4}$ 步进一次，紧接着在 Mini-batch 2 触发超限熔断，随后降速回 $2.5 \times 10^{-4}$ 并跳出。**B 组所有实际执行的 11 次更新，全部是在高达 $3.75 \times 10^{-4}$ 的极高学习率下进行的单步粗暴跨越！**

反观 **A 组（Vanilla PPO）**：
A 组虽在 Mini-batch 1 同样遭遇提速，但在后续 19 个 mini-batch 中未被提前停止中断，自适应调度能够持续发挥作用，连续除以 1.5，在 Iteration 1 结束前将学习率平抑至 **$1.0 \times 10^{-5}$** 的安全区间，从而使后续迭代的策略平稳收敛。

---

## 四、Critic 饥饿假设的因果隔离审计

在 Phase 6A 报告中，作者曾将 B 组的严重退化归因为“Critic 饥饿”（Critic 仅更新 11 次导致价值网络失效）。本轮审计对其因果隔离性进行了严格审验。

### 4.1 权重漂移与更新量的量化对比

通过提取各阶段真实权重张量计算得到的参数漂移量如下：

| 评估对象 | 累计优化器更新数 | Actor 参数漂移模长 $\Delta\theta_A$ | Actor 漂移百分比 | Critic 参数漂移模长 $\Delta\theta_C$ | Critic 漂移百分比 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **基线 `model_25`** | 0 | 0.0000 | 0.00% | 0.0000 | 0.00% |
| **Group A (Vanilla) m1** | 20 | 0.2393 | 0.64% | 0.2209 | 0.45% |
| **Group A (Vanilla) m5** | 100 | 0.5149 | 1.39% | 0.6805 | 1.39% |
| **Group A (Vanilla) m10**| **200** | **0.7623** | **2.05%** | **0.9405** | **1.92%** |
| **Group B (EarlyStop) m1**| 1 | 0.1584 | 0.43% | 0.1509 | 0.31% |
| **Group B (EarlyStop) m5**| 6 | 0.4299 | 1.16% | 0.4050 | 0.83% |
| **Group B (EarlyStop) m10**| **11** | **0.5885** | **1.58%** | **0.5428** | **1.11%** |

### 4.2 为什么“原报告归因为 Critic 饥饿”缺乏因果隔离？

原结论存在两处根本性的因果混淆（Confounding）：

1. **Actor 与 Critic 的优化步骤完全同构与耦合**：
   在 [`ppo.py`](file:///home/xy/IsaacTu/rsl_rl/rsl_rl/algorithms/ppo.py) 中，Actor 与 Critic 的梯度合并在同一个标量损失中：
   $$\mathcal{L} = \mathcal{L}_{surr} + c_1 \mathcal{L}_{val} - c_2 \mathcal{H}$$
   二者通过同一个 `self.optimizer.step()` 更新。当 Early Stopping 触发时，**停止的是整个 mini-batch 循环，Actor 与 Critic 均被同时截断**。
   数据证明：B 组完成 11 次更新时，Critic 更新了 11 次，Actor **也同样只更新了 11 次**。不能仅将后果单独归咎于 Critic。

2. **Actor 遭受了极端的数据欠拟合与采样偏置（Actor Sampling Starvation）**：
   每一轮环境采集了 2,048 个环境 $\times 24$ 步 = 49,152 条转移样本，划分为 4 个 mini-batch（每个 12,288 条）。
   在 B 组中，由于每轮都在 Mini-batch 2 中途退出，**Actor 在整轮训练中仅仅见过了 Mini-batch 1 中的 12,288 条样本，直接丢弃了本轮 75% 的新鲜探索轨迹！**
   并且，每次这唯一的更新都是在一个未经多批次平滑、受离群摔倒样本强烈拉扯的批次上执行的。
   从权重表可以看出，**B 组用区区 11 次更新，就产生了 1.58% 的 Actor 相对位移，达到了 A 组 200 次更新位移的 77%！** 这表明 Actor 经历了极其剧烈且偏置的抖动，并非处于受保护的平稳状态。

**判定结论**：
“Critic 饥饿”是一个**与数据观测一致的现象，但绝非经过因果隔离的唯一根因**。策略崩溃的真实根源是：**高初始 LR 下的单步偏置更新（Actor Update Distortion）与价值网络欠拟合（Critic Starvation）双重作用引发的恶性循环。**

---

## 五、提前停止实现语义与代码缺陷审查

### 5.1 提前停止的时序语义

审查 [`ppo.py:186-224`](file:///home/xy/IsaacTu/rsl_rl/rsl_rl/algorithms/ppo.py#L186-L224)：
```python
# 1. 监测 KL
if self.enable_early_stopping and kl_mean > self.early_stop_kl:
    early_stop_triggered = True
    early_stop_reason = f"..."
    break

# 2. 计算损失与反向传播 (若已 break 则不执行)
loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
self.optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
self.optimizer.step()
num_updates_executed += 1
```
**时序核查**：
- 提前停止检测发生在 `optimizer.step()` **之前**。
- 触发超限的当前 mini-batch **确实未执行** 梯度反向传播和参数更新。
- 整个 Rollout Storage 在跳出后通过 `finally: self.storage.clear()` 被完整清空，未将陈旧样本带入下一轮。

### 5.2 发现的实现缺陷清单

#### 缺陷 1：指标统计分母与采样范围跨组严重失配（统计口径缺陷）
在 [`ppo.py:174`](file:///home/xy/IsaacTu/rsl_rl/rsl_rl/algorithms/ppo.py#L174)：
```python
kl_all.append(kl.detach())
```
该语句位于 `if self.enable_early_stopping: break` 之前。
- **Group A**：完整执行 5 个 epoch $\times 4$ 个 mini-batch = 20 个 batch，`kl_all` 累计了 245,760 个样本，计算得到的 `kl_mean` 是这 20 次更新后策略轨迹的全局平均（0.4785）。
- **Group B**：执行到 Mini-batch 2 时中断，`kl_all` 仅包含了 Mini-batch 1（未更新样本，KL=0）和 Mini-batch 2（超限样本，KL=0.38），总计 24,576 个样本。其日志记录的 `kl_mean` 为 $\frac{0 + 0.38}{2} \approx 0.1906$。
- **后果**：A 组与 B 组日志中的 `kl_mean` 样本总体与采样时刻完全不同，直接拿两者的日志均值对比更新幅度属于口径错误。

#### 缺陷 2：Mini-batch 首步 Adaptive LR 自加速陷阱（算法时序缺陷）
RSL-RL 原生设计的 Adaptive LR 是针对全量 Epoch 迭代的。在每个 Iteration 的第 1 个 mini-batch，策略尚未更新，测得的 KL 必然极小（接近 0）。原有代码直接在 mini-batch 循环内对 $KL \approx 0$ 执行 `lr = lr * 1.5`，导致每一轮的第 1 次更新都被迫承受高学习率冲击。当配合 Early Stopping 时，由于循环在第 2 步被斩断，这个被错误放大的 LR 无法通过后续 mini-batch 得到多步衰减纠正。

#### 缺陷 3：Actor 与 Critic 强行共用早停（架构设计缺陷）
PPO 的信赖域理论（Trust Region）仅约束策略网络 Actor 的分布距离 $D_{\mathrm{KL}}(\pi_\theta \parallel \pi_{\theta_{old}})$。价值函数 Critic 是对贝尔曼期望方程的均方误差拟合，完全不需要受策略 KL 的限制。将 Actor 的早停直接作为 Critic 停止学习的开关，属于架构上的连带误伤。

---

## 六、结论定性分级

依据审查证据的客观性，将所有审计发现定性分级如下：

### A 类：已证明事实（具有直接、精确的实测数据支撑）
1. **KL 解析闭式实现数学正确**，代码与标准对角高斯定义无符号错误。
2. **均值位移主导 KL 散度**：在 Phase 6A 条件下，$\Delta\mu$ 贡献了 **$99.99\%$** 的 KL 散度，$\Delta\sigma$ 贡献不到 $0.01\%$。
3. **驱动轮小方差造成物理敏感度不对称**：轮部方差（$\approx 0.10$）相较膝关节（$\approx 0.93$）将均值偏差放大了 **$86.5$ 倍**，轮部两关节贡献了 **$84.57\%$** 的总 KL。
4. **算术平均受摔倒离群样本（如 #399）主导**：Mini-batch 2 中位数 KL 为 $0.0188$（$<0.02$），但在摔倒大观测样本拉动下，均值飙升至 $0.1504$，$KL_{max} = 454.82$。
5. **Warm Start 导致 Adam 状态清空与 LR 突变**：`model_25.pt` 的 Adam 动量被丢弃，学习率从收敛值 $7.59 \times 10^{-5}$ 跳变至 $2.5 \times 10^{-4}$。
6. **Adaptive LR 逻辑在首步将学习率推升至 $3.75 \times 10^{-4}$**：首步梯度范数达到 $54.16$（Actor 53.79 vs Critic 6.37），引发单步越界。
7. **B 组 10 轮仅执行 11 次更新**，不仅 Critic 极度欠拟合，Actor 同样丢弃了 75% 的环境探索数据，并在粗暴步长下单步漂移达 $1.58\%$。
8. **跨组 `kl_mean` 指标分母与聚合样本口径不一致**（A 组 245,760 样本 vs B 组 24,576 样本）。

### B 类：一致假设（符合实验观测，但此前缺乏因果隔离）
1. **“Critic 欠拟合导致策略性能下降”**：现象与高方差 Value Loss 一致，但由于 Critic 与 Actor 更新步数完全同步被截断，且 Actor 遭受了极端的数据欠拟合与离群步进，**二者高度共线性，不能作为单一独立因果**。
2. **“策略漂移主要由于车轮转速超调引发”**：现象与轮部 84.5% KL 贡献一致，但在闭环多工况仿真中，躯干俯仰失稳与车轮打滑存在双向耦合。

### C 类：未验证疑问（必须在下一阶段通过增量实验确认）
1. **解耦 Actor 与 Critic 更新后，仅由 Actor 提前停止是否能挽救性能？**（需验证单纯补充 Critic 更新是否足够）。
2. **若保留 `model_25.pt` 的 Adam 动量与 $7.59 \times 10^{-5}$ 学习率，Vanilla PPO 是否还会发生技能退化？**
3. **Reference-Policy KL 正则化是否能够在允许充分更新的同时，彻底免疫单步离群样本冲击？**

---

## 七、下一阶段候选技术路线深度对比

针对 Phase 6A 暴露出的根本问题，对四种后续改进路线进行全维度横向比选：

| 维度 | 路线 1：优化器单步更新控制 (LR Warmup / 小步长) | 路线 2：解耦 Actor 提前停止与 Critic 持续拟合 | 路线 3：Reference-Policy KL 正则化 (锚定冻结基线) | 路线 4：Vanilla PPO Warm Start 平滑 (继承 Adam 状态) |
| :--- | :--- | :--- | :--- | :--- |
| **针对的核心问题** | 初始高 LR 冲击与首步无动量越界 | Critic 训练饥饿与更新截断连带效应 | 策略随迭代持续累积漂移、技能退化 | Adam 矩缓存丢失与学习率断层跳变 |
| **代码修改范围** | 极小（修改 LR 调度与 Warmup 配置） | 中等（在 `ppo.py` 中拆分 Actor 与 Critic 更新） | 中等（引入参考策略网络前向与正则损失） | 极小（修改 `task_registry.py` 中 `load_optimizer`） |
| **损失函数变更** | **无变更**（保持原始 PPO 损失） | **无变更**（保持原始 PPO 损失） | **新增损失项**：$\mathcal{L}_{ppo} - \beta D_{\mathrm{KL}}(\pi_{ref} \parallel \pi_\theta)$ | **无变更**（保持原始 PPO 损失） |
| **与 Phase 5.3 可比性**| **100% 完全可比** | **高度可比**（仅调整更新频次） | **算法级改进**（增加了锚定正则项） | **100% 完全可比** |
| **抗离群样本鲁棒性** | 中等（整体步长缩短，但极端样本仍存） | 较差（仍使用 Mini-batch 均值判定早停） | **极高**（连续凸惩罚代替硬截断，梯级软约束） | 中等（Adam 历史二阶矩可压制异常梯度） |
| **能否解决累积漂移** | 否（仅延缓单步漂移速度） | 否（仅缓解 Critic 欠拟合，未限制多轮漂移）| **是**（以 `model_25.pt` 为绝对锚点锁死漂移） | 否（多轮更新后策略仍会偏离 Stage 1） |
| **工程实现风险** | 极低 | 低（需精细处理梯度计算与显存占用） | 低（需常驻一个冻结权重网络） | 极低 |

### 路线综合推荐与排序：

1. **工程基石（前置必选项，不可单独作为算法对比）：路线 4 + 路线 1**
   - 无论后续采用何种算法，**在 Warm Start 中清空优化器状态并调大学习率都是严重的技术隐患**。必须无条件在 `task_registry.py` 中启用 `load_optimizer=True`（或至少对 LR 实施余弦 Warmup 并修复 Mini-batch 1 误加速 Bug）。这是所有微调实验的前提。
2. **算法首选方向：路线 3（Reference-Policy KL 正则化）**
   - **理论深度论证**：
     Phase 6A 尝试的“Mini-batch KL Early Stopping”存在本质局限——它只能约束**当前 mini-batch 相对于当前轮次开始前采样策略**的局部漂移，对跨 Iteration 的全局累积漂移完全失明！在 10 轮乃至 50 轮训练中，即使每轮局部 KL 很小，策略也会像布朗运动一样迅速漂离 Stage 1 的黄金工作点。
     而 **Reference-Policy KL 正则化**（$\beta$-KL penalty）引入了 Stage 1 冻结基线 $\pi_{ref}$ 作为全局锚点，无论训练进行多少步，策略只要偏离已证明优秀的基准技能，就会受到平滑反向梯度的拉回。同时，它**不需要提前停止任何更新**，Critic 可以充分拟合 200 次更新，Actor 也可以平稳遍历所有样本，从数学机理上彻底免疫了“更新饥饿”与“硬熔断”的弊端。
3. **算法备选方向：路线 2（解耦 Actor Early Stopping）**
   - 作为消融对照组。若评审专家希望坚持“提前停止”这一原始技术路线，则必须将 Actor 与 Critic 的更新解耦，保证 Critic 完成全部 20 个 mini-batch，且 Actor 的 Early Stopping 判定必须改用中位数（Median）或剔除离群点后的截断均值。

---

## 八、评审结论与 Phase 6B 准入建议

### 8.1 准入明确判定

> **核心结论**：
> **当前阶段【坚决不建议】直接进入 Phase 6B 全量训练。**

**判定理由**：
1. Phase 6A 暴露出的性能暴跌，其主因并非单纯的“KL Early Stopping 这一概念不可行”，而是受**Warm Start 优化器状态丢失**、**Adaptive LR 首步自加速**、**车轮小方差敏感度失衡**以及**离群样本主导算术平均**等多重实现缺陷与机制错配共同打击所致。
2. 若在未修复上述缺陷的前提下贸然进入 Phase 6B，任何新算法都将在极其不稳定的优化瞬态下运行，无法得出具有科学严谨性的基准结论。

### 8.2 准入前必须完成的最小验证准备（Phase 6A-Fix 验证实验）

在正式申请启动 Phase 6B 之前，应在本地分支执行一个**无需全量训练的最小验证计划（预算 $\le 5$ 迭代）**：

1. **修复 `task_registry.py` 的 Warm Start 状态继承**：
   - 当 `warm_start=True` 时，允许载入 `optimizer_state_dict`，并保持 `model_25.pt` 的原始学习率 $7.59 \times 10^{-5}$。
2. **修复 `ppo.py` 中自适应学习率与早停的执行时序**：
   - 移除在 Mini-batch 内部基于瞬时 KL 粗暴放大 LR 的逻辑，或设置 Mini-batch 1 跳过 LR 调整，防止“首步自加速”；
   - 将 KL 统计聚合范围限定在实际执行过的更新批次内，统一跨组统计口径。
3. **KL 统计改用抗离群鲁棒指标**：
   - 早停判定条件从全局算术平均 `kl_mean > early_stop_kl` 改为剔除前后 5% 离群值后的**截断均值（Trimmed Mean）**或**中位数（Median）**，消除摔倒样本对正常控制策略更新的误杀。
4. **验证准出指标（Go / No-Go Criteria）**：
   - 执行 3 迭代微调：
     - Step 1 的梯度范数需从 54.16 下降至 $< 10.0$；
     - 每迭代执行的更新数应稳定在预期比例（不再是极端的 1/20 熔断）；
     - 策略在 21 工况评估下的表现不得低于 A 组（$\ge 37.24\%$）。
   - 达成上述指标后，方可正式开启 Phase 6B（Reference-Policy KL 正则化与解耦优化）。

---
**报告归档提交**：执行开发 Agent  
**报告受审方**：独立技术顾问与项目总负责人
