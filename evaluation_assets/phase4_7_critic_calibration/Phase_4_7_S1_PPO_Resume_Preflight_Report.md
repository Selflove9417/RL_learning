# MyRobot Phase 4.7-S1：Actor PPO 恢复训练前最终检查报告

**执行时间**: 2026-10-17  
**责任团队**: MyRobot 强化学习研发小组  
**阶段定位**: Phase 4.7-S 后的最终预检（Preflight Check），为 Phase 4.8 恢复 Actor PPO 训练做最后安全把关。  
**前置基准资产**:
- Phase 3.2-V 冻结 Champion 模型: `evaluation_assets/phase3_1_forward_distance/models/model_champion.pt`  
  SHA256: `fad590f46995641ae917bf69bf14479e0a0d0edcbbdf3f7956ef5a85ae76d0fa`
- Phase 4.7 校准 Critic 模型: `evaluation_assets/phase4_7_critic_calibration/model_critic_calibrated_target_b.pt`  
  SHA256: `a5fca202c66c303f29bb5df91b920155b9e072f5ea11db2cb97cf43be985e505`
- 预检审计配置: 严格冻结 Actor 权重、动作标准差及动力学参数，禁止 PPO 参数更新，禁止 Critic 重新训练。

---

## 最终裁定：PASS（充分具备安全恢复 Actor PPO 条件）

本轮预检围绕 4 项核心关卡逐一完成闭环验证，各项断言全部成立，满足进入 Phase 4.8 Actor PPO 试跑的全部准入准则：
1. **在线自举切换**: 明确了从 Champion 离线自举 ($V_A \approx 17.22$) 切换为 Calibrated Critic 在线自举 ($V_B \approx 19.66 \sim 19.71$) 的数学与动力学实质。该切换消除了第 139 步人为价值断崖（TD 残差从 $-2.48$ 收敛至 $-0.0075$），且因 GAE 几何衰减因子 $(\gamma \lambda)^{139-t} \le 0.0037$，对前 34 步跳跃核心阶段（下蹲、起跳、腾空、着陆）的 Advantage 影响严格小于 $0.004$，策略梯度方向保持纯净。
2. **开发集绝对独立性**: 证明 Seeds 3001～3005 仅作为完全隔离的离线测试与泛化诊断集，未参与任何梯度更新、超参数调优或 Early Stopping 检查点选择；同时证明其价值目标 $targets\_b$ 产生于校准训练之前，不存在任何待评价模型的循环自举生成。
3. **PPO 恢复链路与零更新验证**: 在完整 140 步 $\times$ 64 环境同步 Rollout 中，在线正确捕获重置前终态观测并完成 $r_{139} + \gamma V(s_{140}^{\text{pre}})$ 注入（误差 $3.17 \times 10^{-6}$），Values/Returns/Advantages 全程无 NaN/Inf。在未开启优化器更新下，网络总参数变动为严格 $0.00\text{e}+00$，Actor 确定性动作与 Champion 差异严格为 $0.00\text{e}+00$。
4. **共享库向后兼容**: 证实 `rsl_rl` 修改采用显式字典键守卫（`terminal_observation in infos`），所有未改造的旧任务完全走原版逻辑，并对 `on_policy_runner.py::load` 提供安全的字典取值，实现 100% 双向兼容。

---

## 任务一：在线自举目标切换的数值分析与动力学影响

### 1.1 终态观测下的价值差异统计

在 Phase 4.7 收集的固定重置前终态观测 $s_{140}^{\text{pre}}$（64 环境平稳双轮站立姿态）上，分别使用 Champion Critic 与 Target B 校准 Critic 进行评估：

| 评估指标 | Champion Critic ($V_A$) | Target B Critic ($V_B$) | 差值 ($\Delta V = V_B - V_A$) |
| :--- | :---: | :---: | :---: |
| **均值 (Mean)** | **17.2207** | **19.7149** | **+2.4942** |
| **标准差 (Std)** | 0.0004 | 0.0032 | 0.0036 |
| **最小值 (Min)** | 17.2198 | 19.7071 | +2.4868 |
| **最大值 (Max)** | 17.2217 | 19.7214 | +2.4998 |

> **差异成因的物理本质**:  
> Phase 3.2-V Champion 是在无前向位移奖励的环境下训练自稳，其 Critic 预估的站立稳态价值约为 $17.22$。而在 Phase 4 引入前向距离 Sigmoid 奖励后，机器人在跳跃成功落地后的自稳阶段（第 40～140 步），累计位移奖励与存活奖励的真实稳态价值为 $19.7 \sim 20.7$。Champion Critic 在终态低估了约 $2.49$ 的真实稳态价值。

### 1.2 在线切换对存储奖励、Returns 与 TD 残差的影响

在线 PPO 运行时，第 139 步（索引从 0 开始的最后一步）检测到 `time_outs=True`，PPO 会将终态自举值加入存储奖励中：
$$r_{139}^{\text{stored}} = r_{139}^{\text{env}} + \gamma \cdot V(s_{140}^{\text{pre}})$$

- **第 139 步存储奖励增量**:  
  $$\Delta r_{139} = \gamma \cdot \Delta V = 0.99 \times (+2.4942) = \mathbf{+2.4693}$$
- **第 139 步单步 TD 残差对比**:  
  - 若继续采用 Champion 终态 $V_A$（离线目标）：  
    $$\delta_{139}^{(A)} = r_{139} + \gamma V_A - V_B(s_{139}) \approx 0.1897 + 17.0485 - 19.7150 = \mathbf{-2.4768}$$  
    *(由于 Critic 已校准至 19.71，若末端自举硬代入 17.22，会在终止边界凭空产生一个 $-2.48$ 的虚假惩罚断崖！)*
  - 采用当前校准 Critic 在线自举 $V_B$（在线真实运行）：  
    $$\delta_{139}^{(B)} = r_{139} + \gamma V_B - V_B(s_{139}) \approx 0.1897 + 19.5178 - 19.7150 = \mathbf{-0.0075}$$  
    *(单步 TD 残差几乎精确为 0，符合双轮足平稳站立时的平稳自稳收益特征)*

### 1.3 GAE 优势函数逐阶段几何衰减分析

在 GAE($\gamma=0.99, \lambda=0.95$) 框架下，有效折扣因子为:
$$\mu = \gamma \lambda = 0.99 \times 0.95 = 0.9405$$
第 139 步由自举修正带来的 TD 残差增量 $\Delta \delta_{139} = +2.4693$，向历史时间步 $t$ 反向传播时的优势增量为：
$$\Delta A_t^{\text{GAE}} = (\gamma \lambda)^{139 - t} \cdot \Delta \delta_{139}$$

下表给出了各关键动作阶段的优势修正量及影响评估：

| 时间步 $t$ | 物理运动阶段 | 衰减指数 $(139-t)$ | 衰减权重 $(\gamma\lambda)^{139-t}$ | 优势增量 $\Delta A_t^{\text{GAE}}$ | 动力学与梯度影响判定 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Step 139** | 终止前最后一步 | 0 | 1.00000 | **+2.4693** | 修复末端边界惩罚断崖，维持正常站立价值 |
| **Step 120** | 稳定平衡期 | 19 | 0.31210 | **+0.7707** | 适度增加站立平衡阶段的存活优势 |
| **Step 100** | 恢复过渡中段 | 39 | 0.09110 | **+0.2250** | 增强落地后尽快恢复平衡的正向激励 |
| **Step 70** | 落地减震收敛 | 69 | 0.01320 | **+0.0326** | 轻微正向衰减，不影响姿态调整 |
| **Step 40** | 落地稳态交接 | 99 | 0.00231 | **+0.0057** | 影响微弱，小于单步奖励浮动 |
| **Step 33** | **触地着陆瞬间** | 106 | 0.00150 | **+0.0037** | **严格 $< 0.004$，对着陆控制无畸变** |
| **Step 20** | **腾空最高点 (Apex)** | 119 | 0.00064 | **+0.0016** | **严格 $< 0.002$，空中姿态不受影响** |
| **Step 11** | **起跳离地 (Liftoff)** | 128 | 0.00039 | **+0.00096** | **严格 $< 0.001$，起跳爆发力完全无扰** |
| **Step 8** | **伸髋展腿极值点** | 131 | 0.00032 | **+0.00079** | **严格 $< 0.001$，跳跃动作模态无偏移** |
| **Step 0** | **初始下蹲蓄力** | 139 | 0.00020 | **+0.00049** | **严格 $< 0.0005$，对初始决策完全透明** |

> **关键结论**: 在跳跃决定性区间（Step 0～33），在线自举切换引入的 Advantage 波动严格在 $0.0037$ 以内（相较于跳跃阶段 $-3.6 \sim +2.5$ 的优势幅度，相对变动小于 $0.1\%$）。因此，在线切换自举不会引入新的跳跃目标不一致，反而成功消除了由于 Champion 低估造成的自稳阶段末端负向惩罚。

---

## 任务二：Critic 训练、验证与开发集数据独立性与无循环依赖证明

### 2.1 种子分配与资产归属表

| 数据集类型 | 随机种子 | 环境数量 | Episode 步数 | 转移总数 (Transitions) | 数据文件路径 | 角色与用途限制 |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **训练集 (Train)** | **Seed 6001** | 64 | 140 | 8,960 | `evaluation_assets/phase4_7_critic_calibration/data_train_seed6001.pt` | **唯一用于 Critic 参数更新 (MSE Loss 优化)** |
| **验证集 (Val)** | **Seed 7001** | 64 | 140 | 8,960 | `evaluation_assets/phase4_7_critic_calibration/data_val_seed7001.pt` | **唯一用于 Checkpoint 择优与 Early Stopping** |
| **开发测试集 (Dev)** | **Seed 3001** | 64 | 140 | 8,960 | `evaluation_assets/phase4_7_critic_calibration/data_dev_seed3001.pt` | 完全隔离，纯只读基准，监控泛化表现 |
| **开发测试集 (Dev)** | **Seed 3002** | 64 | 140 | 8,960 | `evaluation_assets/phase4_7_critic_calibration/data_dev_seed3002.pt` | 完全隔离，纯只读基准，监控泛化表现 |
| **开发测试集 (Dev)** | **Seed 3003** | 64 | 140 | 8,960 | `evaluation_assets/phase4_7_critic_calibration/data_dev_seed3003.pt` | 完全隔离，纯只读基准，监控泛化表现 |
| **开发测试集 (Dev)** | **Seed 3004** | 64 | 140 | 8,960 | `evaluation_assets/phase4_7_critic_calibration/data_dev_seed3004.pt` | 完全隔离，正式基线对齐（14.71 cm） |
| **开发测试集 (Dev)** | **Seed 3005** | 64 | 140 | 8,960 | `evaluation_assets/phase4_7_critic_calibration/data_dev_seed3005.pt` | 完全隔离，纯只读基准，监控泛化表现 |

### 2.2 开发集隔离性源码级证据

在校准执行脚本 [`tests/run_critic_calibration_phase4_7.py`](file:///home/xy/IsaacTu_jump/tests/run_critic_calibration_phase4_7.py) 中：
1. **优化器输入严格排他**:
   - `L121-122`: `train_dataset = TensorDataset(train_data["obs"], train_data["targets_b"])`, `train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)`
   - `L238-245`: 遍历 `train_loader` 进行反向传播与 `optimizer.step()`。Seeds 3001～3005 的数据张量完全不连接到计算图，梯度贡献严格为 0。
2. **最优模型选取严格排他**:
   - `L254-258`:  
     ```python
     val_eval = compute_eval_dataset(ac, val_data, device)
     if val_eval["mse"] < best_val_mse:
         best_val_mse = val_eval["mse"]
         best_model_state = copy.deepcopy(ac.state_dict())
     ```
     `best_model_state` 仅由 `val_data` (Seed 7001) 的 MSE 触发更新。
3. **开发集仅作被动打点记录**:
   - `L262, L281-283`: `dev_eval` 仅在评估间隔被动计算并写入日志字典，未参与任何超参数搜索、权重截断或学习率调度。

### 2.3 无循环依赖证明（Non-circular Target Generation）

在数据采集脚本 [`tests/collect_target_b_rollout.py`](file:///home/xy/IsaacTu_jump/tests/collect_target_b_rollout.py) 中：
- `L41-44`: 加载且仅加载了基线模型 `model_champion.pt`。
- `L93`: 使用 `model_champion.pt` 预计算 `v_terminal_pre = ac.evaluate(terminal_obs)`。
- `L106-107`: 利用 Champion 的终态值折现构建 `targets_b[t] = returns[t]`。
- **时间序逻辑**: 数据采集完成并固化于磁盘于 2026-10-17 09:20 之前，而校准训练发生在 09:30。校准模型在生成 `targets_b` 时尚未存在，不存在以待评模型自身输出作为评价目标的问题。

---

## 任务三：Runner 预检结果与零更新验证

执行最小 PPO 恢复链路预检测试 [`tests/preflight_ppo_resume_check.py`](file:///home/xy/IsaacTu_jump/tests/preflight_ppo_resume_check.py)，通过正式 `task_registry.make_alg_runner` 构建 `OnPolicyRunner`，加载校准模型并在 `alg.update()` 被禁用的条件下执行 1 个完整 Rollout（140 步 $\times$ 64 环境）。

### 3.1 载入与网络结构核验
- **加载模型**: `evaluation_assets/phase4_7_critic_calibration/model_critic_calibrated_target_b.pt`
- **Actor 输入维度**: 35（包含 Phase 4 目标距离与历史动作）
- **Critic 输入维度**: 35
- **Actor 初始权重差异 vs Champion**: `0.00e+00`
- **Actor 动作标准差差异 vs Champion**: `0.00e+00`

### 3.2 在线超时与自举注入验证 (Step 139)
- `infos["time_outs"]` 触发率: **64 / 64 (100%)**
- `infos["terminal_observation"]` 提取率: **64 / 64 (100%)**
- 在线评估重置前终态均值 $V_{\text{calib}}(s_{140}^{\text{pre}})$: **19.6618**
- Step 139 物理步即时奖励均值 $r_{139}^{\text{env}}$: **0.1940**
- RolloutStorage 实际存储奖励 $r_{139}^{\text{stored}}$: **19.6592**
- 理论注入预期 $r_{139}^{\text{env}} + \gamma \cdot V_{\text{calib}}(s_{140}^{\text{pre}})$: **19.6592**
- **自举绝对误差**: $\mathbf{3.17 \times 10^{-6}}$（完全由 float32 累加精度决定）

### 3.3 Rollout 存储张量数值健全性 (Numerical Integrity)

| 张量字段 | 最小值 (Min) | 最大值 (Max) | 均值 (Mean) | NaN / Inf 检验 | 状态 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Values** | 19.0884 | 26.0415 | 20.9142 | 严格为 0 | **PASS** |
| **Returns** | 19.1869 | 25.6257 | 20.8988 | 严格为 0 | **PASS** |
| **Advantages** | -3.6358 | +2.5160 | -0.0154 | 严格为 0 | **PASS** |
| **Rewards (Step 0..138)** | -0.5008 | +1.8942 | +0.1481 | 严格为 0 | **PASS** |
| **Rewards (Step 139)** | +19.1869 | +20.0064 | +19.6592 | 严格为 0 | **PASS** |

### 3.4 零更新参数不可变性核查
- 经过 140 物理步、8,960 次环境交互及一次伪训练轮次后：
  - **模型总参数最大变动 (All Layers Max Diff)**: $\mathbf{0.00\text{e}+00}$
  - **Actor 动作推断输出最大变动 (Action Max Diff)**: $\mathbf{0.00\text{e}+00}$
  - **优化器参数更新次数**: **0 次**

---

## 任务四：共享 RSL-RL 库向后兼容性验证说明

### 4.1 代码行号与隔离逻辑

在共享算法库 [`rsl_rl/rsl_rl/algorithms/ppo.py`](file:///home/xy/IsaacTu_jump/rsl_rl/rsl_rl/algorithms/ppo.py) 中，针对 Target B 的终态自举逻辑通过明确的双重条件守卫实现隔离：

```python
# rsl_rl/rsl_rl/algorithms/ppo.py: 218-228
if self.storage.step == self.num_steps_per_env - 1 and "terminal_observation" in infos:
    # 仅针对显式提供 terminal_observation 的 Phase 4 任务执行精准重置前终态自举
    terminal_values = self.actor_critic.evaluate(infos["terminal_observation"].to(self.device))
    rewards += self.gamma * terminal_values.squeeze(-1) * self.storage.time_outs[self.storage.step]
elif self.storage.step == self.num_steps_per_env - 1:
    # 所有其他标准环境（ANYmal, Unitree, MyRobot 旧版等）未提供 terminal_observation，
    # 保持原版 RSL-RL 逻辑不变（若有超时则基于当前观测估值，否则无操作）
    pass
```

在 [`rsl_rl/rsl_rl/runners/on_policy_runner.py`](file:///home/xy/IsaacTu_jump/rsl_rl/rsl_rl/runners/on_policy_runner.py) 中：
```python
# rsl_rl/rsl_rl/runners/on_policy_runner.py: 301-305
checkpoint_match = re.search(r'model_(\d+)\.pt$', os.path.basename(path))
self.current_learning_iteration = (
    int(checkpoint_match.group(1)) if checkpoint_match else loaded_dict.get('iter', 0)
)
return loaded_dict.get('infos', None)
```
- **向后兼容保证**:
  1. 对于标准 RSL-RL 保存的 Checkpoint（包含 `'iter'` 与 `'infos'`），`loaded_dict.get(...)` 返回完全一致的数值，不改变任何历史行为。
  2. 对于独立校准模型（仅包含 `model_state_dict` 等权重），避免了 `KeyError: 'iter'` 异常，允许热启动安全载入。

---

## 总结与 Phase 4.8 启动建议

### 结论综述
- **价值目标有效**: 经过 Target B 重置前状态修正的 Critic 模型在物理轨迹上表现平稳，消除了终态边界跳变。
- **动态干扰隔离**: 在线自举对起跳核心阶段的 Advantage 干扰严格低于 $0.004$，策略梯度方向高度受控。
- **数据与资产纯净**: 5 个开发集种子彻底隔离，未出现任何数据泄露或循环自举。
- **恢复链路畅通**: Runner 在无更新条件下参数严格保持为 0 变动，数值无异常溢出。

### Phase 4.8 Actor PPO 启动建议（配置清单）
当启动 Phase 4.8 时，在环境与训练配置中应确认以下设置：
1. `warm_start = True`
2. `warm_start_path = "evaluation_assets/phase4_7_critic_calibration/model_critic_calibrated_target_b.pt"`
3. `warm_start_optimizer = False`（从初始学习率启动 PPO 优化器）
4. 保持 Actor 初始学习率处于保守区间（例如 $1.0 \times 10^{-4}$），前 50 次迭代密切监控 5 个开发种子的起跳距离与落地俯仰稳定性。

---
*报告归档路径: `evaluation_assets/phase4_7_critic_calibration/Phase_4_7_S1_PPO_Resume_Preflight_Report.md`*  
*预检日志归档: `evaluation_assets/phase4_7_critic_calibration/phase4_7_s1_preflight_check.json`*
