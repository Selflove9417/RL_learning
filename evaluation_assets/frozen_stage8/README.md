# Stage 8 资产冻结清单与评测规范归档

## 1. 冻结目的
本目录用于严格封存 Phase 2 成果（Stage 8 `model_40.pt`），作为 Phase 3（指令触发单次跳跃）开发的绝对基准，防止在后续探索中发生意外覆盖或参数漂移。

## 2. 冻结文件清单与校验和 (MD5)
| 文件名 | MD5 校验和 | 描述 |
| :--- | :--- | :--- |
| `model_40.pt` | `97deae5529ee046474d077c014f55c53` | Stage 8 训练完成的模型权重（Early Zero-Velocity Refinement 40 iters） |
| `robot_jump.urdf` | `da1c33919e3b9e086a8f6dffc44ff5e0` | 升级执行器限制的物理模型（髋 140Nm/12rad/s, 膝 120Nm/14rad/s, 轮 12Nm/40rad/s） |
| `myrobot_jump_config.py` | `d1070eb6bcbf3a632dc6943c5b839e41` | Stage 8 训练与环境配置文件 |
| `myrobot_jump_env.py` | `b674dc5c2dc7161b81559a66251bcf2c` | Stage 8 训练环境定义（包含 35 维观测、奖励、周期时钟） |
| `eval_per_jump.py` | `e5ae74919cb0b48aa74ef0029635f6ce` | 逐跳评测脚本（支持 Exact 3D 与简化公式对比、时空位移指标） |
| `run_phase2_5_audit.py` | `cb532e2f3434300d6e6ae9c0cd5e7668` | 5 种子批量严格评测启动器 |

## 3. 双轨评测条件定义（严禁混淆）
在对 Stage 8 策略进行任何评测时，必须明确区分以下两种初始化条件：

### 条件 A：静止初始化（Nominal In-place Static Reset）
- **定义**：环境复位后，将机器人根节点的线速度与角速度显式置零（`root_states[:, 7:13] = 0.0`），机器人从双轮平稳贴地静止状态开始进入策略控制。
- **用途**：标称原地跳跃能力验收、指令触发 jump 执行标准测试。
- **基线指标参考**：5 种子 240 跳 Exact 3D 达标率 **95.83%**（简化公式 97.92%），平均净空 **22.10 cm**，Cycle 1 达标率 88.75%，落地收敛时间 0.291 s。

### 条件 B：原始随机速度初始化（Perturbed Dynamic Velocity Reset）
- **定义**：环境复位时保留 Isaac Gym LeggedRobot 的随机初速度扰动（$v \in [-0.5, +0.5]\text{ m/s}$, $\omega \in [-0.5, +0.5]\text{ rad/s}$）。
- **用途**：抗扰动鲁棒性与异常工况恢复压力测试。
- **基线指标参考**：5 种子 240 跳 Exact 3D 达标率 **93.33%**（简化公式 94.58%），平均净空 **21.90 cm**，Cycle 1 达标率 81.25%，落地收敛时间 0.355 s。
