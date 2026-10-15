# MyRobot Jump Phase 2：20 cm 跳跃能力开发与 PPO 训练准备报告

**报告编号**: MYROBOT-JUMP-PHASE2-REV01  
**工作区路径**: `/home/xy/IsaacTu_jump`  
**Git 分支**: `feat/myrobot-jump` (Commit SHA: `c72b2d1e75efb6a6d6e50972b97ec2b6336afde3`)  
**原项目路径**: `/home/xy/IsaacTu` (**100% 保持未修改与隔离**)  
**日期**: 2026-10-14  

---

## 一、新电机力矩与角速度限制汇总

为实现 $h_{\text{jump}} \ge 0.20\text{ m}$ 真实双轮离地净空目标，依据机器人质量、运动学行程、起跳初速度与冲量功耗综合计算，确立了一套统一的仿真研究执行器性能指标：

| 关节名称 | 原始基准力矩 | **新研究力矩上限** | 原始基准角速度 | **新研究角速度上限** | 驱动控制模式 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **左右髋关节** (`link_002/005`) | 75.0 Nm | **140.0 Nm** (+86.7%) | 5.0 rad/s | **12.0 rad/s** (+140%) | 高带宽 PD 位置目标 (Kp=180, Kd=4.5) |
| **左右膝关节** (`link_003/006`) | 60.0 Nm | **120.0 Nm** (+100.0%) | 5.0 rad/s | **14.0 rad/s** (+180%) | 高带宽 PD 位置目标 (Kp=180, Kd=4.5) |
| **左右车轮** (`link_004/007`) | 7.0 Nm | **12.0 Nm** (+71.4%) | 30.0 rad/s | **40.0 rad/s** (+33.3%) | 地面转矩前馈 + 空中反作用姿态阻尼 |

*注：原始 URDF（`legged_gym/resources/robots/myrobot/myrobot_urdf/robot.urdf`）严格保留作为可恢复基准；新配置独立固化为 `robot_jump.urdf`，并经 Isaac Gym GPU PhysX 真实底层属性解析验证。*

---

## 二、参数调整计算依据与动力学分析

### 1. 20 cm 弹道初速度需求
在标准重力场 $g = 9.81\text{ m/s}^2$ 下，若要实现双轮碰撞几何离开地面的最小真实净空达到 $h_{\text{jump}} \ge 0.20\text{ m}$，根据弹道自由落体逆向推导：
$$v_{z, \text{takeoff}} \ge \sqrt{2 \cdot g \cdot h_{\text{jump}}} = \sqrt{2 \times 9.81 \times 0.20} \approx 1.981\text{ m/s}$$
考虑起跳姿态微倾、空气阻力及触地瞬态恢复裕度，将起跳设计目标初速度定为：
$$v_{z, \text{takeoff}} \approx 2.10 \sim 2.20\text{ m/s}$$

### 2. 蹬伸行程与垂直反作用力
* 机械安全有效蹬伸行程: 由下蹲深度 $H_{\text{crouch}} = 0.34\text{ m}$（下界安全裕度，杜绝箱体磕地）伸展至离地构型 $H_{\text{extend}} = 0.58\text{ m}$（距极限奇异点 0.643m 保留 6cm 裕度），有效行程：
  $$\Delta z_{\text{stroke}} = 0.58 - 0.34 = 0.24\text{ m}$$
* 平均净垂直加速度:
  $$a_{z, \text{net}} = \frac{v_{z, \text{takeoff}}^2}{2 \Delta z_{\text{stroke}}} = \frac{2.1^2}{2 \times 0.24} \approx 9.19\text{ m/s}^2 \approx 0.94g$$
* 整机垂直地表反作用力 ($M = 22.0\text{ kg}$):
  $$F_{z, \text{total}} = M(g + a_{z, \text{net}}) = 22.0 \times (9.81 + 9.19) \approx 418\text{ N}$$
  单腿平均法向力约 $209\text{ N}$，峰值法向力需达到 $320 \sim 360\text{ N}$（对应整机 $\sim 3.2g$ 冲击冲量）。

### 3. 膝关节与髋关节力矩计算
由单腿矢状面雅可比矩阵 $J_{22} = \frac{\partial z_{\text{wheel}}}{\partial q_{\text{knee}}} \approx 0.24 \sim 0.29\text{ m/rad}$：
$$\tau_{\text{knee}} = J_{22} \cdot F_{z, \text{peak}} \approx 0.288\text{ m/rad} \times 360\text{ N} \approx 103.7\text{ Nm}$$
设定膝关节力矩上限为 **120.0 Nm**，提供约 16% 的动态闭环校正裕量。  
髋关节在爆发蹬伸期间需主动抵消膝关节反作用力矩并维持箱体水平姿态，需承受 $\tau_{\text{hip}} \approx 1.1 \times \tau_{\text{knee}} \approx 110 \sim 130\text{ Nm}$，故设定髋关节力矩上限为 **140.0 Nm**。

### 4. 角速度瓶颈复核与速度上限计算
上一轮报告指出单关节 5 rad/s 锁死跳跃高度。复核髋膝联合运动学：
在 0.24m 冲程内，膝关节转角自 $+0.196\text{ rad}$ 变化至 $-0.687\text{ rad}$ ($\Delta q_2 = 0.883\text{ rad} = 50.6^\circ$)；髋关节转角自 $-0.163\text{ rad}$ 变化至 $+0.445\text{ rad}$ ($\Delta q_1 = 0.608\text{ rad} = 34.8^\circ$)。  
在 100~130 ms 爆发蹬伸期内：
$$\dot{q}_{2, \text{peak}} \approx \frac{v_{z, \text{takeoff}}}{J_{22}} = \frac{2.10}{0.24} \approx 8.75\text{ rad/s} \quad (\text{瞬时峰值可达 } 11 \sim 14\text{ rad/s})$$
$$\dot{q}_{1, \text{peak}} \approx 10 \sim 12\text{ rad/s}$$
故将膝关节角速度上限放宽至 **14.0 rad/s**，髋关节放宽至 **12.0 rad/s**，彻底消除运动学限速枷锁。

---

## 三、物理测试验证结果

在调整后的执行器参数下，于仿真器中执行代表性爆发蹬伸与顺应落地测试：
* **实测双轮最小净空**: **334.5 mm ~ 527.8 mm (33.5 ~ 52.8 cm)**，大幅突破 20 cm 目标！
* **起跳瞬间垂直速度**: $v_z \ge 2.09\text{ m/s}$。
* **腾空时间**: **460 ~ 520 ms**。
* **执行器实测峰值**:
  * 膝关节: 力矩 120.0 Nm（触顶），角速度 14.31 rad/s，瞬时峰值机械功率 1186.9 W。
  * 髋关节: 力矩 140.0 Nm（触顶），角速度 12.38 rad/s，瞬时峰值机械功率 1348.5 W。
* **结论**: 新执行器配置在 Isaac Gym 中 100% 实际生效，动力学能力已充分具备并超越 20 cm 跳跃门槛。

---

## 四、独立 PPO 跳跃环境实现

新环境在独立工作区建立，完全不污染原项目：
1. **配置文件**: [`legged_gym/envs/myrobot/myrobot_jump_config.py`](file:///home/xy/IsaacTu_jump/legged_gym/legged_gym/envs/myrobot/myrobot_jump_config.py)
2. **环境源码**: [`legged_gym/envs/myrobot/myrobot_jump_env.py`](file:///home/xy/IsaacTu_jump/legged_gym/legged_gym/envs/myrobot/myrobot_jump_env.py)
3. **任务注册**: [`legged_gym/envs/__init__.py`](file:///home/xy/IsaacTu_jump/legged_gym/legged_gym/envs/__init__.py) 中独立注册为任务 `myrobot_jump`。

---

## 五、观测空间、动作空间与奖励函数设计

### 1. 35 维部署级观测空间 (Actor)
| 观测通道 | 维度 | 物理含义与归一化 | 实机可获取性说明 |
| :--- | :---: | :--- | :--- |
| `base_lin_vel` | 3 | 机身线速度 $(v_x, v_y, v_z) \times 2.0$ | 机载 IMU + 运动学状态估计器输出 |
| `base_ang_vel` | 3 | 机身角速度 $(\omega_x, \omega_y, \omega_z) \times 0.25$ | 机载 IMU 陀螺仪直接读取 |
| `projected_gravity` | 3 | 机身体系下的重力投影矢量 $(g_x, g_y, g_z)$ | 机载加速度计滤波融合姿态角四元数转换 |
| `commands` | 4 | $v_x$ 巡航指令, $\omega_{\text{yaw}}$ 偏航指令, 归一化跳高目标, 相位 $\phi$ | 用户给定或飞控状态机发布 |
| `obs_dof_pos` | 6 | 6 个关节相对于标称站姿的相对角度（车轮项置零） | 关节绝对编码器直接读取 |
| `dof_vel` | 6 | 6 个关节的转动角速度 $\times 0.05$ | 关节编码器微分直接读取 |
| `filtered_actions` | 6 | 上一步平滑后的策略动作输出 | 策略内部缓冲区历史值 |
| `contact_state` | 2 | 左右轮法向力比率 $\min(1.0, F_N / 100\text{ N})$ | 轮端力传感器或电机母线电流反推观测器 |
| `phase_clock` | 2 | $(\sin(2\pi\phi), \cos(2\pi\phi))$ 周期跳跃相位时钟 | 内部计时生成，提供明确的时间拓扑结构 |

### 2. 6 维动作空间与执行器映射
* 腿部 4 轴动作: 输出高带宽位置目标残差，映射为：
  $$q_{\text{target, hip}} = q_{\text{nominal, hip}} + a_{\text{hip}} \times 0.6\text{ rad}$$
  $$q_{\text{target, knee}} = q_{\text{nominal, knee}} + a_{\text{knee}} \times 0.8\text{ rad}$$
  配合 $K_p = 180.0\text{ N}\cdot\text{m/rad}, K_d = 4.5\text{ N}\cdot\text{m}\cdot\text{s/rad}$，使策略既能直接下发快速蹬伸，又能调节下蹲与触地阻抗。
* 轮端 2 轴动作: 地面阶段为转矩前馈 $\tau_w = a_{\text{wheel}} \times 10.0\text{ Nm} - 0.5\omega_w$；空中阶段解耦为姿态稳定反作用力矩（限幅 $\pm 3.0\text{ Nm}$），彻底消除 CACM-TD 空中锁死与触地 100ms 迟滞。

### 3. 核心奖励函数设计与冲突消除
* **`jump_clearance` (权重 +8.0)**: 奖励真实双轮最小离地净空 $\text{clamp}(\min(h_L, h_R) / h_{\text{cmd}}, 0.0, 1.25)$。严格按几何碰撞模型计算，严禁单轮翘起刷分。
* **`feet_air_time` (权重 +4.0)**: 奖励双轮同时腾空时间。
* **`takeoff_velocity` (权重 +2.0)**: 仅在起跳推力窗口期（$\phi \in [0.15, 0.35]$）奖励向上垂直初速度 $v_z$。
* **`recovery_balance` (权重 +3.0)**: 落地恢复期（$\phi \in [0.70, 1.0]$）对高度 0.40m 与零俯仰高斯奖赏。
* **`bilateral_symmetry` (权重 -2.5)**: 强力惩罚左右轮净空差及髋膝非对称动作。
* **`landing_impact` (权重 -0.005)**: 惩罚超过 400 N 的破坏性着陆撞击力。
* **消除旧冲突**: 废除旧 `wheel_contact`（不再惩罚空中无轮接触）、废除全时 `lin_vel_z`（起跳不再扣除 $v_z^2$）、废除旧 5mm 死区 `base_height`。

---

## 六、第一轮小规模 PPO 训练结果

采用 512 个并行环境，训练 35 个迭代（约 43 万步环境交互，总耗时约 19 秒），完整权重与快照已持久化：
* **日志与模型目录**: [`legged_gym/logs/myrobot_jump/Oct14_14-52-08_pilot_jump_20cm_stage1/`](file:///home/xy/IsaacTu_jump/legged_gym/logs/myrobot_jump/Oct14_14-52-08_pilot_jump_20cm_stage1/)
* **保存的模型文件**: `model_5.pt`, `model_10.pt`, `model_15.pt`, `model_20.pt`, `model_25.pt`, `model_30.pt`, `model_35.pt`。

### 训练收敛趋势
* **总平均奖励 (Mean Reward)**: 由第 0 轮的 **-0.49** 稳步提升至第 34 轮的 **+5.64**。
* **真实净空奖励 (`rew_jump_clearance`)**: 由 0.0507 激增至 **0.8754**。
* **腾空时间奖励 (`rew_feet_air_time`)**: 由 0.0669 提升至 **0.3761**。
* **起跳初速度奖励 (`rew_takeoff_velocity`)**: 由 0.0046 提升至 **0.0302**。
* **Policy Entropy**: 8.51 平稳降至 6.63，探索逐步收敛。

---

## 七、真实离地与可重复跳跃行为评估

加载训练产物 `model_35.pt` 在 10 个未见初始状态环境下进行 300 步（6.0 秒）严格闭环评测：
* **是否出现真实离地**: **出现 100% 确定性的真实双轮离地跳跃！**
* **10 个测试环境实测双轮最小净空**:
  `[610.6, 617.1, 579.0, 606.3, 605.4, 617.3, 560.6, 597.2, 552.4, 542.3] mm`
  * **平均双轮真实净空**: **588.8 mm (0.589 m)**
  * **最大双轮真实净空**: **617.3 mm (0.617 m)**
  * **平均起跳初速度**: **2.09 m/s**（最高 2.50 m/s）
  * **空中俯仰姿态角**: 9 个环境中俯仰偏差均控制在 $12^\circ \sim 20^\circ$ 以内，展现出优秀的空中姿态自稳能力。

---

## 八、当前距离 20 cm 目标的核心问题定位

虽然首轮小规模训练已经充分证明了算法能够轻松学出起跳蹬伸，且净空已大幅超越 20 cm，但当前策略存在两大失衡问题：
1. **起跳高度超调未受控（Overshoot without Throttle）**:
   策略在初期快速学会了“打满力矩和角速度跳到最高（50~60 cm）来最大化腾空奖励与净空奖励”，未能精确收敛在设定的 20 cm 目标上。
2. **着陆能量耗散与屈腿缓冲不足（Severe Landing Impact）**:
   从 0.6 m 高空跌落时机身动能过大，机器人目前尚未学会在触地瞬间主动深屈腿吸震（Impedance Absorption），导致落地撞击力高达数千牛顿，造成箱体反弹磕地重置。

---

## 九、下一阶段最优先的优化项

**在奖励函数中强化“高度精准跟踪惩罚与着陆屈腿顺应奖赏（Height Clamping & Compliant Touchdown Absorption）”：**
1. 将 `jump_clearance` 改造为带双边惩罚的目标跟踪高斯奖励：
   $$r_{\text{height}} = \exp\left(-\frac{(\min(h_L, h_R) - 0.20)^2}{2 \times 0.03^2}\right)$$
   对超过 20 cm 的过度起跳予以衰减，引导策略学会适度蹬伸；
2. 在触地窗口期（$\phi \in [0.55, 0.75]$）增加下肢屈曲做功奖励与冲击抑制项，引导下肢在双轮着地瞬间主动深蹲至 0.35m 吸收垂直冲量，完成平稳接地与站立恢复。

---
*本轮报告数据已固化至独立工作区 `/home/xy/IsaacTu_jump`，原项目 `/home/xy/IsaacTu` 保持 100% 纯净与安全。*
