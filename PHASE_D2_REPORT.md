# MyRobot Phase D.2 — GPU 初始化准入复核

基线：`a5a05ed234d15f7eca52d9eb73906d5e2a603e2e`（Phase D.1）。

**结论：训练准入不通过（NOT_ADMITTED）。** 三次独立 GPU 启动均复现 env0 首轮物理异常。已按计划停止，没有启动十轮 A1、A2、PPO 更新或四臂实验。

## 本轮范围与探针修正

- 沿实际 `make_env → make_alg_runner → runner.alg.act → env.step → process_env_step` 链路运行。
- 每次 64 环境、25 个控制步，三个独立进程使用相同 seed=11；这是同条件复现，不是多种子鲁棒性验证。
- 只调用 runner 原有的一次 `reset()`；env0 跌倒后的正常 `reset_idx` 仍保留。不添加双重 reset、不屏蔽或删除异常样本。
- 探针新增全部环境的初始 root 状态、每步轮承载与腿接触、Actor/Critic 输入观测、采样动作、原始/存储奖励、动作掩码、终止和终止前快照。
- 终止帧显式区分 `step()` 返回的重置后 root 状态和 snapshot 的重置前姿态；不能拿重置后的正常姿态否认真实跌倒。
- 检查张量形状与有限值、输入观测/动作/掩码是否正确进入 rollout storage、权重是否不变及 Adam step 计数是否不变。
- 新证据默认独立命名；显式指定已存在文件时拒绝覆盖。旧 Phase D JSON 和 D.1 数据不变。
- 准入退出状态：`0=PASS`、`1=FAIL`（观测到异常）、`2=INCOMPLETE`（证据不足或执行异常）。修复旧探针发现异常仍退出 0 的问题。

纯判定逻辑位于 `tests/training_init_gate.py`，无需 Gym 即可重审证据。判据允许初始悬空沉降，但拒绝：25 步内任何环境终止、连续至少三帧腿承载且双轮均未达到 20 N、25 步内从未连续三帧双轮承载且无腿接触。腿接触阈值沿用旧探针的 1 N；数值/存储错误、重复初始化 reset 或权重/优化器更新同样否决。初始高度相对同批环境的偏差作为证据，不用 `|z-0.400| > 1e-4` 的旧计数作准入门槛。

## 三次实际结果

| 项目 | 第一次 | 第二次 | 第三次 |
|---|---:|---:|---:|
| env0 reset 后高度 | 0.415517 m | 0.415517 m | 0.415517 m |
| 相对 env1–63 高度中位数 | +16.871 mm | +16.871 mm | +16.871 mm |
| 首次终止 step（从 0 计数） | 13 | 13 | 13 |
| 25 步内终止环境 | 仅 env0 | 仅 env0 | 仅 env0 |
| env0 连续腿部单独承载 | 有 | 有 | 有 |
| 终止前俯仰角 | −45.485° | −45.485° | −45.485° |
| 实际进程退出码 | 1 | 1 | 1 |
| 准入判定 | FAIL | FAIL | FAIL |

每次都记录 1600 条 transition，共 4800 条。三次均：

- Actor/Critic 输入、动作、奖励和存储检查未发现 NaN/Inf、形状或存储不一致。
- `policy_weights_unchanged=True`，Adam step 合计 `0 → 0`，初始化 `reset()` 调用计数为 1。
- env0 终止前 snapshot 的 `fell=True`、`failure_reason=NONE`：这是越过 45° 俯仰边界的真实物理跌倒，不是 `CROSSED_WITHOUT_LIFTOFF` 假终止，也不是障碍碰撞触发的任务失败。
- env0 后续自动 reset 后恢复双轮支撑，但不能抹去首次异常；判定仍为 FAIL。

**对训练样本的影响要分开说。** 所有启动样本的 Actor 掩码都是 0（尚处 BALANCE），因此不能声称这 25 步已经直接更新或验证了 Jump Actor。env0 的异常初始轨迹、奖励和真实跌倒终止确实进入了共享 rollout/Critic 数据；若训练继续，会参与 Critic 学习。本轮没有更新，不能报告实际参数受损或训练效果变化。

旧日志中“64 个环境偏离标称 0.400 m”仍只是过严阈值下的历史诊断；正常环境经受控周期后约为 0.39865 m，因此该 64 不代表全部环境异常。新证据明确使用同批相对偏差和真实终止/承载行为。

## 验证与证据

- `python3 tests/test_training_init_gate.py`：**8 个测试方法通过**，含通过、异常、证据不足的真实 CLI 退出码，持续承载计数、非 env0 环境异常、重复 reset、权重/优化器变化及历史异常证据不能获准。
- 三次真实 GPU 探针：均完成 25 步并按异常返回 **1**；这是准入失败的有效测量，不是测试通过。
- 指定既有历史 JSON 的防覆盖检查：返回 **2**，原文件 SHA256 不变；该 CLI 检查未创建仿真场景。
- `python3 -m py_compile tests/verify_training_init_chain.py tests/training_init_gate.py` 与最终 `git diff --check` 通过。
- 未修改任务、奖励、FSM、物理/执行器配置或 PPO 实现，因此本轮未重跑 D.1 的 147 项物理语义测试，也未运行含 optimizer update 的 PPO smoke test。

证据目录：`evaluation_assets/phaseD2/`。`run1_gpu.json` 至 `run3_gpu.json` 保存原始数组，配套 `.log` 保存输出；`summary.json` 保存源文件指纹、证据 SHA256、各次结果；`no_overwrite_check.json` 保存历史文件保护结果。三次证据中的探针/判定器源码指纹均与交付源码一致。

复核命令（不启动仿真）：

```bash
python3 tests/training_init_gate.py evaluation_assets/phaseD2/run1_gpu.json
# 预期退出 1，打印 FAIL
```

## 停止结论与下一步边界

本轮在 30 分钟预算内完成准入复核，未扩展 CPU/GPU 参数矩阵，未尝试用双重 reset 或丢弃 env0 来规避问题。
证据把问题限定在真实 GPU 初始化后 env0 的异常轨迹；**尚未定位到 Isaac Gym 求解器或状态写回的具体根因，未交付物理初始化修复**。

下一步应是单独修复 GPU 初始状态写回与首个受控周期的衔接，并用这套门禁验证。三次准入通过之前，不启动十轮 A1；修复后的干净 A1 完成之前，不进入 A2。无需再改变 D.1 的成功判据来掩盖此异常。
