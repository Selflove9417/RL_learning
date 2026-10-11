# MyRobot Phase 5.3 复现与评估资产索引 (Evaluation Assets Index)

## 一、资产基本信息与环境版本

- **阶段定位**: MyRobot Phase 5.3 阶段封版独立评估与诊断资产包
- **主基线模型 (Primary Baseline Checkpoint)**:
  - 路径: `legged_gym/logs/myrobot/Oct10_15-48-32_turning_height_stage5_3_formal/model_25.pt`
  - SHA256: `d67474d0e47f908a8c863e605b8d5de7ea1bd9f89f41d2c303fbef33d963536c`
  - 文件大小: 4,360,524 bytes
- **Git 提交版本**:
  - Commit Hash: `0c9f35482373f48b56a32891bdedd155d64f5608`
  - Commit Message: `feat: initial commit for IsaacTu reinforcement learning workspace`
- **工作区修改（Uncommitted Changes）影响说明**:
  - 当前仓库 `git status` 显示 7 个修改文件（`myrobot_env.py`, `myrobot_config.py`, `teleop.py`, `train.py`, `task_registry.py`, `on_policy_runner.py`, `__init__.py`）。
  - 这些修改包含了 Phase 5.1~5.3 开发的完整实现（包括 31 维观测扩展、CACM-TD 2.0 接触自适应调制、动态高度命令条件化、交互式遥测工具等）。
  - 与训练日志快照 `Oct10_15-48-32_turning_height_stage5_3_formal/source_snapshot/` 比对，当前工作区代码在高度死区 `height_deadzone=0.015` 时与 Stage 1 正式训练时完全等价，不影响历史评估结果的完全复现。

---

## 二、资产文件清单与 SHA256 校验码

### 1. 评估与诊断脚本 (`scripts/`)

| 文件名 | SHA256 | 功能说明 |
| :--- | :--- | :--- |
| `eval_phase5_3_stage1_comprehensive.py` | `5c822d4f96233204047694490e6d7be41012b776f505e090ef6e7bc552701d83` | Stage 1 50 轮正式训练 4 模型、16 场景全量评估脚本 |
| `eval_phase5_3_stage2_seg1.py` | `4ef0f8cd222b26435213080128855f88e6e31e66e44aa67bf911498f7fb32d39` | Stage 2 首段 25 轮 21 工况扩展评估脚本（Stage 1 m25 vs Stage 2 m10/m25） |
| `diagnose_left_squat_offset.py` | `5e2bc9e691b2c63f9aedd156498158d31082ae38989a1f77e73acc01427e6eec` | 前进/后退左转下蹲稳态高度静差专项诊断与固定观测探针分析 |
| `eval_phase5_3_ab_experiment.py` | `ac2473376671bc51fc6475b1de48ddcaeafcc9717b95cbe251319e41a99c7f1d` | 高度奖励死区 0.015m (A组) vs 0.010m (B组) 21 工况 A/B 对照评估 |
| `eval_phase5_3_low_lr_experiment.py` | `35c1090b6d78e6d5bbcb4243a43e0175b974989dc5375c179d84da778ca80f40` | 低学习率 1.0e-4 对照组 21 工况独立评估脚本 |
| `diagnose_termination_causes.py` | `51bef7d79eed0fea7a3b3d1aee3f09365036af6564ea18934b66a05d8a7ce22c` | 提前终止原因审计与各套件环境步/Episode 统计脚本 |
| `eval_joint_compliance.py` | `a44ace0cca6320789219f81a792748fe56ffbdc8e0fb10488133967832de8577` | 膝关节柔顺度、力矩极值与轮地接触滑移深度分析 |
| `audit_degradation_and_saturation.py` | `9679a5a2751ac5fd4d1c0bd9e95132f1220eb526803dfd1c263ae956ba050bdb` | 微调策略漂移、动作饱和与参数变化审计脚本 |
| `audit_contact_events.py` | `2bb5556024dd9586d1960554f7880cd3bb4b4bbddc73d1dc8234469d7a5a3c54` | 轮地接触力脉动与法向力分布审计脚本 |

### 2. 原始评测结果 (`results/`)

| 文件名 | SHA256 | 对应数据内容 |
| :--- | :--- | :--- |
| `phase5_3_stage1_comprehensive_eval.json` | `aa80bd3e954625716f2f926c035f9769166e2c7d78906ddabd975ac6d8dd1f76` | Stage 1 四模型 16 场景完整遥测指标 JSON |
| `eval_stage2_seg1_results.json` | `93c8a79634cf760db6319dcc63b63a34961b2cc8b46fe390cf52a78d346f55ca` | Stage 2 首段 21 工况评估三模型对比 JSON |
| `diagnose_left_squat_offset_results.json` | `3effffcb8af6f0b67438b7d18b097ba67e9d443d68c79ecd20d61618facceb33` | 左转下蹲物理遥测与固定网络前向输出 JSON |
| `eval_phase5_3_ab_results.json` | `624469048ff504da4f7c79b93d9a86158f1fade831c60d949729e742216c9351` | A/B 死区实验 7 模型 × 21 工况联合达标与遥测 JSON |
| `eval_phase5_3_low_lr_results.json` | `5e56ab4c5a70514f922823ec3a9ca49d6de587304f31a45ebde993c191141acb` | 低学习率 1.0e-4 实验 4 模型 × 21 工况评估 JSON |
| `termination_reasons.json` | `14464179328f4d9bffd2c4afe4c5a0239b3b6caf3bb44e24d5df0d42bbdd5ba8` | 18,000+ 环境步提前终止事件触发机制汇总 JSON |
| `compliance_joint_results.json` | `f332cd720a5b4c9a0c6f6b38d2d2d9c92a484164f0003dd6e7ba5f0fdb1a758c` | 各工况最大关节力矩与轮地纵向滑移度量 JSON |
| `audit_degradation_and_saturation.json` | `2f0fd80c602e1a8b66e4d4415ea88a99854d1d2d5282d27c36df412bfa44c158` | 动作漂移、PPO clip fraction 与 Actor 均值演化数据 |
| `contact_events_deep_audit.json` | `bdf1581d4a0bd9453e6f5abd550524692ccfbc1032e523fec06f7c2445de6a15` | 法向力与轮地微滑移事件统计数据 |

---

## 三、复现调用示例

使用项目统一 Python 运行时（包含 PyTorch 1.10.0+cu113 及 Isaac Gym 绑定）：
```bash
PYTHON_BIN="/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/bin/python"

# 1. 运行主基线 Stage 1 model_25 的全场景独立评估
${PYTHON_BIN} evaluation_assets/phase5_3/scripts/eval_phase5_3_stage1_comprehensive.py

# 2. 诊断左转下蹲高度静差与固定观测探针
${PYTHON_BIN} evaluation_assets/phase5_3/scripts/diagnose_left_squat_offset.py

# 3. 复现高度死区 A/B 对照实验评测
${PYTHON_BIN} evaluation_assets/phase5_3/scripts/eval_phase5_3_ab_experiment.py
```
