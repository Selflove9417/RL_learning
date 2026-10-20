# MyRobot Phase D.1 — 正式恢复语义与版本整理

基于 Phase D `55a1bd428aaef903d696fb7c08123cd2ce4cf9f5`。
本轮只改任务成功语义、评测及反例测试，未启动 PPO，未运行 A1/A2/B1/B2 四臂训练。
FSM、RECOVER 0.40 s 超时、`RECOVERY_HOLD_STEPS=10`、URDF、物理配置和执行器限幅均保持不变。

## 当前前向跳跃能力

最近一次训练仍是 Phase C 的 A1 十轮 pilot；Phase D 和 D.1 都没有更新模型。
Phase D 八种子有障评测中，AFTER 正常落地 6/8、双轮干净通过 3/8，但真实障碍接触 8/8、整机通过 0/8、完整跨障 0/8。
机器人具有部分前向起跳与落地能力，尚未具备完整跨障能力。

## 判据和奖励的最小修改

- **快速恢复**：原 NORMAL_HANDOFF 质量判据独立保留，并显式要求落地到交接不超过 0.40 s；记录原始 handoff 类型和 `fast_recovery_040s`。
- **正式恢复**：任务自有计数跨越 RECOVER 和 BALANCE。从落地后的第一个控制步开始，到落地后 2.0 s（含边界），完成连续至少十步安全巡航。速度至少为目标 0.35 m/s 的 70%，即 0.245 m/s；任一步不合格会清空连续计数。
- **每个合格步**：双轮各承载 >20 N、非腾空；复用现有 running RECOVER 的 Euler x 转角 <6°、y 转角 <12°，|vz|<0.20 m/s、水平速度 <1.20 m/s、双膝速度各 <1.5 rad/s、机身高度 ≥0.300 m。现有偏航角速度上限 0.50 rad/s 同时用于三轴，排除快速俯仰/侧倾；这是任务判据的保守补全，不改控制器。
- **未跌倒**：沿用父类的跌倒边界（高度 <0.20 m、x 转角 >45° 或 y 转角 >40°），按任务闩锁；其后恢复姿态不能抹去跌倒证据。
- **完整任务**：真实离地、双轮干净通过、实际触地时无遮挡、整机通过且当前后缘仍在盒后至少 5 mm、全程至当前零障碍接触、无任务失败、正式恢复、未跌倒必须全部满足。不得只靠速度或旧 `end_to_end_recruise` 布尔值。
- **落地语义修复**：`landing_safe` 只在首次触地时判定，不能先落在障碍投影内、之后驶离时补记为安全落地。
- **奖励契约**：`_mission_completed_mask()` 提供完整物理判据；训练的 `_mission_success_mask()` 只在其上加一次性支付门控。评测从终止前快照读取同一物理判据，并断言逐项评测结果一致。奖励权重、碰撞惩罚及其他 shaping 不变。
- **延后碰撞**：即使之前恢复或支付过成功奖励，后续碰撞仍会使最终物理判据失败；历史已支付的奖励不回滚，后续不再支付成功奖励，碰撞仍按既有逻辑惩罚并终止。

TIMEOUT_HANDOFF 本身不提供成功；BALANCE 的后续真实安全巡航可以提供正式恢复证据。
旧的速度扫描保留为 `m10.within_1s` / `whole_episode` 诊断，正式结果位于 `m10.formal_within_2s` 和共享任务门控。

## 实际验证

CPU pipeline 原有反例基线 **114/114**；本轮完整反例 **147/147**。
新增覆盖 TIMEOUT 后 BALANCE 恢复、TIMEOUT 后始终未恢复、单轮承载、姿态和三轴角速度失稳、0.245 m/s 边界、十步保持中断、2.0 s 边界、后续碰撞/跌倒、完整干净通过、一次支付、周期与 episode reset、落地后驶离不能补写安全落地。

冻结评测：仅 A1 `model_10.pt`，CPU pipeline，确定性动作，种子 **11、13**，每个种子有障/无障各一次；没有重跑完整八种子或四臂实验。

| 指标 | 有障碍 | 无障碍对照 |
|---|---:|---:|
| 实际碰撞 | 2/2 | 0/2 |
| 正式 2.0 s 安全巡航恢复 | 0/2 | 2/2 |
| 快速恢复 ≤0.40 s | 0/2 | 0/2 |
| 完整跨障 | 0/2 | 不适用 |

无障碍两次均 TIMEOUT_HANDOFF 后由 BALANCE 接管，正式恢复完成于触地后 **1.06 s、1.02 s**。这支持新的恢复路径，不代表跨障能力改善。
有障两次均 OBSTACLE_COLLISION；共享奖励/评测判据一致性断言在全部四次评测通过。
无障碍对照的完整跨障字段为 False，因为盒子被移出路径；不得将其解释为跨障失败率。

模型 SHA256 复核不变：

- BEFORE `model_3.pt`：`669da439a00e77d98471acb374cda9468b0ae78443c018edb9f0500f094e4f85`
- AFTER `model_10.pt`：`b6afad50e846deed983988176241b49c091034fe3ad1ecc7517e2095012792c4`

证据：`evaluation_assets/obstacle_jump_phaseD1_after_deterministic_results.json`、`evaluation_assets/phaseD1/semantics_cpu.log`、`evaluation_assets/phaseD1/frozen_after_cpu.log`。
历史 Phase D JSON 未改写。`git diff --check` 通过。

## Phase D 表述更正和 GPU 异常登记

已更正 Phase D 报告：32 次 NORMAL_HANDOFF 为零只说明当前策略在这些样本中未达标，不证明 FSM 逻辑不可达；速度保持可以与缓冲和稳定计数并行。严格成功为零同时受真实碰撞 8/8 和整机未通过 0/8 影响。

GPU 首轮 env0 异常单独保留：Phase D 真实训练初始化链路中，env0 比其余环境高约 16.9 mm、腿着地而轮未承载、第 13 步终止，reset 后恢复。原证据仍在 Phase D 报告与相关初始化结果中；根因未定位。
本轮未采用双重 reset，未扩展 CPU/GPU 探针矩阵，未重新执行训练初始化或 PPO 更新路径测试。

**下一次受控 A1/A2 准入：本轮语义、奖励契约和冻结小样本检查通过；尚不能宣布完整训练准入通过。** GPU 真实初始化异常仍为登记中的未关闭项，训练路径门禁没有在本轮重新验收。下一次另行完成准入裁决后才能启动；本轮到此停止，不以改判据宣称训练改善。

## GitHub 分支整理依据

GitHub 实查共 9 个分支、0 个开放 PR。已保留 `main`、当前 `feat/obstacle-jump-phased` 和 `backup/phase4_9_static_jump_archive`，删除下列 6 个冗余开发分支，远端从 9 个分支减少为 3 个。
逐一使用 `git merge-base --is-ancestor <sha> HEAD` 验证通过；这些提交全部仍可从当前分支访问，不重写提交历史，不删除模型或实验资料。

| 冗余远端分支 | 清理前完整 SHA |
|---|---|
| feat/dynamic-obstacle-clearing-jump | dcbe58c8ee7ac25cd66d39935b20ff77d65833c0 |
| feat/fix-running-jump-action-passthrough | 66a042923d2be8e7dc959c3d6190c1d18c18de37 |
| feat/myrobot-jump | 1e60b036ee49670b2300754055b61d73bf125d49 |
| feat/myrobot-running-jump | 5eab92c1e9a0678c46f333b12cc256052262dd82 |
| feat/obstacle-jump-phaseb | 8c562df633566d3f297cb007d053100f21876167 |
| feat/obstacle-jump-phasec | ba9c8075590c8cd6b46b6f89d2a036b58c5d5f9b |

D.1 首次提交和远端删除使用原子推送与逐个 SHA lease：任一分支被别人更新则整体拒绝，防止误删新工作。本地对应旧分支及已归档的重复分支共 7 个也使用 `git branch -d` 清理，本地从 10 个分支减少为 3 个。可用上表 SHA 随时重建旧分支。
