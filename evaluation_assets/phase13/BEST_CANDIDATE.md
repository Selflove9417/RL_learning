# 正式晋升模型 (BEST_CANDIDATE.md)

- **新任正式 Champion**: **Champion v5** (`champion_v5_m100.pt`)
- **模型文件**: `evaluation_assets/champion_v5/models/champion_v5_m100.pt`
- **SHA256**: `5d006113b09c48e4171db9f9a7ed4f185256d73706544d291b42bd202f9e4499`
- **归档路径**: `evaluation_assets/phase13/models/phase13_exp7_m100.pt`

---

### Champion v5 验收达标成绩全量统计
- **训练迭代**: 100 iters (由 Champion v4 热启动累积 100 轮)
- **目标 A: 速度跟踪全量指标 (eval_speed_ramp.py)**:
  - `+0.20 m/s`: 实测 0.2018 m/s, 稳态误差 0.0071 m/s, 达标率 **100.0%**
  - `+0.30 m/s`: 实测 0.2991 m/s, 稳态误差 0.0072 m/s, 达标率 **100.0%**
  - `+0.40 m/s`: 实测 0.4079 m/s, 稳态误差 0.0094 m/s, 达标率 **100.0%**
  - `+0.50 m/s`: 实测 **0.5210 m/s**, 稳态误差 **0.0218 m/s** (<=0.05m/s 达成), 达标率 **96.2%** (>=90% 达成)
  - `-0.20 m/s`: 实测 -0.1829 m/s, 稳态误差 0.0172 m/s, 达标率 **100.0%**
  - `-0.30 m/s`: 实测 -0.2858 m/s, 稳态误差 0.0145 m/s, 达标率 **100.0%**
  - 稳态巡航制动距离: 0.50 m/s 下刹车距离 0.259 m，制动时间 0.93 s
- **目标 B: 基础技能保持 (eval_21_scenarios.py)**:
  - **21 工况 Global Joint%**: **85.78%** (Champion v4: 66.52%, 提升 **+19.26%**)
  - **21 工况 Global Env%**: **79.02%** (Champion v4: 63.24%, 提升 **+15.78%**)
  - **原地深蹲 Joint%**: **100.0%** (Champion v4: 88.80%, 提升 **+11.20%**)
  - **巡航深蹲 Joint%**: **100.0%**
  - **标称巡航 Joint%**: **100.0%**
  - **标称驻车 Joint%**: **100.0%**
- **目标 C: 连续复合运动序列 (eval_sequential_compound.py)**:
  - **连续复合运动 Sequential Joint%**: **96.85%** (Champion v4: 76.26%, 提升 **+20.59%**)
  - **连续复合运动 Sequential Env%**: **95.31%** (Champion v4: 68.36%, 提升 **+26.95%**)
  - **3 随机种子稳健性 (Seed 1, 42, 100)**:
    - Seed 1: Mean Joint% **96.55%**, Mean Env% **95.31%**, 存活率 **100.0%**
    - Seed 42: Mean Joint% **96.85%**, Mean Env% **95.31%**, 存活率 **100.0%**
    - Seed 100: Mean Joint% **96.27%**, Mean Env% **93.75%**, 存活率 **100.0%**
    - 3 种子平均: Mean Joint% **96.56%**, Mean Env% **94.79%**, 存活率 **100.00%**
- **目标 D: 安全性与执行器限幅**:
  - 全工况存活率: **100.00%** (零倾倒失控)
  - 轮端力矩饱和率: **0.14%** (远低于 2.0% 安全限值)
  - 0.50 m/s 巡航峰值扭矩: 3.03 Nm (远低于电机物理硬限幅 7.00 Nm)
- **部署启动命令**:
  - `python legged_gym/legged_gym/scripts/teleop.py --task myrobot --load_run Oct13_09-50-52_phase13_exp7_speed050_sig002_beta002_resume15iter --checkpoint 100`
