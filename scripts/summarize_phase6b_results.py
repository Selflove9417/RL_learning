import json
import numpy as np

with open("evaluation_assets/phase6b/results/eval_phase6b_pilot_results.json") as f:
    data = json.load(f)

models = ["baseline_m25", "groupA_m1", "groupA_m3", "groupB_m1", "groupB_m3", "groupC_m1", "groupC_m3"]
groups = ["group1_squat", "group2_rise", "group3_regression", "group4_stress"]

print("=" * 115)
print("PHASE 6B-PILOT: 21-SCENARIO BENCHMARK SUMMARY TABLE")
print("=" * 115)
sep = "-" * 115
print(f"| {'Model':<14} | {'Total Jnt%':<10} | {'Total Env%':<10} | {'G1 Squat%':<10} | {'G2 Rise%':<10} | {'G3 Ret%':<10} | {'G4 Stress%':<10} | {'g3_inplace%':<14} | {'Surv%':<6} |")
print(sep)

for m in models:
    m_data = data[m]
    all_jnt = []
    all_env = []
    all_surv = []
    grp_jnt = {}
    
    for g in groups:
        g_jnts = [case["joint_pass_rate"] for case in m_data[g].values()]
        grp_jnt[g] = float(np.mean(g_jnts))
        for case in m_data[g].values():
            all_jnt.append(case["joint_pass_rate"])
            all_env.append(case["env_pass_rate"])
            all_surv.append(case["episode_surv_rate"])
            
    g3_inplace_jnt = m_data["group3_regression"]["g3_inplace_squat"]["joint_pass_rate"]
    g3_inplace_env = m_data["group3_regression"]["g3_inplace_squat"]["env_pass_rate"]
    
    total_jnt = float(np.mean(all_jnt))
    total_env = float(np.mean(all_env))
    total_surv = float(np.mean(all_surv))
    
    inplace_str = f"{g3_inplace_jnt:.1f}% ({g3_inplace_env:.1f}%)"
    print(f"| {m:<14} | {total_jnt:9.2f}% | {total_env:9.2f}% | {grp_jnt['group1_squat']:9.2f}% | {grp_jnt['group2_rise']:9.2f}% | {grp_jnt['group3_regression']:9.2f}% | {grp_jnt['group4_stress']:9.2f}% | {inplace_str:<14} | {total_surv:5.1f}% |")

print("=" * 115)

# Detailed breakdown by group and scenario
print("\n" + "=" * 115)
print("DETAILED BREAKDOWN ACROSS KEY SCENARIOS:")
print("=" * 115)
key_cases = [
    ("group3_regression", "g3_inplace_squat", "In-Place Squat (Historical Regression Target)"),
    ("group3_regression", "g3_cruise_squat",  "Cruise Squat"),
    ("group3_regression", "g3_nominal_cruise", "Nominal Cruise"),
    ("group3_regression", "g3_nominal_stand",  "Nominal Stand"),
    ("group1_squat",      "g1_fwd_straight_squat", "Fwd Straight Squat"),
    ("group1_squat",      "g1_bwd_right_squat", "Bwd Right Squat"),
    ("group2_rise",       "g2_fwd_left_rise", "Fwd Left Rise"),
    ("group2_rise",       "g2_recover_nominal", "Recover Nominal"),
    ("group4_stress",     "g4_exp_fwd_left_rise", "Exp Fwd Left Rise"),
]

for grp, case, label in key_cases:
    print(f"\n--- {case} [{label}] ---")
    header_case = f"{'Model':<14} | {'Jnt Pass%':<10} | {'Env Pass%':<10} | {'H_err (cm)':<10} | {'H_pass%':<10} | {'Vy_rms':<10} | {'Wz_rms':<10} | {'Slip (m/s)':<10}"
    print(header_case)
    print("-" * len(header_case))
    for m in models:
        c = data[m][grp][case]
        print(f"{m:<14} | {c['joint_pass_rate']:9.1f}% | {c['env_pass_rate']:9.1f}% | {c['h_steady_err_cm']:9.2f}cm | {c['h_pass_rate']:9.1f}% | {c['vy_rms_mps']:10.4f} | {c['wz_rms_radps']:10.4f} | {c['true_slip_max_mps']:10.4f}")
