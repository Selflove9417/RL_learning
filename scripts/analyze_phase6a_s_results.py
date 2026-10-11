"""
Analysis script for Phase 6A-S evaluation results.
Computes summary metrics, group-by-group breakdowns, and comparison against baseline model_25.
"""

import json
import os
import numpy as np

def analyze_phase6a_s():
    json_path = "/home/xy/IsaacTu/evaluation_assets/phase6a_s/results/eval_phase6a_s_results.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist yet.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    models = list(data.keys())
    groups = ["group1_squat", "group2_rise", "group3_regression", "group4_stress"]

    print("=" * 105)
    print("PHASE 6A-S: 21-SCENARIO BENCHMARK EVALUATION SUMMARY")
    print("=" * 105)

    headers = [
        "Model", "Overall Joint Pass%", "Squat G1 Pass%", "Rise G2 Pass%",
        "Regression G3%", "Stress G4 Pass%", "H_err (cm)", "Vy_rms (m/s)", "Wz_rms (rad/s)", "Surv%"
    ]
    print(f"{headers[0]:<14} | {headers[1]:<18} | {headers[2]:<15} | {headers[3]:<14} | {headers[4]:<14} | {headers[5]:<16} | {headers[6]:<10} | {headers[7]:<12} | {headers[8]:<14} | {headers[9]:<6}")
    print("-" * 140)

    summary_records = {}

    for m in models:
        m_data = data[m]
        all_joint_passes = []
        all_h_errs = []
        all_vy_rms = []
        all_wz_rms = []
        all_surv = []
        group_passes = {}

        for g in groups:
            g_cases = m_data.get(g, {})
            g_j_passes = [c["joint_pass_rate"] for c in g_cases.values()]
            group_passes[g] = np.mean(g_j_passes) if g_j_passes else 0.0

            for c in g_cases.values():
                all_joint_passes.append(c["joint_pass_rate"])
                all_h_errs.append(c["h_steady_err_cm"])
                all_vy_rms.append(c["vy_rms_mps"])
                all_wz_rms.append(c["wz_rms_radps"])
                all_surv.append(c["episode_surv_rate"])

        overall_pass = np.mean(all_joint_passes)
        mean_h_err = np.mean(all_h_errs)
        mean_vy_rms = np.mean(all_vy_rms)
        mean_wz_rms = np.mean(all_wz_rms)
        mean_surv = np.mean(all_surv)

        summary_records[m] = {
            "overall_pass": overall_pass,
            "group_passes": group_passes,
            "mean_h_err": mean_h_err,
            "mean_vy_rms": mean_vy_rms,
            "mean_wz_rms": mean_wz_rms,
            "mean_surv": mean_surv
        }

        print(f"{m:<14} | {overall_pass:>17.2f}% | {group_passes['group1_squat']:>14.2f}% | {group_passes['group2_rise']:>13.2f}% | {group_passes['group3_regression']:>13.2f}% | {group_passes['group4_stress']:>15.2f}% | {mean_h_err:>9.3f} | {mean_vy_rms:>11.4f} | {mean_wz_rms:>13.4f} | {mean_surv:>5.1f}%")

    print("\n" + "=" * 105)
    print("SPECIFIC SKILL DRILL-DOWN: FORWARD LEFT SQUAT & IN-PLACE REGRESSION")
    print("=" * 105)
    print(f"{'Model':<14} | {'g1_fwd_left_squat Jnt%':<22} | {'g1_fwd_left H_err(cm)':<22} | {'g3_inplace_squat Jnt%':<22} | {'g3_nominal_stand Jnt%':<22}")
    print("-" * 115)
    for m in models:
        m_data = data[m]
        g1_fl = m_data.get("group1_squat", {}).get("g1_fwd_left_squat", {})
        g3_ips = m_data.get("group3_regression", {}).get("g3_inplace_squat", {})
        g3_std = m_data.get("group3_regression", {}).get("g3_nominal_stand", {})
        print(f"{m:<14} | {g1_fl.get('joint_pass_rate', 0.0):>21.1f}% | {g1_fl.get('h_steady_err_cm', 0.0):>21.3f} | {g3_ips.get('joint_pass_rate', 0.0):>21.1f}% | {g3_std.get('joint_pass_rate', 0.0):>21.1f}%")

if __name__ == '__main__':
    analyze_phase6a_s()
