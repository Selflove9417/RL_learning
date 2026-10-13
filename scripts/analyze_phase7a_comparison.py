import json
import numpy as np

def analyze():
    with open("evaluation_assets/phase6b/results/eval_phase6b_pilot_results.json") as f:
        pilot_res = json.load(f)
    with open("evaluation_assets/phase6b/results/eval_phase6b_deadzone010_results.json") as f:
        p6b_champ = json.load(f)
    with open("evaluation_assets/phase7a/results/eval_phase7a_results.json") as f:
        p7a_res = json.load(f)

    all_models = {
        "Baseline (m25)": pilot_res["baseline_m25"],
        "Group C m1 (dz=0.015)": pilot_res["groupC_m1"],
        "Phase 6B Champ v2 (dz=0.010)": p6b_champ["deadzone010_beta020_m1"],
        "Phase 7A (Stand 75%)": p7a_res["phase7a_sampling_m1"],
    }

    groups = ["group1_squat", "group2_rise", "group3_regression", "group4_stress"]

    print(f"{'Model':<30} | {'Joint%':<7} | {'Env%':<7} | {'G1':<6} | {'G2':<6} | {'G3':<6} | {'G4':<6} | {'H_err(cm)':<9} | {'Vy_rms':<7} | {'Wz_rms':<7} | {'Surv%':<6}")
    print("-" * 122)

    for m_name, m_data in all_models.items():
        all_jnt = []
        all_env = []
        all_h_err = []
        all_vy_rms = []
        all_wz_rms = []
        all_surv = []

        group_jnts = {}

        for g in groups:
            g_jnt = []
            for case_name, metrics in m_data[g].items():
                all_jnt.append(metrics["joint_pass_rate"])
                all_env.append(metrics["env_pass_rate"])
                all_h_err.append(metrics["h_steady_err_cm"])
                all_vy_rms.append(metrics["vy_rms_mps"])
                all_wz_rms.append(metrics["wz_rms_radps"])
                all_surv.append(metrics["episode_surv_rate"])
                g_jnt.append(metrics["joint_pass_rate"])
            group_jnts[g] = np.mean(g_jnt)

        total_jnt = np.mean(all_jnt)
        total_env = np.mean(all_env)
        mean_h_err = np.mean(all_h_err)
        mean_vy_rms = np.mean(all_vy_rms)
        mean_wz_rms = np.mean(all_wz_rms)
        mean_surv = np.mean(all_surv)

        print(f"{m_name:<30} | {total_jnt:6.2f}% | {total_env:6.2f}% | {group_jnts['group1_squat']:5.2f}% | {group_jnts['group2_rise']:5.2f}% | {group_jnts['group3_regression']:5.2f}% | {group_jnts['group4_stress']:5.2f}% | {mean_h_err:8.3f}  | {mean_vy_rms:7.4f} | {mean_wz_rms:7.4f} | {mean_surv:5.1f}%")

    print("\n--- Detailed Case: g3_inplace_squat ---")
    print(f"{'Model':<30} | {'Joint%':<7} | {'Env%':<7} | {'H_err(cm)':<9} | {'H_pass%':<7} | {'Vy_rms':<7} | {'Wz_rms':<7} | {'Surv%':<6}")
    print("-" * 100)
    for m_name, m_data in all_models.items():
        case = m_data["group3_regression"]["g3_inplace_squat"]
        print(f"{m_name:<30} | {case['joint_pass_rate']:6.2f}% | {case['env_pass_rate']:6.2f}% | {case['h_steady_err_cm']:8.3f}  | {case['h_pass_rate']:6.2f}% | {case['vy_rms_mps']:7.4f} | {case['wz_rms_radps']:7.4f} | {case['episode_surv_rate']:5.1f}%")

    print("\n--- Detailed Case: g3_nominal_compound (Nominal height turning compound) ---")
    print(f"{'Model':<30} | {'Joint%':<7} | {'Env%':<7} | {'H_err(cm)':<9} | {'H_pass%':<7} | {'Vy_rms':<7} | {'Wz_rms':<7} | {'Surv%':<6}")
    print("-" * 100)
    for m_name, m_data in all_models.items():
        case = m_data["group3_regression"]["g3_nominal_compound"]
        print(f"{m_name:<30} | {case['joint_pass_rate']:6.2f}% | {case['env_pass_rate']:6.2f}% | {case['h_steady_err_cm']:8.3f}  | {case['h_pass_rate']:6.2f}% | {case['vy_rms_mps']:7.4f} | {case['wz_rms_radps']:7.4f} | {case['episode_surv_rate']:5.1f}%")

    print("\n--- Detailed Case: g3_nominal_stand (In-place nominal stand) ---")
    print(f"{'Model':<30} | {'Joint%':<7} | {'Env%':<7} | {'H_err(cm)':<9} | {'H_pass%':<7} | {'Vy_rms':<7} | {'Wz_rms':<7} | {'Surv%':<6}")
    print("-" * 100)
    for m_name, m_data in all_models.items():
        case = m_data["group3_regression"]["g3_nominal_stand"]
        print(f"{m_name:<30} | {case['joint_pass_rate']:6.2f}% | {case['env_pass_rate']:6.2f}% | {case['h_steady_err_cm']:8.3f}  | {case['h_pass_rate']:6.2f}% | {case['vy_rms_mps']:7.4f} | {case['wz_rms_radps']:7.4f} | {case['episode_surv_rate']:5.1f}%")

    print("\n--- Detailed Case-by-Case Delta (Phase 7A vs Phase 6B Champ v2) ---")
    champ = p6b_champ["deadzone010_beta020_m1"]
    p7a = p7a_res["phase7a_sampling_m1"]
    for g in groups:
        print(f"=== {g} ===")
        for case in champ[g]:
            c_j = champ[g][case]["joint_pass_rate"]
            p7_j = p7a[g][case]["joint_pass_rate"]
            c_h = champ[g][case]["h_steady_err_cm"]
            p7_h = p7a[g][case]["h_steady_err_cm"]
            diff_j = p7_j - c_j
            diff_h = p7_h - c_h
            print(f"{case:<24} | Jnt: {c_j:4.1f}% -> {p7_j:4.1f}% ({diff_j:+5.1f}%) | H_err: {c_h:.2f} -> {p7_h:.2f}cm ({diff_h:+.2f}cm)")

if __name__ == "__main__":
    analyze()
