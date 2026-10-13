import json
import numpy as np

def load_results():
    with open("evaluation_assets/phase6b/results/eval_phase6b_pilot_results.json") as f:
        pilot_res = json.load(f)
    with open("evaluation_assets/phase6b/results/eval_phase6b_pilot_groupC_m2_results.json") as f:
        m2_res = json.load(f)
    with open("evaluation_assets/phase6b/results/eval_phase6b_deadzone010_results.json") as f:
        dz010_res = json.load(f)

    all_models = {
        "Baseline (m25)": pilot_res["baseline_m25"],
        "Group C m1 (dz=0.015)": pilot_res["groupC_m1"],
        "Group C m2 (dz=0.015)": m2_res["groupC_m2"],
        "Group C m3 (dz=0.015)": pilot_res["groupC_m3"],
        "Deadzone 0.010 m1": dz010_res["deadzone010_beta020_m1"],
    }

    groups = ["group1_squat", "group2_rise", "group3_regression", "group4_stress"]

    print(f"{'Model':<25} | {'Joint%':<7} | {'Env%':<7} | {'G1':<6} | {'G2':<6} | {'G3':<6} | {'G4':<6} | {'H_err(cm)':<9} | {'Vy_rms':<7} | {'Wz_rms':<7} | {'Surv%':<6}")
    print("-" * 115)

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

        print(f"{m_name:<25} | {total_jnt:6.2f}% | {total_env:6.2f}% | {group_jnts['group1_squat']:5.2f}% | {group_jnts['group2_rise']:5.2f}% | {group_jnts['group3_regression']:5.2f}% | {group_jnts['group4_stress']:5.2f}% | {mean_h_err:8.3f}  | {mean_vy_rms:7.4f} | {mean_wz_rms:7.4f} | {mean_surv:5.1f}%")

    print("\n--- Detailed Case: g3_inplace_squat ---")
    print(f"{'Model':<25} | {'Joint%':<7} | {'Env%':<7} | {'H_err(cm)':<9} | {'H_pass%':<7} | {'Vy_rms':<7} | {'Wz_rms':<7}")
    print("-" * 85)
    for m_name, m_data in all_models.items():
        case = m_data["group3_regression"]["g3_inplace_squat"]
        print(f"{m_name:<25} | {case['joint_pass_rate']:6.2f}% | {case['env_pass_rate']:6.2f}% | {case['h_steady_err_cm']:8.3f}  | {case['h_pass_rate']:6.2f}% | {case['vy_rms_mps']:7.4f} | {case['wz_rms_radps']:7.4f}")

if __name__ == "__main__":
    load_results()
