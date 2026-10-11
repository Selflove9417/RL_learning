import json
import numpy as np

def analyze():
    path = "/home/xy/IsaacTu/evaluation_assets/phase6a/results/eval_phase6a_ab_results.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = list(data.keys())
    groups = ["group1_squat", "group2_rise", "group3_regression", "group4_stress"]

    print("=" * 100)
    print("PHASE 6A BENCHMARK EVALUATION SUMMARY ACROSS 21 SCENARIOS")
    print("=" * 100)

    # 1. Overall Metrics Table
    header = f"{'Model':<16} | {'JointPass%':<10} | {'EnvPass%':<9} | {'H_err(cm)':<10} | {'H_pass%':<8} | {'Vy_RMS':<8} | {'Wz_RMS':<8} | {'Surv%':<6} | {'EarlyDone':<9} | {'WheelSat%':<9}"
    print(header)
    print("-" * len(header))

    summary_stats = {}

    for m in models:
        joint_passes = []
        env_passes = []
        h_errs = []
        h_passes = []
        vy_rmss = []
        wz_rmss = []
        surv_rates = []
        early_dones = []
        wheel_sats = []
        knee_taus = []
        slips = []

        for g in groups:
            for case_name, res in data[m][g].items():
                joint_passes.append(res['joint_pass_rate'])
                env_passes.append(res['env_pass_rate'])
                h_errs.append(res['h_steady_err_cm'])
                h_passes.append(res['h_pass_rate'])
                vy_rmss.append(res['vy_rms_mps'])
                wz_rmss.append(res['wz_rms_radps'])
                surv_rates.append(res['episode_surv_rate'])
                early_dones.append(res['early_dones'])
                wheel_sats.append(res['wheel_sat_rate'])
                knee_taus.append(res['knee_tau_mean_Nm'])
                slips.append(res['true_slip_mean_mps'])

        summary_stats[m] = {
            "joint_pass": np.mean(joint_passes),
            "env_pass": np.mean(env_passes),
            "h_err": np.mean(h_errs),
            "h_pass": np.mean(h_passes),
            "vy_rms": np.mean(vy_rmss),
            "wz_rms": np.mean(wz_rmss),
            "surv": np.mean(surv_rates),
            "early_done_tot": int(np.sum(early_dones)),
            "wheel_sat": np.mean(wheel_sats),
            "knee_tau": np.mean(knee_taus),
            "slip": np.mean(slips),
        }
        s = summary_stats[m]
        print(f"{m:<16} | {s['joint_pass']:10.2f} | {s['env_pass']:9.2f} | {s['h_err']:10.3f} | {s['h_pass']:8.1f} | {s['vy_rms']:8.4f} | {s['wz_rms']:8.4f} | {s['surv']:6.1f} | {s['early_done_tot']:9d} | {s['wheel_sat']:9.2f}")

    print("\n" + "=" * 100)
    print("CATEGORY BREAKDOWN: JOINT PASS RATE (%)")
    print("=" * 100)
    cat_header = f"{'Model':<16} | {'G1 Squat (5)':<14} | {'G2 Rise (5)':<14} | {'G3 Regress (5)':<14} | {'G4 Stress (6)':<14}"
    print(cat_header)
    print("-" * len(cat_header))
    for m in models:
        row = f"{m:<16} | "
        for g in groups:
            vals = [data[m][g][c]['joint_pass_rate'] for c in data[m][g]]
            row += f"{np.mean(vals):14.2f} | "
        print(row)

    print("\n" + "=" * 100)
    print("KEY SCENARIO DETAIL: FORWARD LEFT SQUAT (g1_fwd_left_squat)")
    print("=" * 100)
    fwd_header = f"{'Model':<16} | {'H_err(cm)':<10} | {'H_pass%':<8} | {'Vy_RMS':<8} | {'Wz_RMS':<8} | {'JointPass%':<10} | {'Surv%':<6} | {'EarlyDone':<9}"
    print(fwd_header)
    print("-" * len(fwd_header))
    for m in models:
        res = data[m]['group1_squat']['g1_fwd_left_squat']
        print(f"{m:<16} | {res['h_steady_err_cm']:10.3f} | {res['h_pass_rate']:8.1f} | {res['vy_rms_mps']:8.4f} | {res['wz_rms_radps']:8.4f} | {res['joint_pass_rate']:10.1f} | {res['episode_surv_rate']:6.1f} | {res['early_dones']:9d}")

    print("\n" + "=" * 100)
    print("SCENARIO-BY-SCENARIO DRIFT & DEGRADATION VS BASELINE_M25")
    print("=" * 100)
    # Count improved / degraded / identical scenarios for m10
    base_data = data['baseline_m25']
    for test_m in ["groupA_m10", "groupB_m10"]:
        deg_joint = 0
        imp_joint = 0
        eq_joint = 0
        deg_h = 0
        imp_h = 0
        deg_surv = 0
        imp_surv = 0
        for g in groups:
            for c in base_data[g]:
                b_joint = base_data[g][c]['joint_pass_rate']
                t_joint = data[test_m][g][c]['joint_pass_rate']
                if t_joint < b_joint - 1.0:
                    deg_joint += 1
                elif t_joint > b_joint + 1.0:
                    imp_joint += 1
                else:
                    eq_joint += 1

                b_h = base_data[g][c]['h_steady_err_cm']
                t_h = data[test_m][g][c]['h_steady_err_cm']
                if t_h > b_h + 0.1:
                    deg_h += 1
                elif t_h < b_h - 0.1:
                    imp_h += 1

                b_s = base_data[g][c]['episode_surv_rate']
                t_s = data[test_m][g][c]['episode_surv_rate']
                if t_s < b_s - 1.0:
                    deg_surv += 1
                elif t_s > b_s + 1.0:
                    imp_surv += 1

        print(f"{test_m} vs baseline_m25 (21 scenarios):")
        print(f"  Joint Pass Rate : Improved: {imp_joint}, Degraded: {deg_joint}, Unchanged: {eq_joint}")
        print(f"  Height Steady Err: Improved: {imp_h}, Degraded: {deg_h}")
        print(f"  Survival Rate   : Improved: {imp_surv}, Degraded: {deg_surv}")

if __name__ == "__main__":
    analyze()
