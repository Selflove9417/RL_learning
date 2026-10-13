import json
import numpy as np

p7f_file = "/home/xy/IsaacTu/evaluation_assets/phase7f/results/eval_phase7f_formal_aligned_results.json"
p8a_file = "/home/xy/IsaacTu/evaluation_assets/phase8a/results/eval_phase8a_results.json"

with open(p7f_file, "r") as f:
    d7f = json.load(f)["phase7c_champ_v3_compensated"]
with open(p8a_file, "r") as f:
    d8a = json.load(f)["phase8a_curriculum_ext_m1"]

groups = ["group1_squat", "group2_rise", "group3_regression", "group4_stress"]

def calc_summary(data):
    all_joint = []
    all_env = []
    all_h_err = []
    all_h_pass = []
    all_vy_rms = []
    all_wz_rms = []
    all_surv = []
    all_wheel_sat = []
    
    group_stats = {}
    for g in groups:
        g_joint = []
        g_env = []
        g_h_pass = []
        g_h_err = []
        for case, m in data[g].items():
            g_joint.append(m["joint_pass_rate"])
            g_env.append(m["env_pass_rate"])
            g_h_pass.append(m["h_pass_rate"])
            g_h_err.append(m["h_steady_err_cm"])
            all_joint.append(m["joint_pass_rate"])
            all_env.append(m["env_pass_rate"])
            all_h_pass.append(m["h_pass_rate"])
            all_h_err.append(m["h_steady_err_cm"])
            all_vy_rms.append(m.get("vy_rms_mps", 0.0))
            all_wz_rms.append(m.get("wz_rms_radps", 0.0))
            all_surv.append(m.get("episode_surv_rate", 100.0))
            all_wheel_sat.append(m.get("wheel_sat_rate", 0.0))
        group_stats[g] = {
            "joint": np.mean(g_joint),
            "env": np.mean(g_env),
            "h_pass": np.mean(g_h_pass),
            "h_err": np.mean(g_h_err),
        }
    return {
        "joint": np.mean(all_joint),
        "env": np.mean(all_env),
        "h_pass": np.mean(all_h_pass),
        "h_err": np.mean(all_h_err),
        "vy_rms": np.mean(all_vy_rms),
        "wz_rms": np.mean(all_wz_rms),
        "surv": np.mean(all_surv),
        "wheel_sat": np.mean(all_wheel_sat),
        "groups": group_stats
    }

s7f = calc_summary(d7f)
s8a = calc_summary(d8a)

print("=" * 95)
print("PHASE 8A CURRICULUM EXTENSION BENCHMARK SUMMARY (vs Phase 7F Champion v3 + Comp)")
print("=" * 95)
print(f"{'Metric':<25s} | {'Phase 7F Formal':<22s} | {'Phase 8A (PPO iter1)':<22s} | {'Delta(8A-7F)':<15s}")
print("-" * 95)
print(f"{'Global Joint%':<25s} | {s7f['joint']:21.2f}% | {s8a['joint']:21.2f}% | {s8a['joint'] - s7f['joint']:+14.2f}%")
print(f"{'Global Env%':<25s} | {s7f['env']:21.2f}% | {s8a['env']:21.2f}% | {s8a['env'] - s7f['env']:+14.2f}%")
print(f"{'Global Height Pass%':<25s} | {s7f['h_pass']:21.2f}% | {s8a['h_pass']:21.2f}% | {s8a['h_pass'] - s7f['h_pass']:+14.2f}%")
print(f"{'Mean Height Err (cm)':<25s} | {s7f['h_err']:20.3f}cm | {s8a['h_err']:20.3f}cm | {s8a['h_err'] - s7f['h_err']:+13.3f}cm")
print(f"{'Mean Vy RMS (m/s)':<25s} | {s7f['vy_rms']:22.4f} | {s8a['vy_rms']:22.4f} | {s8a['vy_rms'] - s7f['vy_rms']:+15.4f}")
print(f"{'Mean Wz RMS (rad/s)':<25s} | {s7f['wz_rms']:22.4f} | {s8a['wz_rms']:22.4f} | {s8a['wz_rms'] - s7f['wz_rms']:+15.4f}")
print(f"{'Survival Rate%':<25s} | {s7f['surv']:21.2f}% | {s8a['surv']:21.2f}% | {s8a['surv'] - s7f['surv']:+14.2f}%")
print(f"{'Wheel Saturation%':<25s} | {s7f['wheel_sat']:21.2f}% | {s8a['wheel_sat']:21.2f}% | {s8a['wheel_sat'] - s7f['wheel_sat']:+14.2f}%")

print("\n" + "=" * 95)
print("GROUP-LEVEL BREAKDOWN")
print("=" * 95)
for g in groups:
    j_7f = s7f['groups'][g]['joint']
    j_8a = s8a['groups'][g]['joint']
    h_7f = s7f['groups'][g]['h_pass']
    h_8a = s8a['groups'][g]['h_pass']
    e_7f = s7f['groups'][g]['h_err']
    e_8a = s8a['groups'][g]['h_err']
    env_7f = s7f['groups'][g]['env']
    env_8a = s8a['groups'][g]['env']
    print(f"[{g:18s}] Joint%: {j_7f:5.1f}% -> {j_8a:5.1f}% ({j_8a - j_7f:+5.1f}%) | Env%: {env_7f:5.1f}% -> {env_8a:5.1f}% ({env_8a - env_7f:+5.1f}%) | H_err: {e_7f:.2f}cm->{e_8a:.2f}cm | H%: {h_7f:5.1f}%->{h_8a:5.1f}%")

print("\n" + "=" * 105)
print("FOUR WEAK TARGET SCENARIOS DETAILED AUDIT")
print("=" * 105)
weak_cases = [
    ("group4_stress", "g4_exp_fwd_left_squat"),
    ("group4_stress", "g4_exp_bwd_left_squat"),
    ("group2_rise", "g2_bwd_right_rise"),
    ("group1_squat", "g1_fwd_left_squat"),
]
print(f"{'Target Case':25s} | {'7F J%':6s} {'8A J%':6s} {'Delta J%':8s} | {'7F H%':6s} {'8A H%':6s} | {'7F Vy%':6s} {'8A Vy%':6s} | {'7F Wz%':6s} {'8A Wz%':6s}")
print("-" * 105)
for grp, cname in weak_cases:
    m7 = d7f[grp][cname]
    m8 = d8a[grp][cname]
    dj = m8["joint_pass_rate"] - m7["joint_pass_rate"]
    print(f"{cname:25s} | {m7['joint_pass_rate']:5.1f}% {m8['joint_pass_rate']:5.1f}% {dj:+7.1f}% | {m7['h_pass_rate']:5.1f}% {m8['h_pass_rate']:5.1f}% | {m7['vy_pass_rate']:5.1f}% {m8['vy_pass_rate']:5.1f}% | {m7['wz_pass_rate']:5.1f}% {m8['wz_pass_rate']:5.1f}%")

print("\n" + "=" * 105)
print("IN-PLACE SQUAT AUDIT (g3_inplace_squat)")
print("=" * 105)
sq7 = d7f["group3_regression"]["g3_inplace_squat"]
sq8 = d8a["group3_regression"]["g3_inplace_squat"]
print(f"Joint%:  7F={sq7['joint_pass_rate']:.1f}% -> 8A={sq8['joint_pass_rate']:.1f}% ({sq8['joint_pass_rate']-sq7['joint_pass_rate']:+.1f}%)")
print(f"Env%:    7F={sq7['env_pass_rate']:.1f}% -> 8A={sq8['env_pass_rate']:.1f}% ({sq8['env_pass_rate']-sq7['env_pass_rate']:+.1f}%)")
print(f"Height%: 7F={sq7['h_pass_rate']:.1f}% -> 8A={sq8['h_pass_rate']:.1f}% ({sq8['h_pass_rate']-sq7['h_pass_rate']:+.1f}%)")
print(f"H_err:   7F={sq7['h_steady_err_cm']:.2f}cm -> 8A={sq8['h_steady_err_cm']:.2f}cm")
print(f"Vy RMS:  7F={sq7['vy_rms_mps']:.4f} -> 8A={sq8['vy_rms_mps']:.4f}")

print("\n" + "=" * 135)
print("ALL 21 SCENARIOS DETAILED BREAKDOWN (Phase 7F vs Phase 8A)")
print("=" * 135)
header = f"{'Case Name':25s} | {'H_ref':5s} | {'7F H_act':8s} {'8A H_act':8s} | {'7F H_err':8s} {'8A H_err':8s} | {'7F H%':6s} {'8A H%':6s} | {'7F J%':6s} {'8A J%':6s} {'Delta J%':8s} | {'7F Env%':7s} {'8A Env%':7s}"
print(header)
print("-" * 135)

regressed_cases = []
improved_cases = []
for g in groups:
    print(f"-- {g} --")
    for case in d7f[g]:
        m7 = d7f[g][case]
        m8 = d8a[g][case]
        h_ref = m7["h_ref_target_m"]
        diff_j = m8["joint_pass_rate"] - m7["joint_pass_rate"]
        print(f"{case:25s} | {h_ref:5.3f} | {m7['steady_actual_h_m']:8.4f} {m8['steady_actual_h_m']:8.4f} | {m7['h_steady_err_cm']:7.2f}c {m8['h_steady_err_cm']:7.2f}c | {m7['h_pass_rate']:5.1f}% {m8['h_pass_rate']:5.1f}% | {m7['joint_pass_rate']:5.1f}% {m8['joint_pass_rate']:5.1f}% {diff_j:+7.1f}% | {m7['env_pass_rate']:6.1f}% {m8['env_pass_rate']:6.1f}%")
        if diff_j < -2.0:
            regressed_cases.append((case, diff_j, m7["joint_pass_rate"], m8["joint_pass_rate"]))
        elif diff_j > 2.0:
            improved_cases.append((case, diff_j, m7["joint_pass_rate"], m8["joint_pass_rate"]))

print("\n" + "=" * 80)
print(f"REGRESSION / IMPROVEMENT SUMMARY:")
print(f"  * Improved cases (> +2.0%): {len(improved_cases)}")
for ic in improved_cases:
    print(f"    + {ic[0]:25s}: 7F={ic[2]:.1f}% -> 8A={ic[3]:.1f}% ({ic[1]:+.1f}%)")
print(f"  * Regressed cases (< -2.0%): {len(regressed_cases)}")
for rc in regressed_cases:
    print(f"    - {rc[0]:25s}: 7F={rc[2]:.1f}% -> 8A={rc[3]:.1f}% ({rc[1]:+.1f}%)")
print("=" * 80)
