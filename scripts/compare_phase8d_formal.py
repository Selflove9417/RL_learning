import json
import numpy as np

p8b_file = "/home/xy/IsaacTu/evaluation_assets/phase8b/results/eval_phase8b_m10_formal_results.json"
p8d_file = "/home/xy/IsaacTu/evaluation_assets/phase8d/results/eval_phase8d_m13_formal_results.json"

with open(p8b_file, "r") as f:
    d8b = json.load(f)["phase8b_m10_formal"]
with open(p8d_file, "r") as f:
    d8d = json.load(f)["phase8d_m13_formal"]

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

s8b = calc_summary(d8b)
s8d = calc_summary(d8d)

print("=" * 95)
print("PHASE 8D (m13) FORMAL BENCHMARK SUMMARY (vs Phase 8B Champion v4 m10)")
print("=" * 95)
print(f"{'Metric':<25s} | {'Phase 8B Champ v4':<22s} | {'Phase 8D m13':<22s} | {'Delta(8D-8B)':<15s}")
print("-" * 95)
print(f"{'Global Joint%':<25s} | {s8b['joint']:21.2f}% | {s8d['joint']:21.2f}% | {s8d['joint'] - s8b['joint']:+14.2f}%")
print(f"{'Global Env%':<25s} | {s8b['env']:21.2f}% | {s8d['env']:21.2f}% | {s8d['env'] - s8b['env']:+14.2f}%")
print(f"{'Global Height Pass%':<25s} | {s8b['h_pass']:21.2f}% | {s8d['h_pass']:21.2f}% | {s8d['h_pass'] - s8b['h_pass']:+14.2f}%")
print(f"{'Mean Height Err (cm)':<25s} | {s8b['h_err']:20.3f}cm | {s8d['h_err']:20.3f}cm | {s8d['h_err'] - s8b['h_err']:+13.3f}cm")
print(f"{'Mean Vy RMS (m/s)':<25s} | {s8b['vy_rms']:22.4f} | {s8d['vy_rms']:22.4f} | {s8d['vy_rms'] - s8b['vy_rms']:+15.4f}")
print(f"{'Mean Wz RMS (rad/s)':<25s} | {s8b['wz_rms']:22.4f} | {s8d['wz_rms']:22.4f} | {s8d['wz_rms'] - s8b['wz_rms']:+15.4f}")
print(f"{'Survival Rate%':<25s} | {s8b['surv']:21.2f}% | {s8d['surv']:21.2f}% | {s8d['surv'] - s8b['surv']:+14.2f}%")
print(f"{'Wheel Saturation%':<25s} | {s8b['wheel_sat']:21.2f}% | {s8d['wheel_sat']:21.2f}% | {s8d['wheel_sat'] - s8b['wheel_sat']:+14.2f}%")

print("\n" + "=" * 95)
print("GROUP-LEVEL BREAKDOWN")
print("=" * 95)
for g in groups:
    j_8b = s8b['groups'][g]['joint']
    j_8d = s8d['groups'][g]['joint']
    h_8b = s8b['groups'][g]['h_pass']
    h_8d = s8d['groups'][g]['h_pass']
    e_8b = s8b['groups'][g]['h_err']
    e_8d = s8d['groups'][g]['h_err']
    env_8b = s8b['groups'][g]['env']
    env_8d = s8d['groups'][g]['env']
    print(f"[{g:18s}] Joint%: {j_8b:5.1f}% -> {j_8d:5.1f}% ({j_8d - j_8b:+5.1f}%) | Env%: {env_8b:5.1f}% -> {env_8d:5.1f}% ({env_8d - env_8b:+5.1f}%) | H_err: {e_8b:.2f}cm->{e_8d:.2f}cm | H%: {h_8b:5.1f}%->{h_8d:5.1f}%")

print("\n" + "=" * 105)
print("TARGETED LOW LEFT-TURN SCENARIOS AUDIT")
print("=" * 105)
print(f"{'Target Case':25s} | {'8B J%':7s} {'8D J%':7s} {'Delta J%':8s} | {'8B H%':7s} {'8D H%':7s} | {'8B Vy%':7s} {'8D Vy%':7s} | {'8B Wz%':7s} {'8D Wz%'}")
print("-" * 105)

left_target_cases = [
    ("group4_stress", "g4_exp_fwd_left_squat"),
    ("group4_stress", "g4_exp_bwd_left_squat"),
    ("group1_squat", "g1_fwd_left_squat"),
    ("group1_squat", "g1_bwd_left_squat"),
]

for g, c in left_target_cases:
    m_8b = d8b[g][c]
    m_8d = d8d[g][c]
    print(f"{c:25s} | {m_8b['joint_pass_rate']:6.1f}% {m_8d['joint_pass_rate']:6.1f}% {m_8d['joint_pass_rate'] - m_8b['joint_pass_rate']:+7.1f}% | "
          f"{m_8b['h_pass_rate']:6.1f}% {m_8d['h_pass_rate']:6.1f}% | "
          f"{m_8b['vy_pass_rate']:6.1f}% {m_8d['vy_pass_rate']:6.1f}% | "
          f"{m_8b['wz_pass_rate']:6.1f}% {m_8d['wz_pass_rate']:6.1f}%")

print("\n" + "=" * 135)
print("ALL 21 SCENARIOS DETAILED BREAKDOWN (Phase 8B Champ v4 vs Phase 8D m13)")
print("=" * 135)
print(f"{'Case Name':25s} | {'H_ref':5s} | {'8B H_act':8s} {'8D H_act':8s} | {'8B H_err':8s} {'8D H_err':8s} | {'8B H%':6s} {'8D H%':6s} | {'8B J%':6s} {'8D J%':6s} {'Delta J%':8s} | {'8B Env%':7s} {'8D Env%'}")
print("-" * 135)

improved = []
regressed = []

for g in groups:
    print(f"-- {g} --")
    for c, m_8d in d8d[g].items():
        m_8b = d8b[g][c]
        d_j = m_8d["joint_pass_rate"] - m_8b["joint_pass_rate"]
        if d_j > 2.0:
            improved.append((c, m_8b["joint_pass_rate"], m_8d["joint_pass_rate"], d_j))
        elif d_j < -2.0:
            regressed.append((c, m_8b["joint_pass_rate"], m_8d["joint_pass_rate"], d_j))
            
        print(f"{c:25s} | {m_8d['h_ref_target']:5.3f} | {m_8b['steady_actual_h_m']:8.4f} {m_8d['steady_actual_h_m']:8.4f} | "
              f"{m_8b['h_steady_err_cm']:7.2f}c {m_8d['h_steady_err_cm']:7.2f}c | "
              f"{m_8b['h_pass_rate']:5.1f}% {m_8d['h_pass_rate']:5.1f}% | "
              f"{m_8b['joint_pass_rate']:5.1f}% {m_8d['joint_pass_rate']:5.1f}% {d_j:+7.1f}% | "
              f"{m_8b['env_pass_rate']:6.1f}% {m_8d['env_pass_rate']:6.1f}%")

print("\n" + "=" * 80)
print(f"REGRESSION / IMPROVEMENT SUMMARY:")
print(f"  * Improved cases (> +2.0%): {len(improved)}")
for c, j1, j2, dj in improved:
    print(f"    + {c:25s}: 8B={j1:.1f}% -> 8D={j2:.1f}% ({dj:+.1f}%)")
print(f"  * Regressed cases (< -2.0%): {len(regressed)}")
for c, j1, j2, dj in regressed:
    print(f"    - {c:25s}: 8B={j1:.1f}% -> 8D={j2:.1f}% ({dj:+.1f}%)")
print("=" * 80)
