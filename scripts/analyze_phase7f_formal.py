import json
import numpy as np

p7c_file = "/home/xy/IsaacTu/evaluation_assets/phase7c/results/eval_phase7c_results.json"
p7f_file = "/home/xy/IsaacTu/evaluation_assets/phase7f/results/eval_phase7f_formal_aligned_results.json"

with open(p7c_file, "r") as f:
    d7c = json.load(f)["phase7c_deadzone005_m1"]
with open(p7f_file, "r") as f:
    d7f = json.load(f)["phase7c_champ_v3_compensated"]

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

s7c = calc_summary(d7c)
s7f = calc_summary(d7f)

print("=" * 95)
print("PHASE 7F FORMAL ALIGNED BENCHMARK SUMMARY (vs Phase 7C Champion v3)")
print("=" * 95)
print(f"{'Metric':<25s} | {'Phase 7C Uncompensated':<22s} | {'Phase 7F Compensated':<22s} | {'Delta(7F-7C)':<15s}")
print("-" * 95)
print(f"{'Global Joint%':<25s} | {s7c['joint']:21.2f}% | {s7f['joint']:21.2f}% | {s7f['joint'] - s7c['joint']:+14.2f}%")
print(f"{'Global Env%':<25s} | {s7c['env']:21.2f}% | {s7f['env']:21.2f}% | {s7f['env'] - s7c['env']:+14.2f}%")
print(f"{'Global Height Pass%':<25s} | {s7c['h_pass']:21.2f}% | {s7f['h_pass']:21.2f}% | {s7f['h_pass'] - s7c['h_pass']:+14.2f}%")
print(f"{'Mean Height Err (cm)':<25s} | {s7c['h_err']:20.3f}cm | {s7f['h_err']:20.3f}cm | {s7f['h_err'] - s7c['h_err']:+13.3f}cm")
print(f"{'Mean Vy RMS (m/s)':<25s} | {s7c['vy_rms']:22.4f} | {s7f['vy_rms']:22.4f} | {s7f['vy_rms'] - s7c['vy_rms']:+15.4f}")
print(f"{'Mean Wz RMS (rad/s)':<25s} | {s7c['wz_rms']:22.4f} | {s7f['wz_rms']:22.4f} | {s7f['wz_rms'] - s7c['wz_rms']:+15.4f}")
print(f"{'Survival Rate%':<25s} | {s7c['surv']:21.2f}% | {s7f['surv']:21.2f}% | {s7f['surv'] - s7c['surv']:+14.2f}%")
print(f"{'Wheel Saturation%':<25s} | {s7c['wheel_sat']:21.2f}% | {s7f['wheel_sat']:21.2f}% | {s7f['wheel_sat'] - s7c['wheel_sat']:+14.2f}%")

print("\n" + "=" * 95)
print("GROUP-LEVEL BREAKDOWN")
print("=" * 95)
for g in groups:
    j_7c = s7c['groups'][g]['joint']
    j_7f = s7f['groups'][g]['joint']
    h_7c = s7c['groups'][g]['h_pass']
    h_7f = s7f['groups'][g]['h_pass']
    e_7c = s7c['groups'][g]['h_err']
    e_7f = s7f['groups'][g]['h_err']
    env_7c = s7c['groups'][g]['env']
    env_7f = s7f['groups'][g]['env']
    print(f"[{g:18s}] Joint%: {j_7c:5.1f}% -> {j_7f:5.1f}% ({j_7f - j_7c:+5.1f}%) | Env%: {env_7c:5.1f}% -> {env_7f:5.1f}% ({env_7f - env_7c:+5.1f}%) | H_err: {e_7c:.2f}cm->{e_7f:.2f}cm | H%: {h_7c:5.1f}%->{h_7f:5.1f}%")

print("\n" + "=" * 135)
print("ALL 21 SCENARIOS DETAILED BREAKDOWN (Phase 7C vs Phase 7F Formal)")
print("=" * 135)
header = f"{'Case Name':25s} | {'Cmd (vy, wz, H_ref)':22s} | {'7C H_act':8s} {'7F H_act':8s} | {'7C H_err':8s} {'7F H_err':8s} | {'7C H%':6s} {'7F H%':6s} | {'7C J%':6s} {'7F J%':6s} {'Delta J%':8s} | {'7C Env%':7s} {'7F Env%':7s}"
print(header)
print("-" * 135)

regressed_cases = []
for g in groups:
    print(f"-- {g} --")
    for case in d7c[g]:
        m7c = d7c[g][case]
        m7f = d7f[g][case]
        h_ref = m7f.get("h_ref_target_m", 0.0)
        h_act_7c = 0.360 + m7c['h_steady_err_cm']/100.0 if '0.36' in str(h_ref) else (0.440 - m7c['h_steady_err_cm']/100.0 if '0.44' in str(h_ref) else 0.398 + m7c['h_steady_err_cm']/100.0)
        h_act_7f = m7f.get("steady_actual_h_m", 0.0)
        diff_j = m7f["joint_pass_rate"] - m7c["joint_pass_rate"]
        cmd_str = f"({m7c.get('vy_rms_mps', 0.0):.2f}, ..., {h_ref:.3f})"
        # Look for case commands from scenarios
        print(f"{case:25s} | H_ref:{h_ref:5.3f}             | {h_act_7c:8.4f} {h_act_7f:8.4f} | {m7c['h_steady_err_cm']:7.2f}c {m7f['h_steady_err_cm']:7.2f}c | {m7c['h_pass_rate']:5.1f}% {m7f['h_pass_rate']:5.1f}% | {m7c['joint_pass_rate']:5.1f}% {m7f['joint_pass_rate']:5.1f}% {diff_j:+7.1f}% | {m7c['env_pass_rate']:6.1f}% {m7f['env_pass_rate']:6.1f}%")
        if diff_j < -2.0:
            regressed_cases.append((case, diff_j, m7c["joint_pass_rate"], m7f["joint_pass_rate"]))

print("\n" + "=" * 80)
print(f"REGRESSION CHECK: Found {len(regressed_cases)} cases with Delta Joint% < -2.0%")
for rc in regressed_cases:
    print(f"  * {rc[0]:25s}: 7C={rc[2]:.1f}% -> 7F={rc[3]:.1f}% ({rc[1]:+.1f}%)")
print("=" * 80)
