import json
import numpy as np

# Paths
pbase_file = "/home/xy/IsaacTu/evaluation_assets/phase6b/results/eval_phase6b_pilot_results.json"
p6b_file = "/home/xy/IsaacTu/evaluation_assets/phase6b/results/eval_phase6b_deadzone010_results.json"
p7c_file = "/home/xy/IsaacTu/evaluation_assets/phase7c/results/eval_phase7c_results.json"
p7f_file = "/home/xy/IsaacTu/evaluation_assets/phase7f/results/eval_phase7f_results.json"

with open(pbase_file, "r") as f:
    dbase = json.load(f)["baseline_m25"]
with open(p6b_file, "r") as f:
    d6b = json.load(f)["deadzone010_beta020_m1"]
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

sbase = calc_summary(dbase)
s6b = calc_summary(d6b)
s7c = calc_summary(d7c)
s7f = calc_summary(d7f)

print("=" * 100)
print("PHASE 7F GLOBAL COMPREHENSIVE SUMMARY")
print("=" * 100)
print(f"{'Metric':<25s} | {'Stage 1 Base':<14s} | {'Phase 6B v2':<14s} | {'Phase 7C v3':<14s} | {'Phase 7F Comp':<14s} | {'Delta(7F-7C)':<12s}")
print("-" * 100)
print(f"{'Global Joint%':<25s} | {sbase['joint']:13.2f}% | {s6b['joint']:13.2f}% | {s7c['joint']:13.2f}% | {s7f['joint']:13.2f}% | {s7f['joint'] - s7c['joint']:+11.2f}%")
print(f"{'Global Env%':<25s} | {sbase['env']:13.2f}% | {s6b['env']:13.2f}% | {s7c['env']:13.2f}% | {s7f['env']:13.2f}% | {s7f['env'] - s7c['env']:+11.2f}%")
print(f"{'Global Height Pass%':<25s} | {sbase['h_pass']:13.2f}% | {s6b['h_pass']:13.2f}% | {s7c['h_pass']:13.2f}% | {s7f['h_pass']:13.2f}% | {s7f['h_pass'] - s7c['h_pass']:+11.2f}%")
print(f"{'Mean Height Err (cm)':<25s} | {sbase['h_err']:12.3f}cm | {s6b['h_err']:12.3f}cm | {s7c['h_err']:12.3f}cm | {s7f['h_err']:12.3f}cm | {s7f['h_err'] - s7c['h_err']:+10.3f}cm")
print(f"{'Mean Vy RMS (m/s)':<25s} | {sbase['vy_rms']:14.4f} | {s6b['vy_rms']:14.4f} | {s7c['vy_rms']:14.4f} | {s7f['vy_rms']:14.4f} | {s7f['vy_rms'] - s7c['vy_rms']:+12.4f}")
print(f"{'Mean Wz RMS (rad/s)':<25s} | {sbase['wz_rms']:14.4f} | {s6b['wz_rms']:14.4f} | {s7c['wz_rms']:14.4f} | {s7f['wz_rms']:14.4f} | {s7f['wz_rms'] - s7c['wz_rms']:+12.4f}")
print(f"{'Survival Rate%':<25s} | {sbase['surv']:13.2f}% | {s6b['surv']:13.2f}% | {s7c['surv']:13.2f}% | {s7f['surv']:13.2f}% | {s7f['surv'] - s7c['surv']:+11.2f}%")
print(f"{'Wheel Saturation%':<25s} | {sbase['wheel_sat']:13.2f}% | {s6b['wheel_sat']:13.2f}% | {s7c['wheel_sat']:13.2f}% | {s7f['wheel_sat']:13.2f}% | {s7f['wheel_sat'] - s7c['wheel_sat']:+11.2f}%")

print("\n" + "=" * 100)
print("GROUP-LEVEL JOINT% & HEIGHT PASS% BREAKDOWN")
print("=" * 100)
for g in groups:
    j_base = sbase['groups'][g]['joint']
    j_6b = s6b['groups'][g]['joint']
    j_7c = s7c['groups'][g]['joint']
    j_7f = s7f['groups'][g]['joint']
    h_7c = s7c['groups'][g]['h_pass']
    h_7f = s7f['groups'][g]['h_pass']
    e_7c = s7c['groups'][g]['h_err']
    e_7f = s7f['groups'][g]['h_err']
    print(f"[{g:16s}] Joint%: Base={j_base:5.1f}% | 6B={j_6b:5.1f}% | 7C={j_7c:5.1f}% | 7F={j_7f:5.1f}% ({j_7f - j_7c:+5.1f}%) | H_err: {e_7c:.2f}cm->{e_7f:.2f}cm | H%: {h_7c:5.1f}%->{h_7f:5.1f}%")

print("\n" + "=" * 100)
print("G3 IN-PLACE SQUAT BENCHMARK COMPARISON (vs Phase 7E)")
print("=" * 100)
sq_base = dbase["group3_regression"]["g3_inplace_squat"]
sq_6b = d6b["group3_regression"]["g3_inplace_squat"]
sq_7c = d7c["group3_regression"]["g3_inplace_squat"]
sq_7f = d7f["group3_regression"]["g3_inplace_squat"]
print(f"H_ref: 0.360m | 7C H_act: ~{0.360 + sq_7c['h_steady_err_cm']/100.0:.4f}m -> 7F H_act: {sq_7f.get('steady_actual_h_m', 'N/A')}m (Cmd: {sq_7f.get('h_ctrl_cmd_m', 0.355):.3f}m)")
print(f"H_err_cm:  Base={sq_base['h_steady_err_cm']:.2f}cm | 6B={sq_6b['h_steady_err_cm']:.2f}cm | 7C={sq_7c['h_steady_err_cm']:.2f}cm | 7F={sq_7f['h_steady_err_cm']:.2f}cm")
print(f"Height%:   Base={sq_base['h_pass_rate']:.1f}% | 6B={sq_6b['h_pass_rate']:.1f}% | 7C={sq_7c['h_pass_rate']:.1f}% | 7F={sq_7f['h_pass_rate']:.1f}%")
print(f"Joint%:    Base={sq_base['joint_pass_rate']:.1f}% | 6B={sq_6b['joint_pass_rate']:.1f}% | 7C={sq_7c['joint_pass_rate']:.1f}% | 7F={sq_7f['joint_pass_rate']:.1f}%")
print(f"Env%:      Base={sq_base['env_pass_rate']:.1f}% | 6B={sq_6b['env_pass_rate']:.1f}% | 7C={sq_7c['env_pass_rate']:.1f}% | 7F={sq_7f['env_pass_rate']:.1f}%")

print("\n" + "=" * 120)
print("ALL 21 SCENARIOS DETAILED BREAKDOWN (Phase 7C vs Phase 7F)")
print("=" * 120)
header = f"{'Case Name':25s} | {'H_ref':5s} | {'Cmd':5s} | {'7C H_err':8s} {'7F H_err':8s} | {'7C H%':6s} {'7F H%':6s} | {'7C J%':6s} {'7F J%':6s} {'Delta J%':8s} | {'7C Env%':7s} {'7F Env%':7s}"
print(header)
print("-" * 120)

for g in groups:
    print(f"-- {g} --")
    for case in d7c[g]:
        m7c = d7c[g][case]
        m7f = d7f[g][case]
        h_ref = m7f.get("h_ref_target_m", 0.0)
        h_cmd = m7f.get("h_ctrl_cmd_m", h_ref)
        diff_j = m7f["joint_pass_rate"] - m7c["joint_pass_rate"]
        print(f"{case:25s} | {h_ref:5.3f} | {h_cmd:5.3f} | {m7c['h_steady_err_cm']:7.2f}c {m7f['h_steady_err_cm']:7.2f}c | {m7c['h_pass_rate']:5.1f}% {m7f['h_pass_rate']:5.1f}% | {m7c['joint_pass_rate']:5.1f}% {m7f['joint_pass_rate']:5.1f}% {diff_j:+7.1f}% | {m7c['env_pass_rate']:6.1f}% {m7f['env_pass_rate']:6.1f}%")

