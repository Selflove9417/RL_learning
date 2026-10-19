#!/usr/bin/env python3
# ==============================================================================
# Phase 4.9: ZeroCmd 20-Unseen-Seed Rigorous Paired Batch Runner
# Evaluates 20 brand-new unseen seeds: 8001 to 8020
#
# Compares:
#   Group A: Baseline Pilot B (zero_cmd = False)
#   Group B: ZeroCmd Pilot B (zero_cmd = True)
#
# Model: evaluation_assets/phase4_8_pilot_b/model_pilot_b_step1.pt
# ==============================================================================

import os
import sys
import subprocess
import json
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
eval_script = os.path.join(workspace_root, "tests/eval_phase4_9_zerocmd.py")
activate_script = os.path.join(workspace_root, "activate_jump.sh")
model_path = os.path.join(workspace_root, "evaluation_assets/phase4_8_pilot_b/model_pilot_b_step1.pt")

out_dir = os.path.join(workspace_root, "evaluation_assets/phase4_9_zerocmd")
indiv_dir = os.path.join(out_dir, "individual_runs")
os.makedirs(indiv_dir, exist_ok=True)

# 20 brand-new unseen seeds: 8001 to 8020
seeds = list(range(8001, 8021))

results = {
    "model_path": model_path,
    "seeds": seeds,
    "group_a_baseline": {},
    "group_b_zerocmd": {},
    "paired_comparison": {},
    "summary": {}
}

print("=" * 80)
print("PHASE 4.9: ZEROCMD 20 UNSEEN SEEDS RIGOROUS PAIRED GENERALIZATION BATCH")
print(f"Model Path: {model_path}")
print(f"Total Seeds: {len(seeds)} ({seeds[0]}..{seeds[-1]})")
print("=" * 80)

# 1. Group A: Baseline (zero_cmd = False)
print("\n>>> [GROUP A: BASELINE PILOT B (zero_cmd = False)]")
for idx, seed in enumerate(seeds, 1):
    out_file = os.path.join(indiv_dir, f"baseline_seed{seed}.json")
    cmd = f"source {activate_script} && python {eval_script} --seed {seed} --model \"{model_path}\" --out \"{out_file}\""
    print(f"  [{idx:02d}/20] Running Seed {seed} (Baseline)...", end=" ", flush=True)
    res = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"\nError on seed {seed}:\n{res.stderr[-400:]}")
        sys.exit(1)
    with open(out_file) as f:
        d = json.load(f)
    results["group_a_baseline"][str(seed)] = d
    print(f"Done. Fwd: {d['fwd_dist_cm']:.2f}cm | Clr: {d['max_clearance_cm']:.2f}cm | C4: {d['post_rec_vy_last10']:.4f}m/s | C4 Pass: {d['passed_rec_vy_003']} | Joint: {d['joint_passed_phase4']}")

# 2. Group B: ZeroCmd (zero_cmd = True)
print("\n>>> [GROUP B: ZEROCMD PILOT B (zero_cmd = True)]")
for idx, seed in enumerate(seeds, 1):
    out_file = os.path.join(indiv_dir, f"zerocmd_seed{seed}.json")
    cmd = f"source {activate_script} && python {eval_script} --seed {seed} --model \"{model_path}\" --zero_cmd --out \"{out_file}\""
    print(f"  [{idx:02d}/20] Running Seed {seed} (ZeroCmd)...", end=" ", flush=True)
    res = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"\nError on seed {seed}:\n{res.stderr[-400:]}")
        sys.exit(1)
    with open(out_file) as f:
        d = json.load(f)
    results["group_b_zerocmd"][str(seed)] = d
    print(f"Done. Fwd: {d['fwd_dist_cm']:.2f}cm | Clr: {d['max_clearance_cm']:.2f}cm | C4: {d['post_rec_vy_last10']:.4f}m/s | C4 Pass: {d['passed_rec_vy_003']} | Joint: {d['joint_passed_phase4']}")

# 3. Compute Paired Comparisons & Summary Statistics
def compute_group_stats(group_dict):
    fwd_dists = [d["fwd_dist_cm"] for d in group_dict.values()]
    clrs = [d["max_clearance_cm"] for d in group_dict.values()]
    pitches = [d["touchdown_pitch_deg"] for d in group_dict.values()]
    rolls = [d["touchdown_roll_deg"] for d in group_dict.values()]
    c4_vys = [abs(d["post_rec_vy_last10"]) for d in group_dict.values()]
    tau_Ls = [d["mean_tau_L_last10"] for d in group_dict.values()]
    tau_Rs = [d["mean_tau_R_last10"] for d in group_dict.values()]
    tau_fwds = [d["mean_tau_fwd_last10"] for d in group_dict.values()]
    wvel_Ls = [d["mean_wvel_L_last10"] for d in group_dict.values()]
    wvel_Rs = [d["mean_wvel_R_last10"] for d in group_dict.values()]

    c0_passes = sum(1 for d in group_dict.values() if d["passed_pre_runup"])
    c1_passes = sum(1 for d in group_dict.values() if d["passed_fwd_12cm"])
    c2_passes = sum(1 for d in group_dict.values() if d["passed_clr_20cm"])
    c3_passes = sum(1 for d in group_dict.values() if d["passed_attitude_10deg"])
    c4_passes = sum(1 for d in group_dict.values() if d["passed_rec_vy_003"])
    c5_passes = sum(1 for d in group_dict.values() if d["passed_no_reset"])
    joint_passes = sum(1 for d in group_dict.values() if d["joint_passed_phase4"])

    n = len(group_dict)
    return {
        "n": n,
        "fwd_dist_cm_mean": float(np.mean(fwd_dists)),
        "fwd_dist_cm_std": float(np.std(fwd_dists)),
        "fwd_dist_cm_min": float(np.min(fwd_dists)),
        "fwd_dist_cm_max": float(np.max(fwd_dists)),
        "max_clearance_cm_mean": float(np.mean(clrs)),
        "max_clearance_cm_std": float(np.std(clrs)),
        "touchdown_pitch_deg_mean": float(np.mean(pitches)),
        "touchdown_pitch_deg_std": float(np.std(pitches)),
        "touchdown_roll_deg_mean": float(np.mean(rolls)),
        "touchdown_roll_deg_std": float(np.std(rolls)),
        "c4_recovery_vy_mean": float(np.mean(c4_vys)),
        "c4_recovery_vy_std": float(np.std(c4_vys)),
        "c4_recovery_vy_min": float(np.min(c4_vys)),
        "c4_recovery_vy_max": float(np.max(c4_vys)),
        "mean_tau_L": float(np.mean(tau_Ls)),
        "mean_tau_R": float(np.mean(tau_Rs)),
        "mean_tau_fwd": float(np.mean(tau_fwds)),
        "mean_wvel_L": float(np.mean(wvel_Ls)),
        "mean_wvel_R": float(np.mean(wvel_Rs)),
        "c0_pass_rate": f"{c0_passes}/{n} ({c0_passes/n*100:.1f}%)",
        "c1_pass_rate": f"{c1_passes}/{n} ({c1_passes/n*100:.1f}%)",
        "c2_pass_rate": f"{c2_passes}/{n} ({c2_passes/n*100:.1f}%)",
        "c3_pass_rate": f"{c3_passes}/{n} ({c3_passes/n*100:.1f}%)",
        "c4_pass_rate": f"{c4_passes}/{n} ({c4_passes/n*100:.1f}%)",
        "c5_pass_rate": f"{c5_passes}/{n} ({c5_passes/n*100:.1f}%)",
        "joint_pass_rate": f"{joint_passes}/{n} ({joint_passes/n*100:.1f}%)",
        "c0_passes": c0_passes,
        "c1_passes": c1_passes,
        "c2_passes": c2_passes,
        "c3_passes": c3_passes,
        "c4_passes": c4_passes,
        "c5_passes": c5_passes,
        "joint_passes": joint_passes,
    }

results["summary"]["group_a_baseline"] = compute_group_stats(results["group_a_baseline"])
results["summary"]["group_b_zerocmd"] = compute_group_stats(results["group_b_zerocmd"])

# Paired breakdown
for seed in seeds:
    s = str(seed)
    a = results["group_a_baseline"][s]
    b = results["group_b_zerocmd"][s]
    results["paired_comparison"][s] = {
        "seed": seed,
        "baseline_fwd_cm": a["fwd_dist_cm"],
        "zerocmd_fwd_cm": b["fwd_dist_cm"],
        "fwd_diff_cm": b["fwd_dist_cm"] - a["fwd_dist_cm"],
        "baseline_clr_cm": a["max_clearance_cm"],
        "zerocmd_clr_cm": b["max_clearance_cm"],
        "baseline_pitch_deg": a["touchdown_pitch_deg"],
        "zerocmd_pitch_deg": b["touchdown_pitch_deg"],
        "baseline_roll_deg": a["touchdown_roll_deg"],
        "zerocmd_roll_deg": b["touchdown_roll_deg"],
        "baseline_c4_vy": a["post_rec_vy_last10"],
        "zerocmd_c4_vy": b["post_rec_vy_last10"],
        "c4_vy_reduction": abs(a["post_rec_vy_last10"]) - abs(b["post_rec_vy_last10"]),
        "baseline_c4_pass": a["passed_rec_vy_003"],
        "zerocmd_c4_pass": b["passed_rec_vy_003"],
        "baseline_c1_pass": a["passed_fwd_12cm"],
        "zerocmd_c1_pass": b["passed_fwd_12cm"],
        "baseline_joint_pass": a["joint_passed_phase4"],
        "zerocmd_joint_pass": b["joint_passed_phase4"],
        "baseline_mean_tau_L": a["mean_tau_L_last10"],
        "baseline_mean_tau_R": a["mean_tau_R_last10"],
        "zerocmd_mean_tau_L": b["mean_tau_L_last10"],
        "zerocmd_mean_tau_R": b["mean_tau_R_last10"],
        "baseline_mean_wvel_L": a["mean_wvel_L_last10"],
        "baseline_mean_wvel_R": a["mean_wvel_R_last10"],
        "zerocmd_mean_wvel_L": b["mean_wvel_L_last10"],
        "zerocmd_mean_wvel_R": b["mean_wvel_R_last10"],
    }

# Save full results
summary_file = os.path.join(out_dir, "zerocmd_20seeds_validation_summary.json")
with open(summary_file, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print("PHASE 4.9: 20-SEED GENERALIZATION SUMMARY")
print("=" * 80)
sa = results["summary"]["group_a_baseline"]
sb = results["summary"]["group_b_zerocmd"]
print(f"{'Metric':<28} | {'Group A (Baseline)':<22} | {'Group B (ZeroCmd)':<22}")
print("-" * 78)
print(f"{'Fwd Dist Mean (cm)':<28} | {sa['fwd_dist_cm_mean']:<22.2f} | {sb['fwd_dist_cm_mean']:<22.2f}")
print(f"{'Fwd Dist Min/Max (cm)':<28} | {sa['fwd_dist_cm_min']:.2f} / {sa['fwd_dist_cm_max']:.2f}          | {sb['fwd_dist_cm_min']:.2f} / {sb['fwd_dist_cm_max']:.2f}")
print(f"{'Max Clearance Mean (cm)':<28} | {sa['max_clearance_cm_mean']:<22.2f} | {sb['max_clearance_cm_mean']:<22.2f}")
print(f"{'TD Pitch Mean (deg)':<28} | {sa['touchdown_pitch_deg_mean']:<22.2f} | {sb['touchdown_pitch_deg_mean']:<22.2f}")
print(f"{'TD Roll Mean (deg)':<28} | {sa['touchdown_roll_deg_mean']:<22.2f} | {sb['touchdown_roll_deg_mean']:<22.2f}")
print(f"{'C4 Recovery Speed (|vy|)':<28} | {sa['c4_recovery_vy_mean']:<22.4f} | {sb['c4_recovery_vy_mean']:<22.4f}")
print(f"{'C4 Speed Range (m/s)':<28} | {sa['c4_recovery_vy_min']:.4f} / {sa['c4_recovery_vy_max']:.4f}        | {sb['c4_recovery_vy_min']:.4f} / {sb['c4_recovery_vy_max']:.4f}")
print(f"{'Mean Wheel Torque L (Nm)':<28} | {sa['mean_tau_L']:<22.2f} | {sb['mean_tau_L']:<22.2f}")
print(f"{'Mean Wheel Torque R (Nm)':<28} | {sa['mean_tau_R']:<22.2f} | {sb['mean_tau_R']:<22.2f}")
print(f"{'Mean Wheel Wvel L (rad/s)':<28} | {sa['mean_wvel_L']:<22.2f} | {sb['mean_wvel_L']:<22.2f}")
print(f"{'Mean Wheel Wvel R (rad/s)':<28} | {sa['mean_wvel_R']:<22.2f} | {sb['mean_wvel_R']:<22.2f}")
print("-" * 78)
print(f"{'C0 (Pre-runup) Pass':<28} | {sa['c0_pass_rate']:<22} | {sb['c0_pass_rate']:<22}")
print(f"{'C1 (Fwd >= 12cm) Pass':<28} | {sa['c1_pass_rate']:<22} | {sb['c1_pass_rate']:<22}")
print(f"{'C2 (Clr >= 20cm) Pass':<28} | {sa['c2_pass_rate']:<22} | {sb['c2_pass_rate']:<22}")
print(f"{'C3 (Attitude <= 10deg) Pass':<28} | {sa['c3_pass_rate']:<22} | {sb['c3_pass_rate']:<22}")
print(f"{'C4 (|v_rec| <= 0.03m/s) Pass':<28} | {sa['c4_pass_rate']:<22} | {sb['c4_pass_rate']:<22}")
print(f"{'C5 (No Abnormal Reset) Pass':<28} | {sa['c5_pass_rate']:<22} | {sb['c5_pass_rate']:<22}")
print("=" * 78)
print(f"{'Joint Phase 4 Pass Rate':<28} | {sa['joint_pass_rate']:<22} | {sb['joint_pass_rate']:<22}")
print("=" * 78)
print(f"Summary written to: {summary_file}")
