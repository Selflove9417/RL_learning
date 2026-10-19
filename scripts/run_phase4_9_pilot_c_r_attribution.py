#!/usr/bin/env python3
"""
Phase 4.9-Pilot-C-R: Recovery Attribution Batch Runner
Evaluates 4 counterfactual policy pairings:
  - Group A: B/B (Takeoff = Pilot B, Recovery = Pilot B)
  - Group B: C/C (Takeoff = Pilot C, Recovery = Pilot C)
  - Group C: C/B (Takeoff = Pilot C, Recovery = Pilot B)
  - Group D: B/C (Takeoff = Pilot B, Recovery = Pilot C)

Execution protocol:
1. Run Seeds 3001~3005 minimal check across all 4 modes.
   Assert safety: 0 falls, 0 abnormal resets, acceptable torque rate.
2. If safe, run Seeds 8001~8020 (20 seeds) and Seeds 9001~9020 (20 seeds).
   Total = 45 seeds x 4 modes = 180 evaluations.
3. Compute full statistical attribution & transition torque continuity.
4. Export evaluation_assets/phase4_9_pilot_c_r/isolation_attribution_summary.json.
"""

import os
import sys
import subprocess
import json
import time
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
eval_script = os.path.join(workspace_root, "tests/eval_phase4_9_pilot_c_r_attribution.py")
activate_script = os.path.join(workspace_root, "activate_jump.sh")

out_dir = os.path.join(workspace_root, "evaluation_assets/phase4_9_pilot_c_r")
runs_dir = os.path.join(out_dir, "runs")
os.makedirs(runs_dir, exist_ok=True)

dev_seeds = list(range(3001, 3006))       # 5 seeds
eval_dev_seeds = list(range(8001, 8021))  # 20 seeds
unseen_seeds = list(range(9001, 9021))    # 20 seeds

modes = ["B/B", "C/C", "C/B", "B/C"]

print("=" * 85)
print("PHASE 4.9-PILOT-C-R: RECOVERY ATTRIBUTION & POLICY ISOLATION EXPERIMENT")
print(f"Modes: {modes}")
print(f"Output directory: {out_dir}")
print("=" * 85)


def run_single(seed, mode):
    mode_tag = mode.replace("/", "_")
    out_file = os.path.join(runs_dir, f"run_seed{seed}_mode{mode_tag}.json")
    if os.path.exists(out_file):
        with open(out_file) as f:
            return json.load(f)

    cmd = f"source {activate_script} && python {eval_script} --seed {seed} --mode {mode} --out \"{out_file}\""
    res = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"\n[ERROR] Failed on seed {seed} mode {mode}:\n{res.stderr[-500:]}")
        sys.exit(1)

    with open(out_file) as f:
        return json.load(f)


all_results = {
    "dev_3001_3005": {},
    "eval_dev_8001_8020": {},
    "unseen_9001_9020": {},
}

# --- STEP 1: Minimal Dev Set (3001~3005) Safety Check ---
print("\n>>> [STEP 1: MINIMAL CONTROL CHECK ON SEEDS 3001~3005]")
dev_runs = {}

for s in dev_seeds:
    dev_runs[str(s)] = {}
    print(f"\nSeed {s}:")
    for m in modes:
        res = run_single(s, m)
        dev_runs[str(s)][m] = res
        dtau = res["switching_safety"]["max_dtau_dt"]
        pass_j = "PASS" if res["joint_passed_phase4"] else "FAIL"
        print(f"  Mode {m:4s} | Fwd: {res['fwd_dist_cm']:5.2f}cm | Clr: {res['max_clearance_cm']:5.2f}cm | C4: {res['post_rec_vy_last10']:6.4f}m/s | Switch dtau/dt: {dtau:5.1f} Nm/s | Joint: {pass_j}")

all_results["dev_3001_3005"] = dev_runs

# Check switching safety in dev set
cb_falls = sum(1 for s in dev_seeds if dev_runs[str(s)]["C/B"]["switching_safety"]["has_fallen"])
cb_resets = sum(1 for s in dev_seeds if dev_runs[str(s)]["C/B"]["switching_safety"]["abnormal_resets"] > 0)
cb_max_dtau = max(dev_runs[str(s)]["C/B"]["switching_safety"]["max_dtau_dt"] for s in dev_seeds)

print("\n--- Minimal Dev Set Safety Audit ---")
print(f"  C/B Falls           : {cb_falls}/5")
print(f"  C/B Abnormal Resets : {cb_resets}/5")
print(f"  C/B Max dtau/dt     : {cb_max_dtau:.1f} Nm/s")

if cb_falls > 0 or cb_resets > 0 or cb_max_dtau > 2500.0:
    print("\n[CRITICAL ALERT] Switching instability detected in Dev Set! Stopping.")
    sys.exit(1)

print("  -> Dev Set Switching Safety Confirmed! Proceeding to Full 40 Seeds Batch...")

# --- STEP 2: Full Cohort (Eval Dev 8001~8020 & Unseen 9001~9020) ---
print("\n>>> [STEP 2: RUNNING EVAL DEV SEEDS 8001~8020]")
eval_dev_runs = {}
for idx, s in enumerate(eval_dev_seeds, 1):
    eval_dev_runs[str(s)] = {}
    print(f"[{idx:02d}/20] Seed {s}...", end=" ", flush=True)
    for m in modes:
        res = run_single(s, m)
        eval_dev_runs[str(s)][m] = res
    fwd_b = eval_dev_runs[str(s)]["B/B"]["fwd_dist_cm"]
    fwd_cb = eval_dev_runs[str(s)]["C/B"]["fwd_dist_cm"]
    c4_b = eval_dev_runs[str(s)]["B/B"]["post_rec_vy_last10"]
    c4_c = eval_dev_runs[str(s)]["C/C"]["post_rec_vy_last10"]
    c4_cb = eval_dev_runs[str(s)]["C/B"]["post_rec_vy_last10"]
    print(f"Done. Fwd: B/B={fwd_b:.2f} C/B={fwd_cb:.2f} | C4: B/B={c4_b:.4f} C/C={c4_c:.4f} C/B={c4_cb:.4f}")

all_results["eval_dev_8001_8020"] = eval_dev_runs

print("\n>>> [STEP 3: RUNNING UNSEEN SEEDS 9001~9020]")
unseen_runs = {}
for idx, s in enumerate(unseen_seeds, 1):
    unseen_runs[str(s)] = {}
    print(f"[{idx:02d}/20] Seed {s}...", end=" ", flush=True)
    for m in modes:
        res = run_single(s, m)
        unseen_runs[str(s)][m] = res
    fwd_b = unseen_runs[str(s)]["B/B"]["fwd_dist_cm"]
    fwd_cb = unseen_runs[str(s)]["C/B"]["fwd_dist_cm"]
    c4_b = unseen_runs[str(s)]["B/B"]["post_rec_vy_last10"]
    c4_c = unseen_runs[str(s)]["C/C"]["post_rec_vy_last10"]
    c4_cb = unseen_runs[str(s)]["C/B"]["post_rec_vy_last10"]
    print(f"Done. Fwd: B/B={fwd_b:.2f} C/B={fwd_cb:.2f} | C4: B/B={c4_b:.4f} C/C={c4_c:.4f} C/B={c4_cb:.4f}")

all_results["unseen_9001_9020"] = unseen_runs

# --- STEP 4: Comprehensive Statistics & Attribution Analysis ---
def aggregate_cohort(runs_dict):
    n = len(runs_dict)
    stats = {}
    for m in modes:
        fwds = [runs_dict[str(s)][m]["fwd_dist_cm"] for s in runs_dict]
        clrs = [runs_dict[str(s)][m]["max_clearance_cm"] for s in runs_dict]
        c4s = [abs(runs_dict[str(s)][m]["post_rec_vy_last10"]) for s in runs_dict]
        dtaus = [runs_dict[str(s)][m]["switching_safety"]["max_dtau_dt"] for s in runs_dict]

        c1_passes = sum(1 for s in runs_dict if runs_dict[str(s)][m]["passed_fwd_12cm"])
        c4_passes = sum(1 for s in runs_dict if runs_dict[str(s)][m]["passed_rec_vy_003"])
        joint_passes = sum(1 for s in runs_dict if runs_dict[str(s)][m]["joint_passed_phase4"])

        stats[m] = {
            "n": n,
            "fwd_mean": float(np.mean(fwds)),
            "fwd_std": float(np.std(fwds)),
            "clr_mean": float(np.mean(clrs)),
            "clr_std": float(np.std(clrs)),
            "c4_mean": float(np.mean(c4s)),
            "c4_std": float(np.std(c4s)),
            "dtau_mean": float(np.mean(dtaus)),
            "dtau_max": float(np.max(dtaus)),
            "c1_passes": c1_passes,
            "c1_rate": c1_passes / n,
            "c4_passes": c4_passes,
            "c4_rate": c4_passes / n,
            "joint_passes": joint_passes,
            "joint_rate": joint_passes / n,
        }
    return stats


# Aggregate across cohorts
summary = {
    "dev_3001_3005": aggregate_cohort(dev_runs),
    "eval_dev_8001_8020": aggregate_cohort(eval_dev_runs),
    "unseen_9001_9020": aggregate_cohort(unseen_runs),
}

# Global aggregate of all 45 seeds
all_45_runs = {}
for d in [dev_runs, eval_dev_runs, unseen_runs]:
    all_45_runs.update(d)
summary["global_all_45"] = aggregate_cohort(all_45_runs)

# Print Summary Table
print("\n" + "=" * 105)
print("                       PHASE 4.9-PILOT-C-R: RECOVERY ATTRIBUTION SUMMARY")
print("=" * 105)
print(f"{'Cohort':<18} | {'Mode':<6} | {'Fwd Dist (cm)':<18} | {'Clearance (cm)':<18} | {'C4 Vel (m/s)':<18} | {'C1 Pass':<10} | {'Joint Pass'}")
print("-" * 105)

for c_key, c_stats in summary.items():
    for m in modes:
        st = c_stats[m]
        print(f"{c_key:<18} | {m:<6} | {st['fwd_mean']:5.2f} ± {st['fwd_std']:4.2f}        | {st['clr_mean']:5.2f} ± {st['clr_std']:4.2f}        | {st['c4_mean']:6.4f} ± {st['c4_std']:6.4f}    | {st['c1_passes']:02d}/{st['n']:02d} ({st['c1_rate']*100:4.1f}%) | {st['joint_passes']:02d}/{st['n']:02d} ({st['joint_rate']*100:4.1f}%)")
    print("-" * 105)

# Save summary json
summary_out = os.path.join(out_dir, "isolation_attribution_summary.json")
with open(summary_out, "w") as f:
    json.dump({
        "all_seeds_results": all_45_runs,
        "cohort_summaries": summary,
    }, f, indent=2)

print(f"\nFull Attribution experiment data saved to: {summary_out}")
