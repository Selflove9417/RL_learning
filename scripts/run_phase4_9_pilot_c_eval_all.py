#!/usr/bin/env python3
"""
Phase 4.9-Pilot-C Level 3: Comprehensive 45-Seed Strictly Paired Physical Evaluation
Evaluates:
- Seeds 3001~3005 (Historical Dev Set, 5 seeds)
- Seeds 8001~8020 (Existing Dev Evaluation Set, 20 seeds)
- Seeds 9001~9020 (Brand-New Unseen Evaluation Set, 20 seeds)
Total = 45 seeds!

Strictly pairs:
- Group A (Pre-update Baseline): Pilot B + ZeroCmd (model_pilot_b_step1.pt with zero_cmd=True)
- Group B (Post-update Candidate): Pilot C + ZeroCmd (model_pilot_c_step1.pt with zero_cmd=True)

Both run under verified Phase 15 pre-takeoff settling handoff, falling hip compensation (+0.10),
and standard C0~C5 acceptance criteria.
"""

import os
import sys
import subprocess
import json
import time
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
eval_script = os.path.join(workspace_root, "tests/eval_phase4_9_zerocmd.py")
activate_script = os.path.join(workspace_root, "activate_jump.sh")

pilot_b_path = os.path.join(workspace_root, "evaluation_assets/phase4_8_pilot_b/model_pilot_b_step1.pt")
pilot_c_path = os.path.join(workspace_root, "evaluation_assets/phase4_9_pilot_c/model_pilot_c_step1.pt")

assert os.path.isfile(pilot_b_path), f"Pilot B model not found: {pilot_b_path}"
assert os.path.isfile(pilot_c_path), f"Pilot C model not found: {pilot_c_path}"

out_dir = os.path.join(workspace_root, "evaluation_assets/phase4_9_pilot_c")
eval_dir = os.path.join(out_dir, "eval_runs")
os.makedirs(eval_dir, exist_ok=True)

# 45 seeds partitioned into 3 cohorts
dev_seeds = list(range(3001, 3006))       # 5 seeds
eval_dev_seeds = list(range(8001, 8021))  # 20 seeds
unseen_seeds = list(range(9001, 9021))    # 20 seeds

all_seed_cohorts = {
    "dev_3001_3005": dev_seeds,
    "eval_dev_8001_8020": eval_dev_seeds,
    "unseen_9001_9020": unseen_seeds,
}

all_seeds = dev_seeds + eval_dev_seeds + unseen_seeds
print("=" * 80)
print("PHASE 4.9-PILOT-C: LEVEL 3 STRICTLY PAIRED 45-SEED PHYSICAL EVALUATION")
print(f"  Baseline Model (Pilot B) : {pilot_b_path}")
print(f"  Candidate Model (Pilot C): {pilot_c_path}")
print(f"  Total Seeds              : {len(all_seeds)} (Dev: 5, Eval Dev: 20, Unseen: 20)")
print("=" * 80)


def run_single_eval(seed, model_path, out_file):
    if os.path.exists(out_file):
        with open(out_file) as f:
            return json.load(f)
    cmd = f"source {activate_script} && python {eval_script} --seed {seed} --model \"{model_path}\" --zero_cmd --out \"{out_file}\""
    res = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"\n[ERROR] Failed on seed {seed} with model {os.path.basename(model_path)}:\n{res.stderr[-500:]}")
        sys.exit(1)
    with open(out_file) as f:
        return json.load(f)


results = {
    "pilot_b_model": pilot_b_path,
    "pilot_c_model": pilot_c_path,
    "total_seeds": len(all_seeds),
    "paired_runs": {},
    "cohort_summaries": {},
    "global_summary": {},
}

start_all_time = time.time()

for idx, seed in enumerate(all_seeds, 1):
    cohort_tag = "dev" if seed in dev_seeds else ("eval_dev" if seed in eval_dev_seeds else "unseen")
    print(f"[{idx:02d}/45] Seed {seed} ({cohort_tag})...", end=" ", flush=True)

    # 1. Baseline Pilot B + ZeroCmd
    file_b = os.path.join(eval_dir, f"pilot_b_zerocmd_seed{seed}.json")
    res_b = run_single_eval(seed, pilot_b_path, file_b)

    # 2. Candidate Pilot C + ZeroCmd
    file_c = os.path.join(eval_dir, f"pilot_c_zerocmd_seed{seed}.json")
    res_c = run_single_eval(seed, pilot_c_path, file_c)

    # Compute delta
    delta_fwd = res_c["fwd_dist_cm"] - res_b["fwd_dist_cm"]
    delta_clr = res_c["max_clearance_cm"] - res_b["max_clearance_cm"]
    delta_c4 = abs(res_c["post_rec_vy_last10"]) - abs(res_b["post_rec_vy_last10"])

    results["paired_runs"][str(seed)] = {
        "seed": seed,
        "cohort": cohort_tag,
        "pilot_b": res_b,
        "pilot_c": res_c,
        "delta": {
            "fwd_dist_cm": delta_fwd,
            "max_clearance_cm": delta_clr,
            "c4_vy_abs": delta_c4,
        }
    }

    status_b = "PASS" if res_b["joint_passed_phase4"] else "FAIL"
    status_c = "PASS" if res_c["joint_passed_phase4"] else "FAIL"

    print(f"Pilot B: Fwd={res_b['fwd_dist_cm']:.2f}cm C4={res_b['post_rec_vy_last10']:.4f}m/s [{status_b}] | "
          f"Pilot C: Fwd={res_c['fwd_dist_cm']:.2f}cm (Δ={delta_fwd:+.2f}cm) C4={res_c['post_rec_vy_last10']:.4f}m/s [{status_c}]")

elapsed_total = time.time() - start_all_time
print(f"\nAll 45 paired seeds evaluated in {elapsed_total:.2f}s!")


def compute_cohort_stats(seed_list):
    runs_b = [results["paired_runs"][str(s)]["pilot_b"] for s in seed_list]
    runs_c = [results["paired_runs"][str(s)]["pilot_c"] for s in seed_list]
    n = len(seed_list)

    stats = {}
    for group_name, runs in [("pilot_b", runs_b), ("pilot_c", runs_c)]:
        fwds = [r["fwd_dist_cm"] for r in runs]
        clrs = [r["max_clearance_cm"] for r in runs]
        pitches = [r["touchdown_pitch_deg"] for r in runs]
        rolls = [r["touchdown_roll_deg"] for r in runs]
        c4s = [abs(r["post_rec_vy_last10"]) for r in runs]

        stats[group_name] = {
            "n": n,
            "fwd_dist_mean": float(np.mean(fwds)),
            "fwd_dist_std": float(np.std(fwds)),
            "fwd_dist_min": float(np.min(fwds)),
            "fwd_dist_max": float(np.max(fwds)),
            "clr_mean": float(np.mean(clrs)),
            "clr_std": float(np.std(clrs)),
            "clr_min": float(np.min(clrs)),
            "clr_max": float(np.max(clrs)),
            "pitch_mean": float(np.mean(pitches)),
            "pitch_std": float(np.std(pitches)),
            "roll_mean": float(np.mean(rolls)),
            "roll_std": float(np.std(rolls)),
            "c4_mean": float(np.mean(c4s)),
            "c4_std": float(np.std(c4s)),
            "c4_min": float(np.min(c4s)),
            "c4_max": float(np.max(c4s)),
            "c0_pass_rate": sum(1 for r in runs if r["passed_pre_runup"]) / n,
            "c1_pass_rate": sum(1 for r in runs if r["passed_fwd_12cm"]) / n,
            "c2_pass_rate": sum(1 for r in runs if r["passed_clr_20cm"]) / n,
            "c3_pass_rate": sum(1 for r in runs if r["passed_attitude_10deg"]) / n,
            "c4_pass_rate": sum(1 for r in runs if r["passed_rec_vy_003"]) / n,
            "c5_pass_rate": sum(1 for r in runs if r["passed_no_reset"]) / n,
            "joint_pass_rate": sum(1 for r in runs if r["joint_passed_phase4"]) / n,
            "c0_passes": sum(1 for r in runs if r["passed_pre_runup"]),
            "c1_passes": sum(1 for r in runs if r["passed_fwd_12cm"]),
            "c2_passes": sum(1 for r in runs if r["passed_clr_20cm"]),
            "c3_passes": sum(1 for r in runs if r["passed_attitude_10deg"]),
            "c4_passes": sum(1 for r in runs if r["passed_rec_vy_003"]),
            "c5_passes": sum(1 for r in runs if r["passed_no_reset"]),
            "joint_passes": sum(1 for r in runs if r["joint_passed_phase4"]),
        }

    # Net change
    stats["net_delta"] = {
        "fwd_dist_mean": stats["pilot_c"]["fwd_dist_mean"] - stats["pilot_b"]["fwd_dist_mean"],
        "clr_mean": stats["pilot_c"]["clr_mean"] - stats["pilot_b"]["clr_mean"],
        "c4_mean": stats["pilot_c"]["c4_mean"] - stats["pilot_b"]["c4_mean"],
        "c1_pass_rate_gain": stats["pilot_c"]["c1_pass_rate"] - stats["pilot_b"]["c1_pass_rate"],
        "c4_pass_rate_gain": stats["pilot_c"]["c4_pass_rate"] - stats["pilot_b"]["c4_pass_rate"],
        "joint_pass_rate_gain": stats["pilot_c"]["joint_pass_rate"] - stats["pilot_b"]["joint_pass_rate"],
    }
    return stats


for name, s_list in all_seed_cohorts.items():
    results["cohort_summaries"][name] = compute_cohort_stats(s_list)

results["global_summary"] = compute_cohort_stats(all_seeds)

# Print Detailed Tables
print("\n" + "=" * 95)
print("                           PHASE 4.9-PILOT-C: 45-SEED COHORT SUMMARY")
print("=" * 95)
print(f"{'Cohort':<20} | {'Metric':<18} | {'Pilot B + ZeroCmd':<20} | {'Pilot C + ZeroCmd':<20} | {'Net Gain'}")
print("-" * 95)

for c_name, c_stats in results["cohort_summaries"].items():
    b_st = c_stats["pilot_b"]
    c_st = c_stats["pilot_c"]
    d_st = c_stats["net_delta"]
    print(f"{c_name:<20} | Fwd Dist (cm)      | {b_st['fwd_dist_mean']:5.2f} ± {b_st['fwd_dist_std']:4.2f}          | {c_st['fwd_dist_mean']:5.2f} ± {c_st['fwd_dist_std']:4.2f}          | {d_st['fwd_dist_mean']:+5.2f} cm")
    print(f"{'':<20} | Clearance (cm)     | {b_st['clr_mean']:5.2f} ± {b_st['clr_std']:4.2f}          | {c_st['clr_mean']:5.2f} ± {c_st['clr_std']:4.2f}          | {d_st['clr_mean']:+5.2f} cm")
    print(f"{'':<20} | C4 Vel (|vy|, m/s) | {b_st['c4_mean']:6.4f} ± {b_st['c4_std']:6.4f}      | {c_st['c4_mean']:6.4f} ± {c_st['c4_std']:6.4f}      | {d_st['c4_mean']:+6.4f} m/s")
    print(f"{'':<20} | C1 Pass Rate       | {b_st['c1_passes']:02d}/{b_st['n']:02d} ({b_st['c1_pass_rate']*100:5.1f}%)        | {c_st['c1_passes']:02d}/{c_st['n']:02d} ({c_st['c1_pass_rate']*100:5.1f}%)        | {d_st['c1_pass_rate_gain']*100:+5.1f}%")
    print(f"{'':<20} | C4 Pass Rate       | {b_st['c4_passes']:02d}/{b_st['n']:02d} ({b_st['c4_pass_rate']*100:5.1f}%)        | {c_st['c4_passes']:02d}/{c_st['n']:02d} ({c_st['c4_pass_rate']*100:5.1f}%)        | {d_st['c4_pass_rate_gain']*100:+5.1f}%")
    print(f"{'':<20} | Joint Pass Rate    | {b_st['joint_passes']:02d}/{b_st['n']:02d} ({b_st['joint_pass_rate']*100:5.1f}%)     | {c_st['joint_passes']:02d}/{c_st['n']:02d} ({c_st['joint_pass_rate']*100:5.1f}%)     | {d_st['joint_pass_rate_gain']*100:+5.1f}%")
    print("-" * 95)

# Global Summary
g_b = results["global_summary"]["pilot_b"]
g_c = results["global_summary"]["pilot_c"]
g_d = results["global_summary"]["net_delta"]
print(f"{'GLOBAL (ALL 45 SEEDS)':<20} | Fwd Dist (cm)      | {g_b['fwd_dist_mean']:5.2f} ± {g_b['fwd_dist_std']:4.2f}          | {g_c['fwd_dist_mean']:5.2f} ± {g_c['fwd_dist_std']:4.2f}          | {g_d['fwd_dist_mean']:+5.2f} cm")
print(f"{'':<20} | Clearance (cm)     | {g_b['clr_mean']:5.2f} ± {g_b['clr_std']:4.2f}          | {g_c['clr_mean']:5.2f} ± {g_c['clr_std']:4.2f}          | {g_d['clr_mean']:+5.2f} cm")
print(f"{'':<20} | C4 Vel (|vy|, m/s) | {g_b['c4_mean']:6.4f} ± {g_b['c4_std']:6.4f}      | {g_c['c4_mean']:6.4f} ± {g_c['c4_std']:6.4f}      | {g_d['c4_mean']:+6.4f} m/s")
print(f"{'':<20} | C1 Pass Rate       | {g_b['c1_passes']:02d}/{g_b['n']:02d} ({g_b['c1_pass_rate']*100:5.1f}%)        | {g_c['c1_passes']:02d}/{g_c['n']:02d} ({g_c['c1_pass_rate']*100:5.1f}%)        | {g_d['c1_pass_rate_gain']*100:+5.1f}%")
print(f"{'':<20} | C4 Pass Rate       | {g_b['c4_passes']:02d}/{g_b['n']:02d} ({g_b['c4_pass_rate']*100:5.1f}%)        | {g_c['c4_passes']:02d}/{g_c['n']:02d} ({g_c['c4_pass_rate']*100:5.1f}%)        | {g_d['c4_pass_rate_gain']*100:+5.1f}%")
print(f"{'':<20} | Joint Pass Rate    | {g_b['joint_passes']:02d}/{g_b['n']:02d} ({g_b['joint_pass_rate']*100:5.1f}%)     | {g_c['joint_passes']:02d}/{g_c['n']:02d} ({g_c['joint_pass_rate']*100:5.1f}%)     | {g_d['joint_pass_rate_gain']*100:+5.1f}%")
print("=" * 95)

# Save summary json
summary_out = os.path.join(out_dir, "paired_physical_evaluation_45seeds.json")
with open(summary_out, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nFull 45-seed evaluation data saved to: {summary_out}")


if __name__ == "__main__":
    pass
