#!/usr/bin/env python3
# ==============================================================================
# Phase 4.1 Final Independent Acceptance: 20 Unseen Seeds Paired Evaluation
# Test Seeds: [4001..4010, 5001..5010] (20 brand-new unseen seeds)
# Comparison: Best Phase 4.1 Checkpoint vs Phase 3.2-V Frozen Baseline
# ==============================================================================

import os
import sys
import json
import subprocess
import argparse
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "tests"))

from eval_phase4_seed import eval_seed_full

TEST_SEEDS = [
    4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010,
    5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010
]


def run_single_eval(seed, jump_model_path, out_file):
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                data = json.load(f)
            if "joint_passed_phase4" in data:
                return data
        except Exception:
            pass

    cmd = (
        f"source /home/xy/IsaacTu_jump/activate_jump.sh && "
        f"python /home/xy/IsaacTu_jump/tests/eval_phase4_seed.py "
        f"--seed {seed} --jump_model \"{jump_model_path}\" --out \"{out_file}\""
    )
    res = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[Seed {seed}] ERROR: {res.stderr[-300:]}")
        return None

    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            return json.load(f)
    return None


def run_paired_acceptance(candidate_model_path, out_dir="evaluation_assets/phase4_1_acceptance"):
    os.makedirs(out_dir, exist_ok=True)
    baseline_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"

    print("=" * 80)
    print("PHASE 4.1 FINAL INDEPENDENT ACCEPTANCE: 20 UNSEEN SEEDS PAIRED EVALUATION")
    print(f"Seeds: {TEST_SEEDS}")
    print(f"Baseline: {baseline_path}")
    print(f"Candidate: {candidate_model_path}")
    print("=" * 80)

    baseline_results = []
    candidate_results = []

    for idx, seed in enumerate(TEST_SEEDS):
        print(f"\n--- [{idx+1}/20] Evaluating Seed {seed} ---")
        base_out = os.path.join(out_dir, f"baseline_seed_{seed}.json")
        res_base = run_single_eval(seed, jump_model_path=baseline_path, out_file=base_out)
        baseline_results.append(res_base)
        print(f"  [Baseline]  Fwd = {res_base['fwd_dist_cm']:.2f} cm, Clr = {res_base['max_clearance_cm']:.2f} cm, "
              f"Pitch = {res_base['touchdown_pitch_deg']:.2f} deg, Roll = {res_base['touchdown_roll_deg']:.2f} deg, "
              f"PASS_12cm = {res_base['joint_passed_phase4']}")

        cand_out = os.path.join(out_dir, f"candidate_seed_{seed}.json")
        res_cand = run_single_eval(seed, jump_model_path=candidate_model_path, out_file=cand_out)
        candidate_results.append(res_cand)
        print(f"  [Candidate] Fwd = {res_cand['fwd_dist_cm']:.2f} cm, Clr = {res_cand['max_clearance_cm']:.2f} cm, "
              f"Pitch = {res_cand['touchdown_pitch_deg']:.2f} deg, Roll = {res_cand['touchdown_roll_deg']:.2f} deg, "
              f"PASS_12cm = {res_cand['joint_passed_phase4']}")

    # Detailed Comparison Table
    print("\n" + "=" * 115)
    print("PAIRED 20-SEED DETAILED COMPARISON TABLE")
    print("=" * 115)
    print(f"{'Seed':<6} | {'Base Fwd(cm)':<13} | {'Cand Fwd(cm)':<13} | {'Delta(cm)':<10} | {'Base Clr':<9} | {'Cand Clr':<9} | {'Base Pit':<9} | {'Cand Pit':<9} | {'Base Pass':<10} | {'Cand Pass':<10}")
    print("-" * 115)

    for i, seed in enumerate(TEST_SEEDS):
        b = baseline_results[i]
        c = candidate_results[i]
        delta = c['fwd_dist_cm'] - b['fwd_dist_cm']
        b_pass = "PASS" if b['joint_passed_phase4'] else ("P(8cm)" if b['joint_passed_baseline'] else "FAIL")
        c_pass = "PASS" if c['joint_passed_phase4'] else ("P(8cm)" if c['joint_passed_baseline'] else "FAIL")
        print(f"{seed:<6} | {b['fwd_dist_cm']:<13.2f} | {c['fwd_dist_cm']:<13.2f} | {delta:<+10.2f} | {b['max_clearance_cm']:<9.2f} | {c['max_clearance_cm']:<9.2f} | {abs(b['touchdown_pitch_deg']):<9.2f} | {abs(c['touchdown_pitch_deg']):<9.2f} | {b_pass:<10} | {c_pass:<10}")

    # Aggregates
    b_fwds = [r['fwd_dist_cm'] for r in baseline_results]
    c_fwds = [r['fwd_dist_cm'] for r in candidate_results]
    b_clrs = [r['max_clearance_cm'] for r in baseline_results]
    c_clrs = [r['max_clearance_cm'] for r in candidate_results]
    b_pitches = [abs(r['touchdown_pitch_deg']) for r in baseline_results]
    c_pitches = [abs(r['touchdown_pitch_deg']) for r in candidate_results]
    b_rolls = [abs(r['touchdown_roll_deg']) for r in baseline_results]
    c_rolls = [abs(r['touchdown_roll_deg']) for r in candidate_results]
    b_passes_12 = [r['joint_passed_phase4'] for r in baseline_results]
    c_passes_12 = [r['joint_passed_phase4'] for r in candidate_results]
    b_passes_base = [r['joint_passed_baseline'] for r in baseline_results]
    c_passes_base = [r['joint_passed_baseline'] for r in candidate_results]

    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY ACROSS 20 UNSEEN SEEDS")
    print("=" * 80)
    print(f"Metric                          | Phase 3.2-V Baseline     | Phase 4.1 Candidate")
    print("-" * 80)
    print(f"Strict Pass Rate (>=12cm)       | {sum(b_passes_12)}/20 ({sum(b_passes_12)/20*100:.1f}%)        | {sum(c_passes_12)}/20 ({sum(c_passes_12)/20*100:.1f}%)")
    print(f"Baseline Pass Rate (>=8cm)      | {sum(b_passes_base)}/20 ({sum(b_passes_base)/20*100:.1f}%)        | {sum(c_passes_base)}/20 ({sum(c_passes_base)/20*100:.1f}%)")
    print(f"Forward Distance Mean (cm)      | {np.mean(b_fwds):.2f} +/- {np.std(b_fwds):.2f}         | {np.mean(c_fwds):.2f} +/- {np.std(c_fwds):.2f}")
    print(f"Forward Distance Min (cm)       | {np.min(b_fwds):.2f}                    | {np.min(c_fwds):.2f}")
    print(f"Forward Distance Max (cm)       | {np.max(b_fwds):.2f}                    | {np.max(c_fwds):.2f}")
    print(f"Bilateral Clearance Mean (cm)   | {np.mean(b_clrs):.2f} +/- {np.std(b_clrs):.2f}         | {np.mean(c_clrs):.2f} +/- {np.std(c_clrs):.2f}")
    print(f"Bilateral Clearance Min (cm)    | {np.min(b_clrs):.2f}                    | {np.min(c_clrs):.2f}")
    print(f"Touchdown Pitch Mean (deg)      | {np.mean(b_pitches):.2f} +/- {np.std(b_pitches):.2f}           | {np.mean(c_pitches):.2f} +/- {np.std(c_pitches):.2f}")
    print(f"Touchdown Pitch Max (deg)       | {np.max(b_pitches):.2f}                    | {np.max(c_pitches):.2f}")
    print(f"Touchdown Roll Mean (deg)       | {np.mean(b_rolls):.2f} +/- {np.std(b_rolls):.2f}           | {np.mean(c_rolls):.2f} +/- {np.std(c_rolls):.2f}")
    print(f"Touchdown Roll Max (deg)        | {np.max(b_rolls):.2f}                    | {np.max(c_rolls):.2f}")

    acceptance_summary = {
        "candidate_model_path": candidate_model_path,
        "baseline_model_path": baseline_path,
        "test_seeds": TEST_SEEDS,
        "baseline": {
            "pass_rate_12cm": f"{sum(b_passes_12)}/20",
            "pass_rate_base": f"{sum(b_passes_base)}/20",
            "mean_fwd_cm": float(np.mean(b_fwds)),
            "std_fwd_cm": float(np.std(b_fwds)),
            "min_fwd_cm": float(np.min(b_fwds)),
            "max_fwd_cm": float(np.max(b_fwds)),
            "mean_clr_cm": float(np.mean(b_clrs)),
            "min_clr_cm": float(np.min(b_clrs)),
            "mean_pitch_deg": float(np.mean(b_pitches)),
            "max_pitch_deg": float(np.max(b_pitches)),
            "mean_roll_deg": float(np.mean(b_rolls)),
            "max_roll_deg": float(np.max(b_rolls)),
        },
        "candidate": {
            "pass_rate_12cm": f"{sum(c_passes_12)}/20",
            "pass_rate_base": f"{sum(c_passes_base)}/20",
            "mean_fwd_cm": float(np.mean(c_fwds)),
            "std_fwd_cm": float(np.std(c_fwds)),
            "min_fwd_cm": float(np.min(c_fwds)),
            "max_fwd_cm": float(np.max(c_fwds)),
            "mean_clr_cm": float(np.mean(c_clrs)),
            "min_clr_cm": float(np.min(c_clrs)),
            "mean_pitch_deg": float(np.mean(c_pitches)),
            "max_pitch_deg": float(np.max(c_pitches)),
            "mean_roll_deg": float(np.mean(c_rolls)),
            "max_roll_deg": float(np.max(c_rolls)),
        },
        "paired_deltas": {
            "mean_fwd_gain_cm": float(np.mean(c_fwds) - np.mean(b_fwds)),
            "min_fwd_gain_cm": float(np.min(c_fwds) - np.min(b_fwds)),
            "per_seed_deltas": [float(c['fwd_dist_cm'] - b['fwd_dist_cm']) for b, c in zip(baseline_results, candidate_results)]
        }
    }

    with open(os.path.join(out_dir, "acceptance_summary.json"), "w") as f:
        json.dump(acceptance_summary, f, indent=2)

    return acceptance_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="evaluation_assets/phase4_1_acceptance")
    args = parser.parse_args()
    run_paired_acceptance(args.candidate, args.out_dir)
