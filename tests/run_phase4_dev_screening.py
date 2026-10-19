#!/usr/bin/env python3
# ==============================================================================
# Phase 4.1 Development Screening: Compare Checkpoints across 5 Dev Seeds
# Dev Seeds: [3001, 3002, 3003, 3004, 3005]
# Models: Phase 3.2-V Baseline (Champion), Iter 5, Iter 10, Iter 15
# ==============================================================================

import os
import sys
import glob
import subprocess
import json
import argparse
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "tests"))

from eval_phase4_seed import eval_seed_full

DEV_SEEDS = [3001, 3002, 3003, 3004, 3005]


def find_latest_phase4_log_dir():
    base_dir = "/home/xy/IsaacTu_jump/legged_gym/logs/myrobot_forward_jump_phase4"
    runs = sorted(glob.glob(os.path.join(base_dir, "*_phase4_1_curriculum_0100m")))
    if not runs:
        raise FileNotFoundError(f"No Phase 4 runs found in {base_dir}")
    return runs[-1]


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


def run_screening(log_dir=None):
    if log_dir is None:
        log_dir = find_latest_phase4_log_dir()
    print(f"Screening checkpoints from: {log_dir}")

    out_dir = os.path.join(log_dir, "dev_screening")
    os.makedirs(out_dir, exist_ok=True)

    baseline_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"

    candidates = {
        "Phase 3.2-V Baseline": baseline_path,
        "Phase 4.1 Iter 5": os.path.join(log_dir, "model_5.pt"),
        "Phase 4.1 Iter 10": os.path.join(log_dir, "model_10.pt"),
        "Phase 4.1 Iter 15": os.path.join(log_dir, "model_15.pt"),
    }

    results = {}

    for name, path in candidates.items():
        if not os.path.exists(path):
            print(f"Warning: {name} path not found: {path}, skipping.")
            continue
        print(f"\n==================================================")
        print(f" Evaluating Candidate: {name}")
        print(f" Model Path: {path}")
        print(f"==================================================")
        results[name] = []
        for seed in DEV_SEEDS:
            out_file = os.path.join(out_dir, f"{name.replace(' ', '_').lower()}_seed{seed}.json")
            res = run_single_eval(seed, jump_model_path=path, out_file=out_file)
            if res is None:
                print(f"  Seed {seed}: FAILED TO RUN!")
                continue
            results[name].append(res)
            print(f"  Seed {seed}: Fwd = {res['fwd_dist_cm']:.2f} cm, Clr = {res['max_clearance_cm']:.2f} cm, "
                  f"Pitch = {res['touchdown_pitch_deg']:.2f} deg, Roll = {res['touchdown_roll_deg']:.2f} deg, "
                  f"RecVy = {res['post_rec_vy_last10']:.4f} m/s, PASS_12cm = {res['joint_passed_phase4']}")

    # Summary table
    print("\n" + "="*80)
    print("PHASE 4.1 DEVELOPMENT SCREENING SUMMARY (5 DEV SEEDS: 3001..3005)")
    print("="*80)
    print(f"{'Model':<24} | {'Pass Rate':<10} | {'Fwd Dist (cm)':<18} | {'Min Fwd (cm)':<12} | {'Clr (cm)':<12} | {'Pitch (deg)':<12} | {'Roll (deg)':<10}")
    print("-" * 105)

    summary_data = {}
    for name, res_list in results.items():
        fwds = [r['fwd_dist_cm'] for r in res_list]
        clrs = [r['max_clearance_cm'] for r in res_list]
        pitches = [abs(r['touchdown_pitch_deg']) for r in res_list]
        rolls = [abs(r['touchdown_roll_deg']) for r in res_list]
        passes_12 = [r['joint_passed_phase4'] for r in res_list]
        passes_base = [r['joint_passed_baseline'] for r in res_list]

        pass_rate_str = f"{sum(passes_12)}/5 ({sum(passes_12)/5*100:.0f}%)"
        fwd_str = f"{np.mean(fwds):.2f} +/- {np.std(fwds):.2f}"
        min_fwd = f"{np.min(fwds):.2f}"
        clr_str = f"{np.mean(clrs):.2f}"
        pitch_str = f"{np.mean(pitches):.2f}"
        roll_str = f"{np.mean(rolls):.2f}"

        print(f"{name:<24} | {pass_rate_str:<10} | {fwd_str:<18} | {min_fwd:<12} | {clr_str:<12} | {pitch_str:<12} | {roll_str:<10}")

        summary_data[name] = {
            "pass_rate_12cm": f"{sum(passes_12)}/5",
            "pass_rate_base": f"{sum(passes_base)}/5",
            "mean_fwd_cm": float(np.mean(fwds)),
            "std_fwd_cm": float(np.std(fwds)),
            "min_fwd_cm": float(np.min(fwds)),
            "max_fwd_cm": float(np.max(fwds)),
            "mean_clr_cm": float(np.mean(clrs)),
            "min_clr_cm": float(np.min(clrs)),
            "mean_pitch_deg": float(np.mean(pitches)),
            "max_pitch_deg": float(np.max(pitches)),
            "mean_roll_deg": float(np.mean(rolls)),
            "max_roll_deg": float(np.max(rolls)),
        }

    summary_file = os.path.join(out_dir, "screening_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args()
    run_screening(args.log_dir)
