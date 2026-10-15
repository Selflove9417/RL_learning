#!/usr/bin/env python3
"""
MyRobot Phase 6.1: Multi-Seed Continuous Double Jump Audit (Clean vs Noisy Evaluation)
Evaluates 5 independent seeds x 2 environments = 10 trajectories.
Spawns each seed in an independent subprocess to conform with Isaac Gym singleton architecture.
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEEDS = [42, 100, 200, 300, 400]
NUM_ENVS = 2
DURATION_S = 15.0

def run_phase6_audit(mode="clean", out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(workspace_root, "evaluation_assets", "phase6_clean" if mode == "clean" else "evaluation_assets/phase6_noisy")

    print("=" * 100)
    print(f"  MyRobot Phase 6.1: Multi-Seed Continuous Double Jump Audit [MODE: {mode.upper()}]")
    print(f"  * Seeds: {SEEDS} (Envs per seed: {NUM_ENVS}, Total Trajectories: {len(SEEDS) * NUM_ENVS})")
    print(f"  * Duration per trajectory: {DURATION_S}s")
    print(f"  * Target Output Directory: {out_dir}")
    print("=" * 100)

    os.makedirs(out_dir, exist_ok=True)

    test_script = os.path.join(workspace_root, "tests", "test_phase6_continuous_double_jump.py")

    for seed in SEEDS:
        print(f"\n--> Spawning independent simulation process for Seed {seed} [Mode: {mode}]...")
        cmd = [
            sys.executable, test_script,
            "--seed", str(seed),
            "--num_envs", str(NUM_ENVS),
            "--duration", str(DURATION_S),
            "--mode", mode,
            "--out_dir", out_dir
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{workspace_root}/legged_gym:{workspace_root}/rsl_rl:" + env.get("PYTHONPATH", "")
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            print(f"[ERROR] Process for Seed {seed} exited with code {res.returncode}")

    # Consolidate all trajectories
    all_trajectories = []
    for seed in SEEDS:
        json_path = os.path.join(out_dir, f"double_jump_seed{seed}_summary.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                d = json.load(f)
            all_trajectories.extend(d["trajectories"])

    summary_file = os.path.join(out_dir, "phase6_double_jump_multi_seed_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "eval_mode": mode,
            "seeds": SEEDS,
            "num_envs_per_seed": NUM_ENVS,
            "total_trajectories": len(all_trajectories),
            "trajectories": all_trajectories
        }, f, indent=2)

    total_n = len(all_trajectories)
    succ_n = sum(1 for t in all_trajectories if t["success"])
    j1_clr_pass = sum(1 for t in all_trajectories if t["jump1"]["reached_20cm"])
    j1_ho_norm = sum(1 for t in all_trajectories if t["jump1"]["handoff_type"] == "NORMAL_HANDOFF")
    remove_pass = sum(1 for t in all_trajectories if t["re_move"]["success"])
    j2_clr_pass = sum(1 for t in all_trajectories if t["jump2"]["reached_20cm"])
    j2_ho_norm = sum(1 for t in all_trajectories if t["jump2"]["handoff_type"] == "NORMAL_HANDOFF")
    stand_pass = sum(1 for t in all_trajectories if t["final_standing_duration_s"] >= 3.0)
    crash_n = sum(1 for t in all_trajectories if t["crashed"])
    abnormal_resets = sum(t["abnormal_resets_detected"] for t in all_trajectories)

    j1_clrs = [t["jump1"]["clearance_cm"] for t in all_trajectories]
    j2_clrs = [t["jump2"]["clearance_cm"] for t in all_trajectories]
    remove_dists = [t["re_move"]["distance_m"] for t in all_trajectories]

    print("\n" + "=" * 100)
    print(f"  PHASE 6.1 MULTI-SEED CONTINUOUS DOUBLE JUMP CONSOLIDATED AUDIT SUMMARY [{mode.upper()}]")
    print("=" * 100)
    print(f"Total Trajectories Audited            : {total_n}")
    print(f"Jump 1 Clearance >= 20cm Success Rate : {j1_clr_pass}/{total_n} ({j1_clr_pass/total_n*100:.1f}%) [Mean: {np.mean(j1_clrs):.2f} cm, Min: {np.min(j1_clrs):.2f} cm]")
    print(f"Jump 1 Normal Handoff Rate           : {j1_ho_norm}/{total_n} ({j1_ho_norm/total_n*100:.1f}%)")
    print(f"In-Between Re-Move Success Rate      : {remove_pass}/{total_n} ({remove_pass/total_n*100:.1f}%) [Mean Dist: {np.mean(remove_dists):.3f} m]")
    print(f"Jump 2 Clearance >= 20cm Success Rate : {j2_clr_pass}/{total_n} ({j2_clr_pass/total_n*100:.1f}%) [Mean: {np.mean(j2_clrs):.2f} cm, Min: {np.min(j2_clrs):.2f} cm]")
    print(f"Jump 2 Normal Handoff Rate           : {j2_ho_norm}/{total_n} ({j2_ho_norm/total_n*100:.1f}%)")
    print(f"Post Jump 2 Standing Balance (>=3.0s): {stand_pass}/{total_n} ({stand_pass/total_n*100:.1f}%)")
    print(f"Fatal Crashes                         : {crash_n}/{total_n} ({crash_n/total_n*100:.1f}%)")
    print(f"Abnormal Mid-Episode Resets           : {abnormal_resets}")
    print("-" * 100)
    print(f"FULL CONTINUOUS DOUBLE JUMP SUCCESS   : {succ_n}/{total_n} ({succ_n/total_n*100:.1f}%)")
    print("=" * 100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["clean", "noisy"], default="clean", help="Evaluation mode (clean: no noise, noisy: with noise)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    run_phase6_audit(mode=args.mode, out_dir=args.out_dir)
