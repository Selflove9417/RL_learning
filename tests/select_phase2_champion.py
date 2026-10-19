#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Phase 2 Champion Checkpoint Selection Tool
# Evaluates models 5, 10, 15, 20, 25, 30 across seeds 42, 100, 200, 300, 400
# to select the champion based on authentic physical metrics.
# ==============================================================================

import os
import sys
import json
import subprocess
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
run_dir = os.path.join(workspace_root, "legged_gym", "logs", "myrobot_forward_jump_static", "Oct16_13-08-31_phase2_static_fwd_jump_repaired")
python_bin = sys.executable

checkpoints = [5, 10, 15, 20, 25, 30]
seeds = [42, 100, 200, 300, 400]

worker_script = os.path.join(workspace_root, "tests", "eval_single_seed_fwd_static.py")


def main():
    print("=" * 80)
    print("🏆 Phase 2 Checkpoint Screening & Champion Selection")
    print("=" * 80)

    results = {}

    for ckpt_num in checkpoints:
        ckpt_path = os.path.join(run_dir, f"model_{ckpt_num}.pt")
        print(f"\n--> Evaluating Checkpoint: model_{ckpt_num}.pt")

        seed_data = []
        for seed in seeds:
            out_json = f"/tmp/eval_ckpt_{ckpt_num}_seed_{seed}.json"
            cmd = [
                python_bin, worker_script,
                "--seed", str(seed),
                "--model_path", ckpt_path,
                "--out_json", out_json
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{workspace_root}/legged_gym:{workspace_root}/rsl_rl:" + env.get("PYTHONPATH", "")

            res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if res.returncode != 0:
                print(f"  [ERROR] Seed {seed} failed for model_{ckpt_num}")
                print(res.stderr[:300])
            else:
                with open(out_json) as f:
                    d = json.load(f)
                seed_data.append(d)

        # Aggregate metrics
        valid_count = sum(1 for d in seed_data if d["valid_jump"])
        resets_count = sum(d["abnormal_resets"] for d in seed_data)
        com_dists = [d["com_forward_dist_cm"] for d in seed_data]
        clearances = [d["max_clearance_cm"] for d in seed_data]
        vy_takeoffs = [d["takeoff_com_vy_mps"] for d in seed_data]
        pre_roll_vys = [d["pre_push_vy_mps"] for d in seed_data]
        pitch_touchdowns = [d["touchdown_pitch_deg"] for d in seed_data]
        recovery_vys = [d["recovery_vy_mps"] for d in seed_data]
        yaw_drifts = [d["yaw_drift_deg"] for d in seed_data]

        summary = {
            "checkpoint": ckpt_num,
            "path": ckpt_path,
            "valid_jumps": f"{valid_count}/{len(seeds)}",
            "abnormal_resets": resets_count,
            "com_dist_mean": round(float(np.mean(com_dists)), 2),
            "com_dist_std": round(float(np.std(com_dists)), 2),
            "com_dist_min": round(float(np.min(com_dists)), 2),
            "com_dist_max": round(float(np.max(com_dists)), 2),
            "clearance_mean": round(float(np.mean(clearances)), 2),
            "clearance_min": round(float(np.min(clearances)), 2),
            "takeoff_vy_mean": round(float(np.mean(vy_takeoffs)), 3),
            "pre_roll_vy_mean": round(float(np.mean(pre_roll_vys)), 3),
            "pitch_touchdown_mean": round(float(np.mean(pitch_touchdowns)), 2),
            "recovery_vy_mean": round(float(np.mean(recovery_vys)), 3),
            "yaw_drift_mean": round(float(np.mean(yaw_drifts)), 2),
            "seed_details": seed_data
        }
        results[f"model_{ckpt_num}"] = summary

        print(f"  Summary model_{ckpt_num}: CoM Dist = {summary['com_dist_mean']} ± {summary['com_dist_std']} cm "
              f"[{summary['com_dist_min']} to {summary['com_dist_max']}], "
              f"Clr = {summary['clearance_mean']} cm (min {summary['clearance_min']} cm), "
              f"Takeoff Vy = {summary['takeoff_vy_mean']} m/s, Pre-Roll Vy = {summary['pre_roll_vy_mean']} m/s, "
              f"Rec Vy = {summary['recovery_vy_mean']} m/s, Yaw Drift = {summary['yaw_drift_mean']} deg")

    summary_file = os.path.join(run_dir, "checkpoint_screening_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ All checkpoint screening results saved to: {summary_file}")


if __name__ == "__main__":
    main()
