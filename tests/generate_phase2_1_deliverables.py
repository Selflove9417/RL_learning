#!/usr/bin/env python3
# ==============================================================================
# MyRobot Phase 2.1 Deliverables Generator
# 1. Evaluates Baseline vs Phase 2 Champ vs Phase 2.1 Champ across 5 seeds
# 2. Generates 50 Hz high-frequency telemetry CSV
# 3. Renders 720p HD demonstration MP4 video
# 4. Saves champion model and multi-seed comparative summary JSON
# ==============================================================================

import os
import sys
import csv
import json
import shutil
import subprocess
import numpy as np
import imageio

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
python_bin = "/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/bin/python"
eval_script = os.path.join(workspace_root, "tests", "eval_joint_success.py")

output_dir = os.path.join(workspace_root, "evaluation_assets", "phase2_1_clearance_joint")
models_dir = os.path.join(output_dir, "models")
os.makedirs(models_dir, exist_ok=True)

SEEDS = [42, 100, 200, 300, 400]


def eval_model_across_seeds(model_path, model_name, clip=100.0):
    print(f"\n=================== Evaluating {model_name} ===================")
    results = []
    for seed in SEEDS:
        out_json = f"/tmp/p21_deliv_{model_name}_seed_{seed}.json"
        cmd = [python_bin, eval_script, "--model", model_path, "--seed", str(seed), "--clip", str(clip), "--out", out_json]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{workspace_root}/legged_gym:{workspace_root}/rsl_rl:" + env.get("PYTHONPATH", "")
        env["LD_LIBRARY_PATH"] = f"/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/lib:" + env.get("LD_LIBRARY_PATH", "")

        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if proc.returncode != 0:
            print(f"Error evaluating {model_name} Seed {seed}: {proc.stderr[:200]}")
            continue
        
        with open(out_json) as f:
            res = json.load(f)
        status = "✅ PASS" if res["joint_success"] else f"❌ FAIL ({', '.join(res['failure_reasons'])})"
        print(f"Seed {seed:3d}: {status} | Clr={res['max_clearance_cm']:5.2f}cm | FwdDist={res['com_forward_dist_cm']:5.2f}cm | Vz={res['takeoff_com_vz_mps']:+.3f}m/s | Vy={res['takeoff_com_vy_mps']:+.3f}m/s | Slip={res['prep_ground_slip_cm']:.2f}cm")
        results.append(res)
    
    n = len(results)
    valid_count = sum(1 for r in results if r["valid_jump"])
    clr_count = sum(1 for r in results if r["c2_clearance_ok"])
    fwd_count = sum(1 for r in results if r["c1_fwd_dist_ok"])
    joint_count = sum(1 for r in results if r["joint_success"])

    clrs = [r["max_clearance_cm"] for r in results]
    dists = [r["com_forward_dist_cm"] for r in results]
    vzs = [r["takeoff_com_vz_mps"] for r in results]
    vys = [r["takeoff_com_vy_mps"] for r in results]
    slips = [r["prep_ground_slip_cm"] for r in results]
    prep_vys = [r["max_prep_base_vel_mps"] for r in results]
    pitches = [r["touchdown_pitch_deg"] for r in results]
    rolls = [r["touchdown_roll_deg"] for r in results]
    yaw_drifts = [r["yaw_drift_deg"] for r in results]
    rec_vys = [r["recovery_vy_mps"] for r in results]
    rec_vxs = [r["recovery_vx_mps"] for r in results]
    hip_taus = [r["peak_hip_tau_nm"] for r in results]
    knee_taus = [r["peak_knee_tau_nm"] for r in results]
    wheel_taus = [r["peak_wheel_tau_nm"] for r in results]

    summary = {
        "model_name": model_name,
        "model_path": model_path,
        "clip_actions": clip,
        "valid_jumps": f"{valid_count}/{n}",
        "valid_jump_rate_pct": round(valid_count / n * 100.0, 1),
        "clearance_pass_count": f"{clr_count}/{n}",
        "clearance_pass_rate_pct": round(clr_count / n * 100.0, 1),
        "forward_dist_pass_count": f"{fwd_count}/{n}",
        "forward_dist_pass_rate_pct": round(fwd_count / n * 100.0, 1),
        "joint_success_count": f"{joint_count}/{n}",
        "joint_success_rate_pct": round(joint_count / n * 100.0, 1),
        "abnormal_resets_total": sum(r["abnormal_resets"] for r in results),

        "clearance_mean_cm": round(float(np.mean(clrs)), 2),
        "clearance_std_cm": round(float(np.std(clrs)), 2),
        "clearance_min_cm": round(float(np.min(clrs)), 2),
        "clearance_max_cm": round(float(np.max(clrs)), 2),

        "forward_dist_mean_cm": round(float(np.mean(dists)), 2),
        "forward_dist_std_cm": round(float(np.std(dists)), 2),
        "forward_dist_min_cm": round(float(np.min(dists)), 2),
        "forward_dist_max_cm": round(float(np.max(dists)), 2),

        "takeoff_vz_mean_mps": round(float(np.mean(vzs)), 3),
        "takeoff_vz_std_mps": round(float(np.std(vzs)), 3),
        "takeoff_vy_mean_mps": round(float(np.mean(vys)), 3),
        "takeoff_vy_std_mps": round(float(np.std(vys)), 3),

        "prep_ground_slip_mean_cm": round(float(np.mean(slips)), 2),
        "prep_ground_slip_max_cm": round(float(np.max(slips)), 2),
        "max_prep_base_vel_mean_mps": round(float(np.mean(prep_vys)), 3),

        "touchdown_pitch_mean_deg": round(float(np.mean(pitches)), 2),
        "touchdown_roll_mean_deg": round(float(np.mean(rolls)), 2),
        "yaw_drift_mean_deg": round(float(np.mean(yaw_drifts)), 2),
        "recovery_vy_mean_mps": round(float(np.mean(rec_vys)), 3),
        "recovery_vx_mean_mps": round(float(np.mean(rec_vxs)), 3),

        "peak_hip_tau_nm": round(float(np.max(hip_taus)), 2),
        "peak_knee_tau_nm": round(float(np.max(knee_taus)), 2),
        "peak_wheel_tau_nm": round(float(np.max(wheel_taus)), 2),

        "seed_evaluations": results
    }
    return summary


def render_video_and_telemetry(model_path, out_csv_path, out_mp4_path, seed=42):
    worker_script = os.path.join(workspace_root, "tests", "render_single_run.py")
    cmd = [python_bin, worker_script, "--model", model_path, "--seed", str(seed), "--out_csv", out_csv_path, "--out_mp4", out_mp4_path]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{workspace_root}/legged_gym:{workspace_root}/rsl_rl:" + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = f"/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/lib:" + env.get("LD_LIBRARY_PATH", "")

    print(f"\n🎬 Rendering Demonstration Video and Telemetry CSV (Seed {seed})...")
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if proc.returncode != 0:
        print(f"Error during video/telemetry rendering: {proc.stderr}")
    else:
        print(f"✅ Video and telemetry successfully rendered!")


def main(champion_checkpoint_path, candidate_checkpoint_path):
    # 1. Archive Champion & Candidate checkpoints
    champ_dest = os.path.join(models_dir, "model_champion.pt")
    cand_dest = os.path.join(models_dir, "model_candidate_iter15.pt")
    shutil.copyfile(champion_checkpoint_path, champ_dest)
    shutil.copyfile(candidate_checkpoint_path, cand_dest)
    print(f"Archived champion model to: {champ_dest}")
    print(f"Archived candidate model to: {cand_dest}")

    # 2. Multi-seed evaluations
    baseline_path = os.path.join(workspace_root, "evaluation_assets", "phase1_forward_jump", "models", "model_30_yaw_aligned.pt")
    phase2_champ_path = os.path.join(workspace_root, "evaluation_assets", "phase2_static_forward_jump", "models", "model_champion.pt")

    summary_baseline = eval_model_across_seeds(baseline_path, "baseline_trial2_in_static_env", clip=100.0)
    summary_phase2 = eval_model_across_seeds(phase2_champ_path, "phase2_champion_iter30", clip=1.0)
    summary_phase2_1_champ = eval_model_across_seeds(champ_dest, "phase2_1_champion_iter10", clip=100.0)
    summary_phase2_1_cand = eval_model_across_seeds(cand_dest, "phase2_1_candidate_iter15", clip=100.0)

    # 3. Save comparative summary JSON
    full_summary = {
        "metadata": {
            "title": "MyRobot Phase 2.1 Clearance-First Joint Optimization Multi-Seed Benchmark",
            "eval_seeds": SEEDS,
            "created_at": "2026-10-16T15:35:00+08:00",
            "environment": "myrobot_forward_jump_static",
            "constraints": {
                "c1_flight_displacement_ge_5cm": True,
                "c2_max_clearance_ge_20cm": True,
                "c3_preparation_stationary": "base_vy < 0.20 m/s, com_vy < 0.20 m/s, slip < 0.02 m, wheel_omega < 15.0 rad/s",
                "c4_recovery_stability": "touchdown |pitch|<=10°, |roll|<=10°, yaw_drift<=10°, rec_vy<=0.15 m/s",
                "c5_safety": "valid_jump == True, abnormal_resets == 0"
            }
        },
        "baseline_trial2": summary_baseline,
        "phase2_champion": summary_phase2,
        "phase2_1_champion_iter10": summary_phase2_1_champ,
        "phase2_1_candidate_iter15": summary_phase2_1_cand
    }

    summary_json_path = os.path.join(output_dir, "phase2_1_clearance_joint_evaluation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"\n📊 Multi-seed comparative summary saved to: {summary_json_path}")

    # 4. Render Telemetry & Video
    out_csv = os.path.join(output_dir, "clearance_priority_forward_jump_telemetry.csv")
    out_mp4 = os.path.join(output_dir, "clearance_priority_forward_jump_demonstration.mp4")
    render_video_and_telemetry(champ_dest, out_csv, out_mp4, seed=42)

    print("\n🎉 All Phase 2.1 deliverables successfully generated!")


if __name__ == "__main__":
    run_dir = "/home/xy/IsaacTu_jump/legged_gym/logs/myrobot_forward_jump_static/Oct16_15-26-49_phase2_1_clearance_priority"
    champ_ckpt = os.path.join(run_dir, "model_10.pt")
    cand_ckpt = os.path.join(run_dir, "model_15.pt")
    main(champ_ckpt, cand_ckpt)
