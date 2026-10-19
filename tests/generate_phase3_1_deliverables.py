#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Phase 3.1 Deliverables Generator
# Evaluates Phase 2.1 Iter 10 vs Phase 2.1 Iter 15 vs Phase 3.1 Champion
# across 5 seeds using Level 3.1 acceptance criteria:
#   A. Dual-wheel clearance >= 20.0 cm
#   B. Flight CoM forward displacement >= 8.0 cm
#   C. Preparation ground slip < 2.0 cm (strictly stationary)
#   D. Post-landing recovery |V_y| <= 0.03 m/s
#   E. Zero abnormal resets
# ==============================================================================

import os
import sys
import csv
import json
import shutil
import subprocess
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
python_bin = "/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/bin/python"
eval_script = os.path.join(workspace_root, "tests", "eval_joint_success.py")
render_script = os.path.join(workspace_root, "tests", "render_single_run.py")

output_dir = os.path.join(workspace_root, "evaluation_assets", "phase3_1_forward_distance")
models_dir = os.path.join(output_dir, "models")
os.makedirs(models_dir, exist_ok=True)

SEEDS = [42, 100, 200, 300, 400]


def eval_model_across_seeds(model_path, model_name, clip=100.0, min_dist=8.0, max_rec_vy=0.03):
    print(f"\n=================== Evaluating {model_name} (min_dist={min_dist}cm, max_rec_vy={max_rec_vy}m/s) ===================")
    results = []
    for seed in SEEDS:
        out_json = f"/tmp/p31_deliv_{model_name}_seed_{seed}.json"
        cmd = [
            python_bin, eval_script,
            "--model", model_path,
            "--seed", str(seed),
            "--clip", str(clip),
            "--min_dist", str(min_dist),
            "--max_rec_vy", str(max_rec_vy),
            "--out", out_json
        ]
        
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
        print(f"Seed {seed:3d}: {status} | Clr={res['max_clearance_cm']:5.2f}cm | FwdDist={res['com_forward_dist_cm']:5.2f}cm | Vz={res['takeoff_com_vz_mps']:+.3f}m/s | Vy={res['takeoff_com_vy_mps']:+.3f}m/s | Slip={res['prep_ground_slip_cm']:.2f}cm | RecVy={res['recovery_vy_mps']:+.3f}m/s")
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
    prep_omegas = [r["max_prep_wheel_omega_radps"] for r in results]
    to_pitches = [r["takeoff_pitch_deg"] for r in results]
    to_yaw_rates = [r["takeoff_yaw_rate_radps"] for r in results]
    to_fn_ls = [r["takeoff_fn_L_N"] for r in results]
    to_fn_rs = [r["takeoff_fn_R_N"] for r in results]
    flight_durs = [r["flight_duration_s"] for r in results]
    pitches = [r["touchdown_pitch_deg"] for r in results]
    rolls = [r["touchdown_roll_deg"] for r in results]
    yaw_drifts = [r["yaw_drift_deg"] for r in results]
    td_vys = [r["touchdown_vy_mps"] for r in results]
    td_03s_vys = [r["touchdown_plus_03s_vy_mps"] for r in results]
    td_20s_vys = [r["touchdown_plus_20s_vy_mps"] for r in results]
    rec_vys = [r["recovery_vy_mps"] for r in results]
    rec_vxs = [r["recovery_vx_mps"] for r in results]
    hip_taus = [r["peak_hip_tau_nm"] for r in results]
    knee_taus = [r["peak_knee_tau_nm"] for r in results]
    wheel_taus = [r["peak_wheel_tau_nm"] for r in results]

    summary = {
        "model_name": model_name,
        "model_path": model_path,
        "clip_actions": clip,
        "min_dist_threshold_cm": min_dist,
        "max_rec_vy_threshold_mps": max_rec_vy,
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

        "flight_duration_mean_s": round(float(np.mean(flight_durs)), 3),

        "takeoff_vz_mean_mps": round(float(np.mean(vzs)), 3),
        "takeoff_vz_std_mps": round(float(np.std(vzs)), 3),
        "takeoff_vy_mean_mps": round(float(np.mean(vys)), 3),
        "takeoff_vy_std_mps": round(float(np.std(vys)), 3),
        "takeoff_pitch_mean_deg": round(float(np.mean(to_pitches)), 2),
        "takeoff_yaw_rate_mean_radps": round(float(np.mean(to_yaw_rates)), 3),
        "takeoff_contact_fn_L_mean_N": round(float(np.mean(to_fn_ls)), 2),
        "takeoff_contact_fn_R_mean_N": round(float(np.mean(to_fn_rs)), 2),

        "prep_ground_slip_mean_cm": round(float(np.mean(slips)), 2),
        "prep_ground_slip_max_cm": round(float(np.max(slips)), 2),
        "max_prep_base_vel_mean_mps": round(float(np.mean(prep_vys)), 3),
        "max_prep_wheel_omega_mean_radps": round(float(np.mean(prep_omegas)), 3),

        "touchdown_pitch_mean_deg": round(float(np.mean(pitches)), 2),
        "touchdown_roll_mean_deg": round(float(np.mean(rolls)), 2),
        "touchdown_vy_mean_mps": round(float(np.mean(td_vys)), 3),
        "touchdown_plus_03s_vy_mean_mps": round(float(np.mean(td_03s_vys)), 3),
        "touchdown_plus_20s_vy_mean_mps": round(float(np.mean(td_20s_vys)), 3),
        "yaw_drift_mean_deg": round(float(np.mean(yaw_drifts)), 2),

        "recovery_vy_mean_mps": round(float(np.mean(rec_vys)), 3),
        "recovery_abs_vy_mean_mps": round(float(np.mean([abs(v) for v in rec_vys])), 3),
        "recovery_vx_mean_mps": round(float(np.mean(rec_vxs)), 3),

        "peak_hip_tau_nm": round(float(np.max(hip_taus)), 2),
        "peak_knee_tau_nm": round(float(np.max(knee_taus)), 2),
        "peak_wheel_tau_nm": round(float(np.max(wheel_taus)), 2),

        "seed_evaluations": results
    }
    return summary


def render_video_and_telemetry(model_path, out_csv_path, out_mp4_path, seed=42):
    cmd = [python_bin, render_script, "--model", model_path, "--seed", str(seed), "--clip", "100.0", "--out_csv", out_csv_path, "--out_mp4", out_mp4_path]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{workspace_root}/legged_gym:{workspace_root}/rsl_rl:" + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = f"/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/lib:" + env.get("LD_LIBRARY_PATH", "")

    print(f"\n🎬 Rendering Demonstration Video and Telemetry CSV (Seed {seed})...")
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if proc.returncode != 0:
        print(f"Error during video/telemetry rendering: {proc.stderr}")
    else:
        print(f"✅ Video and telemetry successfully rendered!")


def main(champion_checkpoint_path):
    champ_dest = os.path.join(models_dir, "model_champion.pt")
    shutil.copyfile(champion_checkpoint_path, champ_dest)
    print(f"Archived Phase 3.1 champion model to: {champ_dest}")

    p21_m10_path = os.path.join(workspace_root, "evaluation_assets", "phase2_1_clearance_joint", "models", "model_champion.pt")
    p21_m15_path = os.path.join(workspace_root, "evaluation_assets", "phase2_1_clearance_joint", "models", "model_candidate_iter15.pt")

    summary_p21_m10 = eval_model_across_seeds(p21_m10_path, "phase2_1_iter10", clip=100.0, min_dist=8.0, max_rec_vy=0.03)
    summary_p21_m15 = eval_model_across_seeds(p21_m15_path, "phase2_1_iter15", clip=100.0, min_dist=8.0, max_rec_vy=0.03)
    summary_p31_champ = eval_model_across_seeds(champ_dest, "phase3_1_champion", clip=100.0, min_dist=8.0, max_rec_vy=0.03)

    full_summary = {
        "metadata": {
            "title": "MyRobot Phase 3.1 Level 3.1 (8-10 cm) Forward Jump Benchmark",
            "eval_seeds": SEEDS,
            "created_at": "2026-10-16T16:00:00+08:00",
            "environment": "myrobot_forward_jump_static",
            "target_band": "8.0 ~ 10.0 cm",
            "acceptance_criteria": {
                "A_clearance_ge_20cm": "Dual-wheel maximum mutual clearance >= 20.0 cm",
                "B_forward_displacement_ge_8cm": "Flight CoM forward displacement >= 8.0 cm (Target 8-10 cm)",
                "C_stationary_takeoff": "Prep ground slip < 2.0 cm, max base vel < 0.20 m/s, zero pre-rolling",
                "D_landing_recovery": "Post-landing recovery |V_y| <= 0.03 m/s, pitch <= 10°, roll <= 10°, yaw_drift <= 10°",
                "E_safety": "Zero abnormal resets, valid jump == True"
            }
        },
        "phase2_1_iter10": summary_p21_m10,
        "phase2_1_iter15": summary_p21_m15,
        "phase3_1_champion": summary_p31_champ
    }

    summary_json_path = os.path.join(output_dir, "phase3_1_evaluation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"\n📊 Multi-seed comparative summary saved to: {summary_json_path}")

    out_csv = os.path.join(output_dir, "phase3_1_telemetry.csv")
    out_mp4 = os.path.join(output_dir, "phase3_1_demonstration.mp4")
    render_video_and_telemetry(champ_dest, out_csv, out_mp4, seed=42)

    print("\n🎉 All Phase 3.1 deliverables successfully generated!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    else:
        run_dir = "/home/xy/IsaacTu_jump/legged_gym/logs/myrobot_forward_jump_static/Oct16_15-55-50_phase3_1_fwd_dist_curriculum_level1"
        ckpt_path = os.path.join(run_dir, "model_10.pt")
    main(ckpt_path)
