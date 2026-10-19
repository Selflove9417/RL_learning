#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Phase 3.1 Champion 20-Seed Robustness Evaluation Runner
# Evaluates Phase 3.1 Champion model across 20 deterministic seeds:
#   Original 5 seeds: [42, 100, 200, 300, 400]
#   Deterministic expansion 15 seeds: [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 50, 150, 250, 350, 450]
# Total = 20 deterministic, reproducible seeds.
# ==============================================================================

import os
import sys
import csv
import json
import hashlib
import subprocess
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
python_bin = "/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/bin/python"
eval_script = os.path.join(workspace_root, "tests", "eval_joint_success.py")

champion_model_path = os.path.join(workspace_root, "evaluation_assets", "phase3_1_forward_distance", "models", "model_champion.pt")
training_source_path = os.path.join(workspace_root, "legged_gym", "logs", "myrobot_forward_jump_static", "Oct16_15-55-50_phase3_1_fwd_dist_curriculum_level1", "model_15.pt")

output_dir = os.path.join(workspace_root, "evaluation_assets", "phase3_1_forward_distance", "20seed_robustness")
os.makedirs(output_dir, exist_ok=True)

SEEDS = [
    42, 100, 200, 300, 400,     # Original 5 seeds
    1, 2, 3, 5, 7, 11, 13, 17, 19, 23, # 10 prime seeds
    50, 150, 250, 350, 450      # 5 round seeds
]

def verify_checksums():
    with open(champion_model_path, "rb") as f:
        md5_champ = hashlib.md5(f.read()).hexdigest()
    with open(training_source_path, "rb") as f:
        md5_source = hashlib.md5(f.read()).hexdigest()
    print(f"Champion Checkpoint MD5 : {md5_champ} ({champion_model_path})")
    print(f"Training Source Checkpoint MD5: {md5_source} ({training_source_path})")
    assert md5_champ == md5_source, f"MD5 mismatch! {md5_champ} != {md5_source}"
    return md5_champ

def run_evaluation():
    md5_hash = verify_checksums()
    print(f"\n==============================================================================")
    print(f" Starting Phase 3.1 Champion 20-Seed Robustness Evaluation (20 Seeds)")
    print(f" Model Path: {champion_model_path}")
    print(f" MD5 Hash  : {md5_hash}")
    print(f" Seeds ({len(SEEDS)}): {SEEDS}")
    print(f"==============================================================================\n")

    results = []

    for idx, seed in enumerate(SEEDS, 1):
        out_json = os.path.join(output_dir, f"seed_{seed}.json")
        cmd = [
            python_bin, eval_script,
            "--model", champion_model_path,
            "--seed", str(seed),
            "--clip", "100.0",
            "--min_dist", "8.0",
            "--max_rec_vy", "0.03",
            "--out", out_json
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{workspace_root}/legged_gym:{workspace_root}/rsl_rl:" + env.get("PYTHONPATH", "")
        env["LD_LIBRARY_PATH"] = f"/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/lib:" + env.get("LD_LIBRARY_PATH", "")

        print(f"[{idx:2d}/{len(SEEDS)}] Running Seed {seed:3d}...", end=" ", flush=True)
        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if proc.returncode != 0:
            print(f"FAILED!\nError: {proc.stderr}")
            continue

        with open(out_json, "r") as f:
            res = json.load(f)

        status = "✅ PASS" if res["joint_success"] else f"❌ FAIL ({', '.join(res['failure_reasons'])})"
        print(f"{status} | Clr={res['max_clearance_cm']:5.2f}cm (L={res['max_clearance_L_cm']:5.2f}, R={res['max_clearance_R_cm']:5.2f}) | FwdDist={res['com_forward_dist_cm']:5.2f}cm | Roll_TD={res['touchdown_roll_deg']:+5.2f}° | RecVy={res['recovery_vy_mps']:+.3f}m/s")
        results.append(res)

    # Save detailed CSV
    csv_path = os.path.join(output_dir, "phase3_1_champion_20seed_telemetry.csv")
    fieldnames = [
        "seed", "joint_success", "valid_jump", "abnormal_resets",
        "c1_fwd_dist_ok", "c2_clearance_ok", "c3_prep_stationary_ok", "c4_recovery_ok", "c5_stability_ok",
        "failure_reasons",
        "max_clearance_cm", "max_clearance_L_cm", "max_clearance_R_cm",
        "com_forward_dist_cm", "takeoff_com_y_m", "touchdown_com_y_m", "flight_duration_s",
        "takeoff_com_vz_mps", "takeoff_com_vy_mps", "pre_push_vy_mps",
        "takeoff_pitch_deg", "takeoff_yaw_rate_radps", "takeoff_fn_L_N", "takeoff_fn_R_N",
        "max_prep_base_vel_mps", "max_prep_com_vel_mps", "max_prep_wheel_omega_radps", "prep_ground_slip_cm", "pre_jump_rolling_flag",
        "touchdown_pitch_deg", "touchdown_roll_deg", "touchdown_yaw_deg",
        "max_airborne_pitch_deg", "max_airborne_roll_deg", "yaw_drift_deg",
        "touchdown_vy_mps", "touchdown_plus_03s_vy_mps", "touchdown_plus_20s_vy_mps",
        "recovery_vy_mps", "recovery_vx_mps",
        "peak_hip_tau_nm", "peak_knee_tau_nm", "peak_wheel_tau_nm",
        "hip_tau_saturated", "knee_tau_saturated", "wheel_tau_saturated"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: ("; ".join(r[k]) if isinstance(r[k], list) else r[k]) for k in fieldnames}
            writer.writerow(row)

    print(f"\n📊 CSV telemetry written to: {csv_path}")

    # Compute Statistics across 20 seeds
    n = len(results)
    assert n == 20, f"Expected 20 results, got {n}"

    clrs = [r["max_clearance_cm"] for r in results]
    clrs_L = [r["max_clearance_L_cm"] for r in results]
    clrs_R = [r["max_clearance_R_cm"] for r in results]
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
    td_pitches = [r["touchdown_pitch_deg"] for r in results]
    td_rolls = [r["touchdown_roll_deg"] for r in results]
    td_yaws = [r["touchdown_yaw_deg"] for r in results]
    yaw_drifts = [r["yaw_drift_deg"] for r in results]
    air_rolls = [r["max_airborne_roll_deg"] for r in results]
    air_pitches = [r["max_airborne_pitch_deg"] for r in results]
    td_vys = [r["touchdown_vy_mps"] for r in results]
    td_03s_vys = [r["touchdown_plus_03s_vy_mps"] for r in results]
    td_20s_vys = [r["touchdown_plus_20s_vy_mps"] for r in results]
    rec_vys = [r["recovery_vy_mps"] for r in results]
    rec_abs_vys = [abs(r["recovery_vy_mps"]) for r in results]
    rec_vxs = [r["recovery_vx_mps"] for r in results]
    hip_taus = [r["peak_hip_tau_nm"] for r in results]
    knee_taus = [r["peak_knee_tau_nm"] for r in results]
    wheel_taus = [r["peak_wheel_tau_nm"] for r in results]

    valid_count = sum(1 for r in results if r["valid_jump"])
    clr_count = sum(1 for r in results if r["c2_clearance_ok"])
    fwd_ge_8_count = sum(1 for r in results if r["c1_fwd_dist_ok"])
    fwd_in_band_count = sum(1 for r in results if 8.0 <= r["com_forward_dist_cm"] <= 10.0)
    prep_count = sum(1 for r in results if r["c3_prep_stationary_ok"])
    rec_count = sum(1 for r in results if r["c4_recovery_ok"])
    stab_count = sum(1 for r in results if r["c5_stability_ok"])
    joint_count = sum(1 for r in results if r["joint_success"])
    abnormal_resets_total = sum(r["abnormal_resets"] for r in results)

    # Failure seed analysis
    failed_seeds = [r for r in results if not r["joint_success"]]
    clr_fail_seeds = [r["seed"] for r in results if not r["c2_clearance_ok"]]
    dist_fail_seeds = [r["seed"] for r in results if not r["c1_fwd_dist_ok"]]
    rec_fail_seeds = [r["seed"] for r in results if not r["c4_recovery_ok"]]

    stats = {
        "metadata": {
            "title": "Phase 3.1 Champion 20-Seed Robustness Evaluation",
            "model_path": champion_model_path,
            "model_md5": md5_hash,
            "training_source_path": training_source_path,
            "total_seeds": n,
            "seeds": SEEDS,
            "acceptance_criteria": {
                "clearance_ge_20cm": "Dual-wheel maximum mutual clearance >= 20.0 cm",
                "forward_displacement_ge_8cm": "Flight CoM forward displacement >= 8.0 cm",
                "forward_displacement_target_band": "8.0 ~ 10.0 cm",
                "prep_stationary": "Ground slip < 2.0 cm, base_vy < 0.20 m/s, com_vy < 0.20 m/s, wheel_w < 15 rad/s",
                "touchdown_attitude_recovery": "|pitch| <= 10°, |roll| <= 10°, yaw_drift <= 10°, |recovery_vy| <= 0.03 m/s",
                "safety": "Zero abnormal resets, valid jump == True"
            }
        },
        "pass_rates": {
            "valid_jumps": f"{valid_count}/{n} ({valid_count/n*100:.1f}%)",
            "clearance_ge_20cm": f"{clr_count}/{n} ({clr_count/n*100:.1f}%)",
            "forward_dist_ge_8cm": f"{fwd_ge_8_count}/{n} ({fwd_ge_8_count/n*100:.1f}%)",
            "forward_dist_in_8_to_10cm": f"{fwd_in_band_count}/{n} ({fwd_in_band_count/n*100:.1f}%)",
            "prep_stationary": f"{prep_count}/{n} ({prep_count/n*100:.1f}%)",
            "touchdown_recovery": f"{rec_count}/{n} ({rec_count/n*100:.1f}%)",
            "stability_and_safety": f"{stab_count}/{n} ({stab_count/n*100:.1f}%)",
            "joint_success": f"{joint_count}/{n} ({joint_count/n*100:.1f}%)",
            "abnormal_resets_total": abnormal_resets_total
        },
        "metrics_summary": {
            "clearance_cm": {
                "mean": round(float(np.mean(clrs)), 2),
                "std": round(float(np.std(clrs)), 2),
                "min": round(float(np.min(clrs)), 2),
                "max": round(float(np.max(clrs)), 2)
            },
            "clearance_L_cm": {
                "mean": round(float(np.mean(clrs_L)), 2),
                "std": round(float(np.std(clrs_L)), 2),
                "min": round(float(np.min(clrs_L)), 2),
                "max": round(float(np.max(clrs_L)), 2)
            },
            "clearance_R_cm": {
                "mean": round(float(np.mean(clrs_R)), 2),
                "std": round(float(np.std(clrs_R)), 2),
                "min": round(float(np.min(clrs_R)), 2),
                "max": round(float(np.max(clrs_R)), 2)
            },
            "forward_dist_cm": {
                "mean": round(float(np.mean(dists)), 2),
                "std": round(float(np.std(dists)), 2),
                "min": round(float(np.min(dists)), 2),
                "max": round(float(np.max(dists)), 2)
            },
            "flight_duration_s": {
                "mean": round(float(np.mean(flight_durs)), 3),
                "std": round(float(np.std(flight_durs)), 3),
                "min": round(float(np.min(flight_durs)), 3),
                "max": round(float(np.max(flight_durs)), 3)
            },
            "takeoff_vz_mps": {
                "mean": round(float(np.mean(vzs)), 3),
                "std": round(float(np.std(vzs)), 3),
                "min": round(float(np.min(vzs)), 3),
                "max": round(float(np.max(vzs)), 3)
            },
            "takeoff_vy_mps": {
                "mean": round(float(np.mean(vys)), 3),
                "std": round(float(np.std(vys)), 3),
                "min": round(float(np.min(vys)), 3),
                "max": round(float(np.max(vys)), 3)
            },
            "takeoff_pitch_deg": {
                "mean": round(float(np.mean(to_pitches)), 2),
                "std": round(float(np.std(to_pitches)), 2),
                "min": round(float(np.min(to_pitches)), 2),
                "max": round(float(np.max(to_pitches)), 2)
            },
            "takeoff_yaw_rate_radps": {
                "mean": round(float(np.mean(to_yaw_rates)), 3),
                "std": round(float(np.std(to_yaw_rates)), 3),
                "min": round(float(np.min(to_yaw_rates)), 3),
                "max": round(float(np.max(to_yaw_rates)), 3)
            },
            "prep_ground_slip_cm": {
                "mean": round(float(np.mean(slips)), 2),
                "std": round(float(np.std(slips)), 2),
                "min": round(float(np.min(slips)), 2),
                "max": round(float(np.max(slips)), 2)
            },
            "max_prep_base_vel_mps": {
                "mean": round(float(np.mean(prep_vys)), 3),
                "std": round(float(np.std(prep_vys)), 3),
                "min": round(float(np.min(prep_vys)), 3),
                "max": round(float(np.max(prep_vys)), 3)
            },
            "touchdown_pitch_deg": {
                "mean": round(float(np.mean(td_pitches)), 2),
                "std": round(float(np.std(td_pitches)), 2),
                "min": round(float(np.min(td_pitches)), 2),
                "max": round(float(np.max(td_pitches)), 2)
            },
            "touchdown_roll_deg": {
                "mean": round(float(np.mean(td_rolls)), 2),
                "std": round(float(np.std(td_rolls)), 2),
                "min": round(float(np.min(td_rolls)), 2),
                "max": round(float(np.max(td_rolls)), 2)
            },
            "yaw_drift_deg": {
                "mean": round(float(np.mean(yaw_drifts)), 2),
                "std": round(float(np.std(yaw_drifts)), 2),
                "min": round(float(np.min(yaw_drifts)), 2),
                "max": round(float(np.max(yaw_drifts)), 2)
            },
            "touchdown_vy_mps": {
                "mean": round(float(np.mean(td_vys)), 3),
                "std": round(float(np.std(td_vys)), 3),
                "min": round(float(np.min(td_vys)), 3),
                "max": round(float(np.max(td_vys)), 3)
            },
            "touchdown_plus_03s_vy_mps": {
                "mean": round(float(np.mean(td_03s_vys)), 3),
                "std": round(float(np.std(td_03s_vys)), 3),
                "min": round(float(np.min(td_03s_vys)), 3),
                "max": round(float(np.max(td_03s_vys)), 3)
            },
            "touchdown_plus_20s_vy_mps": {
                "mean": round(float(np.mean(td_20s_vys)), 3),
                "std": round(float(np.std(td_20s_vys)), 3),
                "min": round(float(np.min(td_20s_vys)), 3),
                "max": round(float(np.max(td_20s_vys)), 3)
            },
            "recovery_vy_mps": {
                "mean": round(float(np.mean(rec_vys)), 3),
                "std": round(float(np.std(rec_vys)), 3),
                "min": round(float(np.min(rec_vys)), 3),
                "max": round(float(np.max(rec_vys)), 3)
            },
            "recovery_abs_vy_mps": {
                "mean": round(float(np.mean(rec_abs_vys)), 3),
                "std": round(float(np.std(rec_abs_vys)), 3),
                "min": round(float(np.min(rec_abs_vys)), 3),
                "max": round(float(np.max(rec_abs_vys)), 3)
            },
            "recovery_vx_mps": {
                "mean": round(float(np.mean(rec_vxs)), 3),
                "std": round(float(np.std(rec_vxs)), 3),
                "min": round(float(np.min(rec_vxs)), 3),
                "max": round(float(np.max(rec_vxs)), 3)
            },
            "peak_hip_tau_nm": {
                "max": round(float(np.max(hip_taus)), 2),
                "mean": round(float(np.mean(hip_taus)), 2)
            },
            "peak_knee_tau_nm": {
                "max": round(float(np.max(knee_taus)), 2),
                "mean": round(float(np.mean(knee_taus)), 2)
            },
            "peak_wheel_tau_nm": {
                "max": round(float(np.max(wheel_taus)), 2),
                "mean": round(float(np.mean(wheel_taus)), 2)
            }
        },
        "failure_analysis": {
            "failed_seed_count": len(failed_seeds),
            "clearance_fail_seeds": clr_fail_seeds,
            "forward_dist_fail_seeds": dist_fail_seeds,
            "recovery_attitude_fail_seeds": rec_fail_seeds,
            "failed_seeds_details": [
                {
                    "seed": r["seed"],
                    "failure_reasons": r["failure_reasons"],
                    "clearance_cm": r["max_clearance_cm"],
                    "forward_dist_cm": r["com_forward_dist_cm"],
                    "touchdown_roll_deg": r["touchdown_roll_deg"],
                    "touchdown_pitch_deg": r["touchdown_pitch_deg"],
                    "recovery_vy_mps": r["recovery_vy_mps"],
                    "prep_slip_cm": r["prep_ground_slip_cm"]
                }
                for r in failed_seeds
            ]
        },
        "all_seeds_results": results
    }

    json_path = os.path.join(output_dir, "phase3_1_champion_20seed_summary.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n📈 JSON summary written to: {json_path}")
    print(f"\n==============================================================================")
    print(f" Summary of 20-Seed Robustness:")
    print(f"   Joint Task Success Rate : {joint_count}/{n} ({joint_count/n*100:.1f}%)")
    print(f"   Clearance (>= 20cm) Rate: {clr_count}/{n} ({clr_count/n*100:.1f}%) | Mean: {np.mean(clrs):.2f} ± {np.std(clrs):.2f} cm (Min: {np.min(clrs):.2f})")
    print(f"   Forward Dist (>= 8cm)   : {fwd_ge_8_count}/{n} ({fwd_ge_8_count/n*100:.1f}%) | Mean: {np.mean(dists):.2f} ± {np.std(dists):.2f} cm")
    print(f"   Forward Dist in [8, 10cm: {fwd_in_band_count}/{n} ({fwd_in_band_count/n*100:.1f}%)")
    print(f"   Touchdown & Recovery Ok : {rec_count}/{n} ({rec_count/n*100:.1f}%)")
    print(f"   Abnormal Resets         : {abnormal_resets_total}")
    print(f"==============================================================================\n")

if __name__ == "__main__":
    run_evaluation()
