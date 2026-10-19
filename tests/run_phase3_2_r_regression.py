#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Phase 3.2-R Regression Batch Runner
# Manages multi-seed, multi-mode execution with subprocess isolation
# ==============================================================================

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd

SEEDS_20 = [42, 100, 200, 300, 400, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 50, 150, 250, 350, 450]
SEEDS_UNSEEN = [555, 666, 777, 888, 999]

OUTPUT_DIR = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_2_r_regression"
ACTIVATE_SCRIPT = "/home/xy/IsaacTu_jump/activate_jump.sh"
EVAL_SCRIPT = "/home/xy/IsaacTu_jump/tests/eval_phase3_2_r_single_seed.py"


def run_single_eval(seed, mode, out_json):
    if os.path.exists(out_json):
        try:
            with open(out_json, "r") as f:
                data = json.load(f)
            if "joint_success_level3_1" in data:
                return data
        except Exception:
            pass

    cmd = f"source {ACTIVATE_SCRIPT} && python {EVAL_SCRIPT} --seed {seed} --mode {mode} --out {out_json}"
    res = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"Error executing seed {seed} mode {mode}: returncode {res.returncode}")
        print(res.stderr[-500:] if res.stderr else "No stderr")
        return None

    with open(out_json, "r") as f:
        return json.load(f)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    key_traj_dir = os.path.join(OUTPUT_DIR, "key_trajectories")
    os.makedirs(key_traj_dir, exist_ok=True)

    modes = [
        "Mode_A_control",
        "Mode_B_isolated_lin_vel",
        "Mode_D_counterfactual",
        "Mode_E_physical_consistent"
    ]

    all_mode_results = {}

    for mode in modes:
        mode_dir = os.path.join(OUTPUT_DIR, mode)
        os.makedirs(mode_dir, exist_ok=True)
        print(f"\n==================================================")
        print(f" Running evaluation for {mode} (20 seeds)")
        print(f"==================================================")

        mode_data = []
        for seed in SEEDS_20:
            out_json = os.path.join(mode_dir, f"seed_{seed}.json")
            data = run_single_eval(seed, mode, out_json)
            if data is not None:
                mode_data.append(data)
                print(f"  Seed {seed:3d}: fwd={data['com_forward_dist_cm']:+6.2f}cm, clr={data['max_clearance_cm']:5.2f}cm, roll={data['touchdown_roll_deg']:+5.1f}deg, rec_vy={data['post_recovery_vy_mps']:+6.4f}, Pass={data['joint_success_level3_1']}")

        all_mode_results[mode] = mode_data

        # Export mode CSV
        rows = []
        for d in mode_data:
            rows.append({
                "seed": d["seed"],
                "mode": d["mode"],
                "step0_base_vy": d["step0_base_vy_mps"],
                "step0_com_vy": d["step0_com_vy_mps"],
                "step5_base_vy": d["step5_base_vy_mps"],
                "step5_com_vy": d["step5_com_vy_mps"],
                "takeoff_base_vy": d["takeoff_base_vy_mps"],
                "takeoff_com_vy": d["takeoff_com_vy_mps"],
                "takeoff_base_vz": d["takeoff_base_vz_mps"],
                "takeoff_com_vz": d["takeoff_com_vz_mps"],
                "pushoff_delta_com_vy": d["pushoff_delta_com_vy_mps"],
                "flight_duration_s": d["flight_duration_s"],
                "com_forward_dist_cm": d["com_forward_dist_cm"],
                "max_clearance_cm": d["max_clearance_cm"],
                "touchdown_pitch_deg": d["touchdown_pitch_deg"],
                "touchdown_roll_deg": d["touchdown_roll_deg"],
                "yaw_drift_deg": d["yaw_drift_deg"],
                "post_recovery_vy_mps": d["post_recovery_vy_mps"],
                "prep_ground_slip_y_m": d["prep_ground_slip_y_m"],
                "max_prep_base_vel_mps": d["max_prep_base_vel_mps"],
                "max_prep_com_vel_mps": d["max_prep_com_vel_mps"],
                "max_prep_wheel_omega_radps": d["max_prep_wheel_omega_radps"],
                "peak_hip_tau_nm": d["peak_hip_tau_nm"],
                "peak_knee_tau_nm": d["peak_knee_tau_nm"],
                "peak_wheel_tau_nm": d["peak_wheel_tau_nm"],
                "abnormal_resets": d["abnormal_resets"],
                "jump_class": d["jump_class"],
                "c1_fwd_dist_ok": d["c1_fwd_dist_ok"],
                "c2_clearance_ok": d["c2_clearance_ok"],
                "c3_touchdown_attitude_ok": d["c3_touchdown_attitude_ok"],
                "c4_recovery_ok": d["c4_recovery_ok"],
                "c5_prep_stationary_ok": d["c5_prep_stationary_ok"],
                "c6_no_abnormal_reset": d["c6_no_abnormal_reset"],
                "joint_success_level3_1": d["joint_success_level3_1"],
                "joint_success_relaxed": d["joint_success_relaxed"]
            })
        df = pd.DataFrame(rows)
        csv_path = os.path.join(OUTPUT_DIR, f"{mode}_telemetry.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

    # Evaluate extra unseen seeds for Mode E
    print(f"\n==================================================")
    print(f" Running evaluation for Mode E on UNSEEN seeds (5 seeds)")
    print(f"==================================================")
    unseen_dir = os.path.join(OUTPUT_DIR, "Mode_E_unseen")
    os.makedirs(unseen_dir, exist_ok=True)
    unseen_data = []
    for seed in SEEDS_UNSEEN:
        out_json = os.path.join(unseen_dir, f"seed_{seed}.json")
        data = run_single_eval(seed, "Mode_E_physical_consistent", out_json)
        if data is not None:
            unseen_data.append(data)
            print(f"  Unseen Seed {seed:3d}: fwd={data['com_forward_dist_cm']:+6.2f}cm, clr={data['max_clearance_cm']:5.2f}cm, roll={data['touchdown_roll_deg']:+5.1f}deg, rec_vy={data['post_recovery_vy_mps']:+6.4f}, Pass={data['joint_success_level3_1']}")

    unseen_rows = []
    for d in unseen_data:
        unseen_rows.append({
            "seed": d["seed"],
            "com_forward_dist_cm": d["com_forward_dist_cm"],
            "max_clearance_cm": d["max_clearance_cm"],
            "touchdown_pitch_deg": d["touchdown_pitch_deg"],
            "touchdown_roll_deg": d["touchdown_roll_deg"],
            "yaw_drift_deg": d["yaw_drift_deg"],
            "post_recovery_vy_mps": d["post_recovery_vy_mps"],
            "jump_class": d["jump_class"],
            "joint_success_level3_1": d["joint_success_level3_1"]
        })
    df_unseen = pd.DataFrame(unseen_rows)
    df_unseen.to_csv(os.path.join(OUTPUT_DIR, "unseen_seeds_telemetry.csv"), index=False)

    # Compute comparative summary table across modes
    summary = {}
    for mode in modes:
        data = all_mode_results[mode]
        n = len(data)
        if n == 0:
            continue

        fwds = [d["com_forward_dist_cm"] for d in data]
        clrs = [d["max_clearance_cm"] for d in data]
        rolls = [d["touchdown_roll_deg"] for d in data]
        pitches = [d["touchdown_pitch_deg"] for d in data]
        rec_vys = [d["post_recovery_vy_mps"] for d in data]
        step0_com_vys = [d["step0_com_vy_mps"] for d in data]
        step5_com_vys = [d["step5_com_vy_mps"] for d in data]
        to_com_vys = [d["takeoff_com_vy_mps"] for d in data]
        to_com_vzs = [d["takeoff_com_vz_mps"] for d in data]
        slips = [d["prep_ground_slip_y_m"] * 100.0 for d in data]

        c1_rates = sum(d["c1_fwd_dist_ok"] for d in data) / n * 100.0
        c2_rates = sum(d["c2_clearance_ok"] for d in data) / n * 100.0
        c3_rates = sum(d["c3_touchdown_attitude_ok"] for d in data) / n * 100.0
        c4_rates = sum(d["c4_recovery_ok"] for d in data) / n * 100.0
        c5_rates = sum(d["c5_prep_stationary_ok"] for d in data) / n * 100.0
        c6_rates = sum(d["c6_no_abnormal_reset"] for d in data) / n * 100.0
        joint_strict_rates = sum(d["joint_success_level3_1"] for d in data) / n * 100.0
        joint_relaxed_rates = sum(d["joint_success_relaxed"] for d in data) / n * 100.0

        n_class_a = sum(1 for d in data if d["jump_class"] == "Class_A_normal_forward")
        n_class_b = sum(1 for d in data if d["jump_class"] == "Class_B_short_jump")
        n_class_c = sum(1 for d in data if d["jump_class"] == "Class_C_backward_jump")

        summary[mode] = {
            "n_seeds": n,
            "step0_com_vy_mean": float(np.mean(step0_com_vys)),
            "step0_com_vy_std": float(np.std(step0_com_vys)),
            "step5_com_vy_mean": float(np.mean(step5_com_vys)),
            "step5_com_vy_std": float(np.std(step5_com_vys)),
            "takeoff_com_vy_mean": float(np.mean(to_com_vys)),
            "takeoff_com_vy_std": float(np.std(to_com_vys)),
            "takeoff_com_vz_mean": float(np.mean(to_com_vzs)),
            "takeoff_com_vz_std": float(np.std(to_com_vzs)),
            "fwd_dist_mean_cm": float(np.mean(fwds)),
            "fwd_dist_std_cm": float(np.std(fwds)),
            "clearance_mean_cm": float(np.mean(clrs)),
            "clearance_std_cm": float(np.std(clrs)),
            "touchdown_roll_mean_deg": float(np.mean(rolls)),
            "touchdown_roll_std_deg": float(np.std(rolls)),
            "touchdown_pitch_mean_deg": float(np.mean(pitches)),
            "touchdown_pitch_std_deg": float(np.std(pitches)),
            "recovery_vy_mean_mps": float(np.mean(rec_vys)),
            "recovery_vy_std_mps": float(np.std(rec_vys)),
            "prep_slip_mean_cm": float(np.mean(slips)),
            "c1_fwd_dist_rate_pct": c1_rates,
            "c2_clearance_rate_pct": c2_rates,
            "c3_touchdown_attitude_rate_pct": c3_rates,
            "c4_recovery_rate_pct": c4_rates,
            "c5_prep_stationary_rate_pct": c5_rates,
            "c6_no_abnormal_reset_pct": c6_rates,
            "joint_pass_strict_pct": joint_strict_rates,
            "joint_pass_relaxed_pct": joint_relaxed_rates,
            "class_A_count": n_class_a,
            "class_B_count": n_class_b,
            "class_C_count": n_class_c
        }

    summary_json_path = os.path.join(OUTPUT_DIR, "criteria_breakdown_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_json_path}")

    # Export key trajectories (Seeds 42, 17, 23, 2, 11) for comparison
    key_seeds = [42, 17, 23, 2, 11]
    for ks in key_seeds:
        for mode in modes:
            for d in all_mode_results[mode]:
                if d["seed"] == ks:
                    traj_path = os.path.join(key_traj_dir, f"{mode}_seed_{ks}_traj.csv")
                    pd.DataFrame(d["trajectory"]).to_csv(traj_path, index=False)

    print("\nPhase 3.2-R Regression complete!")


if __name__ == "__main__":
    main()
