#!/usr/bin/env python3
# ==============================================================================
# Phase 3.2-V: 20 Unseen Seeds Paired Batch Runner
# Mode A (Unseen): Dual-policy stable entry, without descent hip compensation (hip_bias = 0.0)
# Mode B (Unseen): Dual-policy stable entry, with Phase 3.2-U hip compensation (hip_bias = 0.10)
# Fixed Seeds: [1001..1010, 2001..2010]
# ==============================================================================

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd

SEEDS_UNSEEN_20 = [
    1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
    2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010
]

BASE_DIR = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_2_v_candidate"
ACTIVATE_SCRIPT = "/home/xy/IsaacTu_jump/activate_jump.sh"
EVAL_SCRIPT = "/home/xy/IsaacTu_jump/tests/eval_phase3_2_v_seed.py"

def run_single(seed, hip_bias, mode_name):
    out_dir = os.path.join(BASE_DIR, mode_name)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"seed_{seed}.json")

    if os.path.exists(out_json):
        try:
            with open(out_json, "r") as f:
                data = json.load(f)
            if "joint_passed_strict" in data:
                return data
        except Exception:
            pass

    cmd = f"source {ACTIVATE_SCRIPT} && python {EVAL_SCRIPT} --seed {seed} --hip_bias {hip_bias} --out {out_json}"
    res = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if res.returncode != 0:
        print(f"[{mode_name} | Seed {seed}] ERROR: {res.stderr[-300:]}")
        return None

    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            return json.load(f)
    return None

def summarize_mode(results, mode_name):
    df = pd.DataFrame(results)
    csv_path = os.path.join(BASE_DIR, f"{mode_name}_telemetry.csv")
    df.to_csv(csv_path, index=False)

    n = len(df)
    strict_count = int(df["joint_passed_strict"].sum())
    fwd_count = int(df["passed_fwd_8cm"].sum())
    clr_count = int(df["passed_clr_20cm"].sum())
    pitch_count = int(df["passed_pitch_10deg"].sum())
    roll_count = int(df["passed_roll_10deg"].sum())
    rec_count = int(df["passed_rec_vy_003"].sum())
    runup_count = int(df["passed_pre_runup"].sum())
    reset_count = int(df["passed_no_reset"].sum())

    summary = {
        "mode": mode_name,
        "num_seeds": n,
        "strict_joint_pass_rate": f"{strict_count}/{n} ({strict_count/n*100:.1f}%)",
        "strict_passed_count": strict_count,
        "passed_pre_runup_rate": f"{runup_count}/{n} ({runup_count/n*100:.1f}%)",
        "passed_fwd_8cm_rate": f"{fwd_count}/{n} ({fwd_count/n*100:.1f}%)",
        "passed_clr_20cm_rate": f"{clr_count}/{n} ({clr_count/n*100:.1f}%)",
        "passed_pitch_10deg_rate": f"{pitch_count}/{n} ({pitch_count/n*100:.1f}%)",
        "passed_roll_10deg_rate": f"{roll_count}/{n} ({roll_count/n*100:.1f}%)",
        "passed_rec_vy_rate": f"{rec_count}/{n} ({rec_count/n*100:.1f}%)",
        "passed_no_reset_rate": f"{reset_count}/{n} ({reset_count/n*100:.1f}%)",
        "fwd_dist_cm": {
            "mean": float(df["fwd_dist_cm"].mean()),
            "std": float(df["fwd_dist_cm"].std()),
            "min": float(df["fwd_dist_cm"].min()),
            "max": float(df["fwd_dist_cm"].max())
        },
        "max_clearance_cm": {
            "mean": float(df["max_clearance_cm"].mean()),
            "std": float(df["max_clearance_cm"].std()),
            "min": float(df["max_clearance_cm"].min()),
            "max": float(df["max_clearance_cm"].max())
        },
        "touchdown_pitch_deg": {
            "mean": float(df["touchdown_pitch_deg"].mean()),
            "std": float(df["touchdown_pitch_deg"].std()),
            "min": float(df["touchdown_pitch_deg"].min()),
            "max": float(df["touchdown_pitch_deg"].max())
        },
        "touchdown_roll_deg": {
            "mean": float(df["touchdown_roll_deg"].mean()),
            "std": float(df["touchdown_roll_deg"].std()),
            "min": float(df["touchdown_roll_deg"].min()),
            "max": float(df["touchdown_roll_deg"].max())
        },
        "post_rec_vy_last10": {
            "mean": float(df["post_rec_vy_last10"].mean()),
            "std": float(df["post_rec_vy_last10"].std()),
            "min": float(df["post_rec_vy_last10"].min()),
            "max": float(df["post_rec_vy_last10"].max())
        },
        "settle_time_s": {
            "mean": float(df["settle_time_s"].mean()),
            "std": float(df["settle_time_s"].std()),
            "min": float(df["settle_time_s"].min()),
            "max": float(df["settle_time_s"].max())
        },
        "dwell_com_disp_cm": {
            "mean": float(df["dwell_com_disp_cm"].mean()),
            "std": float(df["dwell_com_disp_cm"].std()),
            "min": float(df["dwell_com_disp_cm"].min()),
            "max": float(df["dwell_com_disp_cm"].max())
        },
        "dwell_wheel_disp_cm": {
            "mean": float(df["dwell_wheel_disp_cm"].mean()),
            "std": float(df["dwell_wheel_disp_cm"].std()),
            "min": float(df["dwell_wheel_disp_cm"].min()),
            "max": float(df["dwell_wheel_disp_cm"].max())
        },
        "prep_wheel_slip_cm": {
            "mean": float(df["prep_wheel_slip_cm"].mean()),
            "std": float(df["prep_wheel_slip_cm"].std()),
            "min": float(df["prep_wheel_slip_cm"].min()),
            "max": float(df["prep_wheel_slip_cm"].max())
        },
        "results_raw": results
    }

    json_path = os.path.join(BASE_DIR, f"{mode_name}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary, df

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    print("================================================================================")
    print(" Phase 3.2-V: Paired Evaluation on 20 Brand-New Unseen Seeds")
    print(f" Seeds: {SEEDS_UNSEEN_20}")
    print("================================================================================\n")

    # 1. Run Mode A (Unseen, without compensation: hip_bias = 0.0)
    print(">>> Running Mode A (Unseen, WITHOUT hip compensation)...")
    results_A = []
    for idx, seed in enumerate(SEEDS_UNSEEN_20):
        print(f"  [Mode A {idx+1:02d}/20] Seed {seed}...", end="", flush=True)
        res = run_single(seed, hip_bias=0.0, mode_name="Mode_A_unseen")
        if res:
            results_A.append(res)
            st = "PASS" if res.get("joint_passed_strict") else "FAIL"
            print(f" Done. Fwd: {res['fwd_dist_cm']:5.2f} cm | Clr: {res['max_clearance_cm']:5.2f} cm | Pitch: {res['touchdown_pitch_deg']:5.2f}° | Roll: {res['touchdown_roll_deg']:5.2f}° | Strict: {st}")
        else:
            print(" ERROR")

    summary_A, df_A = summarize_mode(results_A, "Mode_A_unseen")

    # 2. Run Mode B (Unseen, WITH hip compensation: hip_bias = 0.10)
    print("\n>>> Running Mode B (Unseen, WITH Phase 3.2-U hip compensation +0.10)...")
    results_B = []
    for idx, seed in enumerate(SEEDS_UNSEEN_20):
        print(f"  [Mode B {idx+1:02d}/20] Seed {seed}...", end="", flush=True)
        res = run_single(seed, hip_bias=0.10, mode_name="Mode_B_unseen")
        if res:
            results_B.append(res)
            st = "PASS" if res.get("joint_passed_strict") else "FAIL"
            print(f" Done. Fwd: {res['fwd_dist_cm']:5.2f} cm | Clr: {res['max_clearance_cm']:5.2f} cm | Pitch: {res['touchdown_pitch_deg']:5.2f}° | Roll: {res['touchdown_roll_deg']:5.2f}° | Strict: {st}")
        else:
            print(" ERROR")

    summary_B, df_B = summarize_mode(results_B, "Mode_B_unseen")

    # 3. Print Paired Comparison
    print("\n========================= PAIRED COMPARISON RESULTS =========================")
    print(f"{'Metric':<32} | {'Mode A (No Comp)':<22} | {'Mode B (With Comp +0.10)':<22} | {'Delta (B - A)':<15}")
    print("-" * 100)
    print(f"{'Strict Joint Pass Rate':<32} | {summary_A['strict_joint_pass_rate']:<22} | {summary_B['strict_joint_pass_rate']:<22} | {summary_B['strict_passed_count'] - summary_A['strict_passed_count']:+d} seeds")
    print(f"{'Forward Distance (cm)':<32} | {summary_A['fwd_dist_cm']['mean']:5.2f} ± {summary_A['fwd_dist_cm']['std']:4.2f}          | {summary_B['fwd_dist_cm']['mean']:5.2f} ± {summary_B['fwd_dist_cm']['std']:4.2f}          | {summary_B['fwd_dist_cm']['mean'] - summary_A['fwd_dist_cm']['mean']:+5.2f} cm")
    print(f"{'Max Clearance (cm)':<32} | {summary_A['max_clearance_cm']['mean']:5.2f} ± {summary_A['max_clearance_cm']['std']:4.2f}          | {summary_B['max_clearance_cm']['mean']:5.2f} ± {summary_B['max_clearance_cm']['std']:4.2f}          | {summary_B['max_clearance_cm']['mean'] - summary_A['max_clearance_cm']['mean']:+5.2f} cm")
    print(f"{'Touchdown Pitch (deg, r_raw)':<32} | {summary_A['touchdown_pitch_deg']['mean']:5.2f} ± {summary_A['touchdown_pitch_deg']['std']:4.2f}          | {summary_B['touchdown_pitch_deg']['mean']:5.2f} ± {summary_B['touchdown_pitch_deg']['std']:4.2f}          | {summary_B['touchdown_pitch_deg']['mean'] - summary_A['touchdown_pitch_deg']['mean']:+5.2f}°")
    print(f"{'Touchdown Roll (deg, p_raw)':<32} | {summary_A['touchdown_roll_deg']['mean']:5.2f} ± {summary_A['touchdown_roll_deg']['std']:4.2f}          | {summary_B['touchdown_roll_deg']['mean']:5.2f} ± {summary_B['touchdown_roll_deg']['std']:4.2f}          | {summary_B['touchdown_roll_deg']['mean'] - summary_A['touchdown_roll_deg']['mean']:+5.2f}°")
    print(f"{'Recovery Vy (m/s)':<32} | {summary_A['post_rec_vy_last10']['mean']:6.4f} ± {summary_A['post_rec_vy_last10']['std']:6.4f}      | {summary_B['post_rec_vy_last10']['mean']:6.4f} ± {summary_B['post_rec_vy_last10']['std']:6.4f}      | {summary_B['post_rec_vy_last10']['mean'] - summary_A['post_rec_vy_last10']['mean']:+6.4f} m/s")
    print("=============================================================================")

if __name__ == "__main__":
    main()
