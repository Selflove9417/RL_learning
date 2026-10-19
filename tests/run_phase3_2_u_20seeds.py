#!/usr/bin/env python3
# ==============================================================================
# Phase 3.2-U: 20-Seed Batch Runner for Dual-Policy and Minimal Correction
# Seeds: [42, 100, 200, 300, 400, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 50, 150, 250, 350, 450]
# Evaluates dual-policy handoff + aerial descent landing preparation hip trim (+0.10)
# ==============================================================================

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd

SEEDS_20 = [42, 100, 200, 300, 400, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 50, 150, 250, 350, 450]
OUTPUT_DIR = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_2_u_dual_policy"
ACTIVATE_SCRIPT = "/home/xy/IsaacTu_jump/activate_jump.sh"
EVAL_SCRIPT = "/home/xy/IsaacTu_jump/tests/eval_single_corrected_seed.py"

def run_seed(seed, hip_bias=0.10, phase_thresh=0.22):
    out_json = os.path.join(OUTPUT_DIR, f"seed_{seed}.json")
    if os.path.exists(out_json):
        try:
            with open(out_json, "r") as f:
                data = json.load(f)
            if "joint_passed_strict" in data:
                print(f"[Seed {seed:3d}] Found cached result.")
                return data
        except Exception:
            pass

    cmd = f"source {ACTIVATE_SCRIPT} && python {EVAL_SCRIPT} --seed {seed} --hip_bias {hip_bias} --phase_threshold {phase_thresh} --out {out_json}"
    res = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[Seed {seed:3d}] ERROR (code {res.returncode})")
        print(res.stderr[-500:] if res.stderr else "No stderr")
        return None

    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            return json.load(f)
    else:
        # Fallback parse from stdout
        out = res.stdout
        start = out.find("===JSON_START===")
        end = out.find("===JSON_END===")
        if start != -1 and end != -1:
            data = json.loads(out[start + len("===JSON_START==="):end].strip())
            with open(out_json, "w") as f:
                json.dump(data, f, indent=2)
            return data
    return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    print(f"Starting 20-Seed Dual-Policy Evaluation with Minimal Intervention (hip_bias=+0.10)...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    for idx, seed in enumerate(SEEDS_20):
        print(f"[{idx+1:02d}/20] Running Seed {seed}...", end="", flush=True)
        res = run_seed(seed)
        if res is not None:
            results.append(res)
            strict_str = "PASS" if res.get("joint_passed_strict") else "FAIL"
            print(f" Done. Fwd: {res['fwd_dist_cm']:5.2f} cm | Clr: {res['max_clearance_cm']:5.2f} cm | r_raw: {res['touchdown_r_raw_deg']:5.2f}° | p_raw: {res['touchdown_p_raw_deg']:5.2f}° | Strict: {strict_str}")
        else:
            print(" FAILED execution!")

    # Convert to DataFrame
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "phase3_2_u_20seeds_telemetry.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nTelemetry saved to: {csv_path}")

    # Summary Statistics
    n = len(df)
    strict_pass_count = int(df["joint_passed_strict"].sum())
    fwd_pass_count = int(df["passed_fwd_8cm"].sum())
    clr_pass_count = int(df["passed_clr_20cm"].sum())
    r_raw_pass_count = int(df["passed_r_raw_10deg"].sum())
    p_raw_pass_count = int(df["passed_p_raw_10deg"].sum())
    rec_vy_pass_count = int(df["passed_rec_vy_003"].sum())

    summary = {
        "num_seeds": n,
        "strict_joint_pass_rate": f"{strict_pass_count}/{n} ({strict_pass_count/n*100:.1f}%)",
        "strict_joint_passed_count": strict_pass_count,
        "fwd_pass_rate": f"{fwd_pass_count}/{n} ({fwd_pass_count/n*100:.1f}%)",
        "clr_pass_rate": f"{clr_pass_count}/{n} ({clr_pass_count/n*100:.1f}%)",
        "touchdown_r_raw_pass_rate_10deg": f"{r_raw_pass_count}/{n} ({r_raw_pass_count/n*100:.1f}%)",
        "touchdown_p_raw_pass_rate_10deg": f"{p_raw_pass_count}/{n} ({p_raw_pass_count/n*100:.1f}%)",
        "rec_vy_pass_rate": f"{rec_vy_pass_count}/{n} ({rec_vy_pass_count/n*100:.1f}%)",
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
        "touchdown_r_raw_deg": {
            "mean": float(df["touchdown_r_raw_deg"].mean()),
            "std": float(df["touchdown_r_raw_deg"].std()),
            "min": float(df["touchdown_r_raw_deg"].min()),
            "max": float(df["touchdown_r_raw_deg"].max())
        },
        "touchdown_p_raw_deg": {
            "mean": float(df["touchdown_p_raw_deg"].mean()),
            "std": float(df["touchdown_p_raw_deg"].std()),
            "min": float(df["touchdown_p_raw_deg"].min()),
            "max": float(df["touchdown_p_raw_deg"].max())
        },
        "post_rec_vy_mps": {
            "mean": float(df["post_rec_vy"].mean()),
            "std": float(df["post_rec_vy"].std()),
            "min": float(df["post_rec_vy"].min()),
            "max": float(df["post_rec_vy"].max())
        },
        "settle_time_s": {
            "mean": float(df["settle_time_s"].mean()),
            "std": float(df["settle_time_s"].std()),
            "min": float(df["settle_time_s"].min()),
            "max": float(df["settle_time_s"].max())
        },
        "seeds_detail": results
    }

    summary_path = os.path.join(OUTPUT_DIR, "phase3_2_u_20seeds_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    print("\n==================== 20-SEED REGRESSION SUMMARY ====================")
    print(f"Strict Joint Pass Rate (All 6 criteria): {summary['strict_joint_pass_rate']}")
    print(f"Forward Distance (>= 8 cm)              : {summary['fwd_pass_rate']} | Mean: {summary['fwd_dist_cm']['mean']:.2f} ± {summary['fwd_dist_cm']['std']:.2f} cm (Range: [{summary['fwd_dist_cm']['min']:.2f}, {summary['fwd_dist_cm']['max']:.2f}])")
    print(f"Vertical Clearance (>= 20 cm)           : {summary['clr_pass_rate']} | Mean: {summary['max_clearance_cm']['mean']:.2f} ± {summary['max_clearance_cm']['std']:.2f} cm (Range: [{summary['max_clearance_cm']['min']:.2f}, {summary['max_clearance_cm']['max']:.2f}])")
    print(f"Touchdown r_raw (<= 10 deg)             : {summary['touchdown_r_raw_pass_rate_10deg']} | Mean: {summary['touchdown_r_raw_deg']['mean']:.2f} ± {summary['touchdown_r_raw_deg']['std']:.2f}° (Range: [{summary['touchdown_r_raw_deg']['min']:.2f}, {summary['touchdown_r_raw_deg']['max']:.2f}])")
    print(f"Touchdown p_raw (<= 10 deg)             : {summary['touchdown_p_raw_pass_rate_10deg']} | Mean: {summary['touchdown_p_raw_deg']['mean']:.2f} ± {summary['touchdown_p_raw_deg']['std']:.2f}° (Range: [{summary['touchdown_p_raw_deg']['min']:.2f}, {summary['touchdown_p_raw_deg']['max']:.2f}])")
    print(f"Post-Recovery Vy (<= 0.03 m/s)          : {summary['rec_vy_pass_rate']} | Mean: {summary['post_rec_vy_mps']['mean']:.4f} ± {summary['post_rec_vy_mps']['std']:.4f} m/s")
    print(f"Balance Settling Time                   : Mean: {summary['settle_time_s']['mean']:.2f} ± {summary['settle_time_s']['std']:.2f} s")
    print("====================================================================")

if __name__ == "__main__":
    main()
