#!/usr/bin/env python3
# ==============================================================================
# Phase 4.8-Pilot-A: Paired 5-Dev-Seed Physical Evaluation
# Compares Model Champion vs. Model Pilot A (Iter 1)
# Across Seeds: 3001, 3002, 3003, 3004, 3005
# Full dual-policy protocol:
#   - Phase 15 stationary settling with cmd_trim
#   - Safe handoff upon meeting stability gate
#   - Descent hip compensation (+0.10 rad)
# ==============================================================================

import os
import sys
import json
import subprocess
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
eval_script = os.path.join(workspace_root, "tests/eval_phase4_seed.py")
activate_script = os.path.join(workspace_root, "activate_jump.sh")

champion_path = os.path.join(workspace_root, "evaluation_assets/phase3_1_forward_distance/models/model_champion.pt")
pilot_path = os.path.join(workspace_root, "evaluation_assets/phase4_8_pilot_a/model_pilot_a_iter1.pt")

output_dir = os.path.join(workspace_root, "evaluation_assets/phase4_8_pilot_a")
runs_dir = os.path.join(output_dir, "eval_runs")
os.makedirs(runs_dir, exist_ok=True)

DEV_SEEDS = [3001, 3002, 3003, 3004, 3005]


def run_eval(seed, model_path, out_file):
    cmd = (
        f"source {activate_script} && "
        f"python {eval_script} --seed {seed} --jump_model \"{model_path}\" --out \"{out_file}\""
    )
    res = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"Error evaluating seed {seed} with {model_path}:\n{res.stderr[-400:]}")
        return None
    with open(out_file, "r") as f:
        return json.load(f)


def main():
    print("=" * 80)
    print("PHASE 4.8-PILOT-A: PAIRED 5-DEV-SEED PHYSICAL EVALUATION")
    print(f"Champion Model : {champion_path}")
    print(f"Pilot A Model  : {pilot_path}")
    print("=" * 80)

    results_champ = {}
    results_pilot = {}

    for seed in DEV_SEEDS:
        print(f"\n--- Evaluating Seed {seed} ---")
        
        # 1. Champion
        champ_out = os.path.join(runs_dir, f"champion_seed{seed}.json")
        print(f"  [1/2] Running Champion Baseline...")
        data_c = run_eval(seed, champion_path, champ_out)
        if data_c is None:
            raise RuntimeError(f"Champion eval failed for seed {seed}")
        results_champ[seed] = data_c
        print(f"        Fwd: {data_c['fwd_dist_cm']:.2f} cm | Clr: {data_c['max_clearance_cm']:.2f} cm | "
              f"Takeoff Pitch: {data_c.get('takeoff_pitch_deg', 0.0):.2f}° | TD Pitch: {data_c['touchdown_pitch_deg']:.2f}° | "
              f"Pass >=12cm: {data_c['joint_passed_phase4']}")

        # 2. Pilot A
        pilot_out = os.path.join(runs_dir, f"pilot_a_seed{seed}.json")
        print(f"  [2/2] Running Pilot A (Iter 1)...")
        data_p = run_eval(seed, pilot_path, pilot_out)
        if data_p is None:
            raise RuntimeError(f"Pilot A eval failed for seed {seed}")
        results_pilot[seed] = data_p
        print(f"        Fwd: {data_p['fwd_dist_cm']:.2f} cm | Clr: {data_p['max_clearance_cm']:.2f} cm | "
              f"Takeoff Pitch: {data_p.get('takeoff_pitch_deg', 0.0):.2f}° | TD Pitch: {data_p['touchdown_pitch_deg']:.2f}° | "
              f"Pass >=12cm: {data_p['joint_passed_phase4']}")

    # Compile paired comparison
    comparison = {
        "seeds": DEV_SEEDS,
        "models": {
            "champion": champion_path,
            "pilot_a_iter1": pilot_path
        },
        "per_seed": {},
        "summary": {}
    }

    c_fwds, p_fwds = [], []
    c_clrs, p_clrs = [], []
    c_p_to, p_p_to = [], []
    c_p_td, p_p_td = [], []
    c_pass12, p_pass12 = 0, 0
    c_pass8, p_pass8 = 0, 0

    print("\n" + "=" * 90)
    print(f"{'Seed':<6} | {'Model':<12} | {'Fwd (cm)':<10} | {'Clr (cm)':<10} | {'TO Pitch':<10} | {'TD Pitch':<10} | {'RecVy (m/s)':<12} | {'Pass>=12cm':<10}")
    print("-" * 90)

    for seed in DEV_SEEDS:
        c = results_champ[seed]
        p = results_pilot[seed]

        c_fwds.append(c['fwd_dist_cm'])
        p_fwds.append(p['fwd_dist_cm'])
        c_clrs.append(c['max_clearance_cm'])
        p_clrs.append(p['max_clearance_cm'])
        c_p_to.append(c.get('takeoff_pitch_deg', 0.0))
        p_p_to.append(p.get('takeoff_pitch_deg', 0.0))
        c_p_td.append(c['touchdown_pitch_deg'])
        p_p_td.append(p['touchdown_pitch_deg'])

        if c['joint_passed_phase4']: c_pass12 += 1
        if p['joint_passed_phase4']: p_pass12 += 1
        if c['joint_passed_baseline']: c_pass8 += 1
        if p['joint_passed_baseline']: p_pass8 += 1

        delta_fwd = p['fwd_dist_cm'] - c['fwd_dist_cm']
        delta_clr = p['max_clearance_cm'] - c['max_clearance_cm']

        comparison["per_seed"][str(seed)] = {
            "champion": c,
            "pilot_a": p,
            "deltas": {
                "fwd_dist_cm": delta_fwd,
                "max_clearance_cm": delta_clr,
                "takeoff_pitch_deg": p.get('takeoff_pitch_deg', 0.0) - c.get('takeoff_pitch_deg', 0.0),
                "touchdown_pitch_deg": p['touchdown_pitch_deg'] - c['touchdown_pitch_deg'],
                "post_rec_vy_last10": p['post_rec_vy_last10'] - c['post_rec_vy_last10'],
            }
        }

        print(f"{seed:<6} | {'Champion':<12} | {c['fwd_dist_cm']:<10.2f} | {c['max_clearance_cm']:<10.2f} | {c.get('takeoff_pitch_deg', 0.0):<10.2f} | {c['touchdown_pitch_deg']:<10.2f} | {c['post_rec_vy_last10']:<12.4f} | {str(c['joint_passed_phase4']):<10}")
        print(f"{'':<6} | {'Pilot A':<12} | {p['fwd_dist_cm']:<10.2f} | {p['max_clearance_cm']:<10.2f} | {p.get('takeoff_pitch_deg', 0.0):<10.2f} | {p['touchdown_pitch_deg']:<10.2f} | {p['post_rec_vy_last10']:<12.4f} | {str(p['joint_passed_phase4']):<10}")
        print(f"{'':<6} | {'Delta (P-C)':<12} | {delta_fwd:<+10.2f} | {delta_clr:<+10.2f} | {p.get('takeoff_pitch_deg', 0.0) - c.get('takeoff_pitch_deg', 0.0):<+10.2f} | {p['touchdown_pitch_deg'] - c['touchdown_pitch_deg']:<+10.2f} | {p['post_rec_vy_last10'] - c['post_rec_vy_last10']:<+12.4f} | {'--':<10}")
        print("-" * 90)

    comparison["summary"] = {
        "champion_fwd_mean": float(np.mean(c_fwds)),
        "pilot_a_fwd_mean": float(np.mean(p_fwds)),
        "delta_fwd_mean": float(np.mean(p_fwds) - np.mean(c_fwds)),
        "champion_clr_mean": float(np.mean(c_clrs)),
        "pilot_a_clr_mean": float(np.mean(p_clrs)),
        "delta_clr_mean": float(np.mean(p_clrs) - np.mean(c_clrs)),
        "champion_pass_rate_12cm": f"{c_pass12}/{len(DEV_SEEDS)} ({c_pass12/len(DEV_SEEDS)*100:.1f}%)",
        "pilot_a_pass_rate_12cm": f"{p_pass12}/{len(DEV_SEEDS)} ({p_pass12/len(DEV_SEEDS)*100:.1f}%)",
        "champion_pass_rate_8cm": f"{c_pass8}/{len(DEV_SEEDS)} ({c_pass8/len(DEV_SEEDS)*100:.1f}%)",
        "pilot_a_pass_rate_8cm": f"{p_pass8}/{len(DEV_SEEDS)} ({p_pass8/len(DEV_SEEDS)*100:.1f}%)",
    }

    summary_file = os.path.join(output_dir, "paired_physical_evaluation.json")
    with open(summary_file, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY (5 SEEDS AVERAGE)")
    print("=" * 80)
    print(f"Forward Distance Mean : Champion = {np.mean(c_fwds):.2f} cm | Pilot A = {np.mean(p_fwds):.2f} cm | Delta = {np.mean(p_fwds) - np.mean(c_fwds):+.2f} cm")
    print(f"Clearance Mean        : Champion = {np.mean(c_clrs):.2f} cm | Pilot A = {np.mean(p_clrs):.2f} cm | Delta = {np.mean(p_clrs) - np.mean(c_clrs):+.2f} cm")
    print(f"Pass Rate >= 12cm     : Champion = {c_pass12}/5 ({c_pass12*20}%) | Pilot A = {p_pass12}/5 ({p_pass12*20}%)")
    print(f"Pass Rate >= 8cm      : Champion = {c_pass8}/5 ({c_pass8*20}%) | Pilot A = {p_pass8}/5 ({p_pass8*20}%)")
    print(f"Summary JSON saved to : {summary_file}")


if __name__ == "__main__":
    main()
