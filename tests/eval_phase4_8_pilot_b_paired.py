#!/usr/bin/env python3
# ==============================================================================
# Phase 4.8-Pilot-B: Three-Model Paired 5-Dev-Seed Physical Evaluation
# Compares:
#   1. Model Champion Baseline (fad590f4...)
#   2. Model Pilot A (c4b4b294...)
#   3. Model Pilot B (64aedcfd...)
# Across Seeds: 3001, 3002, 3003, 3004, 3005
# Full dual-policy protocol:
#   - Phase 15 stationary settling with cmd_trim
#   - Safe handoff upon meeting stability gate
#   - Descent hip compensation (+0.10 normalized -> +0.060 rad)
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
pilot_a_path = os.path.join(workspace_root, "evaluation_assets/phase4_8_pilot_a/model_pilot_a_iter1.pt")
pilot_b_path = os.path.join(workspace_root, "evaluation_assets/phase4_8_pilot_b/model_pilot_b_step1.pt")

output_dir = os.path.join(workspace_root, "evaluation_assets/phase4_8_pilot_b")
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
    print("=" * 90)
    print("PHASE 4.8-PILOT-B: THREE-MODEL PAIRED 5-DEV-SEED PHYSICAL EVALUATION")
    print(f"  Champion Baseline : {champion_path}")
    print(f"  Pilot A Model     : {pilot_a_path}")
    print(f"  Pilot B Model     : {pilot_b_path}")
    print("=" * 90)

    results_champ = {}
    results_pilot_a = {}
    results_pilot_b = {}

    for seed in DEV_SEEDS:
        print(f"\n--- Evaluating Seed {seed} ---")

        # 1. Champion
        champ_out = os.path.join(runs_dir, f"champion_seed{seed}.json")
        # Reuse existing run if available
        if not os.path.isfile(champ_out):
            champ_prev = os.path.join(workspace_root, f"evaluation_assets/phase4_8_pilot_a/eval_runs/champion_seed{seed}.json")
            if os.path.isfile(champ_prev):
                import shutil
                shutil.copy2(champ_prev, champ_out)
        if os.path.isfile(champ_out):
            with open(champ_out, "r") as f:
                data_c = json.load(f)
        else:
            print(f"  [1/3] Running Champion Baseline...")
            data_c = run_eval(seed, champion_path, champ_out)
        results_champ[seed] = data_c
        print(f"  [Champion] Fwd: {data_c['fwd_dist_cm']:.2f} cm | Clr: {data_c['max_clearance_cm']:.2f} cm | "
              f"TO Pitch: {data_c.get('takeoff_pitch_deg', 0.0):.2f}° | TD Pitch: {data_c['touchdown_pitch_deg']:.2f}° | "
              f"RecVy: {data_c['post_rec_vy_last10']:.4f} m/s | C4: {data_c['passed_rec_vy_003']} | PASS: {data_c['joint_passed_phase4']}")

        # 2. Pilot A
        pilot_a_out = os.path.join(runs_dir, f"pilot_a_seed{seed}.json")
        if not os.path.isfile(pilot_a_out):
            pilot_a_prev = os.path.join(workspace_root, f"evaluation_assets/phase4_8_pilot_a/eval_runs/pilot_a_seed{seed}.json")
            if os.path.isfile(pilot_a_prev):
                import shutil
                shutil.copy2(pilot_a_prev, pilot_a_out)
        if os.path.isfile(pilot_a_out):
            with open(pilot_a_out, "r") as f:
                data_pa = json.load(f)
        else:
            print(f"  [2/3] Running Pilot A...")
            data_pa = run_eval(seed, pilot_a_path, pilot_a_out)
        results_pilot_a[seed] = data_pa
        print(f"  [Pilot A ] Fwd: {data_pa['fwd_dist_cm']:.2f} cm | Clr: {data_pa['max_clearance_cm']:.2f} cm | "
              f"TO Pitch: {data_pa.get('takeoff_pitch_deg', 0.0):.2f}° | TD Pitch: {data_pa['touchdown_pitch_deg']:.2f}° | "
              f"RecVy: {data_pa['post_rec_vy_last10']:.4f} m/s | C4: {data_pa['passed_rec_vy_003']} | PASS: {data_pa['joint_passed_phase4']}")

        # 3. Pilot B
        pilot_b_out = os.path.join(runs_dir, f"pilot_b_seed{seed}.json")
        print(f"  [3/3] Running Pilot B (Fixed lr=5e-5, Step 1)...")
        data_pb = run_eval(seed, pilot_b_path, pilot_b_out)
        if data_pb is None:
            raise RuntimeError(f"Pilot B eval failed for seed {seed}")
        results_pilot_b[seed] = data_pb
        print(f"  [Pilot B ] Fwd: {data_pb['fwd_dist_cm']:.2f} cm | Clr: {data_pb['max_clearance_cm']:.2f} cm | "
              f"TO Pitch: {data_pb.get('takeoff_pitch_deg', 0.0):.2f}° | TD Pitch: {data_pb['touchdown_pitch_deg']:.2f}° | "
              f"RecVy: {data_pb['post_rec_vy_last10']:.4f} m/s | C4: {data_pb['passed_rec_vy_003']} | PASS: {data_pb['joint_passed_phase4']}")

    # Compile paired comparison
    comparison = {
        "seeds": DEV_SEEDS,
        "models": {
            "champion": champion_path,
            "pilot_a": pilot_a_path,
            "pilot_b": pilot_b_path,
        },
        "per_seed": {},
        "summary": {}
    }

    c_fwds, pa_fwds, pb_fwds = [], [], []
    c_clrs, pa_clrs, pb_clrs = [], [], []
    c_p_to, pa_p_to, pb_p_to = [], [], []
    c_p_td, pa_p_td, pb_p_td = [], [], []
    c_rec_vys, pa_rec_vys, pb_rec_vys = [], [], []
    c_pass12, pa_pass12, pb_pass12 = 0, 0, 0
    c_c4_pass, pa_c4_pass, pb_c4_pass = 0, 0, 0

    print("\n" + "=" * 115)
    print(f"{'Seed':<6} | {'Model':<10} | {'Fwd (cm)':<10} | {'Clr (cm)':<10} | {'TO Pitch':<10} | {'TD Pitch':<10} | {'RecVy (m/s)':<12} | {'C4 (<=0.03)':<12} | {'Pass>=12cm':<10}")
    print("-" * 115)

    for seed in DEV_SEEDS:
        c = results_champ[seed]
        pa = results_pilot_a[seed]
        pb = results_pilot_b[seed]

        c_fwds.append(c['fwd_dist_cm'])
        pa_fwds.append(pa['fwd_dist_cm'])
        pb_fwds.append(pb['fwd_dist_cm'])

        c_clrs.append(c['max_clearance_cm'])
        pa_clrs.append(pa['max_clearance_cm'])
        pb_clrs.append(pb['max_clearance_cm'])

        c_p_to.append(c.get('takeoff_pitch_deg', 0.0))
        pa_p_to.append(pa.get('takeoff_pitch_deg', 0.0))
        pb_p_to.append(pb.get('takeoff_pitch_deg', 0.0))

        c_p_td.append(c['touchdown_pitch_deg'])
        pa_p_td.append(pa['touchdown_pitch_deg'])
        pb_p_td.append(pb['touchdown_pitch_deg'])

        c_rec_vys.append(c['post_rec_vy_last10'])
        pa_rec_vys.append(pa['post_rec_vy_last10'])
        pb_rec_vys.append(pb['post_rec_vy_last10'])

        if c['passed_rec_vy_003']: c_c4_pass += 1
        if pa['passed_rec_vy_003']: pa_c4_pass += 1
        if pb['passed_rec_vy_003']: pb_c4_pass += 1

        if c['joint_passed_phase4']: c_pass12 += 1
        if pa['joint_passed_phase4']: pa_pass12 += 1
        if pb['joint_passed_phase4']: pb_pass12 += 1

        comparison["per_seed"][str(seed)] = {
            "champion": c,
            "pilot_a": pa,
            "pilot_b": pb,
            "deltas_pb_vs_champ": {
                "fwd_dist_cm": pb['fwd_dist_cm'] - c['fwd_dist_cm'],
                "max_clearance_cm": pb['max_clearance_cm'] - c['max_clearance_cm'],
                "takeoff_pitch_deg": pb.get('takeoff_pitch_deg', 0.0) - c.get('takeoff_pitch_deg', 0.0),
                "touchdown_pitch_deg": pb['touchdown_pitch_deg'] - c['touchdown_pitch_deg'],
                "post_rec_vy_last10": pb['post_rec_vy_last10'] - c['post_rec_vy_last10'],
            },
            "deltas_pb_vs_pa": {
                "fwd_dist_cm": pb['fwd_dist_cm'] - pa['fwd_dist_cm'],
                "max_clearance_cm": pb['max_clearance_cm'] - pa['max_clearance_cm'],
                "takeoff_pitch_deg": pb.get('takeoff_pitch_deg', 0.0) - pa.get('takeoff_pitch_deg', 0.0),
                "touchdown_pitch_deg": pb['touchdown_pitch_deg'] - pa['touchdown_pitch_deg'],
                "post_rec_vy_last10": pb['post_rec_vy_last10'] - pa['post_rec_vy_last10'],
            }
        }

        print(f"{seed:<6} | {'Champion':<10} | {c['fwd_dist_cm']:<10.2f} | {c['max_clearance_cm']:<10.2f} | {c.get('takeoff_pitch_deg', 0.0):<10.2f} | {c['touchdown_pitch_deg']:<10.2f} | {c['post_rec_vy_last10']:<12.4f} | {str(c['passed_rec_vy_003']):<12} | {str(c['joint_passed_phase4']):<10}")
        print(f"{'':<6} | {'Pilot A':<10} | {pa['fwd_dist_cm']:<10.2f} | {pa['max_clearance_cm']:<10.2f} | {pa.get('takeoff_pitch_deg', 0.0):<10.2f} | {pa['touchdown_pitch_deg']:<10.2f} | {pa['post_rec_vy_last10']:<12.4f} | {str(pa['passed_rec_vy_003']):<12} | {str(pa['joint_passed_phase4']):<10}")
        print(f"{'':<6} | {'Pilot B':<10} | {pb['fwd_dist_cm']:<10.2f} | {pb['max_clearance_cm']:<10.2f} | {pb.get('takeoff_pitch_deg', 0.0):<10.2f} | {pb['touchdown_pitch_deg']:<10.2f} | {pb['post_rec_vy_last10']:<12.4f} | {str(pb['passed_rec_vy_003']):<12} | {str(pb['joint_passed_phase4']):<10}")
        print("-" * 115)

    comparison["summary"] = {
        "fwd_mean": {
            "champion": float(np.mean(c_fwds)),
            "pilot_a": float(np.mean(pa_fwds)),
            "pilot_b": float(np.mean(pb_fwds)),
            "delta_pb_vs_champ": float(np.mean(pb_fwds) - np.mean(c_fwds)),
            "delta_pb_vs_pa": float(np.mean(pb_fwds) - np.mean(pa_fwds)),
        },
        "clearance_mean": {
            "champion": float(np.mean(c_clrs)),
            "pilot_a": float(np.mean(pa_clrs)),
            "pilot_b": float(np.mean(pb_clrs)),
            "delta_pb_vs_champ": float(np.mean(pb_clrs) - np.mean(c_clrs)),
            "delta_pb_vs_pa": float(np.mean(pb_clrs) - np.mean(pa_clrs)),
        },
        "takeoff_pitch_mean": {
            "champion": float(np.mean(c_p_to)),
            "pilot_a": float(np.mean(pa_p_to)),
            "pilot_b": float(np.mean(pb_p_to)),
        },
        "touchdown_pitch_mean": {
            "champion": float(np.mean(c_p_td)),
            "pilot_a": float(np.mean(pa_p_td)),
            "pilot_b": float(np.mean(pb_p_td)),
        },
        "rec_vy_mean": {
            "champion": float(np.mean(c_rec_vys)),
            "pilot_a": float(np.mean(pa_rec_vys)),
            "pilot_b": float(np.mean(pb_rec_vys)),
        },
        "c4_pass_rate": {
            "champion": f"{c_c4_pass}/{len(DEV_SEEDS)} ({c_c4_pass/len(DEV_SEEDS)*100:.1f}%)",
            "pilot_a": f"{pa_c4_pass}/{len(DEV_SEEDS)} ({pa_c4_pass/len(DEV_SEEDS)*100:.1f}%)",
            "pilot_b": f"{pb_c4_pass}/{len(DEV_SEEDS)} ({pb_c4_pass/len(DEV_SEEDS)*100:.1f}%)",
        },
        "joint_phase4_pass_rate": {
            "champion": f"{c_pass12}/{len(DEV_SEEDS)} ({c_pass12/len(DEV_SEEDS)*100:.1f}%)",
            "pilot_a": f"{pa_pass12}/{len(DEV_SEEDS)} ({pa_pass12/len(DEV_SEEDS)*100:.1f}%)",
            "pilot_b": f"{pb_pass12}/{len(DEV_SEEDS)} ({pb_pass12/len(DEV_SEEDS)*100:.1f}%)",
        }
    }

    summary_file = os.path.join(output_dir, "paired_physical_evaluation.json")
    with open(summary_file, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 90)
    print("EXECUTIVE SUMMARY (5 SEEDS AVERAGE COMPARISON)")
    print("=" * 90)
    print(f"Forward Distance Mean : Champion = {np.mean(c_fwds):.2f} cm | Pilot A = {np.mean(pa_fwds):.2f} cm | Pilot B = {np.mean(pb_fwds):.2f} cm")
    print(f"                      : Delta(B - Champ) = {np.mean(pb_fwds) - np.mean(c_fwds):+.2f} cm | Delta(B - A) = {np.mean(pb_fwds) - np.mean(pa_fwds):+.2f} cm")
    print(f"Clearance Mean        : Champion = {np.mean(c_clrs):.2f} cm | Pilot A = {np.mean(pa_clrs):.2f} cm | Pilot B = {np.mean(pb_clrs):.2f} cm")
    print(f"RecVy Last10 Mean     : Champion = {np.mean(c_rec_vys):.4f} m/s | Pilot A = {np.mean(pa_rec_vys):.4f} m/s | Pilot B = {np.mean(pb_rec_vys):.4f} m/s")
    print(f"C4 Pass Rate (<=0.03) : Champion = {c_c4_pass}/5 | Pilot A = {pa_c4_pass}/5 | Pilot B = {pb_c4_pass}/5")
    print(f"Joint Pass Rate >=12cm: Champion = {c_pass12}/5 | Pilot A = {pa_pass12}/5 | Pilot B = {pb_pass12}/5")
    print(f"Summary JSON saved to : {summary_file}")


if __name__ == "__main__":
    main()
