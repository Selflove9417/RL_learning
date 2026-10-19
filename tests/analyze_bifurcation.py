#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Phase 3.1 Trajectory Analysis and Bifurcation Detector
# ==============================================================================

import os
import glob
import json
import numpy as np

traj_dir = "/tmp/audit_trajectories"
files = glob.glob(os.path.join(traj_dir, "traj_seed_*.json"))

SEEDS = [
    42, 100, 200, 300, 400,
    1, 2, 3, 5, 7, 11, 13, 17, 19, 23,
    50, 150, 250, 350, 450
]

all_data = {}
for s in SEEDS:
    p = os.path.join(traj_dir, f"traj_seed_{s}.json")
    with open(p, "r") as f:
        all_data[s] = json.load(f)

# Also load telemetry summary
with open("/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/20seed_robustness/phase3_1_champion_20seed_summary.json", "r") as f:
    eval_summary = json.load(f)

results_map = {r["seed"]: r for r in eval_summary["all_seeds_results"]}

print("="*80)
print("1. RANDOMNESS SOURCES AUDIT ACROSS 20 SEEDS")
print("="*80)
for s in SEEDS:
    d = all_data[s]
    r = results_map[s]
    fric = d.get("friction_coefficient")
    cmd_vy = d.get("cmd_vy_mps")
    init_f = d["frames"][0]
    print(f"Seed {s:3d}: fric={fric:.4f} | cmd_vy={cmd_vy:.4f} | init_pitch={init_f['pitch_deg']:+5.2f}° | init_roll={init_f['roll_deg']:+5.2f}° | init_wheel_y={init_f['wheel_pos_y_mean_m']:+.4f} | init_com_z={init_f['com_pos_z_m']:.4f}")

print("\n" + "="*80)
print("2. STEP-BY-STEP KEY TRAJECTORY AUDIT ACROSS REPRESENTATIVE SEEDS")
print("Representative Seeds:")
print("  Normal: Seed 100 (+11.4cm), Seed 200 (+10.1cm), Seed 350 (+11.6cm)")
print("  Low Fwd: Seed 400 (+1.9cm), Seed 2 (+1.5cm), Seed 11 (+4.1cm), Seed 13 (+4.0cm)")
print("  Backward: Seed 17 (-14.1cm), Seed 23 (-4.4cm)")
print("  Low Clr: Seed 5 (16.9cm), Seed 150 (17.9cm), Seed 3 (19.4cm), Seed 450 (19.4cm)")
print("  Roll > 10°: Seed 42 (10.55°), Seed 1 (10.36°)")
print("="*80)

# Key steps to compare:
# Step 0: Initial rest (t=0.00s, phi=0.00)
# Step 3: Early prep (t=0.06s, phi=0.03)
# Step 5: Prep end / Squat starts (t=0.10s, phi=0.05)
# Step 8: Squat deep (t=0.16s, phi=0.08)
# Step 11: Push-off start (t=0.22s, phi=0.11)
# Step 14: Mid push-off (t=0.28s, phi=0.14)
# Step 17: Pre-liftoff (t=0.34s, phi=0.17)
# Step 20: Liftoff window (t=0.40s, phi=0.20)
# Step 25: Mid flight (t=0.50s, phi=0.25)
# Step 32: Apex / Descending (t=0.64s, phi=0.32)
# Step 42: Touchdown window (t=0.84s, phi=0.42)

key_steps = [0, 3, 5, 8, 11, 14, 17, 20, 22, 25, 30]
target_seeds = [100, 200, 350, 400, 2, 11, 13, 17, 23, 5, 150, 42, 1]

header = f"{'Seed':<5} | {'Step':<4} | {'t(s)':<5} | {'phi':<5} | {'CoM_Vy':<8} | {'CoM_Vz':<8} | {'Pitch':<7} | {'Roll':<7} | {'TauKnee_L/R':<14} | {'TauWheel_L/R':<15} | {'ClrMin':<7} | {'Fn_L/R':<14}"
print("\n--- DETAILED STEP COMPARISON ---")
for step_idx in [0, 5, 11, 14, 17, 20, 22]:
    print(f"\n>>> Step {step_idx} (t={step_idx*0.02:.2f}s, phi={step_idx*0.02/2.0:.3f}) <<<")
    print(header)
    print("-" * len(header))
    for s in [100, 200, 400, 2, 17, 23, 5, 150, 42, 1]:
        f = all_data[s]["frames"][step_idx]
        tau_k = f"{f['tau_knee_L_nm']:.1f}/{f['tau_knee_R_nm']:.1f}"
        tau_w = f"{f['tau_wheel_L_nm']:.1f}/{f['tau_wheel_R_nm']:.1f}"
        fn = f"{f['fn_L_N']:.1f}/{f['fn_R_N']:.1f}"
        print(f"{s:<5d} | {f['step']:<4d} | {f['t_s']:<5.2f} | {f['phase']:<5.3f} | {f['com_vel_y_mps']:<+8.3f} | {f['com_vel_z_mps']:<+8.3f} | {f['pitch_deg']:<+7.2f} | {f['roll_deg']:<+7.2f} | {tau_k:<14} | {tau_w:<15} | {f['clr_min_cm']:<7.2f} | {fn:<14}")
