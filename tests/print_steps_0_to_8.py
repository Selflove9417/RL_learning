#!/usr/bin/env python3
import json

seeds = [100, 200, 400, 2, 11, 13, 17, 23, 5, 150, 42, 1]
trajs = {}
for s in seeds:
    with open(f"/tmp/audit_trajectories/traj_seed_{s}.json") as f:
        trajs[s] = json.load(f)

for step in range(9):
    t_s = step * 0.02
    phi = t_s / 2.0
    print(f"\n=================== STEP {step:2d} (t={t_s:.2f}s, phi={phi:.3f}) ===================")
    print(f"{'Seed':<5} | {'CoM_Vy':<8} | {'Base_Vy':<8} | {'CoM_Vz':<8} | {'Pitch':<7} | {'Roll':<7} | {'Wheel_pos_y':<11} | {'WhlOmega_L/R':<15} | {'TauWheel_L/R':<14} | {'TauKnee_L/R':<14} | {'Fn_L/R':<14}")
    print("-" * 125)
    for s in seeds:
        f = trajs[s]["frames"][step]
        tau_k = f"{f['tau_knee_L_nm']:.1f}/{f['tau_knee_R_nm']:.1f}"
        tau_w = f"{f['tau_wheel_L_nm']:.1f}/{f['tau_wheel_R_nm']:.1f}"
        fn = f"{f['fn_L_N']:.1f}/{f['fn_R_N']:.1f}"
        w_lr = f"{f['wheel_omega_L_radps']:.1f}/{f['wheel_omega_R_radps']:.1f}"
        print(f"{s:<5d} | {f['com_vel_y_mps']:<+8.3f} | {f['base_lin_vel_y_mps']:<+8.3f} | {f['com_vel_z_mps']:<+8.3f} | {f['pitch_deg']:<+7.2f} | {f['roll_deg']:<+7.2f} | {f['wheel_pos_y_mean_m']:<+11.4f} | {w_lr:<15} | {tau_w:<14} | {tau_k:<14} | {fn:<14}")
