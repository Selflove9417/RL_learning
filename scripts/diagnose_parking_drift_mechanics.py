"""
Phase 14 Priority 1 Diagnostic Script: Physical Mechanics of Parking Directional Drift
Analyzes steady-state parking (steps 350~600, 5.0 seconds) on model_125.pt across 4 heights:
  - H = 0.30m (extreme lower boundary: severe drift -0.046 m/s, -23cm in 5s)
  - H = 0.32m (regular squat: drift -0.016 m/s, -8cm in 5s)
  - H = 0.398m (nominal stand reference)
  - H = 0.48m (regular rise: drift -0.018 m/s, -8.8cm in 5s)

Quantifies:
1. Sagittal pitch angle (deg) & projected gravity g_y.
2. Wheel driving torque (N*m), wheel velocity, and policy raw wheel action.
3. Sagittal wheel position relative to base CoM: y_wheel_rel_base (m).
4. Hip and knee joint angles (actual vs analytical IK targets) & joint torques.
5. Linear velocity vy, 5s net displacement, total distance, and drift directivity ratio.

Usage:
    python scripts/diagnose_parking_drift_mechanics.py
"""

import os
import sys
import json
import math
import numpy as np

# isaacgym must be imported before torch
import isaacgym
from isaacgym import gymapi, gymtorch
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from rsl_rl.modules import ActorCritic

def run_drift_mechanics_diagnosis():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    model_path = "legged_gym/logs/myrobot/Oct13_15-16-03_phase14_exp3_parking_focus_resume10iter/model_125.pt"
    output_path = "evaluation_assets/phase14/results/parking_drift_mechanics_diagnosis.json"

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False
    env.max_episode_length = 3000

    rad_to_deg = 180.0 / np.pi

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    print(f"\n================ Parking Drift Mechanics Diagnosis (Priority 1) ================")
    print(f"Model Path: {model_path}")
    print(f"Num Envs: {args.num_envs}")

    loaded = torch.load(model_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    pol = ac.act_inference

    # 4 diagnostic heights
    diagnostic_cases = [
        ("H_0.30m (Extreme Squat)", 0.300),
        ("H_0.32m (Regular Squat)", 0.320),
        ("H_0.398m (Nominal Stand)", 0.39774),
        ("H_0.48m (Regular Rise)",  0.480),
    ]

    left_hip_idx = env.left_hip_idx
    left_knee_idx = env.left_knee_idx
    left_wheel_idx = env.left_wheel_idx
    right_hip_idx = env.right_hip_idx
    right_knee_idx = env.right_knee_idx
    right_wheel_idx = env.right_wheel_idx

    # Forward Kinematics constants (matching compute_leg_ik in myrobot_env.py)
    dy1, dz1 = -0.29348091, -0.06220095
    dy2, dz2 = 0.2821087, -0.19553796
    y_hip, z_hip = 0.125, -0.070
    y_wheel_target = 0.11362779 # nominal sagittal wheel offset relative to base

    step_cmd_start = 50     # 1.0s settle
    step_motion_end = 250   # 5.0s motion end
    step_brake_end = 350    # 7.0s brake end
    total_steps = 600       # 12.0s: steps 350~600 is exactly 5.0s (250 steps) steady parking hold

    diagnosis_results = {}

    for case_label, target_H in diagnostic_cases:
        print(f"\n>>> Diagnosing Height: {case_label} (Target H = {target_H:.4f}m)")
        torch.manual_seed(42)
        np.random.seed(42)
        env.reset()

        delta_h = float(np.clip(0.08 * (target_H - 0.39774), -0.010, 0.010))
        h_ctrl_val = round(target_H + delta_h, 4)

        # Theoretical IK targets
        q1_ik_t, q2_ik_t = env.compute_leg_ik(torch.tensor(h_ctrl_val, device=env.device))
        q1_ik = float(q1_ik_t.cpu().numpy()) if isinstance(q1_ik_t, torch.Tensor) else float(q1_ik_t)
        q2_ik = float(q2_ik_t.cpu().numpy()) if isinstance(q2_ik_t, torch.Tensor) else float(q2_ik_t)

        h_current = 0.39774
        step_dh = 0.04 * env.dt

        log_pitch_deg = []
        log_proj_grav_y = []
        log_lin_vel_y = []
        log_wheel_tau_mean = []
        log_wheel_action_mean = []
        log_wheel_ang_vel_mean = []
        log_hip_pos_mean = []
        log_knee_pos_mean = []
        log_knee_tau_mean = []
        log_base_h = []
        log_wheel_base_dy = []

        done_flags = np.zeros(args.num_envs, dtype=bool)

        for step in range(total_steps):
            if step < step_cmd_start:
                tgt_h = 0.39774
            else:
                tgt_h = h_ctrl_val

            if h_current < tgt_h:
                h_current = min(h_current + step_dh, tgt_h)
            elif h_current > tgt_h:
                h_current = max(h_current - step_dh, tgt_h)

            env.commands[:, 0] = 0.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            env.height_cmd_buf[:] = h_current
            if hasattr(env, 'target_height_buf'):
                env.target_height_buf[:] = h_ctrl_val

            env.compute_observations()
            obs = env.get_observations()

            with torch.no_grad():
                actions = pol(obs)

            obs, _, _, dones, _ = env.step(actions)
            done_flags |= dones.cpu().numpy()

            if step >= step_brake_end:
                base_h = env.root_states[:, 2].cpu().numpy()
                base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                proj_gy = env.projected_gravity[:, 1].cpu().numpy()
                pitch_deg = np.arcsin(np.clip(proj_gy, -1.0, 1.0)) * rad_to_deg

                torques = env.torques.cpu().numpy()
                dof_pos = env.dof_pos.cpu().numpy()
                dof_vel = env.dof_vel.cpu().numpy()
                act_arr = actions.cpu().numpy()

                tau_lw = torques[:, left_wheel_idx]
                tau_rw = torques[:, right_wheel_idx]
                tau_w_mean = 0.5 * (tau_lw + tau_rw)

                act_w_mean = 0.5 * (act_arr[:, left_wheel_idx] + act_arr[:, right_wheel_idx])
                w_vel_mean = 0.5 * (dof_vel[:, left_wheel_idx] + dof_vel[:, right_wheel_idx])

                hip_pos = 0.5 * (dof_pos[:, left_hip_idx] + dof_pos[:, right_hip_idx])
                knee_pos = 0.5 * (dof_pos[:, left_knee_idx] + dof_pos[:, right_knee_idx])
                knee_tau = 0.5 * (torques[:, left_knee_idx] + torques[:, right_knee_idx])

                # Analytical Forward Kinematics for sagittal wheel position relative to base
                # in base sagittal coordinates (y forward, z up)
                wy = dy1 + np.cos(knee_pos) * dy2 - np.sin(knee_pos) * dz2
                wz = dz1 + np.sin(knee_pos) * dy2 + np.cos(knee_pos) * dz2
                y_wheel_rel_base = y_hip + np.cos(hip_pos) * wy - np.sin(hip_pos) * wz

                log_base_h.append(base_h)
                log_lin_vel_y.append(base_vy)
                log_pitch_deg.append(pitch_deg)
                log_proj_grav_y.append(proj_gy)
                log_wheel_tau_mean.append(tau_w_mean)
                log_wheel_action_mean.append(act_w_mean)
                log_wheel_ang_vel_mean.append(w_vel_mean)
                log_hip_pos_mean.append(hip_pos)
                log_knee_pos_mean.append(knee_pos)
                log_knee_tau_mean.append(knee_tau)
                log_wheel_base_dy.append(y_wheel_rel_base)

        vy_arr = np.array(log_lin_vel_y)
        pitch_arr = np.array(log_pitch_deg)
        gy_arr = np.array(log_proj_grav_y)
        tau_w_arr = np.array(log_wheel_tau_mean)
        act_w_arr = np.array(log_wheel_action_mean)
        w_vel_arr = np.array(log_wheel_ang_vel_mean)
        knee_tau_arr = np.array(log_knee_tau_mean)
        knee_pos_arr = np.array(log_knee_pos_mean)
        hip_pos_arr = np.array(log_hip_pos_mean)
        h_arr = np.array(log_base_h)
        dy_wb_arr = np.array(log_wheel_base_dy)

        net_disp_per_env = np.sum(vy_arr * env.dt, axis=0)
        tot_dist_per_env = np.sum(np.abs(vy_arr) * env.dt, axis=0)

        mean_net_disp = float(np.mean(net_disp_per_env))
        mean_tot_dist = float(np.mean(tot_dist_per_env))
        directivity_ratio = float(np.mean(np.abs(net_disp_per_env) / np.maximum(tot_dist_per_env, 1e-5)))

        mean_vy = float(np.mean(vy_arr))
        std_vy = float(np.std(vy_arr))
        mean_pitch = float(np.mean(pitch_arr))
        std_pitch = float(np.std(pitch_arr))
        mean_gy = float(np.mean(gy_arr))

        mean_wheel_tau = float(np.mean(tau_w_arr))
        mean_wheel_act = float(np.mean(act_w_arr))
        mean_wheel_w = float(np.mean(w_vel_arr))
        mean_knee_tau = float(np.mean(knee_tau_arr))
        mean_act_h = float(np.mean(h_arr))
        signed_h_err_cm = (mean_act_h - target_H) * 100.0

        actual_hip_rad = float(np.mean(hip_pos_arr))
        actual_knee_rad = float(np.mean(knee_pos_arr))
        mean_dy_wb = float(np.mean(dy_wb_arr))
        sagittal_support_lever_cm = (mean_dy_wb - y_wheel_target) * 100.0

        diagnosis_results[case_label] = {
            "target_H_m": target_H,
            "h_ctrl_sent_m": h_ctrl_val,
            "mean_act_h_m": mean_act_h,
            "signed_h_err_cm": signed_h_err_cm,
            "kinematics": {
                "mean_lin_vel_y_mps": mean_vy,
                "std_lin_vel_y_mps": std_vy,
                "net_displacement_5s_m": mean_net_disp,
                "total_distance_5s_m": mean_tot_dist,
                "directivity_ratio": directivity_ratio,
            },
            "attitude": {
                "mean_pitch_deg": mean_pitch,
                "std_pitch_deg": std_pitch,
                "mean_proj_gravity_y": mean_gy,
            },
            "dynamics": {
                "mean_wheel_torque_nm": mean_wheel_tau,
                "mean_wheel_raw_action": mean_wheel_act,
                "mean_wheel_ang_vel_radps": mean_wheel_w,
                "mean_knee_torque_nm": mean_knee_tau,
                "actual_hip_rad": actual_hip_rad,
                "ik_target_hip_rad": q1_ik,
                "hip_angle_err_deg": (actual_hip_rad - q1_ik) * rad_to_deg,
                "actual_knee_rad": actual_knee_rad,
                "ik_target_knee_rad": q2_ik,
                "knee_angle_err_deg": (actual_knee_rad - q2_ik) * rad_to_deg,
                "wheel_y_rel_base_m": mean_dy_wb,
                "sagittal_support_lever_cm": sagittal_support_lever_cm
            }
        }

        print(f"  -> Act H: {mean_act_h:.4f}m (Err: {signed_h_err_cm:+.2f}cm)")
        print(f"  -> Linear Vy: {mean_vy:+.4f} m/s (std: {std_vy:.4f} m/s)")
        print(f"  -> 5s Net Disp: {mean_net_disp:+.4f} m | Total Dist: {mean_tot_dist:.4f} m | Directivity: {directivity_ratio*100:.1f}%")
        print(f"  -> Steady Pitch: {mean_pitch:+.3f}° (std: {std_pitch:.3f}°) | g_y: {mean_gy:+.5f}")
        print(f"  -> Wheel Torque: {mean_wheel_tau:+.4f} N·m | Action: {mean_wheel_act:+.4f} | Wheel Omega: {mean_wheel_w:+.4f} rad/s")
        print(f"  -> Knee Torque: {mean_knee_tau:.2f} N·m")
        print(f"  -> Hip Angle: {actual_hip_rad:.3f} rad (IK Target: {q1_ik:.3f}, Err: {(actual_hip_rad - q1_ik)*rad_to_deg:+.2f}°)")
        print(f"  -> Knee Angle: {actual_knee_rad:.3f} rad (IK Target: {q2_ik:.3f}, Err: {(actual_knee_rad - q2_ik)*rad_to_deg:+.2f}°)")
        print(f"  -> Wheel-Base y: {mean_dy_wb*100:+.2f} cm (Design Target: {y_wheel_target*100:.2f} cm, Lever: {sagittal_support_lever_cm:+.2f} cm)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(diagnosis_results, f, indent=2)

    print(f"\n================ Drift Diagnosis Complete ================")
    print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    run_drift_mechanics_diagnosis()
