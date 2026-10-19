#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Single-Seed Diagnostic Evaluator for Wheel Brake Modes
# Evaluates a single seed under Mode A (control), Mode B (weak), or Mode C (strong)
# Intervention window: strictly t <= 0.10s (steps 0 to 5)
# Step 6 onward: 100% restored to original Champion policy
# ==============================================================================

import os
import sys
import json
import argparse

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz
import torch
import numpy as np

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.envs.myrobot.myrobot_dual_env import compute_exact_cylinder_clearance

def run_diagnostic(seed, mode, brake_k, out_json):
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["eval_diag", "--task", task_name, "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = seed

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.normalization.clip_actions = 100.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.env.episode_length_s = 2.8
    env_cfg.noise.add_noise = False
    env_cfg.env.enable_camera_sensors = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    ac = ActorCritic(
        num_actor_obs=env_cfg.env.num_observations,
        num_critic_obs=env_cfg.env.num_observations,
        num_actions=env_cfg.env.num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0
    ).to(env.device)
    model_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    ckpt = torch.load(model_path, map_location=env.device)
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    policy = ac.act_inference

    obs, _ = env.reset()
    total_steps = int(2.8 / env.dt) # 140 steps

    takeoff_step = None
    touchdown_step = None
    took_off = False
    landed = False

    max_clearance_cm = 0.0
    max_clearance_L_cm = 0.0
    max_clearance_R_cm = 0.0

    takeoff_com_y = 0.0
    touchdown_com_y = 0.0
    takeoff_com_vy = 0.0
    takeoff_com_vz = 0.0

    step0_vy = 0.0
    step3_vy = 0.0
    step5_vy = 0.0
    step7_vy = 0.0
    step8_vy = 0.0

    step0_pitch = 0.0
    step0_roll = 0.0
    step5_pitch = 0.0
    step5_roll = 0.0

    step5_w_L = 0.0
    step5_w_R = 0.0
    step5_tau_w_L = 0.0
    step5_tau_w_R = 0.0

    step7_tau_w_L = 0.0
    step7_tau_w_R = 0.0
    step7_tau_k_L = 0.0
    step7_tau_k_R = 0.0
    step7_tau_h_L = 0.0
    step7_tau_h_R = 0.0

    step8_tau_w_L = 0.0
    step8_tau_w_R = 0.0
    step8_tau_k_L = 0.0
    step8_tau_k_R = 0.0
    step8_tau_h_L = 0.0
    step8_tau_h_R = 0.0

    touchdown_pitch = 0.0
    touchdown_roll = 0.0
    touchdown_vy = 0.0
    touchdown_plus_20s_vy = 0.0

    prep_base_vels = []
    prep_wheel_omegas = []
    prep_ground_slip_y = 0.0
    initial_wheel_y = None

    airborne_pitches = []
    airborne_rolls = []

    all_base_vys = []
    recovery_vys = []
    abnormal_resets = 0
    trajectory = []

    for step in range(total_steps):
        t_sim = step * env.dt

        true_com_pos, true_com_vel = env._compute_true_com()
        base_lin_vel = env.base_lin_vel[0].cpu().numpy()
        all_base_vys.append(float(base_lin_vel[1]))

        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi

        curr_wheel_y = (env.rb_states[0, env.left_wheel_body, 1].item() + env.rb_states[0, env.right_wheel_body, 1].item()) / 2.0
        if step == 0:
            initial_wheel_y = curr_wheel_y
            step0_vy = float(true_com_vel[0, 1].item())
            step0_pitch = pitch
            step0_roll = roll

        w_L = abs(env.dof_vel[0, env.left_wheel_idx].item())
        w_R = abs(env.dof_vel[0, env.right_wheel_idx].item())

        tau_h_L = float(env.torques[0, env.left_hip_idx].item())
        tau_h_R = float(env.torques[0, env.right_hip_idx].item())
        tau_k_L = float(env.torques[0, env.left_knee_idx].item())
        tau_k_R = float(env.torques[0, env.right_knee_idx].item())
        tau_w_L = float(env.torques[0, env.left_wheel_idx].item())
        tau_w_R = float(env.torques[0, env.right_wheel_idx].item())

        if step <= 5 and not took_off:
            prep_base_vels.append(abs(base_lin_vel[1]))
            prep_wheel_omegas.append(max(w_L, w_R))
            prep_ground_slip_y = abs(curr_wheel_y - initial_wheel_y)

        if step == 3:
            step3_vy = float(true_com_vel[0, 1].item())

        if step == 5:
            step5_vy = float(true_com_vel[0, 1].item())
            step5_pitch = pitch
            step5_roll = roll
            step5_w_L = w_L
            step5_w_R = w_R
            step5_tau_w_L = tau_w_L
            step5_tau_w_R = tau_w_R

        if step == 7:
            step7_vy = float(true_com_vel[0, 1].item())
            step7_tau_w_L = tau_w_L
            step7_tau_w_R = tau_w_R
            step7_tau_k_L = tau_k_L
            step7_tau_k_R = tau_k_R
            step7_tau_h_L = tau_h_L
            step7_tau_h_R = tau_h_R

        if step == 8:
            step8_vy = float(true_com_vel[0, 1].item())
            step8_tau_w_L = tau_w_L
            step8_tau_w_R = tau_w_R
            step8_tau_k_L = tau_k_L
            step8_tau_k_R = tau_k_R
            step8_tau_h_L = tau_h_L
            step8_tau_h_R = tau_h_R

        clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(env)
        clr_min_cm = clr_min[0].item() * 100.0
        clr_L_cm = clr_L[0].item() * 100.0
        clr_R_cm = clr_R[0].item() * 100.0

        is_air = bool(env.is_airborne[0].item())
        has_took = bool(env.has_taken_off[0].item())

        if is_air and has_took:
            max_clearance_cm = max(max_clearance_cm, clr_min_cm)
            max_clearance_L_cm = max(max_clearance_L_cm, clr_L_cm)
            max_clearance_R_cm = max(max_clearance_R_cm, clr_R_cm)
            airborne_pitches.append(pitch)
            airborne_rolls.append(roll)

        if is_air and has_took and not took_off:
            took_off = True
            takeoff_step = step
            takeoff_com_y = float(true_com_pos[0, 1].item())
            takeoff_com_vz = float(true_com_vel[0, 2].item())
            takeoff_com_vy = float(true_com_vel[0, 1].item())

        if took_off and (not is_air) and (not landed):
            landed = True
            touchdown_step = step
            touchdown_com_y = float(true_com_pos[0, 1].item())
            touchdown_pitch = pitch
            touchdown_roll = roll

        if landed and t_sim >= 1.0:
            recovery_vys.append(base_lin_vel[1])

        trajectory.append({
            "step": step,
            "time_s": round(t_sim, 3),
            "com_y": round(float(true_com_pos[0, 1].item()), 4),
            "com_vy": round(float(true_com_vel[0, 1].item()), 4),
            "com_vz": round(float(true_com_vel[0, 2].item()), 4),
            "wheel_y": round(float(curr_wheel_y), 4),
            "pitch_deg": round(pitch, 2),
            "roll_deg": round(roll, 2),
            "w_L_radps": round(w_L, 2),
            "w_R_radps": round(w_R, 2),
            "tau_w_L_nm": round(tau_w_L, 2),
            "tau_w_R_nm": round(tau_w_R, 2),
            "tau_k_L_nm": round(tau_k_L, 2),
            "tau_k_R_nm": round(tau_k_R, 2),
            "tau_h_L_nm": round(tau_h_L, 2),
            "tau_h_R_nm": round(tau_h_R, 2),
            "clr_min_cm": round(clr_min_cm, 2),
            "is_airborne": is_air
        })

        with torch.no_grad():
            actions = policy(obs)

        # DIAGNOSTIC INTERVENTION WINDOW: strictly step <= 5 (t <= 0.10s)
        if step <= 5 and brake_k > 0.0:
            act_mod = actions.clone()
            w_L_vel = env.dof_vel[0, env.left_wheel_idx].item()
            w_R_vel = env.dof_vel[0, env.right_wheel_idx].item()
            act_mod[0, env.left_wheel_idx] -= brake_k * w_L_vel
            act_mod[0, env.right_wheel_idx] -= brake_k * w_R_vel
            obs, _, _, _, _ = env.step(act_mod)
        else:
            obs, _, _, _, _ = env.step(actions)

        if env.reset_buf[0].item() != 0 and step < total_steps - 5:
            abnormal_resets += 1

    valid_jump = took_off and landed and (max_clearance_cm >= 10.0)
    if took_off and landed:
        flight_duration_s = (touchdown_step - takeoff_step) * env.dt
        com_forward_dist_cm = (touchdown_com_y - takeoff_com_y) * 100.0
        touchdown_vy = all_base_vys[touchdown_step]
        step_20s = min(touchdown_step + int(2.0 / env.dt), total_steps - 1)
        touchdown_plus_20s_vy = all_base_vys[step_20s]
    else:
        flight_duration_s = 0.0
        com_forward_dist_cm = 0.0
        touchdown_vy = 0.0
        touchdown_plus_20s_vy = 0.0

    mean_rec_vy = float(np.mean(recovery_vys)) if len(recovery_vys) > 0 else 0.0
    max_prep_base_vel = float(np.max(prep_base_vels)) if prep_base_vels else 0.0
    max_prep_wheel_omega = float(np.max(prep_wheel_omegas)) if prep_wheel_omegas else 0.0

    # Classification logic
    is_normal_jump = (com_forward_dist_cm >= 8.0) and (max_clearance_cm >= 20.0)
    is_short_jump = (com_forward_dist_cm >= 0.0) and (com_forward_dist_cm < 8.0)
    is_backward_jump = (com_forward_dist_cm < 0.0)
    is_low_clearance = (max_clearance_cm < 20.0)
    is_prep_slip = (prep_ground_slip_y >= 0.02) or (max_prep_wheel_omega >= 15.0)

    # Standard acceptance criteria (Level 3.1)
    c1_fwd_dist_ok = (com_forward_dist_cm >= 8.0)
    c2_clearance_ok = (max_clearance_cm >= 20.0)
    c3_prep_stationary_ok = (max_prep_base_vel < 0.20) and (prep_ground_slip_y < 0.02) and (max_prep_wheel_omega < 15.0)
    c4_recovery_ok = (abs(touchdown_pitch) <= 10.0) and (abs(touchdown_roll) <= 10.0) and (abs(mean_rec_vy) <= 0.03)
    c5_stability_ok = valid_jump and (abnormal_resets == 0)
    joint_success = c1_fwd_dist_ok and c2_clearance_ok and c3_prep_stationary_ok and c4_recovery_ok and c5_stability_ok

    delta_push_vy = takeoff_com_vy - step5_vy

    result = {
        "seed": int(seed),
        "mode": mode,
        "brake_k": float(brake_k),
        "joint_success": bool(joint_success),
        "valid_jump": bool(valid_jump),
        "abnormal_resets": int(abnormal_resets),

        # Behavioral Categories
        "is_normal_jump": bool(is_normal_jump),
        "is_short_jump": bool(is_short_jump),
        "is_backward_jump": bool(is_backward_jump),
        "is_low_clearance": bool(is_low_clearance),
        "is_prep_slip": bool(is_prep_slip),

        # Preparation Phase Metrics
        "step0_vy_mps": round(float(step0_vy), 4),
        "step3_vy_mps": round(float(step3_vy), 4),
        "step5_vy_mps": round(float(step5_vy), 4),
        "max_prep_base_vel_mps": round(float(max_prep_base_vel), 4),
        "prep_ground_slip_cm": round(float(prep_ground_slip_y * 100.0), 2),
        "max_prep_wheel_omega_radps": round(float(max_prep_wheel_omega), 2),
        "step0_pitch_deg": round(float(step0_pitch), 2),
        "step0_roll_deg": round(float(step0_roll), 2),
        "step5_pitch_deg": round(float(step5_pitch), 2),
        "step5_roll_deg": round(float(step5_roll), 2),
        "step5_w_L_radps": round(float(step5_w_L), 2),
        "step5_w_R_radps": round(float(step5_w_R), 2),
        "step5_tau_w_L_nm": round(float(step5_tau_w_L), 2),
        "step5_tau_w_R_nm": round(float(step5_tau_w_R), 2),

        # Push-Off Phase Metrics
        "step7_vy_mps": round(float(step7_vy), 4),
        "step8_vy_mps": round(float(step8_vy), 4),
        "takeoff_vy_mps": round(float(takeoff_com_vy), 4),
        "takeoff_vz_mps": round(float(takeoff_com_vz), 4),
        "delta_push_vy_mps": round(float(delta_push_vy), 4),
        "step7_tau_w_L_nm": round(float(step7_tau_w_L), 2),
        "step7_tau_w_R_nm": round(float(step7_tau_w_R), 2),
        "step7_tau_k_L_nm": round(float(step7_tau_k_L), 2),
        "step7_tau_k_R_nm": round(float(step7_tau_k_R), 2),
        "step7_tau_h_L_nm": round(float(step7_tau_h_L), 2),
        "step7_tau_h_R_nm": round(float(step7_tau_h_R), 2),
        "step8_tau_w_L_nm": round(float(step8_tau_w_L), 2),
        "step8_tau_w_R_nm": round(float(step8_tau_w_R), 2),
        "step8_tau_k_L_nm": round(float(step8_tau_k_L), 2),
        "step8_tau_k_R_nm": round(float(step8_tau_k_R), 2),
        "step8_tau_h_L_nm": round(float(step8_tau_h_L), 2),
        "step8_tau_h_R_nm": round(float(step8_tau_h_R), 2),

        # Flight Phase Metrics
        "max_clearance_cm": round(float(max_clearance_cm), 2),
        "max_clearance_L_cm": round(float(max_clearance_L_cm), 2),
        "max_clearance_R_cm": round(float(max_clearance_R_cm), 2),
        "com_forward_dist_cm": round(float(com_forward_dist_cm), 2),
        "flight_duration_s": round(float(flight_duration_s), 3),
        "flight_pitch_max_deg": round(float(np.max(np.abs(airborne_pitches))), 2) if airborne_pitches else 0.0,
        "flight_roll_max_deg": round(float(np.max(np.abs(airborne_rolls))), 2) if airborne_rolls else 0.0,

        # Landing & Recovery Metrics
        "touchdown_pitch_deg": round(float(touchdown_pitch), 2),
        "touchdown_roll_deg": round(float(touchdown_roll), 2),
        "touchdown_vy_mps": round(float(touchdown_vy), 4),
        "touchdown_plus_20s_vy_mps": round(float(touchdown_plus_20s_vy), 4),
        "recovery_vy_mps": round(float(mean_rec_vy), 4),

        # Event Steps and Trajectory
        "takeoff_step": int(takeoff_step) if takeoff_step is not None else None,
        "touchdown_step": int(touchdown_step) if touchdown_step is not None else None,
        "trajectory": trajectory
    }

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    status_icon = "✅ PASS" if joint_success else "❌ FAIL"
    print(f"[{mode}] Seed {seed:3d}: {status_icon} | Clr={result['max_clearance_cm']:5.2f}cm | FwdDist={result['com_forward_dist_cm']:+6.2f}cm | S5_Vy={result['step5_vy_mps']:+6.3f}m/s | TO_Vy={result['takeoff_vy_mps']:+6.3f}m/s | DeltaVy={result['delta_push_vy_mps']:+6.3f}m/s | Vz={result['takeoff_vz_mps']:+5.3f}m/s | TDRoll={result['touchdown_roll_deg']:+5.2f}°")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["Mode_A_control", "Mode_B_weak_brake", "Mode_C_strong_brake"])
    parser.add_argument("--brake_k", type=float, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    run_diagnostic(args.seed, args.mode, args.brake_k, args.out)
