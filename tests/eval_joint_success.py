#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Phase 2.1 Joint Task Success Evaluator
# Evaluates 5 simultaneous constraints across 5 seeds:
# 1. Authentic flight CoM forward displacement >= 5.0 cm
# 2. Dual-wheel maximum mutual clearance >= 20.0 cm
# 3. Preparation phase stationary compliance (base vel, com vel, wheel vel, slip)
# 4. Touchdown attitude, yaw drift, and post-landing velocity recovery
# 5. Zero abnormal resets
# ==============================================================================

import os
import sys
import json
import argparse
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz
import torch

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.envs.myrobot.myrobot_dual_env import compute_exact_cylinder_clearance


def eval_single_seed(model_path, seed, clip_actions=100.0, task_name="myrobot_forward_jump_static", min_dist=8.0, max_rec_vy=0.03):
    sys.argv = ["eval_joint", "--task", task_name, "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = seed

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.normalization.clip_actions = clip_actions
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
    ckpt = torch.load(model_path, map_location=env.device)
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    policy = ac.act_inference

    obs, _ = env.reset()
    total_steps = int(2.8 / env.dt)

    takeoff_step = None
    touchdown_step = None
    max_clearance_cm = 0.0
    max_clearance_L_cm = 0.0
    max_clearance_R_cm = 0.0
    takeoff_com_y = 0.0
    touchdown_com_y = 0.0
    takeoff_com_vy = 0.0
    takeoff_com_vz = 0.0
    pre_push_vy = 0.0
    takeoff_pitch = 0.0
    takeoff_yaw_rate = 0.0
    takeoff_fn_L = 0.0
    takeoff_fn_R = 0.0
    touchdown_pitch = 0.0
    touchdown_roll = 0.0
    touchdown_yaw = 0.0
    initial_yaw = 0.0
    final_yaw = 0.0

    peak_hip_tau = 0.0
    peak_knee_tau = 0.0
    peak_wheel_tau = 0.0

    # Constraint 3 buffers: Prep phase metrics (phi in [0.0, 0.05], step <= 5, t <= 0.10s)
    prep_base_vels = []
    prep_com_vels = []
    prep_wheel_omegas = []
    prep_ground_slip_y = 0.0
    initial_wheel_y = None

    airborne_pitches = []
    airborne_rolls = []
    airborne_yaws = []

    all_base_vys = []
    recovery_vys = []
    recovery_vxs = []
    abnormal_resets = 0

    took_off = False
    landed = False

    for step in range(total_steps):
        t_sim = step * env.dt
        phase = env.jump_phase[0].item()

        true_com_pos, true_com_vel = env._compute_true_com()
        root_pos = env.root_states[0, 0:3].cpu().numpy()
        root_vel = env.root_states[0, 7:10].cpu().numpy()
        base_lin_vel = env.base_lin_vel[0].cpu().numpy()
        all_base_vys.append(float(base_lin_vel[1]))

        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
        yaw = torch.atan2(torch.sin(y_raw[0]), torch.cos(y_raw[0])).item() * 180.0 / np.pi

        if step == 0:
            initial_yaw = yaw
            initial_wheel_y = (env.rb_states[0, env.left_wheel_body, 1].item() + env.rb_states[0, env.right_wheel_body, 1].item()) / 2.0
        final_yaw = yaw

        # Prep phase monitoring (phi in [0.00, 0.05], t <= 0.10s, step <= 5)
        if step <= 5 and not took_off:
            prep_base_vels.append(abs(base_lin_vel[1]))
            prep_com_vels.append(abs(true_com_vel[0, 1].item()))
            w_L = abs(env.dof_vel[0, env.left_wheel_idx].item())
            w_R = abs(env.dof_vel[0, env.right_wheel_idx].item())
            prep_wheel_omegas.append(max(w_L, w_R))

            curr_wheel_y = (env.rb_states[0, env.left_wheel_body, 1].item() + env.rb_states[0, env.right_wheel_body, 1].item()) / 2.0
            prep_ground_slip_y = abs(curr_wheel_y - initial_wheel_y)

        if step == 5:
            pre_push_vy = true_com_vel[0, 1].item()

        clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(env)
        clr_min_cm = clr_min[0].item() * 100.0
        clr_L_cm = clr_L[0].item() * 100.0
        clr_R_cm = clr_R[0].item() * 100.0

        is_air = bool(env.is_airborne[0].item())
        has_took = bool(env.has_taken_off[0].item())

        tau_hip = max(abs(env.torques[0, env.left_hip_idx].item()), abs(env.torques[0, env.right_hip_idx].item()))
        tau_knee = max(abs(env.torques[0, env.left_knee_idx].item()), abs(env.torques[0, env.right_knee_idx].item()))
        tau_wheel = max(abs(env.torques[0, env.left_wheel_idx].item()), abs(env.torques[0, env.right_wheel_idx].item()))

        peak_hip_tau = max(peak_hip_tau, tau_hip)
        peak_knee_tau = max(peak_knee_tau, tau_knee)
        peak_wheel_tau = max(peak_wheel_tau, tau_wheel)

        if is_air and has_took:
            max_clearance_cm = max(max_clearance_cm, clr_min_cm)
            max_clearance_L_cm = max(max_clearance_L_cm, clr_L_cm)
            max_clearance_R_cm = max(max_clearance_R_cm, clr_R_cm)
            airborne_pitches.append(pitch)
            airborne_rolls.append(roll)
            airborne_yaws.append(yaw)

        if is_air and has_took and not took_off:
            took_off = True
            takeoff_step = step
            takeoff_com_y = float(true_com_pos[0, 1].item())
            takeoff_com_vz = float(true_com_vel[0, 2].item())
            takeoff_com_vy = float(true_com_vel[0, 1].item())
            takeoff_pitch = pitch
            takeoff_yaw_rate = float(abs(env.base_ang_vel[0, 2].item()))
            takeoff_fn_L = float(env.contact_forces[0, env.left_wheel_body, 2].item())
            takeoff_fn_R = float(env.contact_forces[0, env.right_wheel_body, 2].item())

        if took_off and (not is_air) and (not landed):
            landed = True
            touchdown_step = step
            touchdown_com_y = float(true_com_pos[0, 1].item())
            touchdown_pitch = pitch
            touchdown_roll = roll
            touchdown_yaw = yaw

        if landed and t_sim >= 1.0:
            recovery_vys.append(base_lin_vel[1])
            recovery_vxs.append(base_lin_vel[0])

        with torch.no_grad():
            actions = policy(obs)
        obs, _, rewards, dones, infos = env.step(actions)

        if env.reset_buf[0].item() != 0 and step < total_steps - 5:
            abnormal_resets += 1

    valid_jump = took_off and landed and (max_clearance_cm >= 10.0)
    if took_off and landed:
        flight_duration_s = (touchdown_step - takeoff_step) * env.dt
        flight_steps = touchdown_step - takeoff_step
        com_forward_dist_cm = (touchdown_com_y - takeoff_com_y) * 100.0

        touchdown_vy = all_base_vys[touchdown_step]
        step_03s = min(touchdown_step + int(0.3 / env.dt), total_steps - 1)
        touchdown_plus_03s_vy = all_base_vys[step_03s]
        step_20s = min(touchdown_step + int(2.0 / env.dt), total_steps - 1)
        touchdown_plus_20s_vy = all_base_vys[step_20s]
    else:
        flight_duration_s = 0.0
        flight_steps = 0
        com_forward_dist_cm = 0.0
        touchdown_vy = 0.0
        touchdown_plus_03s_vy = 0.0
        touchdown_plus_20s_vy = 0.0

    mean_rec_vy = float(np.mean(recovery_vys)) if len(recovery_vys) > 0 else 0.0
    mean_rec_vx = float(np.mean(recovery_vxs)) if len(recovery_vxs) > 0 else 0.0
    yaw_drift = abs(final_yaw - initial_yaw)

    max_prep_base_vel = float(np.max(prep_base_vels)) if prep_base_vels else 0.0
    max_prep_com_vel = float(np.max(prep_com_vels)) if prep_com_vels else 0.0
    max_prep_wheel_omega = float(np.max(prep_wheel_omegas)) if prep_wheel_omegas else 0.0

    # EVALUATE THE 5 JOINT CONSTRAINTS
    # 1. Flight displacement >= min_dist (default 8.0 cm for Level 3.1)
    c1_fwd_dist_ok = (com_forward_dist_cm >= min_dist)
    # 2. Max mutual clearance >= 20.0 cm
    c2_clearance_ok = (max_clearance_cm >= 20.0)
    # 3. Preparation stationary: base vel < 0.20 m/s, com vel < 0.20 m/s, wheel omega < 15.0 rad/s, contact slip < 0.02 m
    c3_prep_stationary_ok = (max_prep_base_vel < 0.20) and (max_prep_com_vel < 0.20) and (max_prep_wheel_omega < 15.0) and (prep_ground_slip_y < 0.02)
    # 4. Touchdown attitude & recovery: |pitch| <= 10 deg, |roll| <= 10 deg, yaw_drift <= 10 deg, rec_vy <= max_rec_vy
    c4_recovery_ok = (abs(touchdown_pitch) <= 10.0) and (abs(touchdown_roll) <= 10.0) and (yaw_drift <= 10.0) and (abs(mean_rec_vy) <= max_rec_vy)
    # 5. Stability: valid jump & zero abnormal resets
    c5_stability_ok = valid_jump and (abnormal_resets == 0)

    joint_success = c1_fwd_dist_ok and c2_clearance_ok and c3_prep_stationary_ok and c4_recovery_ok and c5_stability_ok

    failure_reasons = []
    if not c1_fwd_dist_ok: failure_reasons.append(f"FwdDist {com_forward_dist_cm:.1f}cm < {min_dist:.1f}cm")
    if not c2_clearance_ok: failure_reasons.append(f"Clearance {max_clearance_cm:.1f}cm < 20cm")
    if not c3_prep_stationary_ok: 
        c3_sub = []
        if max_prep_base_vel >= 0.20: c3_sub.append(f"base_v={max_prep_base_vel:.2f}>0.2")
        if max_prep_com_vel >= 0.20: c3_sub.append(f"com_v={max_prep_com_vel:.2f}>0.2")
        if max_prep_wheel_omega >= 15.0: c3_sub.append(f"wheel_w={max_prep_wheel_omega:.1f}>15.0")
        if prep_ground_slip_y >= 0.02: c3_sub.append(f"slip={prep_ground_slip_y*100:.1f}cm>2cm")
        failure_reasons.append(f"PrepFail({', '.join(c3_sub)})")
    if not c4_recovery_ok: failure_reasons.append(f"Attitude/Recovery error (|RecVy|={abs(mean_rec_vy):.3f}>{max_rec_vy:.3f} or tilt)")
    if not c5_stability_ok: failure_reasons.append("Abnormal reset or invalid jump")

    return {
        "seed": seed,
        "valid_jump": valid_jump,
        "abnormal_resets": abnormal_resets,
        "c1_fwd_dist_ok": c1_fwd_dist_ok,
        "c2_clearance_ok": c2_clearance_ok,
        "c3_prep_stationary_ok": c3_prep_stationary_ok,
        "c4_recovery_ok": c4_recovery_ok,
        "c5_stability_ok": c5_stability_ok,
        "joint_success": joint_success,
        "failure_reasons": failure_reasons,
        "max_clearance_cm": round(max_clearance_cm, 2),
        "max_clearance_L_cm": round(max_clearance_L_cm, 2),
        "max_clearance_R_cm": round(max_clearance_R_cm, 2),
        "takeoff_com_y_m": round(takeoff_com_y, 4),
        "touchdown_com_y_m": round(touchdown_com_y, 4),
        "max_airborne_pitch_deg": round(float(np.max(np.abs(airborne_pitches))), 2) if airborne_pitches else 0.0,
        "max_airborne_roll_deg": round(float(np.max(np.abs(airborne_rolls))), 2) if airborne_rolls else 0.0,
        "pre_jump_rolling_flag": bool(max_prep_base_vel > 0.05),
        "hip_tau_saturated": bool(peak_hip_tau >= 139.9),
        "knee_tau_saturated": bool(peak_knee_tau >= 119.9),
        "wheel_tau_saturated": bool(peak_wheel_tau >= 11.9),
        "com_forward_dist_cm": round(com_forward_dist_cm, 2),
        "flight_duration_s": round(flight_duration_s, 3),
        "flight_steps": flight_steps,
        "takeoff_com_vy_mps": round(takeoff_com_vy, 3),
        "takeoff_com_vz_mps": round(takeoff_com_vz, 3),
        "pre_push_vy_mps": round(pre_push_vy, 3),
        "takeoff_pitch_deg": round(takeoff_pitch, 2),
        "takeoff_yaw_rate_radps": round(takeoff_yaw_rate, 3),
        "takeoff_fn_L_N": round(takeoff_fn_L, 2),
        "takeoff_fn_R_N": round(takeoff_fn_R, 2),
        "max_prep_base_vel_mps": round(max_prep_base_vel, 3),
        "max_prep_com_vel_mps": round(max_prep_com_vel, 3),
        "max_prep_wheel_omega_radps": round(max_prep_wheel_omega, 3),
        "prep_ground_slip_cm": round(prep_ground_slip_y * 100.0, 2),
        "touchdown_pitch_deg": round(touchdown_pitch, 2),
        "touchdown_roll_deg": round(touchdown_roll, 2),
        "touchdown_yaw_deg": round(touchdown_yaw, 2),
        "touchdown_vy_mps": round(touchdown_vy, 3),
        "touchdown_plus_03s_vy_mps": round(touchdown_plus_03s_vy, 3),
        "touchdown_plus_20s_vy_mps": round(touchdown_plus_20s_vy, 3),
        "yaw_drift_deg": round(yaw_drift, 2),
        "recovery_vy_mps": round(mean_rec_vy, 3),
        "recovery_vx_mps": round(mean_rec_vx, 3),
        "peak_hip_tau_nm": round(peak_hip_tau, 2),
        "peak_knee_tau_nm": round(peak_knee_tau, 2),
        "peak_wheel_tau_nm": round(peak_wheel_tau, 2)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--clip", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_dist", type=float, default=8.0)
    parser.add_argument("--max_rec_vy", type=float, default=0.03)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    res = eval_single_seed(args.model, args.seed, clip_actions=args.clip, min_dist=args.min_dist, max_rec_vy=args.max_rec_vy)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    status_str = "✅ SUCCESS" if res["joint_success"] else f"❌ FAIL ({', '.join(res['failure_reasons'])})"
    print(f"Seed {args.seed}: Joint={status_str} | Clr={res['max_clearance_cm']}cm | FwdDist={res['com_forward_dist_cm']}cm | Takeoff Vz={res['takeoff_com_vz_mps']}m/s, Vy={res['takeoff_com_vy_mps']}m/s | RecVy={res['recovery_vy_mps']}m/s")


if __name__ == "__main__":
    main()
