#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Phase 3.2-R Single Seed Evaluator
# Supports:
#   - Mode_A_control: Original reset (z=0.4000 drop + self.step(zeros))
#   - Mode_B_isolated_lin_vel: Isolates horizontal lin vel zeroing (vx=vy=0 only)
#   - Mode_D_counterfactual: Complete zeroing of all linear, angular, and joint vels
#   - Mode_E_physical_consistent: Physically consistent ground contact stance init
# ==============================================================================

import os
import sys
import json
import argparse
import math

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse
import torch
import numpy as np

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.envs.myrobot.myrobot_dual_env import compute_exact_cylinder_clearance


def get_state_dict(env):
    """Extract full kinematic state."""
    r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
    roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
    pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
    yaw = torch.atan2(torch.sin(y_raw[0]), torch.cos(y_raw[0])).item() * 180.0 / np.pi

    base_lin_vel = env.base_lin_vel[0].cpu().numpy()
    base_ang_vel = env.base_ang_vel[0].cpu().numpy()
    root_lin_vel = env.root_states[0, 7:10].cpu().numpy()
    root_ang_vel = env.root_states[0, 10:13].cpu().numpy()
    dof_vel = env.dof_vel[0].cpu().numpy()

    _, true_com_vel = env._compute_true_com()
    com_vel = true_com_vel[0].cpu().numpy()

    w_L = float(dof_vel[env.left_wheel_idx])
    w_R = float(dof_vel[env.right_wheel_idx])

    pos_L = env.rb_states[0, env.left_wheel_body, 2].item() - env.wheel_radius
    pos_R = env.rb_states[0, env.right_wheel_body, 2].item() - env.wheel_radius
    fn_L = float(env.contact_forces[0, env.left_wheel_body, 2].item())
    fn_R = float(env.contact_forces[0, env.right_wheel_body, 2].item())

    return {
        "base_vx": float(base_lin_vel[0]),
        "base_vy": float(base_lin_vel[1]),
        "base_vz": float(base_lin_vel[2]),
        "com_vx": float(com_vel[0]),
        "com_vy": float(com_vel[1]),
        "com_vz": float(com_vel[2]),
        "root_vx": float(root_lin_vel[0]),
        "root_vy": float(root_lin_vel[1]),
        "root_vz": float(root_lin_vel[2]),
        "base_pitch_deg": float(pitch),
        "base_roll_deg": float(roll),
        "base_yaw_deg": float(yaw),
        "base_wx": float(base_ang_vel[0]),
        "base_wy": float(base_ang_vel[1]),
        "base_wz": float(base_ang_vel[2]),
        "wheel_w_L_radps": float(w_L),
        "wheel_w_R_radps": float(w_R),
        "fn_L_N": fn_L,
        "fn_R_N": fn_R,
        "clr_L_mm": pos_L * 1000.0,
        "clr_R_mm": pos_R * 1000.0,
        "dof_vel": [float(v) for v in dof_vel]
    }


def run_evaluation(seed, mode, out_json):
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["eval_32r", "--task", task_name, "--num_envs", "1"]
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

    # --------------------------------------------------------------------------
    # 1. State A: Reset initialization complete, before self.step(zeros)
    # --------------------------------------------------------------------------
    env.reset_idx(torch.arange(env.num_envs, device=env.device))

    if mode == "Mode_E_physical_consistent":
        # Physically consistent ground contact placement:
        # Link kinematics shows nominal distance from base origin to wheel bottom is 0.397739 m.
        # Placed at z=0.397739 so wheels are precisely on the ground without drop shock.
        env.root_states[:, 2] = 0.397739
        env.root_states[:, 7:13] = 0.0
        env.dof_vel[:, :] = 0.0

    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env.rb_states[:, :, 7:13] = 0.0
    state_A = get_state_dict(env)

    # --------------------------------------------------------------------------
    # 2. Reset internal pre-execution step: self.step(zeros) (0.02s simulation)
    # --------------------------------------------------------------------------
    obs, privileged_obs, _, _, _ = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    state_B = get_state_dict(env)

    # --------------------------------------------------------------------------
    # 3. State C: Right before formal policy control starts (Step 0)
    # --------------------------------------------------------------------------
    if mode == "Mode_D_counterfactual":
        # Full counterfactual zeroing of all velocities
        env.root_states[:, 7:13] = 0.0
        env.dof_vel[:, :] = 0.0
        env.rb_states[:, :, 7:13] = 0.0
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        env.base_quat[:] = env.root_states[:, 3:7]
        env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
        env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
        env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
        env.jump_timer[:] = 0.0
        env.jump_phase[:] = 0.0
        env.episode_length_buf[:] = 0
        env.filtered_actions[:] = 0.0
        env.compute_observations()
        obs = env.obs_buf.clone()

    elif mode == "Mode_B_isolated_lin_vel":
        # Isolates ONLY horizontal linear velocity zeroing: root vx=0, vy=0
        # Leaves root vz, angular velocities wx, wy, wz, and joint velocities untouched
        env.root_states[:, 7:9] = 0.0
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        env.base_quat[:] = env.root_states[:, 3:7]
        env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
        env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
        env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
        env.jump_timer[:] = 0.0
        env.jump_phase[:] = 0.0
        env.episode_length_buf[:] = 0
        env.filtered_actions[:] = 0.0
        env.compute_observations()
        obs = env.obs_buf.clone()

    elif mode == "Mode_E_physical_consistent":
        # Latch jump timer to phi=0 and clear episode step buffer, retaining physical states
        env.jump_timer[:] = 0.0
        env.jump_phase[:] = 0.0
        env.episode_length_buf[:] = 0
        env.filtered_actions[:] = 0.0
        env.compute_observations()
        obs = env.obs_buf.clone()

    state_C = get_state_dict(env)

    # --------------------------------------------------------------------------
    # 4. Full Episode Rollout (140 steps @ 50Hz = 2.8s)
    # --------------------------------------------------------------------------
    total_steps = int(2.8 / env.dt)

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
    takeoff_base_vy = 0.0
    takeoff_base_vz = 0.0

    step0_base_vy = 0.0
    step0_com_vy = 0.0
    step5_base_vy = 0.0
    step5_com_vy = 0.0

    touchdown_pitch = 0.0
    touchdown_roll = 0.0
    touchdown_yaw = 0.0
    initial_yaw = state_C["base_yaw_deg"]
    final_yaw = 0.0

    peak_hip_tau = 0.0
    peak_knee_tau = 0.0
    peak_wheel_tau = 0.0

    prep_base_vels = []
    prep_com_vels = []
    prep_wheel_omegas = []
    prep_ground_slip_y = 0.0
    initial_wheel_y = None

    all_base_vys = []
    all_com_vys = []
    recovery_vys = []
    abnormal_resets = 0
    trajectory = []

    for step in range(total_steps):
        t_sim = step * env.dt

        true_com_pos, true_com_vel = env._compute_true_com()
        base_lin_vel = env.base_lin_vel[0].cpu().numpy()
        com_vel = true_com_vel[0].cpu().numpy()

        all_base_vys.append(float(base_lin_vel[1]))
        all_com_vys.append(float(com_vel[1]))

        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
        yaw = torch.atan2(torch.sin(y_raw[0]), torch.cos(y_raw[0])).item() * 180.0 / np.pi
        final_yaw = yaw

        curr_wheel_y = (env.rb_states[0, env.left_wheel_body, 1].item() + env.rb_states[0, env.right_wheel_body, 1].item()) / 2.0
        if step == 0:
            step0_base_vy = float(base_lin_vel[1])
            step0_com_vy = float(com_vel[1])
            initial_wheel_y = curr_wheel_y

        w_L = abs(env.dof_vel[0, env.left_wheel_idx].item())
        w_R = abs(env.dof_vel[0, env.right_wheel_idx].item())

        tau_h = max(abs(env.torques[0, env.left_hip_idx].item()), abs(env.torques[0, env.right_hip_idx].item()))
        tau_k = max(abs(env.torques[0, env.left_knee_idx].item()), abs(env.torques[0, env.right_knee_idx].item()))
        tau_w = max(abs(env.torques[0, env.left_wheel_idx].item()), abs(env.torques[0, env.right_wheel_idx].item()))

        peak_hip_tau = max(peak_hip_tau, tau_h)
        peak_knee_tau = max(peak_knee_tau, tau_k)
        peak_wheel_tau = max(peak_wheel_tau, tau_w)

        # Prep phase tracking (step <= 5, t <= 0.10s)
        if step <= 5 and not took_off:
            prep_base_vels.append(abs(base_lin_vel[1]))
            prep_com_vels.append(abs(com_vel[1]))
            prep_wheel_omegas.append(max(w_L, w_R))
            prep_ground_slip_y = abs(curr_wheel_y - initial_wheel_y)

        if step == 5:
            step5_base_vy = float(base_lin_vel[1])
            step5_com_vy = float(com_vel[1])

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

        if is_air and has_took and not took_off:
            took_off = True
            takeoff_step = step
            takeoff_com_y = float(true_com_pos[0, 1].item())
            takeoff_com_vz = float(com_vel[2])
            takeoff_com_vy = float(com_vel[1])
            takeoff_base_vy = float(base_lin_vel[1])
            takeoff_base_vz = float(base_lin_vel[2])

        if took_off and (not is_air) and (not landed):
            landed = True
            touchdown_step = step
            touchdown_com_y = float(true_com_pos[0, 1].item())
            touchdown_pitch = pitch
            touchdown_roll = roll
            touchdown_yaw = yaw

        if landed and t_sim >= 1.0:
            recovery_vys.append(base_lin_vel[1])

        trajectory.append({
            "step": step,
            "time_s": round(t_sim, 3),
            "base_vy": round(float(base_lin_vel[1]), 4),
            "base_vz": round(float(base_lin_vel[2]), 4),
            "com_y": round(float(true_com_pos[0, 1].item()), 4),
            "com_vy": round(float(com_vel[1]), 4),
            "com_vz": round(float(com_vel[2]), 4),
            "pitch_deg": round(pitch, 2),
            "roll_deg": round(roll, 2),
            "clr_min_cm": round(clr_min_cm, 2),
            "is_airborne": is_air
        })

        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _, _ = env.step(actions)

        if env.reset_buf[0].item() != 0 and step < total_steps - 5:
            abnormal_resets += 1

    valid_jump = took_off and landed and (max_clearance_cm >= 10.0)
    if took_off and landed:
        flight_duration_s = (touchdown_step - takeoff_step) * env.dt
        com_forward_dist_cm = (touchdown_com_y - takeoff_com_y) * 100.0
    else:
        flight_duration_s = 0.0
        com_forward_dist_cm = 0.0

    pushoff_delta_com_vy = takeoff_com_vy - step5_com_vy
    pushoff_delta_base_vy = takeoff_base_vy - step5_base_vy
    post_recovery_vy = float(np.mean(recovery_vys[-10:])) if len(recovery_vys) >= 10 else 0.0
    yaw_drift = abs(final_yaw - initial_yaw)

    max_prep_base_vel = float(np.max(prep_base_vels)) if prep_base_vels else 0.0
    max_prep_com_vel = float(np.max(prep_com_vels)) if prep_com_vels else 0.0
    max_prep_wheel_omega = float(np.max(prep_wheel_omegas)) if prep_wheel_omegas else 0.0

    # Strict Level 3.1 Joint Criteria (identical across all modes):
    # 1. Forward displacement >= 8.0 cm
    c1_fwd_dist_ok = bool(com_forward_dist_cm >= 8.0)
    # 2. Maximum clearance >= 20.0 cm
    c2_clearance_ok = bool(max_clearance_cm >= 20.0)
    # 3. Touchdown attitude & recovery: |roll| <= 10 deg, |pitch| <= 10 deg, yaw_drift <= 10 deg, rec_vy <= 0.03 m/s
    c3_touchdown_attitude_ok = bool(abs(touchdown_roll) <= 10.0 and abs(touchdown_pitch) <= 10.0 and yaw_drift <= 10.0)
    c4_recovery_ok = bool(abs(post_recovery_vy) <= 0.03)
    # 4. Preparation phase compliance: base_v < 0.20, com_v < 0.20, wheel_w < 15.0 rad/s, slip < 0.02 m
    c5_prep_stationary_ok = bool((max_prep_base_vel < 0.20) and (max_prep_com_vel < 0.20) and (max_prep_wheel_omega < 15.0) and (prep_ground_slip_y < 0.02))
    # 5. Stability: zero abnormal resets
    c6_no_abnormal_reset = bool(abnormal_resets == 0)

    # Full joint success (Level 3.1)
    joint_success_level3_1 = (
        c1_fwd_dist_ok and
        c2_clearance_ok and
        c3_touchdown_attitude_ok and
        c4_recovery_ok and
        c5_prep_stationary_ok and
        c6_no_abnormal_reset and
        valid_jump
    )

    # Relaxed Level 2.1 Criteria (5cm dist, 12 deg roll, 15 deg pitch, 0.10 rec_vy)
    joint_success_relaxed = (
        max_clearance_cm >= 20.0 and
        com_forward_dist_cm >= 5.0 and
        abs(touchdown_roll) <= 12.0 and
        abs(touchdown_pitch) <= 15.0 and
        abs(post_recovery_vy) <= 0.10 and
        abnormal_resets == 0
    )

    if com_forward_dist_cm >= 8.0 and max_clearance_cm >= 20.0:
        jump_class = "Class_A_normal_forward"
    elif com_forward_dist_cm >= 0.0:
        jump_class = "Class_B_short_jump"
    else:
        jump_class = "Class_C_backward_jump"

    result = {
        "seed": seed,
        "mode": mode,
        "state_A": state_A,
        "state_B": state_B,
        "state_C": state_C,
        "step0_base_vy_mps": step0_base_vy,
        "step0_com_vy_mps": step0_com_vy,
        "step5_base_vy_mps": step5_base_vy,
        "step5_com_vy_mps": step5_com_vy,
        "takeoff_step": takeoff_step,
        "takeoff_base_vy_mps": takeoff_base_vy,
        "takeoff_com_vy_mps": takeoff_com_vy,
        "takeoff_base_vz_mps": takeoff_base_vz,
        "takeoff_com_vz_mps": takeoff_com_vz,
        "pushoff_delta_base_vy_mps": pushoff_delta_base_vy,
        "pushoff_delta_com_vy_mps": pushoff_delta_com_vy,
        "flight_duration_s": flight_duration_s,
        "com_forward_dist_cm": com_forward_dist_cm,
        "max_clearance_cm": max_clearance_cm,
        "touchdown_pitch_deg": touchdown_pitch,
        "touchdown_roll_deg": touchdown_roll,
        "touchdown_yaw_deg": touchdown_yaw,
        "yaw_drift_deg": yaw_drift,
        "post_recovery_vy_mps": post_recovery_vy,
        "prep_ground_slip_y_m": prep_ground_slip_y,
        "max_prep_base_vel_mps": max_prep_base_vel,
        "max_prep_com_vel_mps": max_prep_com_vel,
        "max_prep_wheel_omega_radps": max_prep_wheel_omega,
        "peak_hip_tau_nm": peak_hip_tau,
        "peak_knee_tau_nm": peak_knee_tau,
        "peak_wheel_tau_nm": peak_wheel_tau,
        "abnormal_resets": abnormal_resets,
        "jump_class": jump_class,
        "c1_fwd_dist_ok": c1_fwd_dist_ok,
        "c2_clearance_ok": c2_clearance_ok,
        "c3_touchdown_attitude_ok": c3_touchdown_attitude_ok,
        "c4_recovery_ok": c4_recovery_ok,
        "c5_prep_stationary_ok": c5_prep_stationary_ok,
        "c6_no_abnormal_reset": c6_no_abnormal_reset,
        "joint_success_level3_1": joint_success_level3_1,
        "joint_success_relaxed": joint_success_relaxed,
        "trajectory": trajectory
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3.2-R Single Seed Evaluator")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--mode", type=str, required=True, choices=["Mode_A_control", "Mode_B_isolated_lin_vel", "Mode_D_counterfactual", "Mode_E_physical_consistent"])
    parser.add_argument("--out", type=str, required=True, help="Output JSON path")
    args = parser.parse_args()

    run_evaluation(args.seed, args.mode, args.out)
