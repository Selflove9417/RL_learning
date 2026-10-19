#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Phase 3.2 Static Start Purity Audit Single-Seed Evaluator
# Audits robot physical state at:
#   - State A: reset initialization complete, before self.step(zeros)
#   - State B: after internal pre-execution self.step(zeros)
#   - State C: right before formal policy control starts (Step 0)
# Supports:
#   - Mode A: Control (original Champion + standard reset)
#   - Mode D: Clean-Start Diagnostic (counterfactual true zero-velocity start)
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
    """Extract full kinematic state for State A, B, or C."""
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
        "dof_vel": [float(v) for v in dof_vel],
        "dof_names": env.dof_names
    }


def run_evaluation(seed, mode, out_json):
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

    # --------------------------------------------------------------------------
    # 1. State A: Reset initialization complete, before self.step(zeros)
    # --------------------------------------------------------------------------
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env.rb_states[:, :, 7:13] = 0.0  # Align rigid body velocity buffer with reset zero velocities
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

    # Detailed contact metrics during the 0.02s reset step
    fn_L = float(env.contact_forces[0, env.left_wheel_body, 2].item())
    fn_R = float(env.contact_forces[0, env.right_wheel_body, 2].item())
    f_fric_x_L = float(env.contact_forces[0, env.left_wheel_body, 0].item())
    f_fric_y_L = float(env.contact_forces[0, env.left_wheel_body, 1].item())
    f_fric_x_R = float(env.contact_forces[0, env.right_wheel_body, 0].item())
    f_fric_y_R = float(env.contact_forces[0, env.right_wheel_body, 1].item())
    fric_L = math.sqrt(f_fric_x_L**2 + f_fric_y_L**2)
    fric_R = math.sqrt(f_fric_x_R**2 + f_fric_y_R**2)

    contact_process_0_02s = {
        "delta_base_vy": float(state_B["base_vy"] - state_A["base_vy"]),
        "delta_com_vy": float(state_B["com_vy"] - state_A["com_vy"]),
        "left_wheel_contact": bool(fn_L > 0.1),
        "right_wheel_contact": bool(fn_R > 0.1),
        "fn_L_N": float(fn_L),
        "fn_R_N": float(fn_R),
        "friction_L_N": float(fric_L),
        "friction_R_N": float(fric_R),
        "friction_y_L_N": float(f_fric_y_L),
        "friction_y_R_N": float(f_fric_y_R),
        "w_L_radps": float(state_B["wheel_w_L_radps"]),
        "w_R_radps": float(state_B["wheel_w_R_radps"]),
        "base_pitch_deg": float(state_B["base_pitch_deg"]),
        "base_roll_deg": float(state_B["base_roll_deg"]),
        "base_vy": float(state_B["base_vy"]),
        "com_vy": float(state_B["com_vy"])
    }

    # --------------------------------------------------------------------------
    # 3. State C: Right before formal policy control starts (Step 0)
    # --------------------------------------------------------------------------
    if mode == "Mode_D_clean_start":
        # Mode D: Counterfactual Clean-Start Diagnostic
        # Strictly zero all linear, angular, and joint velocities before policy begins
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
    step3_base_vy = 0.0
    step3_com_vy = 0.0
    step5_base_vy = 0.0
    step5_com_vy = 0.0
    step7_base_vy = 0.0
    step7_com_vy = 0.0
    step8_base_vy = 0.0
    step8_com_vy = 0.0

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
    touchdown_base_vy = 0.0
    touchdown_com_vy = 0.0

    airborne_pitches = []
    airborne_rolls = []

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

        curr_wheel_y = (env.rb_states[0, env.left_wheel_body, 1].item() + env.rb_states[0, env.right_wheel_body, 1].item()) / 2.0
        if step == 0:
            step0_base_vy = float(base_lin_vel[1])
            step0_com_vy = float(com_vel[1])
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

        if step == 3:
            step3_base_vy = float(base_lin_vel[1])
            step3_com_vy = float(com_vel[1])

        if step == 5:
            step5_base_vy = float(base_lin_vel[1])
            step5_com_vy = float(com_vel[1])
            step5_pitch = pitch
            step5_roll = roll
            step5_w_L = w_L
            step5_w_R = w_R
            step5_tau_w_L = tau_w_L
            step5_tau_w_R = tau_w_R

        if step == 7:
            step7_base_vy = float(base_lin_vel[1])
            step7_com_vy = float(com_vel[1])
            step7_tau_w_L = tau_w_L
            step7_tau_w_R = tau_w_R
            step7_tau_k_L = tau_k_L
            step7_tau_k_R = tau_k_R
            step7_tau_h_L = tau_h_L
            step7_tau_h_R = tau_h_R

        if step == 8:
            step8_base_vy = float(base_lin_vel[1])
            step8_com_vy = float(com_vel[1])
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
        obs, _, _, _, _ = env.step(actions)

        if env.reset_buf[0].item() != 0 and step < total_steps - 5:
            abnormal_resets += 1

    valid_jump = took_off and landed and (max_clearance_cm >= 10.0)
    if took_off and landed:
        flight_duration_s = (touchdown_step - takeoff_step) * env.dt
        com_forward_dist_cm = (touchdown_com_y - takeoff_com_y) * 100.0
        touchdown_base_vy = all_base_vys[touchdown_step]
        touchdown_com_vy = all_com_vys[touchdown_step]
    else:
        flight_duration_s = 0.0
        com_forward_dist_cm = 0.0
        touchdown_base_vy = 0.0
        touchdown_com_vy = 0.0

    pushoff_delta_com_vy = takeoff_com_vy - step5_com_vy
    pushoff_delta_base_vy = takeoff_base_vy - step5_base_vy
    post_recovery_vy = float(np.mean(recovery_vys[-10:])) if len(recovery_vys) >= 10 else 0.0

    # Categorization
    if com_forward_dist_cm >= 8.0 and max_clearance_cm >= 20.0:
        jump_class = "Class_A_normal_forward"
    elif com_forward_dist_cm >= 0.0:
        jump_class = "Class_B_short_jump"
    else:
        jump_class = "Class_C_backward_jump"

    joint_pass = (
        max_clearance_cm >= 20.0 and
        com_forward_dist_cm >= 5.0 and
        abs(touchdown_roll) <= 12.0 and
        abs(touchdown_pitch) <= 15.0 and
        abs(post_recovery_vy) <= 0.10 and
        abnormal_resets == 0
    )

    result = {
        "seed": seed,
        "mode": mode,
        "state_A": state_A,
        "state_B": state_B,
        "contact_process_0_02s": contact_process_0_02s,
        "state_C": state_C,
        "step0_base_vy_mps": step0_base_vy,
        "step0_com_vy_mps": step0_com_vy,
        "step3_base_vy_mps": step3_base_vy,
        "step3_com_vy_mps": step3_com_vy,
        "step5_base_vy_mps": step5_base_vy,
        "step5_com_vy_mps": step5_com_vy,
        "step7_base_vy_mps": step7_base_vy,
        "step7_com_vy_mps": step7_com_vy,
        "step8_base_vy_mps": step8_base_vy,
        "step8_com_vy_mps": step8_com_vy,
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
        "touchdown_base_vy_mps": touchdown_base_vy,
        "touchdown_com_vy_mps": touchdown_com_vy,
        "post_recovery_vy_mps": post_recovery_vy,
        "abnormal_resets": abnormal_resets,
        "jump_class": jump_class,
        "joint_pass": joint_pass,
        "trajectory": trajectory
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3.2 Single Seed Evaluator")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--mode", type=str, default="Mode_A_control", choices=["Mode_A_control", "Mode_D_clean_start"])
    parser.add_argument("--out", type=str, required=True, help="Output JSON path")
    args = parser.parse_args()

    run_evaluation(args.seed, args.mode, args.out)
