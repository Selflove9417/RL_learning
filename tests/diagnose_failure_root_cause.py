#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Phase 3.1 Failure Root-Cause Audit Data Collector
# Runs all 20 seeds with identical frozen setup and logs high-frequency
# frame-by-frame telemetry for the critical preparation, push-off, liftoff,
# flight, and touchdown phases.
# ==============================================================================

import os
import sys
import json
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

SEEDS = [
    42, 100, 200, 300, 400,
    1, 2, 3, 5, 7, 11, 13, 17, 19, 23,
    50, 150, 250, 350, 450
]

def audit_seed(model_path, seed, clip_actions=100.0, task_name="myrobot_forward_jump_static"):
    sys.argv = ["eval_audit", "--task", task_name, "--num_envs", "1"]
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
    total_steps = int(2.8 / env.dt) # 140 steps

    # Record environment parameters
    fric_coeff = float(env.friction_coeffs[0, 0].item()) if hasattr(env, "friction_coeffs") else None
    cmd_vy = float(env.commands[0, 1].item())
    cmd_vx = float(env.commands[0, 0].item())
    cmd_yaw = float(env.commands[0, 2].item())

    frames = []

    takeoff_step = None
    touchdown_step = None
    took_off = False
    landed = False

    for step in range(total_steps):
        t_sim = round(step * env.dt, 4)
        phase = round(float(env.jump_phase[0].item()), 4)

        true_com_pos, true_com_vel = env._compute_true_com()
        base_lin_vel = env.base_lin_vel[0].cpu().numpy()
        base_ang_vel = env.base_ang_vel[0].cpu().numpy()

        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
        yaw = torch.atan2(torch.sin(y_raw[0]), torch.cos(y_raw[0])).item() * 180.0 / np.pi

        clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(env)
        clr_min_cm = clr_min[0].item() * 100.0
        clr_L_cm = clr_L[0].item() * 100.0
        clr_R_cm = clr_R[0].item() * 100.0

        is_air = bool(env.is_airborne[0].item())
        has_took = bool(env.has_taken_off[0].item())

        fn_L = float(env.contact_forces[0, env.left_wheel_body, 2].item())
        fn_R = float(env.contact_forces[0, env.right_wheel_body, 2].item())

        w_L = float(env.dof_vel[0, env.left_wheel_idx].item())
        w_R = float(env.dof_vel[0, env.right_wheel_idx].item())

        tau_hip_L = float(env.torques[0, env.left_hip_idx].item())
        tau_hip_R = float(env.torques[0, env.right_hip_idx].item())
        tau_knee_L = float(env.torques[0, env.left_knee_idx].item())
        tau_knee_R = float(env.torques[0, env.right_knee_idx].item())
        tau_wheel_L = float(env.torques[0, env.left_wheel_idx].item())
        tau_wheel_R = float(env.torques[0, env.right_wheel_idx].item())

        pos_wheel_L_y = float(env.rb_states[0, env.left_wheel_body, 1].item())
        pos_wheel_R_y = float(env.rb_states[0, env.right_wheel_body, 1].item())
        pos_wheel_L_z = float(env.rb_states[0, env.left_wheel_body, 2].item())
        pos_wheel_R_z = float(env.rb_states[0, env.right_wheel_body, 2].item())

        dof_pos_hip_L = float(env.dof_pos[0, env.left_hip_idx].item())
        dof_pos_hip_R = float(env.dof_pos[0, env.right_hip_idx].item())
        dof_pos_knee_L = float(env.dof_pos[0, env.left_knee_idx].item())
        dof_pos_knee_R = float(env.dof_pos[0, env.right_knee_idx].item())

        if is_air and has_took and not took_off:
            took_off = True
            takeoff_step = step

        if took_off and (not is_air) and (not landed):
            landed = True
            touchdown_step = step

        frame_data = {
            "step": step,
            "t_s": t_sim,
            "phase": phase,
            "is_airborne": is_air,
            "has_taken_off": has_took,
            "com_pos_y_m": round(float(true_com_pos[0, 1].item()), 4),
            "com_pos_z_m": round(float(true_com_pos[0, 2].item()), 4),
            "com_vel_y_mps": round(float(true_com_vel[0, 1].item()), 4),
            "com_vel_z_mps": round(float(true_com_vel[0, 2].item()), 4),
            "base_lin_vel_y_mps": round(float(base_lin_vel[1]), 4),
            "base_lin_vel_z_mps": round(float(base_lin_vel[2]), 4),
            "roll_deg": round(roll, 2),
            "pitch_deg": round(pitch, 2),
            "yaw_deg": round(yaw, 2),
            "roll_rate_radps": round(float(base_ang_vel[0]), 3),
            "pitch_rate_radps": round(float(base_ang_vel[1]), 3),
            "yaw_rate_radps": round(float(base_ang_vel[2]), 3),
            "clr_min_cm": round(clr_min_cm, 2),
            "clr_L_cm": round(clr_L_cm, 2),
            "clr_R_cm": round(clr_R_cm, 2),
            "fn_L_N": round(fn_L, 2),
            "fn_R_N": round(fn_R, 2),
            "wheel_omega_L_radps": round(w_L, 2),
            "wheel_omega_R_radps": round(w_R, 2),
            "wheel_pos_y_mean_m": round((pos_wheel_L_y + pos_wheel_R_y) / 2.0, 4),
            "tau_hip_L_nm": round(tau_hip_L, 2),
            "tau_hip_R_nm": round(tau_hip_R, 2),
            "tau_knee_L_nm": round(tau_knee_L, 2),
            "tau_knee_R_nm": round(tau_knee_R, 2),
            "tau_wheel_L_nm": round(tau_wheel_L, 2),
            "tau_wheel_R_nm": round(tau_wheel_R, 2),
            "dof_pos_hip_L_rad": round(dof_pos_hip_L, 4),
            "dof_pos_hip_R_rad": round(dof_pos_hip_R, 4),
            "dof_pos_knee_L_rad": round(dof_pos_knee_L, 4),
            "dof_pos_knee_R_rad": round(dof_pos_knee_R, 4),
        }

        with torch.no_grad():
            actions = policy(obs)
            actions_raw = actions[0].cpu().numpy().tolist()
            frame_data["actions"] = [round(a, 4) for a in actions_raw]

        obs, _, _, _, _ = env.step(actions)
        frames.append(frame_data)

    return {
        "seed": seed,
        "friction_coefficient": fric_coeff,
        "cmd_vy_mps": cmd_vy,
        "cmd_vx_mps": cmd_vx,
        "cmd_yaw_radps": cmd_yaw,
        "takeoff_step": takeoff_step,
        "touchdown_step": touchdown_step,
        "frames": frames
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model", type=str, default="/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt")
    args = parser.parse_args()

    data = audit_seed(args.model, args.seed)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Audit data for Seed {args.seed} written to {args.out}")

if __name__ == "__main__":
    main()
