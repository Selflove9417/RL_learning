#!/usr/bin/env python3
# ==============================================================================
# Worker script: Evaluate single seed for Phase 2 Static Forward Jump
# ==============================================================================

import os
import sys
import json
import argparse
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    return parser.parse_known_args()


def main():
    custom_args, _ = parse_args()
    seed = custom_args.seed
    model_path = custom_args.model_path
    out_json = custom_args.out_json

    sys.argv = ["eval_static", "--task", "myrobot_forward_jump_static", "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = seed

    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_forward_jump_static")
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.env.episode_length_s = 2.8
    env_cfg.noise.add_noise = False
    env_cfg.env.enable_camera_sensors = False

    env, _ = task_registry.make_env(name="myrobot_forward_jump_static", args=args, env_cfg=env_cfg)

    # Load policy
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

    total_steps = int(2.8 / env.dt)  # 140 steps

    takeoff_step = None
    touchdown_step = None
    max_clearance_cm = 0.0
    takeoff_root_y = 0.0
    takeoff_com_y = 0.0
    takeoff_root_vz = 0.0
    takeoff_root_vy = 0.0
    takeoff_com_vz = 0.0
    takeoff_com_vy = 0.0
    pre_push_vy = 0.0
    touchdown_pitch = 0.0
    touchdown_roll = 0.0
    touchdown_yaw = 0.0
    initial_yaw = 0.0
    final_yaw = 0.0

    peak_hip_tau = 0.0
    peak_knee_tau = 0.0
    peak_wheel_tau = 0.0

    recovery_vys = []
    recovery_vxs = []
    abnormal_resets = 0

    took_off = False
    landed = False

    for step in range(total_steps):
        t_sim = step * env.dt

        # Track true CoM
        true_com_pos, true_com_vel = env._compute_true_com()
        root_pos = env.root_states[0, 0:3].cpu().numpy()
        root_vel = env.root_states[0, 7:10].cpu().numpy()
        base_lin_vel = env.base_lin_vel[0].cpu().numpy()

        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
        yaw = torch.atan2(torch.sin(y_raw[0]), torch.cos(y_raw[0])).item() * 180.0 / np.pi

        if step == 0:
            initial_yaw = yaw
        final_yaw = yaw

        # End of prep check (step 5, t=0.10s)
        if step == 5:
            pre_push_vy = true_com_vel[0, 1].item()

        clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(env)
        clr_min_cm = clr_min[0].item() * 100.0

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

        # Detect takeoff
        if is_air and has_took and not took_off:
            took_off = True
            takeoff_step = step
            takeoff_root_y = float(root_pos[1])
            takeoff_com_y = float(true_com_pos[0, 1].item())
            takeoff_root_vz = float(root_vel[2])
            takeoff_root_vy = float(root_vel[1])
            takeoff_com_vz = float(true_com_vel[0, 2].item())
            takeoff_com_vy = float(true_com_vel[0, 1].item())

        # Detect touchdown
        if took_off and (not is_air) and (not landed):
            landed = True
            touchdown_step = step
            touchdown_root_y = float(root_pos[1])
            touchdown_com_y = float(true_com_pos[0, 1].item())
            touchdown_pitch = pitch
            touchdown_roll = roll
            touchdown_yaw = yaw

        # Recovery observation (after landing and t >= 1.0s)
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
        root_forward_dist_cm = (touchdown_root_y - takeoff_root_y) * 100.0
        com_forward_dist_cm = (touchdown_com_y - takeoff_com_y) * 100.0
    else:
        flight_duration_s = 0.0
        flight_steps = 0
        root_forward_dist_cm = 0.0
        com_forward_dist_cm = 0.0

    mean_rec_vy = float(np.mean(recovery_vys)) if len(recovery_vys) > 0 else 0.0
    mean_rec_vx = float(np.mean(recovery_vxs)) if len(recovery_vxs) > 0 else 0.0
    yaw_drift = abs(final_yaw - initial_yaw)

    result = {
        "seed": seed,
        "valid_jump": valid_jump,
        "abnormal_resets": abnormal_resets,
        "max_clearance_cm": round(max_clearance_cm, 2),
        "flight_duration_s": round(flight_duration_s, 3),
        "flight_steps": flight_steps,
        "com_forward_dist_cm": round(com_forward_dist_cm, 2),
        "root_forward_dist_cm": round(root_forward_dist_cm, 2),
        "pre_push_vy_mps": round(pre_push_vy, 3),
        "takeoff_com_vy_mps": round(takeoff_com_vy, 3),
        "takeoff_com_vz_mps": round(takeoff_com_vz, 3),
        "takeoff_root_vy_mps": round(takeoff_root_vy, 3),
        "takeoff_root_vz_mps": round(takeoff_root_vz, 3),
        "touchdown_pitch_deg": round(touchdown_pitch, 2),
        "touchdown_roll_deg": round(touchdown_roll, 2),
        "touchdown_yaw_deg": round(touchdown_yaw, 2),
        "yaw_drift_deg": round(yaw_drift, 2),
        "recovery_vy_mps": round(mean_rec_vy, 3),
        "recovery_vx_mps": round(mean_rec_vx, 3),
        "peak_hip_tau_nm": round(peak_hip_tau, 2),
        "peak_knee_tau_nm": round(peak_knee_tau, 2),
        "peak_wheel_tau_nm": round(peak_wheel_tau, 2)
    }

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
