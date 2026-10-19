#!/usr/bin/env python3
# ==============================================================================
# Worker script: Evaluate single seed for Forward Jump / In-place Jump
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


def parse_worker_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--is_forward", action="store_true")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--out_json", type=str, required=True)
    return parser.parse_known_args()


def main():
    custom_args, remaining_args = parse_worker_args()
    seed = custom_args.seed
    model_path = custom_args.model_path
    is_forward = custom_args.is_forward
    duration_s = custom_args.duration
    out_json = custom_args.out_json

    sys.argv = ["eval_worker", "--task", "myrobot_forward_jump", "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = seed

    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_forward_jump")
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.env.episode_length_s = 10.0
    env_cfg.noise.add_noise = False
    env_cfg.env.enable_camera_sensors = False

    env, _ = task_registry.make_env(name="myrobot_forward_jump", args=args, env_cfg=env_cfg)

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

    # Rigid body masses for whole-body CoM
    num_bodies = env.gym.get_actor_rigid_body_count(env.envs[0], 0)
    rb_masses = []
    for b in range(num_bodies):
        prop = env.gym.get_actor_rigid_body_properties(env.envs[0], 0)[b]
        rb_masses.append(prop.mass)
    rb_masses_t = torch.tensor(rb_masses, device=env.device, dtype=torch.float).unsqueeze(0).unsqueeze(-1)
    total_mass = rb_masses_t.sum().item()

    env.reset()
    if not is_forward:
        env.commands[:, 1] = 0.0

    total_steps = int(duration_s / env.dt)

    takeoff_step = None
    touchdown_step = None
    max_clearance_cm = 0.0
    takeoff_root_y = 0.0
    takeoff_com_y = 0.0
    takeoff_root_vz = 0.0
    takeoff_root_vy = 0.0
    takeoff_com_vz = 0.0
    takeoff_com_vy = 0.0
    touchdown_pitch = 0.0
    touchdown_roll = 0.0
    touchdown_yaw = 0.0
    initial_yaw = 0.0
    final_yaw = 0.0

    peak_hip_tau = 0.0
    peak_knee_tau = 0.0
    peak_wheel_tau = 0.0

    hip_sat_count = 0
    knee_sat_count = 0
    wheel_sat_count = 0

    recovery_vys = []
    recovery_vxs = []
    abnormal_resets = 0

    took_off = False
    landed = False

    for step in range(total_steps):
        t_sim = step * env.dt
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        rb_pos = env.rb_states[:, :, 0:3]
        rb_vel = env.rb_states[:, :, 7:10]
        com_pos = (rb_pos * rb_masses_t).sum(dim=1) / total_mass
        com_vel = (rb_vel * rb_masses_t).sum(dim=1) / total_mass

        root_pos = env.root_states[0, 0:3].cpu().numpy()
        root_vel_w = env.root_states[0, 7:10].cpu().numpy()
        base_lin_vel = env.base_lin_vel[0].cpu().numpy()

        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
        yaw = torch.atan2(torch.sin(y_raw[0]), torch.cos(y_raw[0])).item() * 180.0 / np.pi

        if step == 0:
            initial_yaw = yaw
        final_yaw = yaw

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

        if tau_hip >= 119.5: hip_sat_count += 1
        if tau_knee >= 119.5: knee_sat_count += 1
        if tau_wheel >= 7.95: wheel_sat_count += 1

        if is_air and has_took:
            max_clearance_cm = max(max_clearance_cm, clr_min_cm)

        # Detect takeoff
        if is_air and has_took and not took_off:
            took_off = True
            takeoff_step = step
            takeoff_root_y = float(root_pos[1])
            takeoff_com_y = float(com_pos[0, 1].item())
            takeoff_root_vz = float(root_vel_w[2])
            takeoff_root_vy = float(root_vel_w[1])
            takeoff_com_vz = float(com_vel[0, 2].item())
            takeoff_com_vy = float(com_vel[0, 1].item())

        # Detect touchdown
        if took_off and (not is_air) and (not landed):
            landed = True
            touchdown_step = step
            touchdown_root_y = float(root_pos[1])
            touchdown_com_y = float(com_pos[0, 1].item())
            touchdown_pitch = pitch
            touchdown_roll = roll
            touchdown_yaw = yaw

        phase = env.jump_phase[0].item()
        if landed and phase >= 0.45 and phase <= 0.95:
            recovery_vys.append(base_lin_vel[1])
            recovery_vxs.append(base_lin_vel[0])

        obs = env.get_observations()
        with torch.no_grad():
            actions = policy(obs)
        env.step(actions)

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
        "root_forward_dist_cm": round(root_forward_dist_cm, 2),
        "com_forward_dist_cm": round(com_forward_dist_cm, 2),
        "takeoff_root_vy_mps": round(takeoff_root_vy, 3),
        "takeoff_root_vz_mps": round(takeoff_root_vz, 3),
        "takeoff_com_vy_mps": round(takeoff_com_vy, 3),
        "takeoff_com_vz_mps": round(takeoff_com_vz, 3),
        "touchdown_pitch_deg": round(touchdown_pitch, 2),
        "touchdown_roll_deg": round(touchdown_roll, 2),
        "touchdown_yaw_deg": round(touchdown_yaw, 2),
        "yaw_drift_deg": round(yaw_drift, 2),
        "recovery_vy_mps": round(mean_rec_vy, 3),
        "recovery_vx_mps": round(mean_rec_vx, 3),
        "peak_hip_tau_nm": round(peak_hip_tau, 2),
        "peak_knee_tau_nm": round(peak_knee_tau, 2),
        "peak_wheel_tau_nm": round(peak_wheel_tau, 2),
        "knee_saturation_steps": knee_sat_count
    }

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Worker Seed {seed}] Done. Result saved to {out_json}")


if __name__ == "__main__":
    main()
