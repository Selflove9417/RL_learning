#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Single Run Telemetry and 720p Video Renderer
# Runs in an isolated subprocess to prevent IsaacGym Foundation errors
# ==============================================================================

import os
import sys
import csv
import argparse
import numpy as np
import imageio

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


def render(model_path, out_csv_path, out_mp4_path, seed=42, clip=100.0):
    sys.argv = ["render_run", "--task", "myrobot_forward_jump_static", "--num_envs", "1"]
    args = get_args()
    args.headless = False
    args.seed = seed

    env_cfg, _ = task_registry.get_cfgs(name="myrobot_forward_jump_static")
    env_cfg.normalization.clip_actions = clip
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.env.episode_length_s = 2.8
    env_cfg.noise.add_noise = False
    env_cfg.env.enable_camera_sensors = True

    env, _ = task_registry.make_env(name="myrobot_forward_jump_static", args=args, env_cfg=env_cfg)

    # Set camera sensor
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1280
    camera_props.height = 720
    camera_props.enable_tensors = False
    camera_props.horizontal_fov = 65.0
    camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    cam_offset = np.array([-1.4, -2.1, 0.85])

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

    frames = []
    total_steps = int(2.8 / env.dt)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4_path)), exist_ok=True)

    csv_file = open(out_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "time_s", "step", "jump_phase", "jump_latched",
        "root_x", "root_y", "root_z",
        "com_x", "com_y", "com_z",
        "root_vx", "root_vy", "root_vz",
        "com_vx", "com_vy", "com_vz",
        "base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z",
        "roll_deg", "pitch_deg", "yaw_deg",
        "is_airborne", "has_taken_off", "has_touched_down",
        "clr_min_cm", "clr_L_cm", "clr_R_cm",
        "fn_L_N", "fn_R_N",
        "flight_distance_y_cm",
        "tau_hip_L_Nm", "tau_knee_L_Nm", "tau_wheel_L_Nm",
        "tau_hip_R_Nm", "tau_knee_R_Nm", "tau_wheel_R_Nm"
    ])

    for step in range(total_steps):
        t_sim = step * env.dt

        # Compute states
        true_com_pos, true_com_vel = env._compute_true_com()
        root_pos = env.root_states[0, 0:3].cpu().numpy()
        root_vel = env.root_states[0, 7:10].cpu().numpy()
        base_lin_vel = env.base_lin_vel[0].cpu().numpy()

        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
        yaw = torch.atan2(torch.sin(y_raw[0]), torch.cos(y_raw[0])).item() * 180.0 / np.pi

        clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(env)
        clr_min_cm = clr_min[0].item() * 100.0
        clr_L_cm = clr_L[0].item() * 100.0
        clr_R_cm = clr_R[0].item() * 100.0

        fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
        fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()

        is_air = int(env.is_airborne[0].item())
        has_took = int(env.has_taken_off[0].item())
        has_land = int(env.has_touched_down[0].item())
        latched = int(env.jump_latched[0].item())
        phase = env.jump_phase[0].item()

        fwd_dist_cm = env.flight_distance_y[0].item() * 100.0

        t_hip_L = env.torques[0, env.left_hip_idx].item()
        t_knee_L = env.torques[0, env.left_knee_idx].item()
        t_wheel_L = env.torques[0, env.left_wheel_idx].item()
        t_hip_R = env.torques[0, env.right_hip_idx].item()
        t_knee_R = env.torques[0, env.right_knee_idx].item()
        t_wheel_R = env.torques[0, env.right_wheel_idx].item()

        # Telemetry write
        csv_writer.writerow([
            f"{t_sim:.3f}", step, f"{phase:.4f}", latched,
            f"{root_pos[0]:.4f}", f"{root_pos[1]:.4f}", f"{root_pos[2]:.4f}",
            f"{true_com_pos[0, 0].item():.4f}", f"{true_com_pos[0, 1].item():.4f}", f"{true_com_pos[0, 2].item():.4f}",
            f"{root_vel[0]:.4f}", f"{root_vel[1]:.4f}", f"{root_vel[2]:.4f}",
            f"{true_com_vel[0, 0].item():.4f}", f"{true_com_vel[0, 1].item():.4f}", f"{true_com_vel[0, 2].item():.4f}",
            f"{base_lin_vel[0]:.4f}", f"{base_lin_vel[1]:.4f}", f"{base_lin_vel[2]:.4f}",
            f"{roll:.2f}", f"{pitch:.2f}", f"{yaw:.2f}",
            is_air, has_took, has_land,
            f"{clr_min_cm:.2f}", f"{clr_L_cm:.2f}", f"{clr_R_cm:.2f}",
            f"{fn_L:.2f}", f"{fn_R:.2f}",
            f"{fwd_dist_cm:.2f}",
            f"{t_hip_L:.2f}", f"{t_knee_L:.2f}", f"{t_wheel_L:.2f}",
            f"{t_hip_R:.2f}", f"{t_knee_R:.2f}", f"{t_wheel_R:.2f}"
        ])

        # Camera frame
        cam_target = gymapi.Vec3(root_pos[0], root_pos[1], root_pos[2] + 0.12)
        cam_eye = gymapi.Vec3(root_pos[0] + cam_offset[0], root_pos[1] + cam_offset[1], root_pos[2] + cam_offset[2])
        env.gym.set_camera_location(camera_handle, env.envs[0], cam_eye, cam_target)
        env.gym.render_all_camera_sensors(env.sim)
        img = env.gym.get_camera_image(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
        img = img.reshape((camera_props.height, camera_props.width, 4))[:, :, :3]
        frames.append(img)

        with torch.no_grad():
            actions = policy(obs)
        obs, _, rewards, dones, infos = env.step(actions)

    csv_file.close()
    print(f"✅ Telemetry CSV written to: {out_csv_path}")

    print(f"📹 Rendering MP4 video with {len(frames)} frames to: {out_mp4_path}...")
    imageio.mimwrite(out_mp4_path, frames, fps=50, quality=8)
    print(f"✅ Video successfully created ({os.path.getsize(out_mp4_path)/1024/1024:.2f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip", type=float, default=100.0)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_mp4", type=str, required=True)
    args = parser.parse_args()

    render(args.model, args.out_csv, args.out_mp4, seed=args.seed, clip=args.clip)


if __name__ == "__main__":
    main()
