#!/usr/bin/env python3
"""
Offscreen High-Definition MP4 Video Recorder for Phase 6 Continuous Double Jump
Demonstrates:
  1. 0.0s ~ 1.5s  : Normal standing in BALANCE
  2. 1.5s ~ 4.0s  : Low-speed forward motion (vy = 0.30 m/s)
  3. 4.0s         : Trigger Jump 1 -> STOPPING -> PREPARE_JUMP -> JUMP (>=20cm) -> RECOVER -> NORMAL_HANDOFF
  4. ~5.6s ~ 6.5s : Settle in BALANCE post-recovery
  5. 6.5s ~ 9.0s  : Re-engage forward motion (vy = 0.30 m/s)
  6. 9.0s         : Trigger Jump 2 -> STOPPING -> PREPARE_JUMP -> JUMP (>=20cm) -> RECOVER -> NORMAL_HANDOFF
  7. ~10.5s ~ 15.0s: Stable standing in BALANCE for >= 3.0 seconds

Output Video:
  evaluation_assets/phase6/phase6_continuous_double_jump.mp4
"""

import os
import sys
import numpy as np
import imageio
import csv

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from rsl_rl.modules import ActorCritic
from tests.test_phase6_continuous_double_jump import (
    Phase6ContinuousDoubleJumpEnv,
    BALANCE_MODEL_PATH,
    JUMP_MODEL_PATH,
    compute_exact_cylinder_clearance,
    STATE_NAMES
)

def record_phase6_video(seed=42, duration_s=15.0, video_path="evaluation_assets/phase6/phase6_continuous_double_jump.mp4"):
    print("=" * 80)
    print("  Phase 6: Recording High-Definition MP4 Video of Continuous Double Jump")
    print(f"  * Seed: {seed} | Duration: {duration_s}s")
    print(f"  * Output: {video_path}")
    print("=" * 80)

    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    sys.argv = ['record', '--task', 'myrobot_jump']
    args = get_args()
    args.headless = True
    args.seed = seed

    env_cfg, train_cfg = task_registry.get_cfgs(name='myrobot_jump')
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.env.episode_length_s = duration_s + 10.0
    env_cfg.env.enable_camera_sensors = True

    task_name = f"myrobot_record_p6_seed_{seed}"
    task_registry.register(task_name, Phase6ContinuousDoubleJumpEnv, env_cfg, train_cfg)
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # Load policies
    ckpt_bal = torch.load(BALANCE_MODEL_PATH, map_location=env.device)
    ac_bal = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_bal.load_state_dict(ckpt_bal['model_state_dict'])
    ac_bal.eval()

    ckpt_jump = torch.load(JUMP_MODEL_PATH, map_location=env.device)
    ac_jump = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_jump.load_state_dict(ckpt_jump['model_state_dict'])
    ac_jump.eval()

    env.reset()
    env.root_states[:, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.init_dual_policies(ac_bal, ac_jump)

    # Set up Camera Sensor for high-res rendering
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1280
    camera_props.height = 720
    camera_props.enable_tensors = False
    camera_props.horizontal_fov = 65.0

    camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    cam_offset = np.array([-1.5, -2.0, 0.85])

    total_steps = int(duration_s / env.dt)
    frames = []

    print("\nStarting simulation stepping and video capture...")

    for step in range(total_steps):
        t_sim = step * env.dt

        if 1.5 <= t_sim < 4.0:
            vy = 0.30
        elif 6.5 <= t_sim < 9.0:
            vy = 0.30
        else:
            vy = 0.0

        env.set_teleop_targets(vy=vy, yaw=0.0, height=0.39774)

        if step == int(4.0 / env.dt):
            env.request_teleop_jump()
            print(f"[{t_sim:5.2f}s] >>> Triggered Jump 1")
        elif step == int(9.0 / env.dt):
            env.request_teleop_jump()
            print(f"[{t_sim:5.2f}s] >>> Triggered Jump 2")

        env.step_dual_policy()

        # Follow camera
        root_pos = env.root_states[0, 0:3].cpu().numpy()
        cam_target = gymapi.Vec3(root_pos[0], root_pos[1], root_pos[2] + 0.15)
        cam_eye = gymapi.Vec3(root_pos[0] + cam_offset[0], root_pos[1] + cam_offset[1], root_pos[2] + cam_offset[2])
        env.gym.set_camera_location(camera_handle, env.envs[0], cam_eye, cam_target)
        env.gym.render_all_camera_sensors(env.sim)

        img = env.gym.get_camera_image(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
        img = img.reshape((camera_props.height, camera_props.width, 4))[:, :, :3]
        frames.append(img)

        if step % 50 == 0:
            clr_L, clr_R, clr_exact = compute_exact_cylinder_clearance(env)
            fsm_st = env.fsm_state[0].item()
            print(f"[{t_sim:5.2f}s] State: {STATE_NAMES[fsm_st]:<13} | Clr: {clr_exact[0].item()*100:5.1f}cm | Height: {root_pos[2]:.3f}m")

    print(f"\nEncoding {len(frames)} frames into MP4 video (50 FPS)...")
    imageio.mimsave(video_path, frames, fps=50, quality=8)
    print(f"--> Video successfully saved to: {video_path} ({os.path.getsize(video_path) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    record_phase6_video()
