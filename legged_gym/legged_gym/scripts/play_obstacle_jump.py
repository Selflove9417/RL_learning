#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# MyRobot Obstacle Clearance Running Jump: Interactive 3D Viewer
# ======================================================================================
# Genuine physical obstacle with collision properties, fixed lateral side-view camera,
# and real-time collision/clearance telemetry.
#
# Camera Perspectives (Toggle with [C]):
#   Mode 0 (Default): Fixed Lateral Side-View (Optimal perspective framing takeoff,
#                     5cm red hurdle, and landing impact).
#   Mode 1          : Moving Lateral Side-View (Follows robot Y translation from the side).
#   Mode 2          : Isometric 45° Chase View.
#
# Interactive Controls:
#   [Space]     : Force manual jump trigger in-motion
#   [W] / [↑]   : Forward cruise speed up (+0.05 m/s, max 0.50 m/s)
#   [S] / [↓]   : Forward cruise speed down (-0.05 m/s)
#   [A] / [D]   : Steer left / right (yaw)
#   [C]         : Cycle camera perspective (Fixed side -> Moving side -> 45° chase)
#   [R]         : Reset back to starting line (Y = 0)
#   [ESC]       : Exit cleanly
# ======================================================================================

import os
import sys
import time
import math
import select
import tty
import termios
import argparse
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from rsl_rl.modules import ActorCritic
from legged_gym.envs.myrobot.myrobot_dual_env import STATE_NAMES, compute_exact_cylinder_clearance


class NonBlockingKeyboard:
    """Reads non-blocking single keystrokes from terminal without requiring Enter."""
    def __init__(self):
        self.is_tty = sys.stdin.isatty()
        self.old_settings = None
        if self.is_tty:
            try:
                self.fd = sys.stdin.fileno()
                self.old_settings = termios.tcgetattr(self.fd)
                tty.setraw(self.fd)
            except Exception:
                self.is_tty = False

    def read_key(self):
        if not self.is_tty:
            return None
        if select.select([sys.stdin], [], [], 0.0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                if select.select([sys.stdin], [], [], 0.005)[0]:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[' and select.select([sys.stdin], [], [], 0.005)[0]:
                        ch3 = sys.stdin.read(1)
                        return {'A': 'UP', 'B': 'DOWN', 'C': 'RIGHT', 'D': 'LEFT'}.get(ch3, None)
                return 'ESC'
            return ch
        return None

    def restore(self):
        if self.is_tty and self.old_settings is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass


def play_obstacle_viewer(args):
    # Parse CLI overrides
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="myrobot_obstacle_jump")
    parser.add_argument("--load_run", type=str, default="Oct19_21-52-03_running_jump_020ms_iter3_controlled")
    parser.add_argument("--checkpoint", type=int, default=3)
    parser.add_argument("--target_speed", type=float, default=0.35)
    parser.add_argument("--trigger_mode", type=str, default="spatial", choices=["spatial", "step"])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--continuous", action="store_true", default=True)
    parser.add_argument("--headless", action="store_true", default=False)
    parsed_args, unknown = parser.parse_known_args()

    args.headless = parsed_args.headless
    args.task = parsed_args.task

    env_cfg, train_cfg = task_registry.get_cfgs(name=parsed_args.task)
    env_cfg.env.num_envs = 1
    env_cfg.seed = parsed_args.seed
    env_cfg.env.continuous_mode = parsed_args.continuous
    env_cfg.trigger.mode = parsed_args.trigger_mode
    env_cfg.commands.target_forward_vel = parsed_args.target_speed
    env_cfg.commands.ranges.lin_vel_y = [parsed_args.target_speed, parsed_args.target_speed]

    # Model resolution
    log_dir = os.path.join(workspace_root, "legged_gym", "logs", "myrobot_running_jump", parsed_args.load_run)
    model_file = f"model_{parsed_args.checkpoint}.pt"
    model_path = os.path.join(log_dir, model_file)

    if not os.path.exists(model_path):
        print(f"[ERROR] Checkpoint not found at: {model_path}")
        sys.exit(1)

    print("\n" + "=" * 95)
    print(" === MyRobot 真实物理障碍跨越 3D Viewer 已启动 ===")
    print(f" 加载基准模型 : {model_path}")
    print(f" 障碍物物理属性: 高度 H = {env_cfg.obstacle.height*100:.1f} cm | 宽度 W = {env_cfg.obstacle.width*100:.1f} cm | 位置 Y = {env_cfg.obstacle.pos_y:.2f} m")
    print(f" 目标巡航车速 : {parsed_args.target_speed:.2f} m/s | 起跳触发模式: {parsed_args.trigger_mode}")
    print(" 默认观察机位 : [固定纯侧视机位] 位于 X = 1.3m, 聚焦在起跳点、障碍物与落地点")
    print(" 快捷交互按键 :")
    print("   [Space]     : 立即动态起跳 (在行进中保持前进速度直接起飞)")
    print("   [W] / [↑]   : 提高巡航车速 (+0.05 m/s, 最大 +0.50 m/s)")
    print("   [S] / [↓]   : 降低巡航车速 (-0.05 m/s, 最小 -0.10 m/s)")
    print("   [A] / [D]   : 左/右偏航转向")
    print("   [C]         : 切换摄像机机位 (0: 固定侧视 -> 1: 跟踪侧视 -> 2: 45°追随视角)")
    print("   [R]         : 复位仿真回到起跑线 (Y = 0)")
    print("   [ESC]       : 退出演示")
    print("=" * 95 + "\n")

    env, _ = task_registry.make_env(name=parsed_args.task, args=args, env_cfg=env_cfg)
    env.reset()
    obs = env.get_observations()

    # Load ActorCritic policy
    ckpt = torch.load(model_path, map_location=env.device)
    policy = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()

    # Register keyboard events in Isaac Gym Viewer
    if env.viewer is not None:
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_SPACE, "JUMP")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_W, "SPEED_UP")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_UP, "SPEED_UP")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_S, "SPEED_DOWN")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_DOWN, "SPEED_DOWN")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_A, "STEER_LEFT")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_D, "STEER_RIGHT")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_C, "TOGGLE_CAM")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R, "RESET")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_ESCAPE, "QUIT")

    kb = NonBlockingKeyboard()
    user_vy = parsed_args.target_speed
    user_yaw = 0.0
    cam_mode = 0  # 0: Fixed side-view, 1: Tracking side-view, 2: 45 deg chase view

    obs_front_y = env.obstacle_py - 0.5 * env.obstacle_w
    step = 0

    try:
        while (env.viewer is None or not env.gym.query_viewer_has_closed(env.viewer)):
            t_sim = step * env.dt

            # 1. Terminal keyboard polling
            term_key = kb.read_key()
            if term_key is not None:
                k = term_key.upper()
                if k == 'ESC':
                    break
                elif k in ['W', 'UP']:
                    user_vy = round(min(0.50, user_vy + 0.05), 2)
                    env.user_cmd_vy[:] = user_vy
                    env.commands[:, 1] = user_vy
                elif k in ['S', 'DOWN']:
                    user_vy = round(max(-0.10, user_vy - 0.05), 2)
                    env.user_cmd_vy[:] = user_vy
                    env.commands[:, 1] = user_vy
                elif k in ['A', 'LEFT']:
                    user_yaw = round(min(0.35, user_yaw + 0.05), 2)
                    env.user_cmd_yaw[:] = user_yaw
                    env.commands[:, 2] = user_yaw
                elif k in ['D', 'RIGHT']:
                    user_yaw = round(max(-0.35, user_yaw - 0.05), 2)
                    env.user_cmd_yaw[:] = user_yaw
                    env.commands[:, 2] = user_yaw
                elif k == ' ':
                    env.request_teleop_jump()
                elif k == 'C':
                    cam_mode = (cam_mode + 1) % 3
                    mode_names = ["固定纯侧视机位 (Fixed Side-View)", "跟踪侧视机位 (Tracking Side-View)", "45°追随视角 (Chase View)"]
                    print(f" [视角切换] -> {mode_names[cam_mode]}")
                elif k == 'R':
                    env.reset_idx(torch.tensor([0], device=env.device, dtype=torch.long))
                    obs = env.get_observations()
                    print(f" [手动复位] 机器人已重置至起点 Y = 0.000m")

            # 2. Isaac Gym Viewer keyboard polling
            if env.viewer is not None:
                for evt in env.gym.query_viewer_action_events(env.viewer):
                    if evt.value <= 0:
                        continue
                    if evt.action == "QUIT":
                        return
                    elif evt.action == "JUMP":
                        env.request_teleop_jump()
                    elif evt.action == "SPEED_UP":
                        user_vy = round(min(0.50, user_vy + 0.05), 2)
                        env.user_cmd_vy[:] = user_vy
                        env.commands[:, 1] = user_vy
                    elif evt.action == "SPEED_DOWN":
                        user_vy = round(max(-0.10, user_vy - 0.05), 2)
                        env.user_cmd_vy[:] = user_vy
                        env.commands[:, 1] = user_vy
                    elif evt.action == "STEER_LEFT":
                        user_yaw = round(min(0.35, user_yaw + 0.05), 2)
                        env.user_cmd_yaw[:] = user_yaw
                        env.commands[:, 2] = user_yaw
                    elif evt.action == "STEER_RIGHT":
                        user_yaw = round(max(-0.35, user_yaw - 0.05), 2)
                        env.user_cmd_yaw[:] = user_yaw
                        env.commands[:, 2] = user_yaw
                    elif evt.action == "TOGGLE_CAM":
                        cam_mode = (cam_mode + 1) % 3
                    elif evt.action == "RESET":
                        env.reset_idx(torch.tensor([0], device=env.device, dtype=torch.long))
                        obs = env.get_observations()

            # 3. Policy inference and step simulation
            with torch.no_grad():
                actions = policy.act_inference(obs)
            obs, _, rews, dones, infos = env.step(actions)

            # 4. Telemetry readout
            pos_x = env.root_states[0, 0].item()
            pos_y = env.root_states[0, 1].item()
            pos_z = env.root_states[0, 2].item()
            vw_y = env.root_states[0, 8].item()
            clr_cm = env.bilateral_clearance[0].item() * 100.0
            fsm_str = STATE_NAMES.get(env.fsm_state[0].item(), "UNKNOWN")
            dist_to_obs = obs_front_y - (pos_y + env.wheel_radius)
            is_collision = env.has_obstacle_collision[0].item()
            obs_fn = env.peak_obstacle_contact_force[0].item()

            # 5. Dynamic Camera Positioning
            if env.viewer is not None:
                if cam_mode == 0:
                    # Fixed lateral side-view (X=1.3, Y=0.32, Z=0.28)
                    cam_pos = gymapi.Vec3(1.3, 0.32, 0.28)
                    cam_target = gymapi.Vec3(0.0, 0.32, 0.16)
                elif cam_mode == 1:
                    # Tracking lateral side-view (moves along Y with robot)
                    cam_pos = gymapi.Vec3(1.3, pos_y, 0.28)
                    cam_target = gymapi.Vec3(0.0, pos_y, 0.16)
                elif cam_mode == 2:
                    # 45 deg chase view
                    cam_pos = gymapi.Vec3(pos_x - 0.8, pos_y - 1.2, pos_z + 0.6)
                    cam_target = gymapi.Vec3(pos_x, pos_y + 0.5, pos_z)

                env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)

            # Console Telemetry update (every 10 steps = 0.2s)
            if step % 10 == 0:
                col_status = f"COLLISION! (F={obs_fn:.1f}N)" if is_collision else "CLEAN"
                clr_str = f"{clr_cm:.1f}cm" if clr_cm > 0.1 else "GROUND"
                print(f"[{t_sim:5.2f}s | Step {step:4d}] Y: {pos_y:+6.3f}m | v_Y: {vw_y:+5.2f}m/s | "
                      f"To Obstacle: {dist_to_obs*100:+5.1f}cm | Clr: {clr_str:>7s} | Status: {col_status:<18s} | FSM: {fsm_str}")

            step += 1
            time.sleep(0.01)

    finally:
        kb.restore()
        print("\n[INFO] Obstacle viewer demo closed successfully.")


if __name__ == "__main__":
    play_obstacle_viewer(get_args())
