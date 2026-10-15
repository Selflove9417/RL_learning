#!/usr/bin/env python3
"""
MyRobot Formal Dual-Policy Continuous Jumping & Teleoperation Interface
========================================================================
Interactive 3D viewer and teleoperation for the formal MyRobotDualEnv.
Supports 100% manual keyboard driving, continuous jump triggering, and height control.

Keybindings:
  [W] / [↑]   : Forward acceleration (+0.05 m/s, max +0.50 m/s)
  [S] / [↓]   : Backward acceleration (-0.05 m/s, min -0.30 m/s)
  [A] / [←]   : Turn left (+0.05 rad/s, max +0.35 rad/s)
  [D] / [→]   : Turn right (-0.05 rad/s, min -0.35 rad/s)
  [Q]         : Raise body (+0.01 m, max 0.52 m)
  [E]         : Lower body (-0.01 m, min 0.34 m)
  [Space]     : Trigger jump (STOPPING -> PREPARE -> JUMP -> RECOVER -> BALANCE)
  [X]         : Active brake / parking (clamp to zero speed with PI lock)
  [R]         : Reset simulation and restore nominal height
  [ESC]       : Exit and save 50 Hz CSV telemetry
"""

import os
import sys
import time
import math
import select
import tty
import termios
import csv
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
from legged_gym.envs.myrobot.myrobot_dual_env import (
    MyRobotDualEnv,
    compute_exact_cylinder_clearance,
    STATE_NAMES
)

class NonBlockingKeyboard:
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

def main():
    parser = argparse.ArgumentParser(description="MyRobot Dual-Policy Interactive Viewer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--csv_path", type=str, default="evaluation_assets/phase6/viewer_telemetry.csv", help="CSV telemetry output path")
    parsed_args, _ = parser.parse_known_args()

    sys.argv = ['teleop_dual', '--task', 'myrobot_dual']
    args = get_args()
    args.headless = False
    args.seed = parsed_args.seed

    env_cfg, train_cfg = task_registry.get_cfgs(name='myrobot_dual')
    env_cfg.env.num_envs = 1
    env_cfg.seed = parsed_args.seed
    env_cfg.env.episode_length_s = 1000.0

    task_name = f"myrobot_dual_viewer_{parsed_args.seed}"
    task_registry.register(task_name, MyRobotDualEnv, env_cfg, train_cfg)
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # Subscribe 3D viewer keyboard events
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_W, "FORWARD")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_S, "BACKWARD")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_A, "LEFT")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_D, "RIGHT")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_Q, "RISE")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_E, "LOWER")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_SPACE, "JUMP")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_X, "BRAKE")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R, "RESET")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_ESCAPE, "QUIT")

    # Initialize policies from default frozen checkpoints
    env.reset()
    env.root_states[:, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.init_dual_policies()

    kb = NonBlockingKeyboard()
    user_vy = 0.0
    user_yaw = 0.0
    user_height = 0.39774

    os.makedirs(os.path.dirname(parsed_args.csv_path), exist_ok=True)
    csv_file = open(parsed_args.csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        "timestamp_s", "fsm_state", "cmd_vy", "act_vy",
        "roll_deg", "pitch_deg", "height_m", "clr_exact_cm",
        "tau_knee_L", "tau_knee_R", "fn_L", "fn_R", "event"
    ])

    print("\n" + "=" * 90)
    print(" === MyRobot: 双策略连续跳跃与巡航正式交互系统已启动 ===")
    print(" 运行模式 : 100% 纯手动控制 (默认在 BALANCE 状态原地站立待命)")
    print(" 键盘支持 : 3D 渲染视窗与终端控制台均可直接响应按键")
    print(" 控制按键 :")
    print("   [W] / [↑]   : 前进加速 (+0.05 m/s, 最大 +0.50 m/s)")
    print("   [S] / [↓]   : 后退加速 (-0.05 m/s, 最小 -0.30 m/s)")
    print("   [A] / [←]   : 左转偏航 (+0.05 rad/s, 最大 +0.35 rad/s)")
    print("   [D] / [→]   : 右转偏航 (-0.05 rad/s, 最小 -0.35 rad/s)")
    print("   [Q]         : 升高机身 (+0.01 m, 最大 0.52 m)")
    print("   [E]         : 降低机身 (-0.01 m, 最小 0.34 m)")
    print("   [Space]     : 请求跳跃 (随时按下，自动平滑刹车并爆发起跳)")
    print("   [X]         : 驻车刹停 (指令归零，激活主动防滑 PI 积分)")
    print("   [R]         : 复位仿真与高度")
    print("   [ESC]       : 退出并保存遥测数据")
    print("=" * 90 + "\n")

    step = 0
    try:
        while not env.gym.query_viewer_has_closed(env.viewer):
            t_sim = step * env.dt

            # 1. Read terminal keyboard
            term_key = kb.read_key()
            if term_key is not None:
                k = term_key.upper()
                if k == 'ESC':
                    break
                elif k in ['W', 'UP']:
                    user_vy = round(min(0.50, user_vy + 0.05), 2)
                elif k in ['S', 'DOWN']:
                    user_vy = round(max(-0.30, user_vy - 0.05), 2)
                elif k in ['A', 'LEFT']:
                    user_yaw = round(min(0.35, user_yaw + 0.05), 2)
                elif k in ['D', 'RIGHT']:
                    user_yaw = round(max(-0.35, user_yaw - 0.05), 2)
                elif k == 'Q':
                    user_height = round(min(0.520, user_height + 0.01), 3)
                elif k == 'E':
                    user_height = round(max(0.340, user_height - 0.01), 3)
                elif k == 'X':
                    user_vy = 0.0
                    user_yaw = 0.0
                elif k == ' ':
                    env.request_teleop_jump()
                elif k == 'R':
                    env.reset()
                    env.root_states[:, 7:13] = 0.0
                    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
                    env.init_dual_policies()
                    step = 0
                    user_vy = 0.0
                    user_yaw = 0.0
                    user_height = 0.39774

            # 2. Read 3D viewer window events
            for evt in env.gym.query_viewer_action_events(env.viewer):
                act = evt.action
                val = evt.value
                if val > 0.5:
                    if act == "QUIT":
                        return
                    elif act == "FORWARD":
                        user_vy = round(min(0.50, user_vy + 0.05), 2)
                    elif act == "BACKWARD":
                        user_vy = round(max(-0.30, user_vy - 0.05), 2)
                    elif act == "LEFT":
                        user_yaw = round(min(0.35, user_yaw + 0.05), 2)
                    elif act == "RIGHT":
                        user_yaw = round(max(-0.35, user_yaw - 0.05), 2)
                    elif act == "RISE":
                        user_height = round(min(0.520, user_height + 0.01), 3)
                    elif act == "LOWER":
                        user_height = round(max(0.340, user_height - 0.01), 3)
                    elif act == "BRAKE":
                        user_vy = 0.0
                        user_yaw = 0.0
                    elif act == "JUMP":
                        env.request_teleop_jump()
                    elif act == "RESET":
                        env.reset()
                        env.root_states[:, 7:13] = 0.0
                        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
                        env.init_dual_policies()
                        step = 0
                        user_vy = 0.0
                        user_yaw = 0.0
                        user_height = 0.39774

            env.set_teleop_targets(vy=user_vy, yaw=user_yaw, height=user_height)
            env.step_dual_policy()

            # Render graphics
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)
            env.gym.sync_frame_time(env.sim)

            # Camera follow
            root_pos = env.root_states[0, 0:3].cpu().numpy()
            cam_target = gymapi.Vec3(root_pos[0], root_pos[1], root_pos[2] + 0.15)
            cam_eye = gymapi.Vec3(root_pos[0] - 1.5, root_pos[1] - 2.0, root_pos[2] + 0.85)
            env.gym.viewer_camera_look_at(env.viewer, None, cam_eye, cam_target)

            # Telemetry logging
            clr_exact, clr_L, clr_R = compute_exact_cylinder_clearance(env)
            r_raw, p_raw, _ = get_euler_xyz(env.base_quat)
            roll = torch.atan2(torch.sin(r_raw), torch.cos(r_raw))
            pitch = torch.atan2(torch.sin(p_raw), torch.cos(p_raw))

            writer.writerow([
                f"{t_sim:.4f}", STATE_NAMES[env.fsm_state[0].item()],
                f"{env.actual_cmd_vy[0].item():.3f}", f"{env.base_lin_vel[0, 1].item():.3f}",
                f"{np.degrees(roll[0].item()):.2f}", f"{np.degrees(pitch[0].item()):.2f}",
                f"{root_pos[2]:.3f}", f"{clr_exact[0].item()*100:.2f}",
                f"{env.torques[0, env.left_knee_idx].item():.2f}", f"{env.torques[0, env.right_knee_idx].item():.2f}",
                f"{env.contact_forces[0, env.feet_indices[0], 2].item():.1f}", f"{env.contact_forces[0, env.feet_indices[1], 2].item():.1f}",
                env.last_transition_event[0]
            ])

            if step % 25 == 0:
                print(f"\r[{t_sim:5.2f}s] State: {STATE_NAMES[env.fsm_state[0].item()]:<13} | CmdVy: {user_vy:+.2f} ActVy: {env.base_lin_vel[0, 1].item():+.2f} | CmdH: {user_height:.3f}m ActH: {root_pos[2]:.3f}m | Clr: {clr_exact[0].item()*100:5.1f}cm | Roll: {np.degrees(roll[0].item()):+5.2f}° | Evt: {env.last_transition_event[0]:<22}", end="")

            step += 1
    finally:
        kb.restore()
        csv_file.close()
        print("\n\n--> 交互视窗已退出。遥测日志保存至:", parsed_args.csv_path)

if __name__ == "__main__":
    main()
