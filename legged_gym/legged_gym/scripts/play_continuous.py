#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# MyRobot Running Jump: Continuous Multi-Cycle Interactive Viewer & Teleoperation Demo
# ======================================================================================
# Continuous running jump demonstration mode without the 2.8s automatic reset.
# Loads Iter 3 weights (Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt).
#
# Features:
#   1. Independent Continuous Mode: No 2.8s timeout resets. Robot cruises and jumps continuously.
#      Resets ONLY if the robot falls (height < 0.20m, pitch > 45°, roll > 40°) or user presses [R].
#   2. World-Frame Ground Reference Markers:
#      Draws a high-visibility track runway with 10 cm tick marks, 50 cm marks, and 1.0 m gold lines
#      that stay fixed in the world frame and DO NOT move with the robot.
#   3. Real-Time Heads-Up Telemetry HUD:
#      Outputs world-frame position Y, forward velocity v_Y^world, bilateral cylinder clearance,
#      per-jump horizontal flight distance, touchdown impact, and FSM lifecycle events.
#   4. Dual-Focus Keyboard Controls (Viewer window + Terminal):
#      [Space]     : Trigger dynamic jump immediately in-motion
#      [W] / [↑]   : Forward speed up (+0.05 m/s, max 0.50 m/s)
#      [S] / [↓]   : Forward speed down (-0.05 m/s)
#      [A] / [D]   : Steer left / right (yaw rate)
#      [C]         : Toggle camera perspective (Chase 45° -> Lateral side-view -> Fixed world)
#      [R]         : Reset simulation back to start line (Y=0)
#      [ESC]       : Exit cleanly
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


def generate_runway_lines(y_min=-2.0, y_max=60.0):
    """
    Builds fixed ground reference markings in the world coordinate frame:
    - Left and right track boundary rails at X = -0.6m and X = +0.6m (cyan).
    - Start line at Y = 0.0m (bright red, X in [-1.0, 1.0]).
    - Major 1-meter distance markers (bright yellow, X in [-0.7, 0.7]).
    - Intermediate 0.5-meter markers (cyan, X in [-0.5, 0.5]).
    - High-density 0.1-meter (10 cm) tick marks (light grey, X in [-0.25, 0.25]).
    """
    lines = []
    colors = []

    # Longitudinal boundary rails
    for x_rail in [-0.6, 0.6]:
        lines.extend([x_rail, y_min, 0.001, x_rail, y_max, 0.001])
        colors.extend([0.2, 0.8, 1.0])

    # Transverse distance markers
    y = y_min
    while y <= y_max + 1e-4:
        is_zero = abs(y) < 1e-4
        is_meter = abs(round(y) - y) < 1e-4
        is_half_meter = abs(round(y * 2) - y * 2) < 1e-4

        if is_zero:
            lines.extend([-1.0, 0.0, 0.003, 1.0, 0.0, 0.003])
            colors.extend([1.0, 0.1, 0.1])  # Red start line
        elif is_meter:
            lines.extend([-0.7, y, 0.0025, 0.7, y, 0.0025])
            colors.extend([1.0, 0.85, 0.0])  # Gold 1.0m lines
        elif is_half_meter:
            lines.extend([-0.5, y, 0.002, 0.5, y, 0.002])
            colors.extend([0.0, 0.9, 0.9])  # Cyan 0.5m lines
        else:
            lines.extend([-0.25, y, 0.0015, 0.25, y, 0.0015])
            colors.extend([0.65, 0.65, 0.65])  # Light Grey 10cm ticks

        y = round(y + 0.1, 2)

    num_lines = len(lines) // 6
    return num_lines, np.array(lines, dtype=np.float32), np.array(colors, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="MyRobot Running Jump Continuous Demonstration Viewer")
    parser.add_argument("--load_run", type=str, default="Oct19_21-52-03_running_jump_020ms_iter3_controlled", help="Run folder name under legged_gym/logs/myrobot_running_jump")
    parser.add_argument("--checkpoint", type=int, default=3, help="Checkpoint iteration number (default: 3)")
    parser.add_argument("--target_speed", type=float, default=0.20, help="Initial cruise forward speed in m/s (default: 0.20)")
    parser.add_argument("--auto_jump", action="store_true", default=True, help="Automatically trigger jumps every cycle (default: True)")
    parser.add_argument("--manual_jump_only", action="store_true", default=False, help="Disable auto-jump and only jump on Space bar")
    parser.add_argument("--seed", type=int, default=42, help="Simulation seed")
    parser.add_argument("--headless", action="store_true", default=False, help="Run headless without 3D window (for testing)")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps to run (-1 for infinite until closed/ESC)")
    parsed_args, _ = parser.parse_known_args()

    # Pass minimal args to legged_gym get_args
    sys.argv = ['play_continuous.py', '--task', 'myrobot_running_jump']
    args = get_args()
    args.headless = parsed_args.headless
    args.seed = parsed_args.seed

    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_running_jump")
    env_cfg.env.num_envs = 1
    env_cfg.seed = parsed_args.seed
    env_cfg.env.continuous_mode = True  # CANCELS 2.8s automatic reset!
    env_cfg.env.auto_jump = not parsed_args.manual_jump_only
    env_cfg.commands.target_forward_vel = parsed_args.target_speed
    env_cfg.commands.ranges.lin_vel_y = [parsed_args.target_speed, parsed_args.target_speed]

    # Model resolution
    log_dir = os.path.join(workspace_root, "legged_gym", "logs", "myrobot_running_jump", parsed_args.load_run)
    model_file = f"model_{parsed_args.checkpoint}.pt"
    model_path = os.path.join(log_dir, model_file)

    if not os.path.exists(model_path):
        print(f"\n[ERROR] Checkpoint not found at: {model_path}")
        print("Available runs in logs/myrobot_running_jump/:")
        available = os.listdir(os.path.join(workspace_root, "legged_gym", "logs", "myrobot_running_jump"))
        for a in available:
            print(f"  - {a}")
        sys.exit(1)

    print("\n" + "=" * 90)
    print(" === MyRobot Running Jump: 连续多周期动态跳跃演示视窗已启动 ===")
    print(f" 加载模型权重 : {model_path}")

    # Verify weights
    ckpt = torch.load(model_path, map_location="cpu")
    iter_loaded = ckpt.get('iter', parsed_args.checkpoint)
    print(f" 确认权重版本 : Checkpoint Iteration #{iter_loaded} (确定加载 Iter 3 受控探索权重)")
    print(f" 目标巡航速度 : {parsed_args.target_speed:.2f} m/s | 自动周期跳跃: {not parsed_args.manual_jump_only}")
    print(" 连续演示模式 : [已开启] 彻底取消 2.8 秒自动 Reset，只在跌倒或按 R 时重置！")
    print(" 地面标尺参照 : [已绘制] 世界坐标系 10cm 细刻度、0.5m 分米线、1.0m 金色米标及原点起跑线")
    print(" 交互控制按键 :")
    print("   [Space]     : 立即动态起跳 (在行进中保持前进速度直接起飞)")
    print("   [W] / [↑]   : 提高巡航车速 (+0.05 m/s, 最大 +0.50 m/s)")
    print("   [S] / [↓]   : 降低巡航车速 (-0.05 m/s, 最小 -0.10 m/s)")
    print("   [A] / [D]   : 左/右偏航转向")
    print("   [C]         : 切换摄像机视角 (45°追随 -> 纯侧视跟随 -> 世界固定机位)")
    print("   [R]         : 复位仿真回到起跑线 (Y=0)")
    print("   [ESC]       : 退出演示")
    print("=" * 90 + "\n")

    env, _ = task_registry.make_env(name="myrobot_running_jump", args=args, env_cfg=env_cfg)
    env.reset()
    obs = env.get_observations()

    # Load ActorCritic policy
    policy = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()

    # Register 3D viewer keyboard events and draw lines if viewer is active
    num_runway_lines, line_verts, line_colors = generate_runway_lines(y_min=-2.0, y_max=60.0)
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

        env.gym.clear_lines(env.viewer)
        env.gym.add_lines(env.viewer, None, num_runway_lines, line_verts, line_colors)

    kb = NonBlockingKeyboard()
    user_vy = parsed_args.target_speed
    user_yaw = 0.0
    cam_mode = 0  # 0: Chase 45°, 1: Lateral Side-view, 2: Fixed World Cam

    step = 0
    jump_count = 0
    current_jump = None
    completed_jumps = []

    try:
        while (env.viewer is None or not env.gym.query_viewer_has_closed(env.viewer)):
            if parsed_args.max_steps > 0 and step >= parsed_args.max_steps:
                break
            t_sim = step * env.dt

            # 1. Read terminal keyboard events
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
                elif k in ['D', 'RIGHT']:
                    user_yaw = round(max(-0.35, user_yaw - 0.05), 2)
                    env.user_cmd_yaw[:] = user_yaw
                elif k == ' ':
                    env.request_teleop_jump()
                elif k == 'C':
                    cam_mode = (cam_mode + 1) % 3
                elif k == 'R':
                    env.reset()
                    obs = env.get_observations()
                    step = 0
                    jump_count = 0
                    current_jump = None
                    env.gym.clear_lines(env.viewer)
                    env.gym.add_lines(env.viewer, None, num_runway_lines, line_verts, line_colors)
                    print("\n--> [RESET] 仿真已重置回原点 Y=0.00m")

            # 2. Read 3D viewer window events
            if env.viewer is not None:
                for evt in env.gym.query_viewer_action_events(env.viewer):
                    if evt.value > 0.5:
                        act = evt.action
                        if act == "QUIT":
                            return
                        elif act == "SPEED_UP":
                            user_vy = round(min(0.50, user_vy + 0.05), 2)
                            env.user_cmd_vy[:] = user_vy
                            env.commands[:, 1] = user_vy
                        elif act == "SPEED_DOWN":
                            user_vy = round(max(-0.10, user_vy - 0.05), 2)
                            env.user_cmd_vy[:] = user_vy
                            env.commands[:, 1] = user_vy
                        elif act == "STEER_LEFT":
                            user_yaw = round(min(0.35, user_yaw + 0.05), 2)
                            env.user_cmd_yaw[:] = user_yaw
                        elif act == "STEER_RIGHT":
                            user_yaw = round(max(-0.35, user_yaw - 0.05), 2)
                            env.user_cmd_yaw[:] = user_yaw
                        elif act == "JUMP":
                            env.request_teleop_jump()
                        elif act == "TOGGLE_CAM":
                            cam_mode = (cam_mode + 1) % 3
                        elif act == "RESET":
                            env.reset()
                            obs = env.get_observations()
                            step = 0
                            jump_count = 0
                            current_jump = None
                            env.gym.clear_lines(env.viewer)
                            env.gym.add_lines(env.viewer, None, num_runway_lines, line_verts, line_colors)
                            print("\n--> [RESET] 仿真已重置回原点 Y=0.00m")

            # 3. Policy inference and environment step
            with torch.no_grad():
                actions = policy.act_inference(obs)

            obs, _, _, dones, _ = env.step(actions)

            # Check if fall occurred (the only way reset triggers in continuous mode)
            if dones[0].item():
                print(f"\n--> [FALL RESET] 机器人失衡或倾角超限，已自动复位回起点！")
                step = 0
                jump_count = 0
                current_jump = None
                env.gym.clear_lines(env.viewer)
                env.gym.add_lines(env.viewer, None, num_runway_lines, line_verts, line_colors)
                continue

            # 4. Kinematics & jump telemetry
            fsm = env.fsm_state[0].item()
            root_pos = env.root_states[0, 0:3].cpu().numpy()
            v_world_y = env.root_states[0, 8].item()
            v_body_y = env.base_lin_vel[0, 1].item()
            clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(env)
            clr_cm = clr_min[0].item() * 100.0

            r_raw, p_raw, _ = get_euler_xyz(env.base_quat)
            pitch_deg = math.degrees(torch.atan2(torch.sin(r_raw), torch.cos(r_raw))[0].item())
            roll_deg = math.degrees(torch.atan2(torch.sin(p_raw), torch.cos(p_raw))[0].item())

            fn_L = env.contact_forces[0, env.feet_indices[0], 2].item()
            fn_R = env.contact_forces[0, env.feet_indices[1], 2].item()
            fn_max = max(fn_L, fn_R)

            # Track jump events
            if fsm == env.STATE_JUMP and current_jump is None:
                jump_count += 1
                current_jump = {
                    "idx": jump_count,
                    "trigger_step": step,
                    "takeoff_y": None,
                    "takeoff_vy": None,
                    "touchdown_y": None,
                    "flight_dist": 0.0,
                    "peak_clr": 0.0,
                    "peak_impact": 0.0
                }

            if current_jump is not None:
                if clr_cm > current_jump["peak_clr"]:
                    current_jump["peak_clr"] = clr_cm

                # Liftoff
                if env.has_taken_off[0].item() and current_jump["takeoff_y"] is None:
                    current_jump["takeoff_y"] = root_pos[1]
                    current_jump["takeoff_vy"] = v_world_y

                # Touchdown
                if fsm == env.STATE_RECOVER and current_jump["touchdown_y"] is None:
                    current_jump["touchdown_y"] = root_pos[1]
                    if current_jump["takeoff_y"] is not None:
                        current_jump["flight_dist"] = root_pos[1] - current_jump["takeoff_y"]

                if fsm == env.STATE_RECOVER:
                    if fn_max > current_jump["peak_impact"]:
                        current_jump["peak_impact"] = fn_max

                # Recovered to BALANCE
                if fsm == env.STATE_BALANCE and current_jump["touchdown_y"] is not None:
                    completed_jumps.append(current_jump)
                    print(f"\n >>> [跳跃 #{current_jump['idx']} 完成] 起跳位置: {current_jump['takeoff_y']:.2f}m | 落地位置: {current_jump['touchdown_y']:.2f}m | 腾空净距离: {current_jump['flight_dist']*100:.1f} cm | 最大净空: {current_jump['peak_clr']:.1f} cm | 触地冲击: {current_jump['peak_impact']:.0f} N | 恢复巡航中...")
                    current_jump = None

            # 5. Dynamic Camera Control & Rendering
            if env.viewer is not None:
                if cam_mode == 0:
                    # Mode 0: Smooth 45° chase cam (follows robot against fixed ground lines)
                    cam_target = gymapi.Vec3(0.0, root_pos[1] + 0.35, root_pos[2] + 0.15)
                    cam_eye = gymapi.Vec3(root_pos[0] - 1.8, root_pos[1] - 1.2, root_pos[2] + 0.75)
                elif cam_mode == 1:
                    # Mode 1: Pure lateral side view (shows height clearance and flight arc against runway)
                    cam_target = gymapi.Vec3(0.0, root_pos[1] + 0.2, 0.35)
                    cam_eye = gymapi.Vec3(-2.6, root_pos[1] + 0.2, 0.45)
                else:
                    # Mode 2: Fixed world spectator camera (robot rushes past from start line into distance)
                    cam_target = gymapi.Vec3(0.0, 1.5, 0.35)
                    cam_eye = gymapi.Vec3(-2.5, -0.5, 0.85)

                env.gym.viewer_camera_look_at(env.viewer, env.envs[0], cam_eye, cam_target)

                # 6. Render frame
                if env.device != 'cpu':
                    env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.draw_viewer(env.viewer, env.sim, True)
                env.gym.sync_frame_time(env.sim)

            # 7. Real-time HUD Telemetry
            if step % 5 == 0:
                cam_names = ["追随45°", "纯侧视", "世界固定机位"]
                last_dist_str = f"{completed_jumps[-1]['flight_dist']*100:.1f}cm" if completed_jumps else "--"
                print(
                    f"\r[{t_sim:5.2f}s] FSM: {STATE_NAMES[fsm]:<9} | "
                    f"位置Y: {root_pos[1]:+5.2f}m | "
                    f"世界速度Vy: {v_world_y:+.2f}m/s (指令 {user_vy:+.2f}) | "
                    f"净空: {clr_cm:4.1f}cm | "
                    f"跳跃#{jump_count} (上次距离: {last_dist_str}) | "
                    f"俯仰: {pitch_deg:+4.1f}° | 视角: {cam_names[cam_mode]}",
                    end="", flush=True
                )

            step += 1

    finally:
        kb.restore()
        print("\n\n" + "=" * 80)
        print(" === 演示结束：连续跳跃总结报告 ===")
        print(f" 总运行时长 : {step * env.dt:.2f} s")
        print(f" 最终行驶距离: Y = {env.root_states[0, 1].item():.3f} m")
        print(f" 累计跳跃次数: {len(completed_jumps)} 次")
        for j in completed_jumps:
            print(f"   跳跃 #{j['idx']}: 起飞 Y={j['takeoff_y']:.2f}m -> 落地 Y={j['touchdown_y']:.2f}m | 腾空位移 = {j['flight_dist']*100:.1f} cm | 净空 = {j['peak_clr']:.1f} cm | 冲击 = {j['peak_impact']:.0f} N")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
