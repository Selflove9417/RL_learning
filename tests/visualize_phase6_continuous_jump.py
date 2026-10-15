#!/usr/bin/env python3
"""
MyRobot Phase 6: 3D GUI Interactive Visualization & Teleoperation Dashboard
=============================================================================
Pure Manual Teleoperation by Default:
  The robot starts in stable standing BALANCE state, waiting for user input.
  Supports simultaneous dual-channel keyboard input from BOTH:
    1. Isaac Gym 3D GUI Window (Direct OpenGL keyboard events)
    2. Terminal stdin (Non-blocking keyboard reader)

Keybindings (Both in 3D Viewer & Terminal):
  [W] / [↑]   : Forward acceleration (+0.05 m/s per tap, max +0.50 m/s)
  [S] / [↓]   : Backward acceleration (-0.05 m/s per tap, min -0.30 m/s)
  [A] / [←]   : Turn left (+0.05 rad/s per tap, max +0.35 rad/s)
  [D] / [→]   : Turn right (-0.05 rad/s per tap, min -0.35 rad/s)
  [Q]         : Raise base height (+0.01 m per tap, max 0.52 m)
  [E]         : Lower base height (-0.01 m per tap, min 0.34 m)
  [Space]     : Trigger Jump (STOPPING -> PREPARE -> JUMP -> RECOVER -> BALANCE)
  [X]         : Active Brake / Parking (Target speed -> 0, active PI clamp)
  [R]         : Reset robot, height & policy states
  [ESC]       : Exit viewer & save 50 Hz CSV telemetry
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
    STATE_NAMES,
    get_euler_xyz
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

def run_phase6_viewer(seed=42, auto_mission=False):
    sys.argv = ['viewer', '--task', 'myrobot_jump']
    args = get_args()
    args.headless = False
    args.seed = seed

    env_cfg, train_cfg = task_registry.get_cfgs(name='myrobot_jump')
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.env.episode_length_s = 1000.0

    task_name = f"myrobot_p6_viewer_seed_{seed}"
    task_registry.register(task_name, Phase6ContinuousDoubleJumpEnv, env_cfg, train_cfg)
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # Subscribe 3D Viewer window keyboard events
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

    # Load frozen policies
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

    kb = NonBlockingKeyboard()
    user_vy = 0.0
    user_yaw = 0.0
    user_height = 0.39774

    csv_path = "evaluation_assets/phase6/viewer_telemetry.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        "timestamp_s", "fsm_state", "cmd_vy", "act_vy",
        "roll_deg", "pitch_deg", "height_m", "clr_exact_cm",
        "tau_knee_L", "tau_knee_R", "fn_L", "fn_R", "event"
    ])

    print("\n" + "=" * 90)
    print(" === MyRobot Phase 6: 纯手动交互 3D Viewer 已启动 ===")
    print(" 运行模式 : 100% 纯手动控制 (默认在 BALANCE 状态原地站立待命)")
    print(" 键盘支持 : 3D 渲染视窗与终端控制台均可直接响应按键")
    print(" 控制按键 :")
    print("   [W] / [↑]   : 前进加速 (+0.05 m/s, 最大 +0.50 m/s)")
    print("   [S] / [↓]   : 后退加速 (-0.05 m/s, 最小 -0.30 m/s)")
    print("   [A] / [←]   : 左转偏航 (+0.05 rad/s, 最大 +0.35 rad/s)")
    print("   [D] / [→]   : 右转偏航 (-0.05 rad/s, 最小 -0.35 rad/s)")
    print("   [Q]         : 升高机身 (+0.01 m, 最大 0.52 m)")
    print("   [E]         : 降低机身 (-0.01 m, 最小 0.34 m)")
    print("   [Space]     : 请求跳跃 (随时按下，自动平滑停车并爆发起跳)")
    print("   [X]         : 驻车刹停 (指令归零，激活主动防滑 PI 积分)")
    print("   [R]         : 复位仿真与高度")
    print("   [ESC]       : 退出 Viewer")
    print("=" * 90 + "\n")

    step = 0
    t_start = time.time()
    try:
        while not env.gym.query_viewer_has_closed(env.viewer):
            t_sim = step * env.dt

            # 1. Read Terminal stdin key
            term_key = kb.read_key()
            if term_key is not None:
                k = term_key.upper()
                if k == 'ESC':
                    break
                elif k in ['W', 'UP']:
                    user_vy = round(min(0.50, user_vy + 0.05), 2)
                    auto_mission = False
                elif k in ['S', 'DOWN']:
                    user_vy = round(max(-0.30, user_vy - 0.05), 2)
                    auto_mission = False
                elif k in ['A', 'LEFT']:
                    user_yaw = round(min(0.35, user_yaw + 0.05), 2)
                    auto_mission = False
                elif k in ['D', 'RIGHT']:
                    user_yaw = round(max(-0.35, user_yaw - 0.05), 2)
                    auto_mission = False
                elif k == 'Q':
                    user_height = round(min(0.520, user_height + 0.01), 3)
                elif k == 'E':
                    user_height = round(max(0.340, user_height - 0.01), 3)
                elif k == 'X':
                    user_vy = 0.0
                    user_yaw = 0.0
                    auto_mission = False
                elif k == ' ':
                    env.request_teleop_jump()
                    auto_mission = False
                elif k == 'R':
                    env.reset()
                    env.root_states[:, 7:13] = 0.0
                    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
                    env.init_dual_policies(ac_bal, ac_jump)
                    step = 0
                    user_vy = 0.0
                    user_yaw = 0.0
                    user_height = 0.39774

            # 2. Read 3D Viewer window events (Direct OpenGL GUI focus)
            for evt in env.gym.query_viewer_action_events(env.viewer):
                act = evt.action
                val = evt.value
                if val > 0.5:
                    if act == "QUIT":
                        return
                    elif act == "FORWARD":
                        user_vy = round(min(0.50, user_vy + 0.05), 2)
                        auto_mission = False
                    elif act == "BACKWARD":
                        user_vy = round(max(-0.30, user_vy - 0.05), 2)
                        auto_mission = False
                    elif act == "LEFT":
                        user_yaw = round(min(0.35, user_yaw + 0.05), 2)
                        auto_mission = False
                    elif act == "RIGHT":
                        user_yaw = round(max(-0.35, user_yaw - 0.05), 2)
                        auto_mission = False
                    elif act == "RISE":
                        user_height = round(min(0.520, user_height + 0.01), 3)
                    elif act == "LOWER":
                        user_height = round(max(0.340, user_height - 0.01), 3)
                    elif act == "BRAKE":
                        user_vy = 0.0
                        user_yaw = 0.0
                        auto_mission = False
                    elif act == "JUMP":
                        env.request_teleop_jump()
                        auto_mission = False
                    elif act == "RESET":
                        env.reset()
                        env.root_states[:, 7:13] = 0.0
                        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
                        env.init_dual_policies(ac_bal, ac_jump)
                        step = 0
                        user_vy = 0.0
                        user_yaw = 0.0
                        user_height = 0.39774

            # Optional automated mission flow if explicitly requested via --auto
            if auto_mission:
                if 1.5 <= t_sim < 4.0:
                    user_vy = 0.30
                elif 6.5 <= t_sim < 9.0:
                    user_vy = 0.30
                else:
                    user_vy = 0.0

                if step == int(4.0 / env.dt):
                    env.request_teleop_jump()
                elif step == int(9.0 / env.dt):
                    env.request_teleop_jump()

            env.set_teleop_targets(vy=user_vy, yaw=user_yaw, height=user_height)
            env.step_dual_policy()

            # Camera follow behind robot
            root_pos = env.root_states[0, 0:3].cpu().numpy()
            cam_target = gymapi.Vec3(root_pos[0], root_pos[1], root_pos[2] + 0.10)
            cam_eye = gymapi.Vec3(root_pos[0] - 1.5, root_pos[1] - 1.8, root_pos[2] + 0.70)
            env.gym.viewer_camera_look_at(env.viewer, env.envs[0], cam_eye, cam_target)

            # Render graphics (fetch GPU PhysX results to graphics engine)
            if env.device != 'cpu':
                env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)
            env.gym.sync_frame_time(env.sim)

            # Telemetry logging: clr_exact is bilateral minimum clearance
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

            if step % 25 == 0:  # 10 Hz refresh
                print(f"\r[{t_sim:5.2f}s] State: {STATE_NAMES[env.fsm_state[0].item()]:<13} | CmdVy: {user_vy:+.2f} ActVy: {env.base_lin_vel[0, 1].item():+.2f} | CmdH: {user_height:.3f}m ActH: {root_pos[2]:.3f}m | Clr: {clr_exact[0].item()*100:5.1f}cm | Roll: {np.degrees(roll[0].item()):+5.2f}° | Evt: {env.last_transition_event[0]:<22}", end="")

            step += 1
    finally:
        kb.restore()
        csv_file.close()
        print("\n\n--> Viewer session closed. Telemetry saved to:", csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Simulation random seed")
    parser.add_argument("--auto", action="store_true", help="Run automated demonstration (Default: False, pure manual teleop)")
    parsed, _ = parser.parse_known_args()
    run_phase6_viewer(seed=parsed.seed, auto_mission=parsed.auto)
