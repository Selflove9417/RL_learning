# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import types
import tty
import termios
import select
from legged_gym import LEGGED_GYM_ROOT_DIR

# 重要: isaacgym 必须在 torch 之前导入
import isaacgym
from isaacgym import gymapi

import numpy as np
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger


class KeyboardReader:
    """
    从终端原始模式读取键盘按键，非阻塞，不依赖 Isaac Gym 事件系统。
    支持: W/S/A/D, 方向键 (ESC [ A/B/C/D), Space, R
    """
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        # 切换到 raw 模式: 按键立即可读，不需要回车
        tty.setraw(self.fd)

    def read_key(self):
        """非阻塞读取按键, 返回字符串 key 或 None"""
        if select.select([sys.stdin], [], [], 0.0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                # 方向键: ESC [ A/B/C/D
                if select.select([sys.stdin], [], [], 0.005)[0]:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[' and select.select([sys.stdin], [], [], 0.005)[0]:
                        ch3 = sys.stdin.read(1)
                        return {'A': 'UP', 'B': 'DOWN', 'C': 'RIGHT', 'D': 'LEFT'}.get(ch3, None)
                return None  # 单独 ESC，忽略
            return ch
        return None

    def restore(self):
        """恢复终端设置"""
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def teleop(args):
    print("\n" + "=" * 72)
    print("🎮 双足轮足机器人实时键盘遥控系统 (Phase 5.3 运动与高度全态遥控)")
    print("=" * 72)
    print("  【运动控制 (速度/转向)】:")
    print("    [W] 或 [↑]   : 前进加速 (+0.04 m/s, 上限 +0.20 m/s [平滑滤波])")
    print("    [S] 或 [↓]   : 减速 / 倒车 (-0.04 m/s, 下限 -0.20 m/s [平滑滤波])")
    print("    [A] 或 [←]   : 向左旋转转向 (+0.05 rad/s, 上限 +0.35 rad/s)")
    print("    [D] 或 [→]   : 向右旋转转向 (-0.05 rad/s, 下限 -0.35 rad/s)")
    print("  【高度调节 (动态升降/下蹲)】:")
    print("    [E] 或 [U]   : 升高机身 (+0.01 m, 上限 0.440 m)")
    print("    [C] 或 [J]   : 降低/下蹲 (-0.01 m, 下限 0.360 m)")
    print("    [Z] 或 [N]   : 恢复标称高度 (0.398 m)")
    print("  【辅助控制】:")
    print("    [Space]      : 刹车驻车 (速度清零，保持当前高度平衡)")
    print("    [R]          : 重置机器人到初始状态")
    print("    [Q] 或 [ESC] : 退出遥控系统")
    print("=" * 72)
    print("💡 提示: 既可以在【终端窗口】按键，也可以直接点击【3D 仿真窗口】按键！\n")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # 部署推理必须使用最终执行器范围
    if hasattr(env_cfg.control, 'wheel_action_limit_curriculum'):
        env_cfg.control.wheel_action_limit_curriculum = False
    if hasattr(env_cfg.control, 'wheel_damping_curriculum'):
        env_cfg.control.wheel_damping_curriculum = False
    if hasattr(env_cfg.control, 'wheel_torque_curriculum'):
        env_cfg.control.wheel_torque_curriculum = False
    if hasattr(env_cfg.control, 'wheel_velocity_damping_curriculum'):
        env_cfg.control.wheel_velocity_damping_curriculum = False
    if hasattr(env_cfg.control, 'wheel_yaw_damping_curriculum'):
        env_cfg.control.wheel_yaw_damping_curriculum = False
    if hasattr(env_cfg.control, 'wheel_diff_damping_curriculum'):
        env_cfg.control.wheel_diff_damping_curriculum = False
    if hasattr(env_cfg.rewards, 'tracking_lin_vel_curriculum'):
        env_cfg.rewards.tracking_lin_vel_curriculum = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # 禁用仿真器内部随机重采样
    env._resample_commands = types.MethodType(lambda self, env_ids: None, env)

    # 注册 3D Viewer 键盘事件 (使得点击 3D 窗口焦点也能按键遥控)
    if env.viewer is not None:
        key_mappings = [
            (gymapi.KEY_W, "FORWARD"),
            (gymapi.KEY_S, "BACKWARD"),
            (gymapi.KEY_A, "LEFT"),
            (gymapi.KEY_D, "RIGHT"),
            (gymapi.KEY_UP, "FORWARD"),
            (gymapi.KEY_DOWN, "BACKWARD"),
            (gymapi.KEY_LEFT, "LEFT"),
            (gymapi.KEY_RIGHT, "RIGHT"),
            (gymapi.KEY_E, "RISE"),
            (gymapi.KEY_U, "RISE"),
            (gymapi.KEY_C, "SQUAT"),
            (gymapi.KEY_J, "SQUAT"),
            (gymapi.KEY_Z, "NOMINAL_H"),
            (gymapi.KEY_N, "NOMINAL_H"),
            (gymapi.KEY_SPACE, "BRAKE"),
            (gymapi.KEY_R, "RESET"),
            (gymapi.KEY_Q, "QUIT"),
        ]
        for key_code, act_name in key_mappings:
            env.gym.subscribe_viewer_keyboard_event(env.viewer, key_code, act_name)

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # 目标控制值与平滑过渡指令 (Ramp Limiter)
    target_ctrl = {'vel_y': 0.0, 'ang': 0.0, 'height': 0.398}
    actual_cmd = {'vel_y': 0.0, 'ang': 0.0, 'height': 0.398}
    max_accel_y = 0.8    # 最大线性加速度 0.8 m/s^2，平滑不冲撞
    max_accel_ang = 2.0  # 最大角加速度 2.0 rad/s^2
    max_speed_h = 0.04   # 最大变高速度 0.04 m/s (4 cm/s)，平滑升降

    def process_key_event(action_name):
        if action_name is None:
            return False
        changed = True
        act = action_name.upper()

        if act in ['W', 'UP', 'FORWARD']:
            target_ctrl['vel_y'] = round(min(target_ctrl['vel_y'] + 0.04, 0.20), 2)
        elif act in ['S', 'DOWN', 'BACKWARD']:
            target_ctrl['vel_y'] = round(max(target_ctrl['vel_y'] - 0.04, -0.20), 2)
        elif act in ['A', 'LEFT']:
            target_ctrl['ang'] = round(min(target_ctrl['ang'] + 0.05, 0.35), 2)
        elif act in ['D', 'RIGHT']:
            target_ctrl['ang'] = round(max(target_ctrl['ang'] - 0.05, -0.35), 2)
        elif act in ['E', 'U', 'RISE']:
            target_ctrl['height'] = round(min(target_ctrl['height'] + 0.01, 0.440), 3)
        elif act in ['C', 'J', 'SQUAT']:
            target_ctrl['height'] = round(max(target_ctrl['height'] - 0.01, 0.360), 3)
        elif act in ['Z', 'N', 'NOMINAL_H']:
            target_ctrl['height'] = 0.398
        elif act in [' ', 'BRAKE']:
            target_ctrl['vel_y'] = 0.0
            target_ctrl['ang'] = 0.0
        elif act in ['R', 'RESET']:
            target_ctrl['vel_y'] = 0.0
            target_ctrl['ang'] = 0.0
            target_ctrl['height'] = 0.398
            actual_cmd['vel_y'] = 0.0
            actual_cmd['ang'] = 0.0
            actual_cmd['height'] = 0.398
            env.reset()
        elif act in ['Q', 'QUIT', '\x03']:
            return 'quit'
        else:
            changed = False
        return changed

    def print_hud():
        status_items = []
        if abs(actual_vy) < 0.03 and abs(actual_yaw) < 0.03:
            status_items.append("🛑 驻车静止")
        elif actual_vy > 0.03:
            status_items.append("🏎️ 前进巡航")
        else:
            status_items.append("🔙 倒车行驶")

        if abs(actual_yaw) > 0.03:
            status_items.append("🔄 差动转向")

        if actual_h < 0.380:
            status_items.append("⬇️ 低蹲姿态")
        elif actual_h > 0.420:
            status_items.append("⬆️ 抬高姿态")
        else:
            status_items.append("⚖️ 标称高度")

        h_err_cm = (actual_h - actual_cmd['height']) * 100.0
        status_str = " | ".join(status_items)
        sys.stdout.write(
            f"\r\033[K[{status_str}] "
            f"高度目标:{actual_cmd['height']:.3f}m(实测:{actual_h:.3f}m, 差:{h_err_cm:+.1f}cm) | "
            f"线速目标:{actual_cmd['vel_y']:+.2f}m/s(实测:{actual_vy:+.2f}) | "
            f"转向目标:{actual_cmd['ang']:+.2f}rad/s(实测:{actual_yaw:+.2f})"
        )
        sys.stdout.flush()

    kb = KeyboardReader()
    actual_vy = 0.0
    actual_yaw = 0.0
    actual_h = 0.398
    print_hud()

    step_count = 0
    import math
    try:
        while True:
            step_count += 1
            key_changed = False

            # 1. 优先读取终端按键 (non-blocking)
            key = kb.read_key()
            if key is not None:
                res = process_key_event(key)
                if res == 'quit':
                    break
                if res:
                    key_changed = True

            # 2. 读取 3D 仿真窗口按键事件
            if env.viewer is not None:
                events = env.gym.query_viewer_action_events(env.viewer)
                for event in events:
                    if event.value > 0.5: # Key press
                        res = process_key_event(event.action)
                        if res == 'quit':
                            return
                        if res:
                            key_changed = True

            # 指令斜坡发生器 (平滑加速度过渡)
            dt = env.dt
            # 纵向速度平滑
            target_vy = target_ctrl['vel_y']
            delta_y = target_vy - actual_cmd['vel_y']
            step_max_y = max_accel_y * dt
            if abs(delta_y) <= step_max_y:
                actual_cmd['vel_y'] = target_vy
            else:
                actual_cmd['vel_y'] += math.copysign(step_max_y, delta_y)

            # 偏航角速度平滑
            target_ang = target_ctrl['ang']
            delta_ang = target_ang - actual_cmd['ang']
            step_max_ang = max_accel_ang * dt
            if abs(delta_ang) <= step_max_ang:
                actual_cmd['ang'] = target_ang
            else:
                actual_cmd['ang'] += math.copysign(step_max_ang, delta_ang)

            # 机身高度平滑
            target_h = target_ctrl['height']
            delta_h = target_h - actual_cmd['height']
            step_max_h = max_speed_h * dt
            if abs(delta_h) <= step_max_h:
                actual_cmd['height'] = target_h
            else:
                actual_cmd['height'] += math.copysign(step_max_h, delta_h)

            # 注入平滑指令到仿真环境
            env.commands[:, 0] = 0.0
            env.commands[:, 1] = actual_cmd['vel_y']
            env.commands[:, 2] = actual_cmd['ang']
            env.height_cmd_buf[:] = actual_cmd['height']

            # 刷新观测向量，确保网络推理即时获取最新指令与高度特征
            env.compute_observations()
            obs = env.get_observations()

            # 推理并步进
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())

            actual_vy = env.base_lin_vel[0, 1].item()
            actual_yaw = env.base_ang_vel[0, 2].item()
            actual_h = env.root_states[0, 2].item()

            # 检查 3D 窗口是否关闭
            if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
                break

            if step_count % 10 == 0 or key_changed:
                print_hud()

    finally:
        kb.restore()
        sys.stdout.write('\n')
        print("遥控系统已安全退出。")


if __name__ == '__main__':
    args = get_args()
    teleop(args)
