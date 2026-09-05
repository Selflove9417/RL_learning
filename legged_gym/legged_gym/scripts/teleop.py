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
    print("\n" + "=" * 65)
    print("🎮 双足轮足机器人实时键盘遥控系统 (Teleoperation)")
    print("=" * 65)
    print("  [W] 或 [↑]   : 前进加速 (+0.20 m/s, 上限 1.00 m/s [平滑滤波])")
    print("  [S] 或 [↓]   : 减速 / 倒车 (-0.20 m/s, 下限 -0.80 m/s [平滑滤波])")
    print("  [A] 或 [←]   : 向左旋转转向 (+0.25 rad/s, 上限 1.00 rad/s)")
    print("  [D] 或 [→]   : 向右旋转转向 (-0.25 rad/s, 下限 -1.00 rad/s)")
    print("  [Space]      : 紧急刹车 / 驻车自平衡")
    print("  [R]          : 重置机器人到原点")
    print("  [Q]          : 退出遥控")
    print("=" * 65)
    print("\n⚠️  注意: 请在终端窗口按键（不需要点击 3D 仿真窗口！）\n")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # Inference/deployment must use the final actuator range, not the temporary
    # training curriculum that starts from the legacy policy's wide actions.
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

    # 禁用仿真器内部随机速度重采样
    env._resample_commands = types.MethodType(lambda self, env_ids: None, env)

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # 目标速度与平滑实际指令 (Ramp Limiter)
    target_ctrl = {'vel_y': 0.0, 'ang': 0.0}
    actual_cmd = {'vel_y': 0.0, 'ang': 0.0}
    max_accel_y = 1.0    # 最大线性加速度 1.0 m/s^2，杜绝阶跃冲撞
    max_accel_ang = 2.5  # 最大角加速度 2.5 rad/s^2

    def process_key(key):
        if key is None:
            return False
        key_lower = key.lower()
        changed = True
        # 限制在训练分布区间内: vel_y in [-0.8, 1.0], ang in [-1.0, 1.0]
        if key_lower == 'w' or key == 'UP':
            target_ctrl['vel_y'] = round(min(target_ctrl['vel_y'] + 0.20, 1.0), 2)
        elif key_lower == 's' or key == 'DOWN':
            target_ctrl['vel_y'] = round(max(target_ctrl['vel_y'] - 0.20, -0.8), 2)
        elif key_lower == 'a' or key == 'LEFT':
            target_ctrl['ang'] = round(min(target_ctrl['ang'] + 0.25, 1.0), 2)
        elif key_lower == 'd' or key == 'RIGHT':
            target_ctrl['ang'] = round(max(target_ctrl['ang'] - 0.25, -1.0), 2)
        elif key == ' ':
            target_ctrl['vel_y'] = 0.0
            target_ctrl['ang'] = 0.0
        elif key_lower == 'r':
            target_ctrl['vel_y'] = 0.0
            target_ctrl['ang'] = 0.0
            actual_cmd['vel_y'] = 0.0
            actual_cmd['ang'] = 0.0
            env.reset()
        elif key_lower == 'q' or key == '\x03':  # Q 或 Ctrl+C
            return 'quit'
        else:
            changed = False
        return changed

    def print_hud(force=False):
        vy, ang = target_cmd_y, target_ang
        if abs(actual_vy) < 0.05 and abs(actual_yaw) < 0.05:
            status = "🛑 驻车静止"
        elif actual_vy > 0.05:
            status = "🏎️ 前进巡航"
        else:
            status = "🔙 倒车行驶"
        if abs(actual_yaw) > 0.05:
            status += " (转向中)"
        sys.stdout.write(
            f"\r\033[K[{status}] 目标:{vy:+.2f}m/s (平滑:{actual_cmd['vel_y']:+.2f}) | "
            f"实际:{actual_vy:+.2f}m/s | "
            f"转向目标:{ang:+.2f} | "
            f"实际:{actual_yaw:+.2f} rad/s"
        )
        sys.stdout.flush()

    kb = KeyboardReader()
    target_cmd_y = 0.0
    target_ang = 0.0
    actual_vy = 0.0
    actual_yaw = 0.0
    print_hud()

    step_count = 0
    import math
    try:
        while True:
            step_count += 1

            # 读取键盘 (终端 raw 模式，不依赖 3D 窗口焦点)
            key = kb.read_key()
            result = process_key(key)
            if result == 'quit':
                break

            target_cmd_y = target_ctrl['vel_y']
            target_ang = target_ctrl['ang']

            # 指令斜坡发生器 (平滑加速度过渡)
            dt = env.dt
            delta_y = target_cmd_y - actual_cmd['vel_y']
            step_max_y = max_accel_y * dt
            if abs(delta_y) <= step_max_y:
                actual_cmd['vel_y'] = target_cmd_y
            else:
                actual_cmd['vel_y'] += math.copysign(step_max_y, delta_y)

            delta_ang = target_ang - actual_cmd['ang']
            step_max_ang = max_accel_ang * dt
            if abs(delta_ang) <= step_max_ang:
                actual_cmd['ang'] = target_ang
            else:
                actual_cmd['ang'] += math.copysign(step_max_ang, delta_ang)

            # 注入平滑指令
            env.commands[:, 0] = 0.0
            env.commands[:, 1] = actual_cmd['vel_y']
            env.commands[:, 2] = actual_cmd['ang']

            # 刷新观测向量，确保网络推理即时获取最新指令
            env.compute_observations()
            obs = env.get_observations()

            # 推理并步进
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())

            actual_vy = env.base_lin_vel[0, 1].item()
            actual_yaw = env.base_ang_vel[0, 2].item()

            # 检查 3D 窗口是否关闭
            if env.gym.query_viewer_has_closed(env.viewer):
                break

            if step_count % 10 == 0 or result:
                print_hud()

    finally:
        kb.restore()
        sys.stdout.write('\n')
        print("遥控已退出。")


if __name__ == '__main__':
    args = get_args()
    teleop(args)
