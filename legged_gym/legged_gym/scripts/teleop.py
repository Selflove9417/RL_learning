# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import types
import tty
import termios
import select

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

from legged_gym import LEGGED_GYM_ROOT_DIR

# 重要: isaacgym 必须在 torch 之前导入
import isaacgym
from isaacgym import gymapi

import numpy as np
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger


# 用户在低蹲仿真中观察到机械干涉；先收紧用户目标高度，间隙仍需实测确认。
# 这些是补偿前的用户高度边界，内部 IK 高度继续使用既有柔性补偿。
TELEOP_MIN_HEIGHT = 0.340
TELEOP_MAX_HEIGHT = 0.520


class KeyboardReader:
    """
    从终端原始模式读取键盘按键，非阻塞，不依赖 Isaac Gym 事件系统。
    支持: W/S/A/D, 方向键 (ESC [ A/B/C/D), Space, R
    """
    def __init__(self):
        self.is_tty = sys.stdin.isatty()
        if self.is_tty:
            try:
                self.fd = sys.stdin.fileno()
                self.old_settings = termios.tcgetattr(self.fd)
                tty.setraw(self.fd)
            except Exception:
                self.is_tty = False

    def read_key(self):
        """非阻塞读取按键, 返回字符串 key 或 None"""
        if not self.is_tty:
            return None
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
        if getattr(self, 'is_tty', False):
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass


class ParkingPIController:
    """
    驻车速度闭环 PI 补偿器:
    - 仅在用户目标零速驻车时生效，通过小幅速度指令修正消除稳态漂移。
    - 包含平滑模式切换、积分防饱和与接地保护。
    """
    def __init__(self, dt=0.02, Kp=0.35, Ki=0.60, trim_max=0.045, deadband=0.003):
        self.dt = dt
        self.Kp = Kp
        self.Ki = Ki
        self.trim_max = trim_max
        self.deadband = deadband
        self.integral = None
        self.filtered_vy = None
        self.park_settle_counter = None
        self.alpha_lpf = 0.15
        self.i_max = self.trim_max / max(1e-4, self.Ki)

    def init_buffers(self, num_envs, device):
        self.integral = torch.zeros(num_envs, device=device, dtype=torch.float)
        self.filtered_vy = torch.zeros(num_envs, device=device, dtype=torch.float)
        self.park_settle_counter = torch.zeros(num_envs, device=device, dtype=torch.long)

    def reset_env(self, env_ids=None):
        if self.integral is None:
            return
        if env_ids is None:
            self.integral.zero_()
            self.filtered_vy.zero_()
            self.park_settle_counter.zero_()
        else:
            self.integral[env_ids] = 0.0
            self.filtered_vy[env_ids] = 0.0
            self.park_settle_counter[env_ids] = 0

    def compute_trim(self, actual_vy, user_cmd_vy, user_cmd_yaw, is_elevating=False, is_grounded=True):
        self.filtered_vy = (1.0 - self.alpha_lpf) * self.filtered_vy + self.alpha_lpf * actual_vy
        is_zero_cmd = (torch.abs(user_cmd_vy) < 0.005) & (torch.abs(user_cmd_yaw) < 0.010)
        is_slow = torch.abs(self.filtered_vy) < 0.12

        can_engage = is_zero_cmd & is_slow
        if isinstance(is_elevating, torch.Tensor):
            can_engage = can_engage & (~is_elevating)
        elif is_elevating:
            can_engage = torch.zeros_like(can_engage)

        if isinstance(is_grounded, torch.Tensor):
            can_engage = can_engage & is_grounded
        elif not is_grounded:
            can_engage = torch.zeros_like(can_engage)

        can_engage = can_engage.bool()

        self.park_settle_counter = torch.where(
            can_engage,
            self.park_settle_counter + 1,
            torch.zeros_like(self.park_settle_counter)
        )
        is_active = (self.park_settle_counter >= 15).bool()

        err_v = -self.filtered_vy
        in_deadband = torch.abs(err_v) < self.deadband
        err_int = torch.where(in_deadband, torch.zeros_like(err_v), err_v)

        new_integral = torch.clamp(self.integral + err_int * self.dt, -self.i_max, self.i_max)
        self.integral = torch.where(is_active, new_integral, torch.zeros_like(self.integral))

        p_term = self.Kp * err_v
        i_term = self.Ki * self.integral
        trim_raw = p_term + i_term
        trim_clamped = torch.clamp(trim_raw, -self.trim_max, self.trim_max)

        delta_v_trim = torch.where(is_active, trim_clamped, torch.zeros_like(trim_clamped))
        return delta_v_trim, is_active


def teleop(args):
    print("\n" + "=" * 72)
    print("🎮 双足轮足机器人实时键盘遥控系统 (Champion v6 运动与高度全态遥控)")
    print("=" * 72)
    print("  【运动控制 (速度/转向)】:")
    print("    [W] 或 [↑]   : 前进加速 (+0.04 m/s, 上限 +0.50 m/s [平滑滤波])")
    print("    [S] 或 [↓]   : 减速 / 倒车 (-0.04 m/s, 下限 -0.30 m/s [平滑滤波])")
    print("    [A] 或 [←]   : 向左旋转转向 (+0.05 rad/s, 上限 +0.35 rad/s)")
    print("    [D] 或 [→]   : 向右旋转转向 (-0.05 rad/s, 下限 -0.35 rad/s)")
    print(f"  【高度调节 (动态升降/下蹲 - {(TELEOP_MAX_HEIGHT - TELEOP_MIN_HEIGHT) * 100:.0f}cm 行程 {TELEOP_MIN_HEIGHT:.2f}~{TELEOP_MAX_HEIGHT:.2f}m)】:")
    print(f"    [E] 或 [U]   : 升高机身 (+0.01 m, 上限 {TELEOP_MAX_HEIGHT:.3f} m)")
    print(f"    [C] 或 [J]   : 降低/下蹲 (-0.01 m, 下限 {TELEOP_MIN_HEIGHT:.3f} m)")
    print("    [Z] 或 [N]   : 恢复标称高度 (0.398 m)")
    print("  【辅助控制】:")
    print("    [Space]      : 刹车驻车 (速度清零，保持当前高度平衡)")
    print("    [P]          : 驻车闭环 PI 补偿开关 (消除稳态漂移，默认开启)")
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
            (gymapi.KEY_P, "TOGGLE_PI"),
            (gymapi.KEY_R, "RESET"),
            (gymapi.KEY_Q, "QUIT"),
        ]
        for key_code, act_name in key_mappings:
            env.gym.subscribe_viewer_keyboard_event(env.viewer, key_code, act_name)

    train_cfg.runner.resume = True
    custom_model = getattr(args, 'model_path', None)
    if custom_model is not None:
        if not os.path.exists(custom_model):
            raise FileNotFoundError(f"指定的模型文件不存在: {custom_model}")
    elif not getattr(args, 'load_run', None):
        default_phase15 = 'evaluation_assets/phase15/models/phase15_m130.pt'
        default_champ = 'evaluation_assets/champion_v6/models/champion_v6_m115.pt'
        if os.path.exists(default_phase15):
            custom_model = default_phase15
        elif os.path.exists(default_champ):
            custom_model = default_champ

    if custom_model:
        from rsl_rl.modules import ActorCritic
        print(f"📦 正在加载指定策略模型: {custom_model}")
        loaded = torch.load(custom_model, map_location=env.device)
        ac = ActorCritic(env.num_obs, env.num_obs, env.num_actions, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
        ac.load_state_dict(loaded['model_state_dict'])
        ac.eval()
        policy = ac.act_inference
    else:
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        policy = ppo_runner.get_inference_policy(device=env.device)

    # 驻车闭环 PI 控制器初始化
    enable_parking_pi = [True]
    pi_controller = ParkingPIController(dt=env.dt, Kp=0.35, Ki=0.60, trim_max=0.045, deadband=0.003)
    pi_controller.init_buffers(1, env.device)
    current_trim = 0.0
    pi_is_active = False

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
            target_ctrl['vel_y'] = round(min(target_ctrl['vel_y'] + 0.04, 0.50), 2)
        elif act in ['S', 'DOWN', 'BACKWARD']:
            target_ctrl['vel_y'] = round(max(target_ctrl['vel_y'] - 0.04, -0.30), 2)
        elif act in ['A', 'LEFT']:
            target_ctrl['ang'] = round(min(target_ctrl['ang'] + 0.05, 0.35), 2)
        elif act in ['D', 'RIGHT']:
            target_ctrl['ang'] = round(max(target_ctrl['ang'] - 0.05, -0.35), 2)
        elif act in ['E', 'U', 'RISE']:
            target_ctrl['height'] = round(min(target_ctrl['height'] + 0.01, TELEOP_MAX_HEIGHT), 3)
        elif act in ['C', 'J', 'SQUAT']:
            target_ctrl['height'] = round(max(target_ctrl['height'] - 0.01, TELEOP_MIN_HEIGHT), 3)
        elif act in ['Z', 'N', 'NOMINAL_H']:
            target_ctrl['height'] = 0.398
        elif act in [' ', 'BRAKE']:
            target_ctrl['vel_y'] = 0.0
            target_ctrl['ang'] = 0.0
        elif act in ['P', 'TOGGLE_PI']:
            enable_parking_pi[0] = not enable_parking_pi[0]
            if not enable_parking_pi[0]:
                pi_controller.reset_env()
        elif act in ['R', 'RESET']:
            target_ctrl['vel_y'] = 0.0
            target_ctrl['ang'] = 0.0
            target_ctrl['height'] = 0.398
            actual_cmd['vel_y'] = 0.0
            actual_cmd['ang'] = 0.0
            actual_cmd['height'] = 0.398
            pi_controller.reset_env()
            env.reset()
        elif act in ['Q', 'QUIT', '\x03']:
            return 'quit'
        else:
            changed = False
        return changed

    def print_hud():
        status_items = []
        if abs(actual_vy) < 0.03 and abs(actual_yaw) < 0.03:
            if pi_is_active:
                status_items.append(f"🛑 闭环驻车(trim:{current_trim:+.3f})")
            elif enable_parking_pi[0]:
                status_items.append("🛑 驻车平息中")
            else:
                status_items.append("🛑 开环驻车")
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

        target_h_user = target_ctrl['height']
        h_smooth = actual_cmd['height']
        delta_comp = float(np.clip(0.08 * (h_smooth - 0.39774), -0.010, 0.010))
        h_ctrl_cmd = round(h_smooth + delta_comp, 4)

        if abs(h_smooth - target_h_user) > 0.002:
            status_items.append("⏳ 变高过渡中")
        else:
            status_items.append("🎯 稳态保持")

        h_err_cm = (actual_h - target_h_user) * 100.0
        status_str = " | ".join(status_items)
        sys.stdout.write(
            f"\r\033[K[{status_str}] "
            f"目标:{target_h_user:.3f}m | 平滑:{h_smooth:.3f}m | 补偿:{h_ctrl_cmd:.3f}m | 实测:{actual_h:.3f}m(误差:{h_err_cm:+.1f}cm) | "
            f"线速目标:{actual_cmd['vel_y']:+.2f}(实测:{actual_vy:+.2f}) | "
            f"转向目标:{actual_cmd['ang']:+.2f}(实测:{actual_yaw:+.2f})"
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
            target_h = min(max(target_ctrl['height'], TELEOP_MIN_HEIGHT), TELEOP_MAX_HEIGHT)
            target_ctrl['height'] = target_h
            delta_h = target_h - actual_cmd['height']
            step_max_h = max_speed_h * dt
            if abs(delta_h) <= step_max_h:
                actual_cmd['height'] = target_h
            else:
                actual_cmd['height'] += math.copysign(step_max_h, delta_h)

            # 驻车闭环 PI 速度补偿
            fn_L = env.contact_forces[0, env.feet_indices[0], 2]
            fn_R = env.contact_forces[0, env.feet_indices[1], 2]
            is_grounded = (fn_L > 10.0) & (fn_R > 10.0)
            is_elevating = abs(actual_cmd['height'] - target_ctrl['height']) > 0.003

            if enable_parking_pi[0]:
                v_trim_t, is_act_t = pi_controller.compute_trim(
                    actual_vy=env.base_lin_vel[:, 1],
                    user_cmd_vy=torch.tensor([actual_cmd['vel_y']], device=env.device),
                    user_cmd_yaw=torch.tensor([actual_cmd['ang']], device=env.device),
                    is_elevating=is_elevating,
                    is_grounded=is_grounded
                )
                current_trim = v_trim_t.item()
                pi_is_active = is_act_t.item()
            else:
                current_trim = 0.0
                pi_is_active = False

            # 注入平滑指令与驻车修整到仿真环境
            env.commands[:, 0] = 0.0
            env.commands[:, 1] = actual_cmd['vel_y'] + current_trim
            env.commands[:, 2] = actual_cmd['ang']
            # 前馈柔性软补偿: 保证物理实际高度精准对齐用户目标高度 (消除有限 PD 刚度下的重力下沉)
            h_smooth = actual_cmd['height']
            delta_comp = float(np.clip(0.08 * (h_smooth - 0.39774), -0.010, 0.010))
            h_ctrl = round(h_smooth + delta_comp, 4)
            env.height_cmd_buf[:] = h_ctrl
            if hasattr(env, 'target_height_buf'):
                env.target_height_buf[:] = h_ctrl

            # 刷新观测向量，确保网络推理即时获取最新指令与高度特征
            env.compute_observations()
            obs = env.get_observations()

            # 推理并步进
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())

            # 仿真环境自动重置时清空 PI 状态，并将平滑指令置为零速与标称高度重新平滑过渡
            if dones[0]:
                pi_controller.reset_env()
                actual_cmd['vel_y'] = 0.0
                actual_cmd['ang'] = 0.0
                actual_cmd['height'] = 0.398

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
    custom_model_path = None
    has_explicit_task = False
    cleaned_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--model_path" and i + 1 < len(sys.argv):
            custom_model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i].startswith("--model_path="):
            custom_model_path = sys.argv[i].split("=", 1)[1]
            i += 1
        elif sys.argv[i] == "--task":
            has_explicit_task = True
            cleaned_argv.append(sys.argv[i])
            i += 1
        elif sys.argv[i].startswith("--task="):
            has_explicit_task = True
            cleaned_argv.append(sys.argv[i])
            i += 1
        else:
            cleaned_argv.append(sys.argv[i])
            i += 1
    sys.argv = cleaned_argv

    args = get_args()
    if not has_explicit_task:
        args.task = 'myrobot'
    if custom_model_path:
        if not os.path.exists(custom_model_path):
            raise FileNotFoundError(f"指定的模型文件不存在: {custom_model_path}")
        args.model_path = custom_model_path
    elif getattr(args, 'model_path', None):
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"指定的模型文件不存在: {args.model_path}")
    teleop(args)
