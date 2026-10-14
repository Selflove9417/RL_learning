"""
Phase 15 Step 1 Diagnostic: Baseline Parking Drift & Small-Command Sensitivity on model_130.pt
Measures:
1. 20-second steady parking drift at zero command across 6 heights: [0.34, 0.38, 0.42, 0.46, 0.50, 0.52] m.
2. Response to small positive velocity commands (+0.01, +0.02, +0.04, +0.06 m/s) to verify control authority.
"""

import os
import sys
import json
import numpy as np

# isaacgym must be imported before torch
import isaacgym
from isaacgym import gymapi, gymtorch
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from rsl_rl.modules import ActorCritic

def run_parking_baseline():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    model_path = "legged_gym/logs/myrobot/Oct14_00-28-55_phase15_height_034_060_exploration/model_130.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False
    env.max_episode_length = 3000

    loaded = torch.load(model_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    pol = ac.act_inference

    rad_to_deg = 180.0 / np.pi
    dt = env.dt # 0.02s
    hold_duration_s = 20.0
    hold_steps = int(hold_duration_s / dt) # 1000 steps
    warmup_steps = 150 # 3.0s warmup to reach target height

    test_heights = [0.340, 0.380, 0.420, 0.460, 0.500, 0.520]
    drift_records = {}

    print(f"\n================ STEP 1: QUANTIFY BASELINE PARKING DRIFT (m130) ================")
    print(f"Model: {model_path}")
    print(f"Hold duration: {hold_duration_s:.1f}s ({hold_steps} steps) at zero command (cmd_vy=0.0, cmd_yaw=0.0)\n")

    for h_user in test_heights:
        torch.manual_seed(42)
        np.random.seed(42)
        env.reset()

        delta_h = float(np.clip(0.08 * (h_user - 0.39774), -0.010, 0.010))
        h_ctrl_val = round(h_user + delta_h, 4)

        # Transition height gradually
        h_current = 0.39774
        step_dh = 0.04 * dt

        # Warmup phase: transition height to h_ctrl_val
        for _ in range(warmup_steps):
            if h_current < h_ctrl_val:
                h_current = min(h_current + step_dh, h_ctrl_val)
            elif h_current > h_ctrl_val:
                h_current = max(h_current - step_dh, h_ctrl_val)

            env.commands[:, 0] = 0.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            env.height_cmd_buf[:] = h_current
            if hasattr(env, 'target_height_buf'):
                env.target_height_buf[:] = h_ctrl_val

            env.compute_observations()
            obs = env.get_observations()
            with torch.no_grad():
                actions = pol(obs)
            env.step(actions)

        # Record starting state
        start_y = env.root_states[:, 1].clone().cpu().numpy()
        start_x = env.root_states[:, 0].clone().cpu().numpy()

        vy_history = []
        pitch_history = []
        actual_h_history = []
        wheel_tau_history = []
        wheel_omega_history = []

        for step in range(hold_steps):
            env.commands[:, 0] = 0.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            env.height_cmd_buf[:] = h_ctrl_val
            if hasattr(env, 'target_height_buf'):
                env.target_height_buf[:] = h_ctrl_val

            env.compute_observations()
            obs = env.get_observations()
            with torch.no_grad():
                actions = pol(obs)
            env.step(actions)

            vy = env.base_lin_vel[:, 1].cpu().numpy()
            proj_gy = env.projected_gravity[:, 1].cpu().numpy()
            pitch_deg = np.arcsin(np.clip(proj_gy, -1.0, 1.0)) * rad_to_deg
            act_h = env.root_states[:, 2].cpu().numpy()
            torques = env.torques.cpu().numpy()
            dof_vel = env.dof_vel.cpu().numpy()
            wheel_tau = 0.5 * (torques[:, env.left_wheel_idx] + torques[:, env.right_wheel_idx])
            wheel_omega = 0.5 * (dof_vel[:, env.left_wheel_idx] + dof_vel[:, env.right_wheel_idx])

            vy_history.append(vy)
            pitch_history.append(pitch_deg)
            actual_h_history.append(act_h)
            wheel_tau_history.append(wheel_tau)
            wheel_omega_history.append(wheel_omega)

        end_y = env.root_states[:, 1].clone().cpu().numpy()
        end_x = env.root_states[:, 0].clone().cpu().numpy()

        disp_y = end_y - start_y
        disp_x = end_x - start_x

        mean_vy = np.mean(vy_history)
        std_vy = np.std(vy_history)
        mean_pitch = np.mean(pitch_history)
        std_pitch = np.std(pitch_history)
        mean_act_h = np.mean(actual_h_history)
        mean_wheel_tau = np.mean(wheel_tau_history)
        mean_wheel_omega = np.mean(wheel_omega_history)
        mean_disp_y = np.mean(disp_y)
        mean_disp_x = np.mean(disp_x)

        drift_records[f"H_{h_user:.3f}m"] = {
            "target_H": h_user,
            "actual_H_mean": float(mean_act_h),
            "H_err_cm": float((mean_act_h - h_user) * 100.0),
            "mean_vy_mps": float(mean_vy),
            "std_vy_mps": float(std_vy),
            "disp_20s_y_m": float(mean_disp_y),
            "disp_20s_x_m": float(mean_disp_x),
            "mean_pitch_deg": float(mean_pitch),
            "std_pitch_deg": float(std_pitch),
            "mean_wheel_tau_Nm": float(mean_wheel_tau),
            "mean_wheel_omega_radps": float(mean_wheel_omega),
        }

        print(f"H = {h_user:.3f}m | 实测H: {mean_act_h:.3f}m | 20s位移ΔY: {mean_disp_y:+.4f}m | 平均线速vy: {mean_vy:+.4f} m/s (std: {std_vy:.4f}) | 俯仰角: {mean_pitch:+.2f}° (std: {std_pitch:.2f}°)")

    print(f"\n================ SMALL POSITIVE COMMAND SENSITIVITY TEST ================")
    print(f"Testing response to small positive cmd_vy at H = 0.34m and H = 0.40m (5.0s hold each)...")
    test_cmds = [0.00, +0.01, +0.02, +0.03, +0.04, +0.06]
    cmd_sensitivity = {}

    for h_test in [0.340, 0.400]:
        cmd_sensitivity[f"H_{h_test:.3f}m"] = {}
        for cmd_val in test_cmds:
            torch.manual_seed(42)
            np.random.seed(42)
            env.reset()

            delta_h = float(np.clip(0.08 * (h_test - 0.39774), -0.010, 0.010))
            h_ctrl = round(h_test + delta_h, 4)

            # settle at target height with cmd=0
            for _ in range(150):
                env.commands[:, 1] = 0.0
                env.height_cmd_buf[:] = h_ctrl
                env.compute_observations()
                obs = env.get_observations()
                with torch.no_grad():
                    actions = pol(obs)
                env.step(actions)

            # apply test cmd for 250 steps (5.0s)
            vy_list = []
            for _ in range(250):
                env.commands[:, 1] = cmd_val
                env.height_cmd_buf[:] = h_ctrl
                env.compute_observations()
                obs = env.get_observations()
                with torch.no_grad():
                    actions = pol(obs)
                env.step(actions)
                vy_list.append(env.base_lin_vel[:, 1].cpu().numpy())

            steady_vy = float(np.mean(vy_list[100:])) # last 3 seconds
            cmd_sensitivity[f"H_{h_test:.3f}m"][f"cmd_{cmd_val:+.2f}"] = steady_vy
            print(f"H = {h_test:.3f}m | 输入指令 cmd_vy = {cmd_val:+.2f} m/s --> 稳态实测速度 vy = {steady_vy:+.4f} m/s (响应差值: {steady_vy - cmd_val:+.4f})")

    out_file = "evaluation_assets/phase15/parking_drift_step1_m130.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({"drift_records": drift_records, "cmd_sensitivity": cmd_sensitivity}, f, indent=2)
    print(f"\nStep 1 baseline results saved to: {out_file}")

if __name__ == '__main__':
    run_parking_baseline()
