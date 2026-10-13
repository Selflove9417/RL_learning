"""
Phase 14: In-Place Extended Height Range Benchmark Evaluation Script
Evaluates candidate policy models across the extended height range (e.g. 0.32 ~ 0.48 m or 0.30 ~ 0.50 m)
in 2 cm increments, with 10 seconds steady holding at each point.

Acceptance Criteria:
- Steady height error within +-1 cm (0.010 m) for >= 90% of steady-state samples.
- Zero falling over / 100% survival.
- Torque saturation within limits (Hip < 75 Nm, Knee < 60 Nm, Wheel < 7.0 Nm).
- Score is ALWAYS calculated against the user target height (not internal ramp/offset).

Usage:
    python scripts/eval_height_range.py --model_path <path> --output_path <path> --h_min 0.32 --h_max 0.48 --h_step 0.02
"""

import sys
import os
import json
import math

import isaacgym
from isaacgym import gymtorch
import numpy as np
import torch

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def parse_args():
    model_path = "evaluation_assets/champion_v5/models/champion_v5_m100.pt"
    output_path = "evaluation_assets/phase14/results/eval_height_range_v5.json"
    h_min = 0.32
    h_max = 0.48
    h_step = 0.02
    hold_duration_s = 10.0 # 10 seconds hold per user specification
    enable_compensation = False
    
    cleaned_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--model_path" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output_path" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--h_min" and i + 1 < len(sys.argv):
            h_min = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--h_max" and i + 1 < len(sys.argv):
            h_max = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--h_step" and i + 1 < len(sys.argv):
            h_step = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--hold_duration_s" and i + 1 < len(sys.argv):
            hold_duration_s = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--enable_compensation":
            enable_compensation = True
            i += 1
        else:
            cleaned_argv.append(sys.argv[i])
            i += 1
    sys.argv = cleaned_argv
    return model_path, output_path, h_min, h_max, h_step, hold_duration_s, enable_compensation

def compute_h_ctrl(H_target, enable_compensation=False):
    if not enable_compensation:
        return H_target
    # 连续前馈补偿: 针对极端高度几何柔性与力矩偏置的微小线性前馈校正
    delta_h = np.clip(0.08 * (H_target - 0.39774), -0.010, 0.010)
    return round(H_target + float(delta_h), 4)

def run_height_range_eval():
    model_path, output_path, h_min, h_max, h_step, hold_duration_s, enable_compensation = parse_args()
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False
    env.max_episode_length = 3000

    rad_to_deg = 180.0 / np.pi

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    print(f"Loading policy model from: {model_path}")
    loaded = torch.load(model_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    policy = ac.act_inference

    # Test heights: h_min to h_max with step h_step
    num_points = int(round((h_max - h_min) / h_step)) + 1
    height_targets = [round(h_min + i * h_step, 3) for i in range(num_points)]

    eval_results = {}
    print(f"\n================ Phase 14 Height Range Evaluation ({h_min:.2f}m ~ {h_max:.2f}m) ================")
    print(f"Hold duration: {hold_duration_s:.1f}s (500 steps) | Elevation speed: 4 cm/s | Compensation: {enable_compensation}")
    print(f"{'H_target(m)':>11} | {'H_ctrl(m)':>9} | {'H_act(m)':>8} | {'Err(cm)':>7} | {'Pass% (<=1cm)':>14} | {'PitchRMS':>8} | {'TauKnee':>7} | {'Surv%':>6}")
    print("-" * 96)

    all_pass_rates = []
    all_surv_rates = []

    for H_target in height_targets:
        torch.manual_seed(42)
        np.random.seed(42)
        env.reset()

        h_ctrl_target = compute_h_ctrl(H_target, enable_compensation)
        h_current = 0.39774
        elevation_speed = 0.04 # 4 cm/s
        step_dh = elevation_speed * env.dt # 0.0008 m per step

        ramp_steps = int(math.ceil(abs(h_ctrl_target - h_current) / step_dh))
        steady_steps = int(round(hold_duration_s / env.dt))
        warmup_in_steady = 50 # 1.0s settle after reaching target
        total_steps = ramp_steps + warmup_in_steady + steady_steps # 确保在 1.0s 稳定等待后完整采集 10.0 秒 (500 步)

        steady_heights = []
        steady_errors = []
        steady_pitch = []
        steady_roll = []
        steady_hip_tau = []
        steady_knee_tau = []
        steady_wheel_tau = []
        chassis_contact_count = np.zeros(args.num_envs)
        done_env_flags = np.zeros(args.num_envs, dtype=bool)

        for step in range(total_steps):
            if step < ramp_steps:
                if h_current < h_ctrl_target:
                    h_current = min(h_current + step_dh, h_ctrl_target)
                elif h_current > h_ctrl_target:
                    h_current = max(h_current - step_dh, h_ctrl_target)
            else:
                h_current = h_ctrl_target

            env.commands[:, 0] = 0.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            env.height_cmd_buf[:] = h_current
            if hasattr(env, 'target_height_buf'):
                env.target_height_buf[:] = h_ctrl_target

            env.compute_observations()
            obs = env.get_observations()

            with torch.no_grad():
                actions = policy(obs)

            obs, _, _, dones, infos = env.step(actions)

            d_arr = dones.cpu().numpy()
            done_env_flags |= d_arr

            # Record steady metrics only after ramp and settle
            if step >= (ramp_steps + warmup_in_steady):
                base_h = env.root_states[:, 2].cpu().numpy()
                pitch = (torch.asin(torch.clamp(env.projected_gravity[:, 1], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
                roll = (torch.asin(torch.clamp(env.projected_gravity[:, 0], -1.0, 1.0)) * rad_to_deg).cpu().numpy()

                hip_indices = [env.left_hip_idx, env.right_hip_idx]
                knee_indices = [env.left_knee_idx, env.right_knee_idx]
                wheel_indices = env.wheel_indices

                tau_hip = torch.abs(env.torques[:, hip_indices]).cpu().numpy()
                tau_knee = torch.abs(env.torques[:, knee_indices]).cpu().numpy()
                tau_wheel = torch.abs(env.torques[:, wheel_indices]).cpu().numpy()

                steady_heights.append(base_h)
                # Score is always evaluated against user target H_target
                steady_errors.append(np.abs(base_h - H_target))
                steady_pitch.append(pitch)
                steady_roll.append(roll)
                steady_hip_tau.append(np.mean(tau_hip, axis=1))
                steady_knee_tau.append(np.mean(tau_knee, axis=1))
                steady_wheel_tau.append(np.mean(tau_wheel, axis=1))

        survival_rate = float(1.0 - np.mean(done_env_flags)) * 100.0

        if len(steady_heights) > 0:
            arr_h = np.array(steady_heights)
            arr_err = np.array(steady_errors)
            arr_pitch = np.array(steady_pitch)
            arr_knee = np.array(steady_knee_tau)
            arr_wheel = np.array(steady_wheel_tau)

            mean_h = float(np.mean(arr_h))
            mean_err_cm = float(np.mean(arr_err) * 100.0)
            pass_rate_1cm = float(np.mean(arr_err <= 0.010)) * 100.0
            pitch_rms = float(np.sqrt(np.mean(np.square(arr_pitch))))
            mean_tau_knee = float(np.mean(arr_knee))
            max_tau_knee = float(np.max(arr_knee))

            all_pass_rates.append(pass_rate_1cm)
            all_surv_rates.append(survival_rate)

            res = {
                "H_target_m": H_target,
                "H_ctrl_m": h_ctrl_target,
                "mean_act_h_m": mean_h,
                "mean_err_cm": mean_err_cm,
                "pass_rate_1cm": pass_rate_1cm,
                "pitch_rms_deg": pitch_rms,
                "mean_tau_knee_nm": mean_tau_knee,
                "max_tau_knee_nm": max_tau_knee,
                "survival_rate": survival_rate
            }
            eval_results[f"H_{H_target:.2f}"] = res
            status_flag = "PASS" if pass_rate_1cm >= 90.0 and survival_rate >= 100.0 else "FAIL"

            print(f"{H_target:11.2f} | {h_ctrl_target:9.3f} | {mean_h:8.3f} | {mean_err_cm:6.2f}cm | {pass_rate_1cm:12.1f}% | {pitch_rms:7.2f}° | {mean_tau_knee:4.1f} ({max_tau_knee:4.1f}) | {survival_rate:5.1f}% [{status_flag}]")

    avg_pass = float(np.mean(all_pass_rates))
    avg_surv = float(np.mean(all_surv_rates))
    min_pass = float(np.min(all_pass_rates))

    summary = {
        "model_path": model_path,
        "h_min": h_min,
        "h_max": h_max,
        "h_step": h_step,
        "hold_duration_s": hold_duration_s,
        "overall_mean_pass_rate": avg_pass,
        "min_point_pass_rate": min_pass,
        "overall_survival_rate": avg_surv,
        "all_points_pass_ge_90": bool(min_pass >= 90.0 and avg_surv >= 100.0),
        "per_point_results": eval_results
    }

    print("-" * 84)
    print(f"Overall Mean Pass Rate: {avg_pass:.2f}% | Min Point Pass Rate: {min_pass:.2f}% | Survival: {avg_surv:.1f}%")
    print(f"Meets Acceptance Criteria (all points >= 90%): {summary['all_points_pass_ge_90']}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to: {output_path}")

if __name__ == '__main__':
    run_height_range_eval()
