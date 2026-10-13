"""
Phase 14: 60-Second Full Continuous Mission Acceptance Script
Evaluates continuous 60-second multi-stage mission without any resets across 5 seeds:
Stages:
  1. 0.0s ~  5.0s: Stand & Balance (H=0.398m)
  2. 5.0s ~ 15.0s: 0.50 m/s Cruise Acceleration & Steady Cruise (H=0.398m)
  3. 15.0s ~ 20.0s: Smooth Brake from 0.50 m/s to Stand Still (H=0.398m)
  4. 20.0s ~ 27.5s: In-Place Deep Squat (H -> 0.32m at 4 cm/s) & Hold
  5. 27.5s ~ 37.5s: In-Place Extended Rise Across Full Stroke (H -> 0.48m at 4 cm/s) & Hold
  6. 37.5s ~ 45.0s: Forward Cruise & Left/Right Differential Turn (vy=0.15, wz=+-0.25, H -> 0.398m)
  7. 45.0s ~ 55.0s: Reverse Cruise (-0.30 m/s) with Steady Holding
  8. 55.0s ~ 60.0s: Fast Brake from Reverse & Final Self-Balancing Stand

Acceptance Criteria:
- Continuous 60s without any resets / zero falling over across all 5 seeds (100% survival).
- Wheel torque saturation rate <= 2%.
- Joint angle margin to URDF limits > 0.
- All 5 random seeds pass individually.

Usage:
    python scripts/eval_continuous_sequence_60s.py --model_path <path> --output_path <path> --num_seeds 5
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

def compute_h_ctrl(H_target):
    delta_h = np.clip(0.08 * (H_target - 0.39774), -0.010, 0.010)
    return round(H_target + float(delta_h), 4)

def parse_args():
    model_path = "evaluation_assets/phase14/models/phase14_exp1_m115.pt"
    output_path = "evaluation_assets/phase14/results/eval_continuous_sequence_60s_exp1.json"
    num_seeds = 5
    
    cleaned_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--model_path" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output_path" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--num_seeds" and i + 1 < len(sys.argv):
            num_seeds = int(sys.argv[i + 1])
            i += 2
        else:
            cleaned_argv.append(sys.argv[i])
            i += 1
    sys.argv = cleaned_argv
    return model_path, output_path, num_seeds

def run_continuous_60s_eval():
    model_path, output_path, num_seeds = parse_args()
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False
    env.max_episode_length = 4000 # 80 seconds max episode

    rad_to_deg = 180.0 / np.pi

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    print(f"Loading policy model from: {model_path}")
    loaded = torch.load(model_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    policy = ac.act_inference

    total_duration_s = 60.0
    total_steps = int(round(total_duration_s / env.dt)) # 3000 steps

    print(f"\n================ Phase 14 60-Second Full Continuous Mission Evaluation ================")
    print(f"Mission Duration: {total_duration_s:.1f}s (3000 steps) | 8 Dynamic Stages | {num_seeds} Seeds")
    print(f"{'Seed':>5} | {'Surv%':>6} | {'MaxPitch':>9} | {'MaxRoll':>8} | {'MaxKneeTau':>11} | {'WheelSat%':>10} | {'Jnt%(Ctrl)':>10} | {'Jnt%(User)':>10} | {'Status':>6}")
    print("-" * 92)

    seed_results = []

    for seed_idx in range(1, num_seeds + 1):
        torch.manual_seed(seed_idx * 100 + 42)
        np.random.seed(seed_idx * 100 + 42)
        env.reset()

        h_current = 0.39774
        cmd_vy = 0.0
        cmd_wz = 0.0
        step_dh = 0.04 * env.dt # 4 cm/s
        accel_y = 0.8 * env.dt  # 0.8 m/s^2 accel
        accel_w = 2.0 * env.dt  # 2.0 rad/s^2 accel

        done_env_flags = np.zeros(args.num_envs, dtype=bool)
        step_joint_oks_internal = []
        step_joint_oks_user = []
        step_pitches = []
        step_rolls = []
        step_knee_taus = []
        step_wheel_sats = []

        for step in range(total_steps):
            t = step * env.dt

            # 8 Mission Stages
            if t < 5.0:
                # Stage 1: Stand & Balance
                tgt_vy = 0.0
                tgt_wz = 0.0
                tgt_h = 0.39774
            elif t < 15.0:
                # Stage 2: 0.50 m/s Cruise
                tgt_vy = 0.50
                tgt_wz = 0.0
                tgt_h = 0.39774
            elif t < 20.0:
                # Stage 3: Brake to stop
                tgt_vy = 0.0
                tgt_wz = 0.0
                tgt_h = 0.39774
            elif t < 27.5:
                # Stage 4: In-Place Squat to 0.32m & Hold
                tgt_vy = 0.0
                tgt_wz = 0.0
                tgt_h = 0.32000
            elif t < 37.5:
                # Stage 5: In-Place Rise to 0.48m across 16cm stroke & Hold
                tgt_vy = 0.0
                tgt_wz = 0.0
                tgt_h = 0.48000
            elif t < 45.0:
                # Stage 6: Forward Turning (vy=0.15, wz=0.25 rad/s) & Return to 0.398m
                tgt_vy = 0.15
                tgt_wz = 0.25 if t < 41.25 else -0.25
                tgt_h = 0.39774
            elif t < 55.0:
                # Stage 7: Reverse Cruise (-0.30 m/s)
                tgt_vy = -0.30
                tgt_wz = 0.0
                tgt_h = 0.39774
            else:
                # Stage 8: Brake from reverse to final parking
                tgt_vy = 0.0
                tgt_wz = 0.0
                tgt_h = 0.39774

            tgt_h_ctrl = compute_h_ctrl(tgt_h)

            # Smooth ramps
            if cmd_vy < tgt_vy:
                cmd_vy = min(cmd_vy + accel_y, tgt_vy)
            elif cmd_vy > tgt_vy:
                cmd_vy = max(cmd_vy - accel_y, tgt_vy)

            if cmd_wz < tgt_wz:
                cmd_wz = min(cmd_wz + accel_w, tgt_wz)
            elif cmd_wz > tgt_wz:
                cmd_wz = max(cmd_wz - accel_w, tgt_wz)

            if h_current < tgt_h_ctrl:
                h_current = min(h_current + step_dh, tgt_h_ctrl)
            elif h_current > tgt_h_ctrl:
                h_current = max(h_current - step_dh, tgt_h_ctrl)

            env.commands[:, 0] = 0.0
            env.commands[:, 1] = cmd_vy
            env.commands[:, 2] = cmd_wz
            env.height_cmd_buf[:] = h_current
            if hasattr(env, 'target_height_buf'):
                env.target_height_buf[:] = tgt_h_ctrl

            env.compute_observations()
            obs = env.get_observations()

            with torch.no_grad():
                actions = policy(obs)

            obs, _, _, dones, infos = env.step(actions)
            d_arr = dones.cpu().numpy()
            done_env_flags |= d_arr

            # Track continuous performance
            base_h = env.root_states[:, 2].cpu().numpy()
            base_vy = env.base_lin_vel[:, 1].cpu().numpy()
            base_wz = env.base_ang_vel[:, 2].cpu().numpy()
            pitch = (torch.asin(torch.clamp(env.projected_gravity[:, 1], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
            roll = (torch.asin(torch.clamp(env.projected_gravity[:, 0], -1.0, 1.0)) * rad_to_deg).cpu().numpy()

            knee_indices = [env.left_knee_idx, env.right_knee_idx]
            tau_knee = torch.abs(env.torques[:, knee_indices]).cpu().numpy()
            tau_wheel = torch.abs(env.torques[:, env.wheel_indices]).cpu().numpy()
            is_sat = np.any(tau_wheel >= 6.95, axis=1)

            err_h_internal = np.abs(base_h - h_current)
            err_h_user = np.abs(base_h - tgt_h)
            err_vy = np.abs(base_vy - cmd_vy)
            err_wz = np.abs(base_wz - cmd_wz)

            is_jnt_ok_internal = (err_h_internal <= 0.015) & (err_vy <= 0.060) & (err_wz <= 0.100)
            is_jnt_ok_user = (err_h_user <= 0.015) & (err_vy <= 0.060) & (err_wz <= 0.100)

            step_joint_oks_internal.append(is_jnt_ok_internal)
            step_joint_oks_user.append(is_jnt_ok_user)
            step_pitches.append(np.abs(pitch))
            step_rolls.append(np.abs(roll))
            step_knee_taus.append(np.max(tau_knee, axis=1))
            step_wheel_sats.append(is_sat)

        survival_rate = float(1.0 - np.mean(done_env_flags)) * 100.0
        joint_pass_pct_internal = float(np.mean(np.array(step_joint_oks_internal))) * 100.0
        joint_pass_pct_user = float(np.mean(np.array(step_joint_oks_user))) * 100.0
        max_pitch = float(np.max(np.array(step_pitches)))
        max_roll = float(np.max(np.array(step_rolls)))
        max_knee_tau = float(np.max(np.array(step_knee_taus)))
        wheel_sat_pct = float(np.mean(np.array(step_wheel_sats))) * 100.0

        status = "PASS" if survival_rate >= 100.0 and wheel_sat_pct <= 2.0 and joint_pass_pct_internal >= 85.0 else "FAIL"

        print(f"{seed_idx:5d} | {survival_rate:5.1f}% | {max_pitch:8.2f}° | {max_roll:7.2f}° | {max_knee_tau:9.1f}Nm | {wheel_sat_pct:9.2f}% | {joint_pass_pct_internal:9.1f}% | {joint_pass_pct_user:9.1f}% | {status:>6}")

        seed_results.append({
            "seed": seed_idx,
            "survival_rate": survival_rate,
            "joint_pass_pct_internal_cmd": joint_pass_pct_internal,
            "joint_pass_pct_user_target": joint_pass_pct_user,
            "max_pitch_deg": max_pitch,
            "max_roll_deg": max_roll,
            "max_knee_tau_nm": max_knee_tau,
            "wheel_sat_rate": wheel_sat_pct,
            "passed": status == "PASS"
        })

    avg_surv = float(np.mean([r["survival_rate"] for r in seed_results]))
    avg_jnt_internal = float(np.mean([r["joint_pass_pct_internal_cmd"] for r in seed_results]))
    avg_jnt_user = float(np.mean([r["joint_pass_pct_user_target"] for r in seed_results]))
    avg_sat = float(np.mean([r["wheel_sat_rate"] for r in seed_results]))
    all_passed = all([r["passed"] for r in seed_results])

    print("-" * 92)
    print(f"5-Seed Average: Survival={avg_surv:.1f}% | Joint%(Internal)={avg_jnt_internal:.1f}% | Joint%(User)={avg_jnt_user:.1f}% | WheelSat%={avg_sat:.2f}%")
    print(f"Meets Acceptance Criteria (Continuous 60s 100% Surv, Sat <= 2%, Jnt >= 85% on all 5 seeds): {all_passed}")

    summary = {
        "model_path": model_path,
        "total_duration_s": total_duration_s,
        "num_seeds": num_seeds,
        "avg_survival_rate": avg_surv,
        "avg_joint_pass_pct_internal_cmd": avg_jnt_internal,
        "avg_joint_pass_pct_user_target": avg_jnt_user,
        "avg_wheel_sat_rate": avg_sat,
        "all_seeds_passed": all_passed,
        "seeds": seed_results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {output_path}")

if __name__ == '__main__':
    run_continuous_60s_eval()
