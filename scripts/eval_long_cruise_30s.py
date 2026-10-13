"""
Phase 14: Long-Duration 30-Second High-Speed Cruise (0.50 m/s) Verification Script
Acceptance Requirement:
- 0.50 m/s continuous cruise for 30.0 seconds (1500 steps).
- Mean absolute speed tracking error <= 0.05 m/s.
- Steady pass rate (|v_y - 0.50| <= 0.05 m/s) >= 90%.
- Zero falling over (100% survival).
- Tested across multiple seeds (default 5 seeds).

Usage:
    python scripts/eval_long_cruise_30s.py --model_path <path> --output_path <path> --num_seeds 5
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
    model_path = "evaluation_assets/phase14/models/phase14_exp1_m115.pt"
    output_path = "evaluation_assets/phase14/results/eval_long_cruise_30s_exp1.json"
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

def run_long_cruise_eval():
    model_path, output_path, num_seeds = parse_args()
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

    target_speed = 0.50
    ramp_accel = 0.50 # 0.5 m/s^2 smooth ramp up (1.0s to 0.5 m/s)
    cruise_duration_s = 30.0 # 30 seconds steady cruise
    steady_steps = int(round(cruise_duration_s / env.dt)) # 1500 steps
    ramp_steps = int(round(target_speed / (ramp_accel * env.dt))) # 50 steps
    warmup_steps = 50 # 1.0s settle after reaching 0.50
    total_steps = ramp_steps + warmup_steps + steady_steps

    print(f"\n================ Phase 14 30-Second Continuous 0.50 m/s Cruise Evaluation ================")
    print(f"Duration: {cruise_duration_s:.1f}s (1500 steps) | Seeds: {num_seeds}")
    print(f"{'Seed':>5} | {'Actual Vy':>10} | {'Err(m/s)':>9} | {'Pass%(±0.05)':>13} | {'PitchRMS':>8} | {'TauWheelSat':>11} | {'Surv%':>6}")
    print("-" * 78)

    seed_results = []

    for seed_idx in range(1, num_seeds + 1):
        torch.manual_seed(seed_idx * 100 + 42)
        np.random.seed(seed_idx * 100 + 42)
        env.reset()

        curr_cmd_vy = 0.0
        step_dvy = ramp_accel * env.dt

        steady_vys = []
        steady_errors = []
        steady_pitch = []
        steady_height_err = []
        steady_wheel_sat = []
        done_env_flags = np.zeros(args.num_envs, dtype=bool)

        for step in range(total_steps):
            if step < ramp_steps:
                curr_cmd_vy = min(curr_cmd_vy + step_dvy, target_speed)
            else:
                curr_cmd_vy = target_speed

            env.commands[:, 0] = 0.0
            env.commands[:, 1] = curr_cmd_vy
            env.commands[:, 2] = 0.0
            env.height_cmd_buf[:] = 0.39774
            if hasattr(env, 'target_height_buf'):
                env.target_height_buf[:] = 0.39774

            env.compute_observations()
            obs = env.get_observations()

            with torch.no_grad():
                actions = policy(obs)

            obs, _, _, dones, infos = env.step(actions)
            d_arr = dones.cpu().numpy()
            done_env_flags |= d_arr

            # Steady evaluation after ramp and settle
            if step >= (ramp_steps + warmup_steps):
                base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                base_h = env.root_states[:, 2].cpu().numpy()
                pitch = (torch.asin(torch.clamp(env.projected_gravity[:, 1], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
                tau_wheel = torch.abs(env.torques[:, env.wheel_indices]).cpu().numpy()
                is_sat = np.any(tau_wheel >= 6.95, axis=1)

                steady_vys.append(base_vy)
                steady_errors.append(np.abs(base_vy - target_speed))
                steady_pitch.append(pitch)
                steady_height_err.append(np.abs(base_h - 0.39774))
                steady_wheel_sat.append(is_sat)

        survival_rate = float(1.0 - np.mean(done_env_flags)) * 100.0

        arr_vy = np.array(steady_vys)
        arr_err = np.array(steady_errors)
        arr_pitch = np.array(steady_pitch)
        arr_sat = np.array(steady_wheel_sat)

        mean_vy = float(np.mean(arr_vy))
        mean_err = float(np.mean(arr_err))
        pass_rate_005 = float(np.mean(arr_err <= 0.05)) * 100.0
        pitch_rms = float(np.sqrt(np.mean(np.square(arr_pitch))))
        sat_rate = float(np.mean(arr_sat)) * 100.0

        status = "PASS" if mean_err <= 0.05 and pass_rate_005 >= 90.0 and survival_rate >= 100.0 else "FAIL"

        print(f"{seed_idx:5d} | {mean_vy:+10.4f} | {mean_err:9.4f} | {pass_rate_005:12.1f}% | {pitch_rms:7.2f}° | {sat_rate:10.2f}% | {survival_rate:5.1f}% [{status}]")

        seed_results.append({
            "seed": seed_idx,
            "mean_vy": mean_vy,
            "mean_err_ms": mean_err,
            "pass_rate_005": pass_rate_005,
            "pitch_rms_deg": pitch_rms,
            "wheel_sat_rate": sat_rate,
            "survival_rate": survival_rate,
            "passed": status == "PASS"
        })

    avg_vy = float(np.mean([r["mean_vy"] for r in seed_results]))
    avg_err = float(np.mean([r["mean_err_ms"] for r in seed_results]))
    avg_pass = float(np.mean([r["pass_rate_005"] for r in seed_results]))
    avg_surv = float(np.mean([r["survival_rate"] for r in seed_results]))
    all_seeds_passed = all([r["passed"] for r in seed_results])

    print("-" * 78)
    print(f"5-Seed Average: Vy={avg_vy:+.4f} m/s | Err={avg_err:.4f} m/s | Pass%={avg_pass:.1f}% | Surv%={avg_surv:.1f}%")
    print(f"Meets Acceptance Criteria (Err <= 0.05, Pass >= 90%, 100% Surv on all seeds): {all_seeds_passed}")

    summary = {
        "model_path": model_path,
        "cruise_duration_s": cruise_duration_s,
        "num_seeds": num_seeds,
        "avg_vy": avg_vy,
        "avg_err_ms": avg_err,
        "avg_pass_rate": avg_pass,
        "avg_survival_rate": avg_surv,
        "all_seeds_passed": all_seeds_passed,
        "seeds": seed_results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {output_path}")

if __name__ == '__main__':
    run_long_cruise_eval()
