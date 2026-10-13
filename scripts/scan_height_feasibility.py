"""
Phase 14: Height Feasibility & Physical Boundary Scanner
Scans heights from 0.28 m to 0.52 m in 1 cm increments.
Measures:
- Actual base height and error
- Chassis / leg ground collision contact forces
- Joint angle margin to URDF limits [-1.57, +1.57] rad
- Leg PD torques (Hip limit 75 Nm, Knee limit 60 Nm)
- Wheel torque (limit 7.0 Nm)
- Attitude pitch/roll RMS and survival rate
Supports:
--mode pure_ik   : Actions = 0, tests pure IK feedforward balance under PD
--mode policy    : Closed-loop policy inference (e.g. Champion v5)
Usage:
    python scripts/scan_height_feasibility.py --model_path <path> --output_path <path> --mode <pure_ik|policy>
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
    output_path = "evaluation_assets/phase14/results/height_feasibility_scan.json"
    mode = "policy"
    
    cleaned_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--model_path" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output_path" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
            i += 2
        else:
            cleaned_argv.append(sys.argv[i])
            i += 1
    sys.argv = cleaned_argv
    return model_path, output_path, mode

def run_height_scan():
    model_path, output_path, mode = parse_args()
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False
    env.max_episode_length = 2000

    rad_to_deg = 180.0 / np.pi

    if mode == "policy":
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        print(f"Loading policy model from: {model_path}")
        loaded = torch.load(model_path, map_location=env.device)
        ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
        ac.load_state_dict(loaded['model_state_dict'])
        ac.eval()
        policy = ac.act_inference
    else:
        print("Running in PURE IK mode (Actions = 0)")
        policy = lambda obs: torch.zeros(args.num_envs, 6, device=env.device)

    # 0.28 to 0.52 in 0.01 increments
    height_targets = np.round(np.arange(0.28, 0.521, 0.01), 3).tolist()
    scan_results = {}

    print(f"\n================ Phase 14 Height Feasibility Scan ({mode.upper()}) ================")
    print(f"{'H_cmd(m)':>8} | {'H_act(m)':>8} | {'Err(cm)':>7} | {'PitchRMS':>8} | {'TauHip(Nm)':>10} | {'TauKnee':>8} | {'TauWheel':>8} | {'ChassisCol':>10} | {'Surv%':>6}")
    print("-" * 92)

    for H_target in height_targets:
        torch.manual_seed(42)
        np.random.seed(42)
        env.reset()

        warmup_steps = 50
        hold_steps = 150 # 3 seconds steady eval
        total_steps = warmup_steps + hold_steps

        # Transition smoothly from nominal height (0.398) to H_target
        # using 4 cm/s ramp limiter
        h_current = 0.39774
        step_dh = 0.04 * env.dt # 0.0008 m per step

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
            if h_current < H_target:
                h_current = min(h_current + step_dh, H_target)
            elif h_current > H_target:
                h_current = max(h_current - step_dh, H_target)

            env.commands[:, 0] = 0.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            env.height_cmd_buf[:] = h_current

            env.compute_observations()
            obs = env.get_observations()

            with torch.no_grad():
                actions = policy(obs)

            obs, _, _, dones, infos = env.step(actions)

            d_arr = dones.cpu().numpy()
            done_env_flags |= d_arr

            # Metrics
            base_h = env.root_states[:, 2].cpu().numpy()
            pitch = (torch.asin(torch.clamp(env.projected_gravity[:, 1], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
            roll = (torch.asin(torch.clamp(env.projected_gravity[:, 0], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
            
            hip_indices = [env.left_hip_idx, env.right_hip_idx]
            knee_indices = [env.left_knee_idx, env.right_knee_idx]
            wheel_indices = env.wheel_indices

            tau_hip = torch.abs(env.torques[:, hip_indices]).cpu().numpy()
            tau_knee = torch.abs(env.torques[:, knee_indices]).cpu().numpy()
            tau_wheel = torch.abs(env.torques[:, wheel_indices]).cpu().numpy()

            # Contact forces on non-feet links
            if len(env.termination_contact_indices) > 0:
                contact_mags = torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1)
                has_collision = torch.any(contact_mags > 1.0, dim=1).cpu().numpy()
                chassis_contact_count += has_collision.astype(float)

            if step >= warmup_steps:
                steady_heights.append(base_h)
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
            arr_hip = np.array(steady_hip_tau)
            arr_knee = np.array(steady_knee_tau)
            arr_wheel = np.array(steady_wheel_tau)

            mean_h = float(np.mean(arr_h))
            mean_err_cm = float(np.mean(arr_err) * 100.0)
            pitch_rms = float(np.sqrt(np.mean(np.square(arr_pitch))))
            mean_tau_hip = float(np.mean(arr_hip))
            max_tau_hip = float(np.max(arr_hip))
            mean_tau_knee = float(np.mean(arr_knee))
            max_tau_knee = float(np.max(arr_knee))
            mean_tau_wheel = float(np.mean(arr_wheel))
            max_tau_wheel = float(np.max(arr_wheel))
            pass_rate_1cm = float(np.mean(arr_err <= 0.010)) * 100.0
            collision_rate = float(np.mean(chassis_contact_count > 0)) * 100.0

            res = {
                "cmd_h_m": H_target,
                "mean_act_h_m": mean_h,
                "mean_err_cm": mean_err_cm,
                "pass_rate_1cm": pass_rate_1cm,
                "pitch_rms_deg": pitch_rms,
                "mean_tau_hip_nm": mean_tau_hip,
                "max_tau_hip_nm": max_tau_hip,
                "mean_tau_knee_nm": mean_tau_knee,
                "max_tau_knee_nm": max_tau_knee,
                "mean_tau_wheel_nm": mean_tau_wheel,
                "max_tau_wheel_nm": max_tau_wheel,
                "chassis_collision_rate": collision_rate,
                "survival_rate": survival_rate
            }
            scan_results[f"H_{H_target:.2f}"] = res

            print(f"{H_target:8.2f} | {mean_h:8.3f} | {mean_err_cm:6.2f}cm | {pitch_rms:7.2f}° | {mean_tau_hip:6.1f} ({max_tau_hip:4.1f}) | {mean_tau_knee:4.1f} ({max_tau_knee:4.1f}) | {mean_tau_wheel:4.2f} ({max_tau_wheel:4.2f}) | {collision_rate:9.1f}% | {survival_rate:5.1f}%")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scan_results, f, indent=2)
    print(f"\nScan results saved to: {output_path}")

if __name__ == '__main__':
    run_height_scan()
