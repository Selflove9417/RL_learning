"""
Phase 13: Phase B - Champion v4 Speed Ramp & Dynamic Limit Benchmark Test
Tests Champion v4 policy across target velocities:
[+0.20, +0.30, +0.40, +0.50, -0.20, -0.30] m/s
Without modifying any model weights.
Uses acceleration/deceleration speed ramps (0.8 m/s^2).
Evaluates steady-state tracking, attitude stability, wheel torque saturation,
braking distance, and termination causes.
"""

import isaacgym
import os
import json
import numpy as np
import torch
from isaacgym import gymtorch

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def run_speed_ramp_evaluation():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False
    env.max_episode_length = 2000

    model_path = "/home/xy/IsaacTu/evaluation_assets/phase8b/models/phase8b_m10.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Champion v4 model not found at {model_path}")

    print(f"Loading Champion v4 from: {model_path}")
    loaded = torch.load(model_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    pol = ac.act_inference

    dt = env.dt # 0.02s
    max_accel = 0.8 # m/s^2
    step_accel = max_accel * dt # 0.016 m/s per step
    rad_to_deg = 180.0 / np.pi
    h_nom = 0.39774

    test_targets = [0.20, 0.30, 0.40, 0.50, -0.20, -0.30]

    all_test_results = {}

    for v_target in test_targets:
        test_name = f"vel_{v_target:+.2f}"
        print(f"\n================ Running Speed Ramp Test: target = {v_target:+.2f} m/s ================")

        torch.manual_seed(42)
        np.random.seed(42)
        env.reset()

        # Phases definition:
        # 1. Stand initial: 100 steps (2.0s)
        # 2. Ramp to target: int(ceil(abs(v_target) / step_accel)) steps
        # 3. Cruise steady: 250 steps (5.0s)
        # 4. Ramp down to 0: int(ceil(abs(v_target) / step_accel)) steps
        # 5. Stand post-brake: 100 steps (2.0s)

        ramp_steps = int(np.ceil(abs(v_target) / step_accel))
        stand_pre_steps = 100
        cruise_steps = 250
        brake_ramp_steps = ramp_steps
        stand_post_steps = 100

        total_steps = stand_pre_steps + ramp_steps + cruise_steps + brake_ramp_steps + stand_post_steps

        # Timeline boundaries
        t_stand_pre_end = stand_pre_steps
        t_ramp_up_end = t_stand_pre_end + ramp_steps
        t_cruise_end = t_ramp_up_end + cruise_steps
        t_brake_end = t_cruise_end + brake_ramp_steps
        t_total_end = total_steps

        # Steady-state evaluation window: last 150 steps (3.0s) of cruise
        steady_eval_start = t_cruise_end - 150
        steady_eval_end = t_cruise_end

        # Tracking variables
        current_cmd_vy = 0.0
        done_env_flags = np.zeros(args.num_envs, dtype=bool)
        first_fail_step = {}

        # Cruise steady-state data
        cruise_actual_vy = []
        cruise_vy_errors = []
        cruise_pitch_deg = []
        cruise_roll_deg = []
        cruise_height = []
        cruise_wheel_tau_mean = []
        cruise_wheel_tau_max = []
        cruise_wheel_sat_count = np.zeros(args.num_envs)

        # Braking distance tracking
        pos_at_brake_start = None
        pos_at_stop = None
        stop_step = np.full(args.num_envs, -1)

        for step in range(total_steps):
            # Compute commanded vy according to ramp
            if step < t_stand_pre_end:
                current_cmd_vy = 0.0
            elif step < t_ramp_up_end:
                if v_target > 0:
                    current_cmd_vy = min(current_cmd_vy + step_accel, v_target)
                else:
                    current_cmd_vy = max(current_cmd_vy - step_accel, v_target)
            elif step < t_cruise_end:
                current_cmd_vy = v_target
            elif step < t_brake_end:
                if current_cmd_vy > 0:
                    current_cmd_vy = max(current_cmd_vy - step_accel, 0.0)
                else:
                    current_cmd_vy = min(current_cmd_vy + step_accel, 0.0)
            else:
                current_cmd_vy = 0.0

            # Set environment commands
            env.commands[:, 0] = 0.0
            env.commands[:, 1] = current_cmd_vy
            env.commands[:, 2] = 0.0
            env.height_cmd_buf[:] = h_nom

            # Synchronize observations with updated commands
            env.compute_observations()
            obs = env.get_observations()

            with torch.no_grad():
                actions = pol(obs)
            obs, _, _, dones, infos = env.step(actions)

            d_arr = dones.cpu().numpy()
            if np.any(d_arr):
                for e_idx in range(args.num_envs):
                    if d_arr[e_idx] and not done_env_flags[e_idx]:
                        first_fail_step[e_idx] = step
                done_env_flags |= d_arr

            # Extract metrics
            root_y_pos = env.root_states[:, 1].cpu().numpy()
            base_vy = env.base_lin_vel[:, 1].cpu().numpy()
            base_h = env.root_states[:, 2].cpu().numpy()
            pitch = (torch.asin(torch.clamp(env.projected_gravity[:, 1], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
            roll = (torch.asin(torch.clamp(env.projected_gravity[:, 0], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
            wheel_tau = torch.abs(env.torques[:, env.wheel_indices]).cpu().numpy()
            wheel_sat = np.any(wheel_tau >= 6.95, axis=1)

            # Record cruise steady state
            if steady_eval_start <= step < steady_eval_end:
                cruise_actual_vy.append(base_vy)
                cruise_vy_errors.append(np.abs(base_vy - v_target))
                cruise_pitch_deg.append(pitch)
                cruise_roll_deg.append(roll)
                cruise_height.append(base_h)
                cruise_wheel_tau_mean.append(np.mean(wheel_tau, axis=1))
                cruise_wheel_tau_max.append(np.max(wheel_tau, axis=1))
                cruise_wheel_sat_count += wheel_sat.astype(float)

            # Braking distance tracking
            if step == t_cruise_end:
                pos_at_brake_start = root_y_pos.copy()

            if step >= t_cruise_end:
                for e_idx in range(args.num_envs):
                    if stop_step[e_idx] == -1 and abs(base_vy[e_idx]) < 0.03:
                        stop_step[e_idx] = step

        # Summarize test results
        survival_rate = float(1.0 - np.mean(done_env_flags))

        if len(cruise_actual_vy) > 0:
            arr_vy = np.array(cruise_actual_vy) # [T, N]
            arr_err = np.array(cruise_vy_errors)
            arr_pitch = np.array(cruise_pitch_deg)
            arr_roll = np.array(cruise_roll_deg)
            arr_h = np.array(cruise_height)
            arr_tau_mean = np.array(cruise_wheel_tau_mean)
            arr_tau_max = np.array(cruise_wheel_tau_max)

            mean_actual_vy = float(np.mean(arr_vy))
            mean_abs_err_vy = float(np.mean(arr_err))
            std_vy = float(np.std(arr_vy))

            # Pass rate: |vy - v_target| <= 0.05 m/s
            pass_mask_005 = (arr_err <= 0.05)
            pass_rate_005 = float(np.mean(pass_mask_005))

            pitch_rms = float(np.sqrt(np.mean(np.square(arr_pitch))))
            pitch_max = float(np.max(np.abs(arr_pitch)))
            roll_rms = float(np.sqrt(np.mean(np.square(arr_roll))))
            roll_max = float(np.max(np.abs(arr_roll)))

            mean_height = float(np.mean(arr_h))
            height_err_cm = float(np.mean(np.abs(arr_h - h_nom)) * 100.0)

            mean_wheel_tau = float(np.mean(arr_tau_mean))
            max_wheel_tau = float(np.max(arr_tau_max))
            steady_steps = steady_eval_end - steady_eval_start
            sat_rate = float(np.mean(cruise_wheel_sat_count) / steady_steps)
        else:
            mean_actual_vy = 0.0
            mean_abs_err_vy = 999.0
            pass_rate_005 = 0.0
            pitch_rms = 999.0
            pitch_max = 999.0
            roll_rms = 999.0
            roll_max = 999.0
            mean_height = 0.0
            height_err_cm = 999.0
            mean_wheel_tau = 0.0
            max_wheel_tau = 0.0
            sat_rate = 1.0

        # Braking metrics
        brake_distances = []
        brake_times = []
        if pos_at_brake_start is not None:
            final_y = env.root_states[:, 1].cpu().numpy()
            for e_idx in range(args.num_envs):
                if not done_env_flags[e_idx] and stop_step[e_idx] > 0:
                    dist = abs(final_y[e_idx] - pos_at_brake_start[e_idx])
                    time_s = (stop_step[e_idx] - t_cruise_end) * dt
                    brake_distances.append(dist)
                    brake_times.append(time_s)

        mean_brake_dist = float(np.mean(brake_distances)) if len(brake_distances) > 0 else -1.0
        mean_brake_time = float(np.mean(brake_times)) if len(brake_times) > 0 else -1.0

        res_dict = {
            "v_target_mps": v_target,
            "survival_rate": survival_rate,
            "num_envs": args.num_envs,
            "failed_envs": int(np.sum(done_env_flags)),
            "first_fail_step": first_fail_step,
            "steady_cruise": {
                "mean_actual_vy_mps": mean_actual_vy,
                "mean_abs_err_mps": mean_abs_err_vy,
                "pass_rate_within_005": pass_rate_005,
                "target_met_goal_a": bool(pass_rate_005 >= 0.90 and mean_abs_err_vy <= 0.05 and survival_rate == 1.0),
                "pitch_rms_deg": pitch_rms,
                "pitch_max_deg": pitch_max,
                "roll_rms_deg": roll_rms,
                "roll_max_deg": roll_max,
                "mean_height_m": mean_height,
                "height_err_cm": height_err_cm,
                "mean_wheel_torque_nm": mean_wheel_tau,
                "max_wheel_torque_nm": max_wheel_tau,
                "wheel_saturation_rate": sat_rate,
            },
            "braking": {
                "mean_brake_distance_m": mean_brake_dist,
                "mean_brake_time_s": mean_brake_time,
            }
        }

        all_test_results[test_name] = res_dict

        print(f"--- Results for target {v_target:+.2f} m/s ---")
        print(f"  Survival Rate: {survival_rate * 100:.1f}%")
        print(f"  Actual Vy: {mean_actual_vy:+.4f} m/s (Cmd: {v_target:+.2f} m/s, Err: {mean_abs_err_vy:.4f} m/s)")
        print(f"  Steady Pass Rate (±0.05 m/s): {pass_rate_005 * 100:.1f}% (Goal >= 90%: {res_dict['steady_cruise']['target_met_goal_a']})")
        print(f"  Pitch RMS: {pitch_rms:.2f}° (Max: {pitch_max:.2f}°)")
        print(f"  Height Err: {height_err_cm:.2f} cm")
        print(f"  Wheel Torque: Mean {mean_wheel_tau:.2f} Nm, Max {max_wheel_tau:.2f} Nm, Sat: {sat_rate*100:.2f}%")
        print(f"  Brake Dist: {mean_brake_dist:.3f} m, Brake Time: {mean_brake_time:.2f} s")

    out_file = "/home/xy/IsaacTu/evaluation_assets/phase13/results/champion_v4_speed_ramp_test.json"
    with open(out_file, 'w') as f:
        json.dump(all_test_results, f, indent=2)
    print(f"\nAll results saved to {out_file}")

if __name__ == "__main__":
    run_speed_ramp_evaluation()
