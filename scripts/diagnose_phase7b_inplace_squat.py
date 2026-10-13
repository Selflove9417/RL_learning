"""
Phase 7B: In-place Squat (g3_inplace_squat) Diagnostic Script.
Compares Phase 6B Champion v2 vs Phase 7A model_1.pt under identical 32 envs and seed 42.
Records:
- Left/Right wheel velocity and variance
- Left/Right wheel torque and saturation rate
- Pitch angle and pitch rate
- Height trajectory and error curve
- In-place position drift (X, Y)
- Pre-instability states and reset attribution
"""

import isaacgym
import torch
import os
import json
import numpy as np

from isaacgym import gymtorch
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def run_inplace_squat_diagnosis():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    models = {
        "Phase6B_Champ_v2": "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct11_21-24-38_phase6b_pilot_deadzone010_beta020_iter1/model_1.pt",
        "Phase7A_Sampling": "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct12_10-13-34_phase7a_inplace_squat_sampling_beta020_iter1/model_1.pt",
    }

    device = env.device
    num_envs = args.num_envs
    total_steps = 300
    step_cmd_start = 100
    steady_start = 200
    target_h = 0.36
    v_target = 0.0
    w_target = 0.0

    rad_to_deg = 180.0 / np.pi

    results = {}

    for m_name, m_path in models.items():
        print(f"\n=======================================================")
        print(f"Diagnosing Model: {m_name}")
        print(f"Checkpoint: {m_path}")
        print(f"=======================================================")

        # Load policy
        policy = ActorCritic(
            num_actor_obs=env.cfg.env.num_observations,
            num_critic_obs=env.cfg.env.num_observations,
            num_actions=env.cfg.env.num_actions,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation='elu',
            init_noise_std=1.0
        ).to(device)

        loaded = torch.load(m_path, map_location=device)
        policy.load_state_dict(loaded['model_state_dict'])
        policy.eval()

        # Reset environment with fixed seed 42
        torch.manual_seed(42)
        np.random.seed(42)
        env.reset()

        # Record initial position at step 100 (when squat command is issued)
        init_pos_xy = None

        # Data collection containers (time-series [total_steps, num_envs])
        ts_base_h = np.zeros((total_steps, num_envs))
        ts_h_err = np.zeros((total_steps, num_envs))
        ts_vy = np.zeros((total_steps, num_envs))
        ts_wz = np.zeros((total_steps, num_envs))
        ts_pitch = np.zeros((total_steps, num_envs))
        ts_pitch_rate = np.zeros((total_steps, num_envs))
        ts_roll = np.zeros((total_steps, num_envs))
        ts_drift_y = np.zeros((total_steps, num_envs))
        ts_drift_x = np.zeros((total_steps, num_envs))

        ts_wheel_vel_L = np.zeros((total_steps, num_envs))
        ts_wheel_vel_R = np.zeros((total_steps, num_envs))
        ts_wheel_tau_L = np.zeros((total_steps, num_envs))
        ts_wheel_tau_R = np.zeros((total_steps, num_envs))
        ts_wheel_act_L = np.zeros((total_steps, num_envs))
        ts_wheel_act_R = np.zeros((total_steps, num_envs))

        ts_knee_tau_L = np.zeros((total_steps, num_envs))
        ts_knee_tau_R = np.zeros((total_steps, num_envs))
        ts_hip_tau_L = np.zeros((total_steps, num_envs))
        ts_hip_tau_R = np.zeros((total_steps, num_envs))

        reset_events = []
        done_flags = np.zeros(num_envs, dtype=bool)

        for step in range(total_steps):
            curr_h = 0.39774 if step < step_cmd_start else target_h
            env.commands[:, 0] = 0.0
            env.commands[:, 1] = v_target
            env.commands[:, 2] = w_target
            env.height_cmd_buf[:] = curr_h

            if step == step_cmd_start:
                init_pos_xy = env.root_states[:, 0:2].cpu().numpy().copy()

            obs = env.get_observations()
            with torch.no_grad():
                actions = policy.act_inference(obs)
            obs, _, _, dones, _ = env.step(actions)

            # Record state
            base_pos = env.root_states[:, 0:3].cpu().numpy()
            base_h = base_pos[:, 2]
            base_vy = env.base_lin_vel[:, 1].cpu().numpy()
            base_wz = env.base_ang_vel[:, 2].cpu().numpy()

            pitch = (torch.asin(torch.clamp(env.projected_gravity[:, 1], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
            pitch_rate = (env.base_ang_vel[:, 0] * rad_to_deg).cpu().numpy()
            roll = (torch.asin(torch.clamp(env.projected_gravity[:, 0], -1.0, 1.0)) * rad_to_deg).cpu().numpy()

            wheel_vel_L = env.dof_vel[:, env.left_wheel_idx].cpu().numpy()
            wheel_vel_R = env.dof_vel[:, env.right_wheel_idx].cpu().numpy()
            wheel_tau_L = env.torques[:, env.left_wheel_idx].cpu().numpy()
            wheel_tau_R = env.torques[:, env.right_wheel_idx].cpu().numpy()
            wheel_act_L = actions[:, env.left_wheel_idx].cpu().numpy()
            wheel_act_R = actions[:, env.right_wheel_idx].cpu().numpy()

            knee_tau_L = env.torques[:, env.left_knee_idx].cpu().numpy()
            knee_tau_R = env.torques[:, env.right_knee_idx].cpu().numpy()
            hip_tau_L = env.torques[:, env.left_hip_idx].cpu().numpy()
            hip_tau_R = env.torques[:, env.right_hip_idx].cpu().numpy()

            ts_base_h[step] = base_h
            ts_h_err[step] = np.abs(base_h - curr_h)
            ts_vy[step] = base_vy
            ts_wz[step] = base_wz
            ts_pitch[step] = pitch
            ts_pitch_rate[step] = pitch_rate
            ts_roll[step] = roll

            if init_pos_xy is not None:
                ts_drift_x[step] = base_pos[:, 0] - init_pos_xy[:, 0]
                ts_drift_y[step] = base_pos[:, 1] - init_pos_xy[:, 1]

            ts_wheel_vel_L[step] = wheel_vel_L
            ts_wheel_vel_R[step] = wheel_vel_R
            ts_wheel_tau_L[step] = wheel_tau_L
            ts_wheel_tau_R[step] = wheel_tau_R
            ts_wheel_act_L[step] = wheel_act_L
            ts_wheel_act_R[step] = wheel_act_R

            ts_knee_tau_L[step] = knee_tau_L
            ts_knee_tau_R[step] = knee_tau_R
            ts_hip_tau_L[step] = hip_tau_L
            ts_hip_tau_R[step] = hip_tau_R

            # Check reset events
            d_arr = dones.cpu().numpy()
            if np.any(d_arr):
                for e in range(num_envs):
                    if d_arr[e] and not done_flags[e]:
                        done_flags[e] = True
                        reset_events.append({
                            "env_id": int(e),
                            "step": int(step),
                            "base_h": float(base_h[e]),
                            "pitch": float(pitch[e]),
                            "roll": float(roll[e]),
                            "vy": float(base_vy[e]),
                            "wheel_vel_L": float(wheel_vel_L[e]),
                            "wheel_vel_R": float(wheel_vel_R[e]),
                            "wheel_tau_L": float(wheel_tau_L[e]),
                            "wheel_tau_R": float(wheel_tau_R[e]),
                        })

        # Calculate steady-state statistics (steps 200..300)
        s_slice = slice(steady_start, total_steps)

        # Non-reset env mask
        valid_envs = ~done_flags

        # Height stats
        steady_h_mean = float(np.mean(ts_base_h[s_slice, valid_envs]))
        steady_h_err_mean = float(np.mean(ts_h_err[s_slice, valid_envs]) * 100.0) # cm
        pass_h_rate = float(np.mean(ts_h_err[s_slice] <= 0.010) * 100.0)
        pass_vy_rate = float(np.mean(np.abs(ts_vy[s_slice]) <= 0.030) * 100.0)
        pass_wz_rate = float(np.mean(np.abs(ts_wz[s_slice]) <= 0.050) * 100.0)
        joint_pass_rate = float(np.mean((ts_h_err[s_slice] <= 0.010) & (np.abs(ts_vy[s_slice]) <= 0.030) & (np.abs(ts_wz[s_slice]) <= 0.050)) * 100.0)

        # Wheel velocity stats (steady state)
        wheel_vel_mag = 0.5 * (np.abs(ts_wheel_vel_L[s_slice]) + np.abs(ts_wheel_vel_R[s_slice]))
        wheel_vel_mean = float(np.mean(wheel_vel_mag))
        wheel_vel_std = float(np.std(wheel_vel_mag))
        wheel_vel_rms = float(np.sqrt(np.mean(wheel_vel_mag**2)))
        wheel_vel_zero_fraction = float(np.mean(wheel_vel_mag < 0.05) * 100.0) # fraction of time wheel is near zero (<0.05 rad/s)

        # Wheel torque & saturation stats
        wheel_tau_mag = 0.5 * (np.abs(ts_wheel_tau_L[s_slice]) + np.abs(ts_wheel_tau_R[s_slice]))
        wheel_tau_mean = float(np.mean(wheel_tau_mag))
        wheel_tau_max = float(np.max(wheel_tau_mag))
        wheel_sat_rate = float(np.mean((np.abs(ts_wheel_tau_L[s_slice]) >= 6.95) | (np.abs(ts_wheel_tau_R[s_slice]) >= 6.95)) * 100.0)

        # Pitch & Roll stats
        pitch_mean = float(np.mean(ts_pitch[s_slice, valid_envs]))
        pitch_abs_mean = float(np.mean(np.abs(ts_pitch[s_slice, valid_envs])))
        pitch_max = float(np.max(np.abs(ts_pitch[s_slice, valid_envs])))
        pitch_rate_rms = float(np.sqrt(np.mean(ts_pitch_rate[s_slice, valid_envs]**2)))

        roll_abs_mean = float(np.mean(np.abs(ts_roll[s_slice, valid_envs])))

        # Position drift stats (at step 300)
        drift_y_end = float(np.mean(np.abs(ts_drift_y[-1, valid_envs])))
        drift_x_end = float(np.mean(np.abs(ts_drift_x[-1, valid_envs])))
        drift_dist_end = float(np.mean(np.sqrt(ts_drift_x[-1, valid_envs]**2 + ts_drift_y[-1, valid_envs]**2)))

        # Knee & Hip torque stats
        knee_tau_mean = float(np.mean(0.5 * (ts_knee_tau_L[s_slice] + ts_knee_tau_R[s_slice])))
        hip_tau_mean = float(np.mean(0.5 * (ts_hip_tau_L[s_slice] + ts_hip_tau_R[s_slice])))

        # Trajectory profile across steps (sampled every 20 steps after step 100)
        traj_profile = {}
        for st in range(100, 301, 20):
            idx = min(st, 299)
            w_v = 0.5 * (np.abs(ts_wheel_vel_L[idx]) + np.abs(ts_wheel_vel_R[idx]))
            w_t = 0.5 * (np.abs(ts_wheel_tau_L[idx]) + np.abs(ts_wheel_tau_R[idx]))
            traj_profile[f"step_{st}"] = {
                "h_mean": round(float(np.mean(ts_base_h[idx, valid_envs])), 4),
                "h_err_cm": round(float(np.mean(ts_h_err[idx, valid_envs]) * 100.0), 3),
                "pitch_deg": round(float(np.mean(ts_pitch[idx, valid_envs])), 3),
                "wheel_vel": round(float(np.mean(w_v)), 4),
                "wheel_tau": round(float(np.mean(w_t)), 3),
            }

        res = {
            "joint_pass_rate": round(joint_pass_rate, 2),
            "pass_h_rate": round(pass_h_rate, 2),
            "pass_vy_rate": round(pass_vy_rate, 2),
            "pass_wz_rate": round(pass_wz_rate, 2),
            "early_resets_count": int(np.sum(done_flags)),
            "survival_rate": round(float(np.mean(~done_flags) * 100.0), 2),
            "reset_events": reset_events,
            
            # Height metrics
            "steady_h_mean_m": round(steady_h_mean, 4),
            "steady_h_err_cm": round(steady_h_err_mean, 3),

            # Wheel kinematics & dynamics
            "wheel_vel_mean_rad_s": round(wheel_vel_mean, 4),
            "wheel_vel_std_rad_s": round(wheel_vel_std, 4),
            "wheel_vel_rms_rad_s": round(wheel_vel_rms, 4),
            "wheel_vel_zero_fraction_pct": round(wheel_vel_zero_fraction, 2),
            "wheel_tau_mean_Nm": round(wheel_tau_mean, 3),
            "wheel_tau_max_Nm": round(wheel_tau_max, 3),
            "wheel_sat_rate_pct": round(wheel_sat_rate, 2),

            # Attitude metrics
            "pitch_mean_deg": round(pitch_mean, 3),
            "pitch_abs_mean_deg": round(pitch_abs_mean, 3),
            "pitch_max_deg": round(pitch_max, 3),
            "pitch_rate_rms_deg_s": round(pitch_rate_rms, 3),
            "roll_abs_mean_deg": round(roll_abs_mean, 3),

            # Drift metrics
            "drift_x_end_m": round(drift_x_end, 4),
            "drift_y_end_m": round(drift_y_end, 4),
            "drift_dist_end_m": round(drift_dist_end, 4),

            # Leg joint torques
            "knee_tau_mean_Nm": round(knee_tau_mean, 3),
            "hip_tau_mean_Nm": round(hip_tau_mean, 3),

            # Trajectory profile
            "trajectory_profile": traj_profile
        }

        results[m_name] = res

    # Save diagnosis results
    out_dir = "evaluation_assets/phase7b/diagnosis"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase7b_inplace_squat_diagnosis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n=======================================================")
    print(f"Diagnosis Report Saved: {out_path}")
    print(f"=======================================================\n")

    # Print summary comparison table
    c6b = results["Phase6B_Champ_v2"]
    p7a = results["Phase7A_Sampling"]

    print(f"{'Metric':<35} | {'Phase 6B Champ v2':<20} | {'Phase 7A Sampling':<20} | {'Delta':<15}")
    print("-" * 98)
    metrics_to_print = [
        ("Joint Pass Rate (%)", "joint_pass_rate", "%"),
        ("Height Pass Rate (%)", "pass_h_rate", "%"),
        ("Vy Pass Rate (%)", "pass_vy_rate", "%"),
        ("Wz Pass Rate (%)", "pass_wz_rate", "%"),
        ("Early Resets (envs)", "early_resets_count", ""),
        ("Survival Rate (%)", "survival_rate", "%"),
        ("Steady Base Height (m)", "steady_h_mean_m", "m"),
        ("Steady Height Error (cm)", "steady_h_err_cm", "cm"),
        ("Wheel Vel Mean (rad/s)", "wheel_vel_mean_rad_s", "rad/s"),
        ("Wheel Vel RMS (rad/s)", "wheel_vel_rms_rad_s", "rad/s"),
        ("Wheel Vel Near-Zero (<0.05)", "wheel_vel_zero_fraction_pct", "%"),
        ("Wheel Torque Mean (Nm)", "wheel_tau_mean_Nm", "Nm"),
        ("Wheel Torque Max (Nm)", "wheel_tau_max_Nm", "Nm"),
        ("Wheel Torque Saturation (%)", "wheel_sat_rate_pct", "%"),
        ("Pitch Mean (deg)", "pitch_mean_deg", "deg"),
        ("Pitch Abs Mean (deg)", "pitch_abs_mean_deg", "deg"),
        ("Pitch Max (deg)", "pitch_max_deg", "deg"),
        ("Pitch Rate RMS (deg/s)", "pitch_rate_rms_deg_s", "deg/s"),
        ("Position Drift Y (m)", "drift_y_end_m", "m"),
        ("Total Position Drift (m)", "drift_dist_end_m", "m"),
        ("Knee Torque Mean (Nm)", "knee_tau_mean_Nm", "Nm"),
        ("Hip Torque Mean (Nm)", "hip_tau_mean_Nm", "Nm"),
    ]

    for label, key, unit in metrics_to_print:
        v1 = c6b[key]
        v2 = p7a[key]
        delta = v2 - v1
        print(f"{label:<35} | {v1:10.4f} {unit:<5} | {v2:10.4f} {unit:<5} | {delta:+10.4f} {unit}")

if __name__ == '__main__':
    run_inplace_squat_diagnosis()
