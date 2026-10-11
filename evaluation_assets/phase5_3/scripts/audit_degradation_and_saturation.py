import isaacgym
import torch
import os
import json
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def audit():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 20

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None

    m0_path = "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct09_transplanted_phase5_1/model_0.pt"
    m150_path = "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_10-32-49_in_place_height_stage5_1_formal/model_150.pt"

    def load_policy(ckpt_path):
        loaded = torch.load(ckpt_path, map_location=env.device)
        ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
        ac.load_state_dict(loaded['model_state_dict'])
        ac.eval()
        return ac.act_inference

    policy_m0 = load_policy(m0_path)
    policy_m150 = load_policy(m150_path)

    # 1. Audit Issue 1: fwd_right_turn and bwd_right_turn time series
    print("\n=== AUDITING ISSUE 1: OLD SKILL TRACKING DEGRADATION ===")
    turns = [
        ("fwd_right_turn", 0.20, -0.35),
        ("bwd_right_turn", -0.20, -0.35),
    ]

    issue1_results = {}
    for task_name, cmd_vy, cmd_wz in turns:
        issue1_results[task_name] = {}
        for m_name, pol in [("model_0", policy_m0), ("model_150", policy_m150)]:
            torch.manual_seed(42)
            env.reset()
            env.commands[:, 0] = 0.0
            env.commands[:, 1] = cmd_vy
            env.commands[:, 2] = cmd_wz
            env.height_cmd_buf[:] = 0.39774

            vy_all = [] # [steps, envs]
            wz_all = []
            for step in range(200):
                obs = env.get_observations()
                with torch.no_grad():
                    actions = pol(obs)
                obs, _, _, dones, _ = env.step(actions)
                vy_all.append(env.base_lin_vel[:, 1].cpu().numpy())
                wz_all.append(env.base_ang_vel[:, 2].cpu().numpy())

            vy_arr = np.array(vy_all) # [200, 20]
            wz_arr = np.array(wz_all)

            # Analyze full vs settled
            full_vy_err = np.abs(vy_arr - cmd_vy)
            settled_vy_err = np.abs(vy_arr[30:] - cmd_vy)
            settled_wz_err = np.abs(wz_arr[30:] - cmd_wz)

            issue1_results[task_name][m_name] = {
                "mean_vy_settled": round(float(np.mean(vy_arr[30:])), 4),
                "std_vy_settled": round(float(np.std(vy_arr[30:])), 4),
                "rms_vy_err_settled": round(float(np.sqrt(np.mean(settled_vy_err**2))), 4),
                "mean_wz_settled": round(float(np.mean(wz_arr[30:])), 4),
                "rms_wz_err_settled": round(float(np.sqrt(np.mean(settled_wz_err**2))), 4),
                "min_vy_settled": round(float(np.min(vy_arr[30:])), 4),
                "max_vy_settled": round(float(np.max(vy_arr[30:])), 4),
            }
            print(f"[{task_name} | {m_name}] Settled Vy: {issue1_results[task_name][m_name]['mean_vy_settled']:.3f} +/- {issue1_results[task_name][m_name]['std_vy_settled']:.3f} | RMS Err: {issue1_results[task_name][m_name]['rms_vy_err_settled']:.4f}")

    # 2. Audit Issue 2: Wheel torque saturation (7.0 Nm) in backward cruise height switching
    print("\n=== AUDITING ISSUE 2: WHEEL TORQUE SATURATION (7.0 Nm) ===")
    sat_scenarios = [
        ("bwd_0.10_squat", -0.10, [(0, 100, 0.39774), (100, 200, 0.35000), (200, 300, 0.39774)]),
        ("bwd_0.10_raise", -0.10, [(0, 100, 0.39774), (100, 200, 0.45000), (200, 300, 0.39774)]),
        ("bwd_0.20_raise", -0.20, [(0, 100, 0.39774), (100, 200, 0.45000), (200, 300, 0.39774)]),
        ("fwd_0.10_squat", 0.10, [(0, 100, 0.39774), (100, 200, 0.35000), (200, 300, 0.39774)]),
    ]

    issue2_results = {}
    for s_name, cmd_vy, h_stages in sat_scenarios:
        torch.manual_seed(42)
        env.reset()
        env.commands[:, 0] = 0.0
        env.commands[:, 1] = cmd_vy
        env.commands[:, 2] = 0.0

        sat_events = []
        tau_L_all, tau_R_all, vy_all = [], [], []

        # Track saturation per env
        consecutive_sat_L = np.zeros(env.num_envs, dtype=int)
        consecutive_sat_R = np.zeros(env.num_envs, dtype=int)
        max_consecutive_sat = 0

        total_steps = 300
        total_env_steps = total_steps * env.num_envs

        for step in range(total_steps):
            for s_start, s_end, s_h in h_stages:
                if s_start <= step < s_end:
                    curr_h = s_h
                    break
            env.height_cmd_buf[:] = curr_h

            obs = env.get_observations()
            with torch.no_grad():
                actions = policy_m150(obs)
            obs, _, _, dones, _ = env.step(actions)

            tau_L = torch.abs(env.torques[:, env.left_wheel_idx]).cpu().numpy()
            tau_R = torch.abs(env.torques[:, env.right_wheel_idx]).cpu().numpy()
            vy = env.base_lin_vel[:, 1].cpu().numpy()

            tau_L_all.append(tau_L)
            tau_R_all.append(tau_R)
            vy_all.append(vy)

            is_sat_L = tau_L >= 6.99
            is_sat_R = tau_R >= 6.99
            is_sat_either = is_sat_L | is_sat_R

            for i in range(env.num_envs):
                if is_sat_L[i]:
                    consecutive_sat_L[i] += 1
                else:
                    consecutive_sat_L[i] = 0

                if is_sat_R[i]:
                    consecutive_sat_R[i] += 1
                else:
                    consecutive_sat_R[i] = 0

                c_max = max(consecutive_sat_L[i], consecutive_sat_R[i])
                if c_max > max_consecutive_sat:
                    max_consecutive_sat = c_max

                if is_sat_either[i]:
                    sat_events.append({
                        "step": step,
                        "time_s": round(step * 0.02, 2),
                        "env_id": i,
                        "tau_L": float(round(tau_L[i], 3)),
                        "tau_R": float(round(tau_R[i], 3)),
                        "vy": float(round(vy[i], 3)),
                        "stage_h": curr_h,
                    })

        tau_L_arr = np.array(tau_L_all) # [300, 20]
        tau_R_arr = np.array(tau_R_all)

        sat_L_count = np.sum(tau_L_arr >= 6.99)
        sat_R_count = np.sum(tau_R_arr >= 6.99)
        sat_total = len(sat_events)

        issue2_results[s_name] = {
            "total_env_steps": total_env_steps,
            "sat_events_count": sat_total,
            "sat_pct": round(float(sat_total / total_env_steps * 100.0), 3),
            "sat_L_pct": round(float(sat_L_count / total_env_steps * 100.0), 3),
            "sat_R_pct": round(float(sat_R_count / total_env_steps * 100.0), 3),
            "max_consecutive_steps": int(max_consecutive_sat),
            "max_consecutive_duration_s": round(float(max_consecutive_sat * 0.02), 3),
            "sample_events": sat_events[:10],
        }

        print(f"[{s_name}] Sat events: {sat_total}/{total_env_steps} ({issue2_results[s_name]['sat_pct']}%) | Max consecutive: {max_consecutive_sat} steps ({issue2_results[s_name]['max_consecutive_duration_s']}s)")

    out_file = "/home/xy/.gemini/antigravity/brain/853aaed2-8860-4930-be3f-1bb608c17795/scratch/audit_degradation_and_saturation_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "issue1_old_skill_degradation": issue1_results,
            "issue2_wheel_torque_saturation": issue2_results
        }, f, indent=2)
    print(f"Results saved to {out_file}")

if __name__ == '__main__':
    audit()
