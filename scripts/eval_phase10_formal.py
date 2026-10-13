"""
Phase 10: Formal Full 21-Scenario Benchmark Evaluation Script.
Evaluates the candidate model on all 21 formal benchmark scenarios with:
- 32 parallel environments
- seed = 42
- 300 simulation steps
- Strict evaluation against h_ref
- Phase 7F feedforward height compensation:
    delta_h = -0.005 * clamp((0.39774 - h_ref) / (0.39774 - 0.360), 0.0, 1.0)
    h_ctrl = h_ref + delta_h
- Wheel saturation threshold: 6.95 Nm
- Output saved to evaluation_assets/phase10/results/eval_phase10_formal_results.json
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

def run_phase10_full_evaluation(model_name="phase10_candidate", model_path=None):
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    body_states_tensor = env.gym.acquire_rigid_body_state_tensor(env.sim)
    bs = gymtorch.wrap_tensor(body_states_tensor).view(args.num_envs, env.num_bodies, 13)

    if model_path is None:
        raise ValueError("model_path must be provided!")

    scenarios = {
        # 第一组：重点下蹲测试 (H=0.36m)
        "group1_squat": [
            ("g1_fwd_left_squat",     0.10,  0.20, 0.36),
            ("g1_bwd_left_squat",    -0.10,  0.20, 0.36),
            ("g1_fwd_right_squat",    0.10, -0.20, 0.36),
            ("g1_bwd_right_squat",   -0.10, -0.20, 0.36),
            ("g1_fwd_straight_squat", 0.10,  0.00, 0.36),
        ],
        # 第二组：升高与恢复能力 (H=0.44m / 0.39774m)
        "group2_rise": [
            ("g2_fwd_left_rise",      0.10,  0.20, 0.44),
            ("g2_fwd_right_rise",     0.10, -0.20, 0.44),
            ("g2_bwd_left_rise",     -0.10,  0.20, 0.44),
            ("g2_bwd_right_rise",    -0.10, -0.20, 0.44),
            ("g2_recover_nominal",    0.10,  0.20, 0.39774),
        ],
        # 第三组：旧任务保持 (防遗忘)
        "group3_regression": [
            ("g3_inplace_squat",      0.00,  0.00, 0.36),
            ("g3_cruise_squat",       0.15,  0.00, 0.36),
            ("g3_nominal_cruise",     0.20,  0.00, 0.39774),
            ("g3_nominal_compound",   0.10,  0.20, 0.39774),
            ("g3_nominal_stand",      0.00,  0.00, 0.39774),
        ],
        # 第四组：扩展域压力测试 (原 Stage 2 首段范围)
        "group4_stress": [
            ("g4_exp_fwd_left_squat",   0.16,  0.30, 0.36),
            ("g4_exp_fwd_right_squat",  0.16, -0.30, 0.36),
            ("g4_exp_bwd_left_squat",  -0.16,  0.30, 0.36),
            ("g4_exp_bwd_right_squat", -0.16, -0.30, 0.36),
            ("g4_exp_fwd_left_rise",    0.16,  0.30, 0.44),
            ("g4_exp_fwd_right_rise",   0.16, -0.30, 0.44),
        ]
    }

    tol_h = 0.010       # 1cm
    tol_vy = 0.030      # 0.03 m/s
    tol_wz = 0.050      # 0.05 rad/s
    rad_to_deg = 180.0 / np.pi
    R_wheel = 0.070
    r_vec = np.array([0.0, 0.0, -R_wheel])

    total_steps = 300
    step_cmd_start = 100
    steady_start = 200

    print(f"\n================ Evaluating Checkpoint: {model_name} ================")
    loaded = torch.load(model_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    pol = ac.act_inference

    model_results = {}

    for group_name, cases in scenarios.items():
        group_metrics = {}
        for case_name, cmd_vy, cmd_wz, target_h in cases:
            h_ref = target_h
            delta_h = -0.005 * float(np.clip((0.39774 - h_ref) / (0.39774 - 0.360), 0.0, 1.0))
            h_ctrl = h_ref + delta_h

            torch.manual_seed(42)
            np.random.seed(42)
            env.reset()

            h_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
            vy_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
            wz_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
            early_done_count = 0
            done_env_flags = np.zeros(args.num_envs, dtype=bool)

            steady_actual_h = []
            steady_h_signed = []
            steady_h_abs = []
            steady_h_sq = []
            steady_vy_sq = []
            steady_wz_sq = []
            steady_pitch_abs = []
            steady_pitch_max = []
            steady_roll_abs = []
            steady_roll_max = []
            knee_tau_mean_list = []
            knee_tau_max_list = []
            true_slip_mean_list = []
            true_slip_max_list = []
            wheel_sat_steps = np.zeros(args.num_envs)

            for step in range(total_steps):
                curr_h_ctrl = 0.39774 if step < step_cmd_start else h_ctrl
                env.commands[:, 0] = 0.0
                env.commands[:, 1] = cmd_vy
                env.commands[:, 2] = cmd_wz
                env.height_cmd_buf[:] = curr_h_ctrl

                obs = env.get_observations()
                with torch.no_grad():
                    actions = pol(obs)

                obs, _, _, dones, _ = env.step(actions)

                d_arr = dones.cpu().numpy()
                early_done_count += int(np.sum(d_arr))
                done_env_flags |= d_arr

                tau_wheels = torch.abs(env.torques[:, env.wheel_indices]).cpu().numpy()
                is_sat = np.any(tau_wheels >= 6.95, axis=1)
                wheel_sat_steps += is_sat.astype(float)

                if step >= step_cmd_start:
                    idx = step - step_cmd_start
                    base_h = env.root_states[:, 2].cpu().numpy()
                    base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                    base_wz = env.base_ang_vel[:, 2].cpu().numpy()

                    h_err_traj[:, idx] = np.abs(base_h - h_ref)
                    vy_err_traj[:, idx] = np.abs(base_vy - cmd_vy)
                    wz_err_traj[:, idx] = np.abs(base_wz - cmd_wz)

                if step >= steady_start:
                    base_h = env.root_states[:, 2].cpu().numpy()
                    base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                    base_wz = env.base_ang_vel[:, 2].cpu().numpy()
                    gx = env.projected_gravity[:, 0].cpu().numpy()
                    gy = env.projected_gravity[:, 1].cpu().numpy()
                    pitch = np.arcsin(np.clip(gy, -1.0, 1.0)) * rad_to_deg
                    roll = np.arcsin(np.clip(gx, -1.0, 1.0)) * rad_to_deg

                    steady_actual_h.append(base_h)
                    steady_h_signed.append(base_h - h_ref)
                    steady_h_abs.append(np.abs(base_h - h_ref))
                    steady_h_sq.append((base_h - h_ref) ** 2)
                    steady_vy_sq.append((base_vy - cmd_vy) ** 2)
                    steady_wz_sq.append((base_wz - cmd_wz) ** 2)
                    steady_pitch_abs.append(np.abs(pitch))
                    steady_pitch_max.append(np.max(np.abs(pitch)))
                    steady_roll_abs.append(np.abs(roll))
                    steady_roll_max.append(np.max(np.abs(roll)))

                    knee_tau = torch.abs(env.torques[:, [env.left_knee_idx, env.right_knee_idx]]).cpu().numpy()
                    knee_tau_mean_list.append(np.mean(knee_tau))
                    knee_tau_max_list.append(np.max(knee_tau))

                    v_wheel_L = bs[:, env.wheel_indices[0], 7:10].cpu().numpy()
                    w_wheel_L = bs[:, env.wheel_indices[0], 10:13].cpu().numpy()
                    v_contact_L = v_wheel_L + np.cross(w_wheel_L, r_vec)
                    slip_L = np.linalg.norm(v_contact_L[:, :2], axis=1)

                    v_wheel_R = bs[:, env.wheel_indices[1], 7:10].cpu().numpy()
                    w_wheel_R = bs[:, env.wheel_indices[1], 10:13].cpu().numpy()
                    v_contact_R = v_wheel_R + np.cross(w_wheel_R, r_vec)
                    slip_R = np.linalg.norm(v_contact_R[:, :2], axis=1)

                    steady_slip = np.maximum(slip_L, slip_R)
                    true_slip_mean_list.append(np.mean(steady_slip))
                    true_slip_max_list.append(np.max(steady_slip))

            pass_h_mat = (h_err_traj <= tol_h)
            pass_vy_mat = (vy_err_traj <= tol_vy)
            pass_wz_mat = (wz_err_traj <= tol_wz)
            pass_joint_mat = pass_h_mat & pass_vy_mat & pass_wz_mat

            steady_slice = slice(100, 200)
            sample_pass_h = float(np.mean(pass_h_mat[:, steady_slice]) * 100.0)
            sample_pass_vy = float(np.mean(pass_vy_mat[:, steady_slice]) * 100.0)
            sample_pass_wz = float(np.mean(pass_wz_mat[:, steady_slice]) * 100.0)
            sample_pass_joint = float(np.mean(pass_joint_mat[:, steady_slice]) * 100.0)

            env_pass_count = 0
            for e in range(args.num_envs):
                if not done_env_flags[e] and np.mean(pass_joint_mat[e, steady_slice]) >= 0.80:
                    env_pass_count += 1
            env_pass_rate = float(env_pass_count / args.num_envs * 100.0)

            settle_times = []
            for e in range(args.num_envs):
                if done_env_flags[e]:
                    continue
                stable_idx = -1
                for t in range(total_steps - step_cmd_start - 25):
                    if np.all(pass_joint_mat[e, t:t + 25]):
                        stable_idx = t
                        break
                if stable_idx != -1:
                    settle_times.append(stable_idx * env.dt)
            settle_p50 = float(np.percentile(settle_times, 50)) if len(settle_times) > 0 else -1.0
            settle_p95 = float(np.percentile(settle_times, 95)) if len(settle_times) > 0 else -1.0

            h_steady_signed_err = float(np.mean(steady_h_signed))
            h_steady_abs_err = float(np.mean(steady_h_abs))
            h_rms = float(np.sqrt(np.mean(steady_h_sq)))
            vy_rms = float(np.sqrt(np.mean(steady_vy_sq)))
            wz_rms = float(np.sqrt(np.mean(steady_wz_sq)))
            pitch_rms = float(np.mean(steady_pitch_abs))
            pitch_peak = float(np.max(steady_pitch_max))
            roll_rms = float(np.mean(steady_roll_abs))
            roll_peak = float(np.max(steady_roll_max))
            knee_tau_mean = float(np.mean(knee_tau_mean_list))
            knee_tau_max = float(np.max(knee_tau_max_list))
            wheel_sat_rate = float(np.mean(wheel_sat_steps / total_steps) * 100.0)
            true_slip_mean = float(np.mean(true_slip_mean_list))
            true_slip_max = float(np.max(true_slip_max_list))
            surv_rate = float(np.mean(~done_env_flags) * 100.0)

            metrics = {
                "h_ref_target_m": h_ref,
                "h_ctrl_cmd_m": h_ctrl,
                "steady_actual_h_m": round(float(np.mean(steady_actual_h)), 4),
                "h_signed_err_cm": round(h_steady_signed_err * 100.0, 3),
                "h_steady_err_cm": round(h_steady_abs_err * 100.0, 3),
                "h_rms_cm": round(h_rms * 100.0, 3),
                "h_pass_rate": round(sample_pass_h, 1),
                "vy_rms_mps": round(vy_rms, 4),
                "vy_pass_rate": round(sample_pass_vy, 1),
                "wz_rms_radps": round(wz_rms, 4),
                "wz_pass_rate": round(sample_pass_wz, 1),
                "joint_pass_rate": round(sample_pass_joint, 1),
                "env_pass_rate": round(env_pass_rate, 1),
                "failed_envs": int(args.num_envs - env_pass_count),
                "settle_p50_s": round(settle_p50, 2),
                "settle_p95_s": round(settle_p95, 2),
                "pitch_rms_deg": round(pitch_rms, 2),
                "pitch_peak_deg": round(pitch_peak, 2),
                "roll_rms_deg": round(roll_rms, 2),
                "roll_peak_deg": round(roll_peak, 2),
                "knee_tau_mean_Nm": round(knee_tau_mean, 2),
                "knee_tau_max_Nm": round(knee_tau_max, 2),
                "wheel_sat_rate": round(wheel_sat_rate, 2),
                "true_slip_mean_mps": round(true_slip_mean, 4),
                "true_slip_max_mps": round(true_slip_max, 4),
                "early_dones": early_done_count,
                "episode_surv_rate": round(surv_rate, 1)
            }
            group_metrics[case_name] = metrics
            print(f"  {case_name:<24}: Joint%={metrics['joint_pass_rate']:>5.1f}%, Env%={metrics['env_pass_rate']:>5.1f}%, H_err={metrics['h_signed_err_cm']:>+6.2f}cm")

        model_results[group_name] = group_metrics

    # Calculate overall summary across all 21 cases
    all_joint = []
    all_env = []
    all_h_pass = []
    all_h_err = []
    all_surv = []
    all_wheel_sat = []
    group_summaries = {}

    for g_name, cases in model_results.items():
        g_joint = [m["joint_pass_rate"] for m in cases.values()]
        g_env = [m["env_pass_rate"] for m in cases.values()]
        g_h_pass = [m["h_pass_rate"] for m in cases.values()]
        g_h_err = [m["h_steady_err_cm"] for m in cases.values()]
        group_summaries[g_name] = {
            "joint": float(np.mean(g_joint)),
            "env": float(np.mean(g_env)),
            "h_pass": float(np.mean(g_h_pass)),
            "h_err": float(np.mean(g_h_err)),
        }
        all_joint.extend(g_joint)
        all_env.extend(g_env)
        all_h_pass.extend(g_h_pass)
        all_h_err.extend(g_h_err)
        all_surv.extend([m["episode_surv_rate"] for m in cases.values()])
        all_wheel_sat.extend([m["wheel_sat_rate"] for m in cases.values()])

    summary = {
        "joint_pass_rate": float(np.mean(all_joint)),
        "env_pass_rate": float(np.mean(all_env)),
        "h_pass_rate": float(np.mean(all_h_pass)),
        "h_steady_err_cm": float(np.mean(all_h_err)),
        "episode_surv_rate": float(np.mean(all_surv)),
        "wheel_sat_rate": float(np.mean(all_wheel_sat)),
        "group_summaries": group_summaries,
    }
    model_results["_summary"] = summary

    print(f"\n========================================================")
    print(f" Formal 21-Scenario Benchmark Summary ({model_name}):")
    print(f" Global Joint%       : {summary['joint_pass_rate']:.2f}% (vs Champ v4: 66.52%)")
    print(f" Global Env%         : {summary['env_pass_rate']:.2f}% (vs Champ v4: 63.24%)")
    print(f" Group 1 (Squat)     : Joint%={group_summaries['group1_squat']['joint']:.2f}%, Env%={group_summaries['group1_squat']['env']:.2f}%")
    print(f" Group 2 (Rise)      : Joint%={group_summaries['group2_rise']['joint']:.2f}%, Env%={group_summaries['group2_rise']['env']:.2f}%")
    print(f" Group 3 (Regression): Joint%={group_summaries['group3_regression']['joint']:.2f}%, Env%={group_summaries['group3_regression']['env']:.2f}%")
    print(f" Group 4 (Stress)    : Joint%={group_summaries['group4_stress']['joint']:.2f}%, Env%={group_summaries['group4_stress']['env']:.2f}%")
    print(f" Inplace Squat       : Joint%={model_results['group3_regression']['g3_inplace_squat']['joint_pass_rate']:.2f}% (vs Champ v4: 88.80%)")
    print(f" Wheel Saturation    : {summary['wheel_sat_rate']:.2f}%")
    print(f" Survival Rate       : {summary['episode_surv_rate']:.2f}%")
    print(f"========================================================\n")

    out_file = "evaluation_assets/phase10/results/eval_phase10_formal_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({model_name: model_results}, f, indent=2)
    print(f"Saved full formal results to {out_file}")
    return summary

if __name__ == '__main__':
    default_model = "phase10_m20"
    default_path = "/home/xy/IsaacTu/evaluation_assets/phase10/models/phase10_m20.pt"
    run_phase10_full_evaluation(default_model, default_path)
