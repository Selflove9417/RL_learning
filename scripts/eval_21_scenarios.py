"""
Standard 21-Scenario Benchmark Evaluation Script (Phase 7C / Phase 8B aligned protocol).
Evaluates any model checkpoint across the 21 standard benchmark scenarios with
continuous -5mm height feedforward compensation:
    delta_h = -0.005 * clamp((0.39774 - h_ref) / (0.39774 - 0.360), 0.0, 1.0)
    h_ctrl = h_ref + delta_h

Usage:
    python scripts/eval_21_scenarios.py --model_path <path> --output_path <path>
"""

import sys
import isaacgym
import torch
import os
import json
import numpy as np

from isaacgym import gymtorch
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def parse_custom_args():
    model_path = "evaluation_assets/champion_v6/models/champion_v6_m115.pt"
    output_path = "evaluation_assets/phase14/results/eval_21_results_v6_refined.json"
    cleaned_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--model_path" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output_path" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        else:
            cleaned_argv.append(sys.argv[i])
            i += 1
    sys.argv = cleaned_argv
    return model_path, output_path

def run_21_evaluation():
    model_path, out_file = parse_custom_args()
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print(f"\n================ 21-Scenario Benchmark Evaluation ================")
    print(f"Model Path: {model_path}")

    loaded = torch.load(model_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    pol = ac.act_inference

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
        # 第四组：扩展域压力测试
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
    hold_steps_req = 25 # 0.5s at 50Hz
    rad_to_deg = 180.0 / np.pi
    total_steps = 300
    step_cmd_start = 100
    steady_start = 200

    model_results = {}
    all_joint_rates = []
    all_env_rates = []
    all_height_rates = []
    all_survival_rates = []
    all_wheel_sat_rates = []
    group_joint_rates = {}
    group_env_rates = {}

    inplace_squat_metrics = {}

    for group_name, cases in scenarios.items():
        group_metrics = {}
        grp_j_list = []
        grp_e_list = []

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
            wheel_sat_steps = np.zeros(args.num_envs)

            for step in range(total_steps):
                curr_h_ctrl = 0.39774 if step < step_cmd_start else h_ctrl
                env.commands[:, 0] = 0.0
                env.commands[:, 1] = cmd_vy
                env.commands[:, 2] = cmd_wz
                env.height_cmd_buf[:] = curr_h_ctrl
                if hasattr(env, 'target_height_buf'):
                    env.target_height_buf[:] = curr_h_ctrl

                env.compute_observations()
                obs = env.get_observations()
                with torch.no_grad():
                    actions = pol(obs)
                obs, _, _, dones, _ = env.step(actions)

                d_arr = dones.cpu().numpy()
                if np.any(d_arr):
                    early_done_count += np.sum(d_arr)
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
                    h_signed_err = base_h - h_ref
                    h_err = np.abs(h_signed_err)

                    steady_actual_h.append(base_h)
                    steady_h_signed.append(h_signed_err)
                    steady_h_abs.append(h_err)
                    steady_h_sq.append(np.square(h_err))

                    base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                    base_wz = env.base_ang_vel[:, 2].cpu().numpy()
                    steady_vy_sq.append(np.square(base_vy - cmd_vy))
                    steady_wz_sq.append(np.square(base_wz - cmd_wz))

                    pitch = (torch.asin(torch.clamp(env.projected_gravity[:, 1], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
                    roll = (torch.asin(torch.clamp(env.projected_gravity[:, 0], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
                    steady_pitch_abs.append(np.abs(pitch))
                    steady_pitch_max.append(np.max(np.abs(pitch)))
                    steady_roll_abs.append(np.abs(roll))
                    steady_roll_max.append(np.max(np.abs(roll)))

                    tau_knee_L = np.abs(env.torques[:, env.left_knee_idx].cpu().numpy())
                    tau_knee_R = np.abs(env.torques[:, env.right_knee_idx].cpu().numpy())
                    knee_tau_mean_list.append(0.5 * (tau_knee_L + tau_knee_R))
                    knee_tau_max_list.append(np.maximum(tau_knee_L, tau_knee_R))

            steady_slice = slice(steady_start - step_cmd_start, total_steps - step_cmd_start)
            h_pass_matrix = (h_err_traj[:, steady_slice] <= tol_h)
            vy_pass_matrix = (vy_err_traj[:, steady_slice] <= tol_vy)
            wz_pass_matrix = (wz_err_traj[:, steady_slice] <= tol_wz)

            alive_envs = ~done_env_flags
            if np.any(alive_envs):
                sample_pass_h = float(np.mean(h_pass_matrix[alive_envs]) * 100.0)
                sample_pass_vy = float(np.mean(vy_pass_matrix[alive_envs]) * 100.0)
                sample_pass_wz = float(np.mean(wz_pass_matrix[alive_envs]) * 100.0)
                joint_pass_matrix = h_pass_matrix & vy_pass_matrix & wz_pass_matrix
                sample_pass_joint = float(np.mean(joint_pass_matrix[alive_envs]) * 100.0)
            else:
                sample_pass_h = sample_pass_vy = sample_pass_wz = sample_pass_joint = 0.0

            settling_times = []
            failed_envs = 0
            for e in range(args.num_envs):
                if done_env_flags[e]:
                    failed_envs += 1
                    continue
                cur_consec = 0
                found_time = None
                for t_idx in range(total_steps - step_cmd_start):
                    ok = (h_err_traj[e, t_idx] <= tol_h) and (vy_err_traj[e, t_idx] <= tol_vy) and (wz_err_traj[e, t_idx] <= tol_wz)
                    if ok:
                        cur_consec += 1
                        if cur_consec >= hold_steps_req:
                            found_time = (t_idx - hold_steps_req + 1) * 0.02
                            settling_times.append(found_time)
                            break
                    else:
                        cur_consec = 0
                if found_time is None:
                    failed_envs += 1

            env_pass_rate = float((args.num_envs - failed_envs) / args.num_envs * 100.0)
            surv_rate = float((args.num_envs - np.sum(done_env_flags)) / args.num_envs * 100.0)
            wheel_sat_rate = float(np.mean(wheel_sat_steps) / total_steps * 100.0)

            case_res = {
                "h_ref_target_m": round(float(h_ref), 4),
                "steady_actual_h_m": round(float(np.mean(steady_actual_h)), 4),
                "h_steady_err_cm": round(float(np.mean(steady_h_abs) * 100.0), 3),
                "h_pass_rate": round(sample_pass_h, 1),
                "vy_rms_mps": round(float(np.sqrt(np.mean(steady_vy_sq))), 4),
                "wz_rms_radps": round(float(np.sqrt(np.mean(steady_wz_sq))), 4),
                "joint_pass_rate": round(sample_pass_joint, 1),
                "env_pass_rate": round(env_pass_rate, 1),
                "wheel_sat_rate": round(wheel_sat_rate, 2),
                "episode_surv_rate": round(surv_rate, 2),
            }
            group_metrics[case_name] = case_res
            grp_j_list.append(sample_pass_joint)
            grp_e_list.append(env_pass_rate)

            all_joint_rates.append(sample_pass_joint)
            all_env_rates.append(env_pass_rate)
            all_height_rates.append(sample_pass_h)
            all_survival_rates.append(surv_rate)
            all_wheel_sat_rates.append(wheel_sat_rate)

            if case_name == "g3_inplace_squat":
                inplace_squat_metrics = case_res

            print(f"[{group_name}] {case_name:24s} | H_err:{case_res['h_steady_err_cm']:4.2f}cm | Jnt:{case_res['joint_pass_rate']:5.1f}% | Env:{case_res['env_pass_rate']:5.1f}% | Surv:{case_res['episode_surv_rate']:5.1f}%")

        group_joint_rates[group_name] = round(float(np.mean(grp_j_list)), 2)
        group_env_rates[group_name] = round(float(np.mean(grp_e_list)), 2)
        model_results[group_name] = group_metrics

    summary = {
        "global_joint_pass_rate": round(float(np.mean(all_joint_rates)), 2),
        "global_env_pass_rate": round(float(np.mean(all_env_rates)), 2),
        "global_height_pass_rate": round(float(np.mean(all_height_rates)), 2),
        "global_survival_rate": round(float(np.mean(all_survival_rates)), 2),
        "mean_wheel_sat_rate": round(float(np.mean(all_wheel_sat_rates)), 2),
        "group_joint_pass_rates": group_joint_rates,
        "group_env_pass_rates": group_env_rates,
        "g3_inplace_squat": inplace_squat_metrics,
    }

    full_output = {
        "summary": summary,
        "details": model_results,
    }

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    print("\n================ 21-Scenario Benchmark Summary ================")
    print(f"  Global Joint% : {summary['global_joint_pass_rate']:.2f}% (Champion v4: 66.52%)")
    print(f"  Global Env%   : {summary['global_env_pass_rate']:.2f}% (Champion v4: 63.24%)")
    print(f"  Global Height%: {summary['global_height_pass_rate']:.2f}%")
    print(f"  In-place Squat Joint%: {inplace_squat_metrics.get('joint_pass_rate', 0.0):.1f}% (Threshold: >= 85.0%, Champ: 88.8%)")
    print(f"  In-place Squat Env%  : {inplace_squat_metrics.get('env_pass_rate', 0.0):.1f}%")
    print(f"  Survival Rate : {summary['global_survival_rate']:.2f}%")
    print(f"  Wheel Sat Rate: {summary['mean_wheel_sat_rate']:.2f}%")
    print(f"Results saved to: {out_file}\n")

if __name__ == '__main__':
    run_21_evaluation()
