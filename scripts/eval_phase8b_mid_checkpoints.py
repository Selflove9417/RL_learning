"""
Phase 8B Mid-training Evaluation across 7 key scenarios for m3, m5, m10.
Scenarios:
Weak cases:
- g4_exp_fwd_left_squat (0.16, 0.30, 0.360)
- g4_exp_bwd_left_squat (-0.16, 0.30, 0.360)
- g2_bwd_right_rise (-0.10, -0.20, 0.440)
- g1_fwd_left_squat (0.10, 0.20, 0.360)
Preservation cases:
- g3_inplace_squat (0.00, 0.00, 0.360)
- g3_nominal_cruise (0.20, 0.00, 0.39774)
- g3_nominal_compound (0.10, 0.20, 0.39774)

Evaluation Interface:
    Continuous -5 mm height feedforward compensation.
    Strictly evaluated against h_ref.
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

def run_phase8b_mid_evaluation():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    models_to_test = {
        "m3": "evaluation_assets/phase8b/models/phase8b_m3.pt",
        "m5": "evaluation_assets/phase8b/models/phase8b_m5.pt",
        "m10": "evaluation_assets/phase8b/models/phase8b_m10.pt",
    }

    test_cases = [
        # Weak cases
        ("g4_exp_fwd_left_squat",  0.16,  0.30, 0.36),
        ("g4_exp_bwd_left_squat", -0.16,  0.30, 0.36),
        ("g2_bwd_right_rise",    -0.10, -0.20, 0.44),
        ("g1_fwd_left_squat",     0.10,  0.20, 0.36),
        # Preservation cases
        ("g3_inplace_squat",      0.00,  0.00, 0.36),
        ("g3_nominal_cruise",     0.20,  0.00, 0.39774),
        ("g3_nominal_compound",   0.10,  0.20, 0.39774),
    ]

    tol_h = 0.010
    tol_vy = 0.030
    tol_wz = 0.050
    hold_steps_req = 25
    rad_to_deg = 180.0 / np.pi
    total_steps = 300
    step_cmd_start = 100
    steady_start = 200

    results_all = {}

    for m_label, ckpt_path in models_to_test.items():
        print(f"\n================ Evaluating Model: {m_label} ({ckpt_path}) ================")
        loaded = torch.load(ckpt_path, map_location=env.device)
        ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
        ac.load_state_dict(loaded['model_state_dict'])
        ac.eval()
        pol = ac.act_inference

        model_res = {}

        for case_name, cmd_vy, cmd_wz, target_h in test_cases:
            h_ref = target_h
            delta_h = -0.005 * float(np.clip((0.39774 - h_ref) / (0.39774 - 0.360), 0.0, 1.0))
            h_ctrl = h_ref + delta_h

            torch.manual_seed(42)
            np.random.seed(42)
            env.reset()

            h_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
            vy_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
            wz_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
            done_env_flags = np.zeros(args.num_envs, dtype=bool)

            steady_actual_h = []
            steady_h_abs = []
            steady_vy_sq = []
            steady_wz_sq = []
            steady_pitch_abs = []

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
                done_env_flags |= d_arr

                base_h = env.root_states[:, 2].cpu().numpy()
                base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                base_wz = env.base_ang_vel[:, 2].cpu().numpy()

                if step >= step_cmd_start:
                    idx = step - step_cmd_start
                    h_err_traj[:, idx] = np.abs(base_h - h_ref)
                    vy_err_traj[:, idx] = np.abs(base_vy - cmd_vy)
                    wz_err_traj[:, idx] = np.abs(base_wz - cmd_wz)

                if step >= steady_start:
                    h_err = np.abs(base_h - h_ref)
                    steady_actual_h.append(base_h)
                    steady_h_abs.append(h_err)
                    steady_vy_sq.append(np.square(base_vy - cmd_vy))
                    steady_wz_sq.append(np.square(base_wz - cmd_wz))
                    pitch = (torch.asin(torch.clamp(env.projected_gravity[:, 1], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
                    steady_pitch_abs.append(np.abs(pitch))

            pass_h_mat = (h_err_traj <= tol_h)
            pass_vy_mat = (vy_err_traj <= tol_vy)
            pass_wz_mat = (wz_err_traj <= tol_wz)
            pass_joint_mat = pass_h_mat & pass_vy_mat & pass_wz_mat

            steady_slice = slice(100, 200)
            sample_pass_h = float(np.mean(pass_h_mat[:, steady_slice]) * 100.0)
            sample_pass_vy = float(np.mean(pass_vy_mat[:, steady_slice]) * 100.0)
            sample_pass_wz = float(np.mean(pass_wz_mat[:, steady_slice]) * 100.0)
            sample_pass_joint = float(np.mean(pass_joint_mat[:, steady_slice]) * 100.0)

            failed_envs = 0
            for e in range(args.num_envs):
                if done_env_flags[e]:
                    failed_envs += 1
                    continue
                j_seq = pass_joint_mat[e]
                cur_consec = 0
                found_time = None
                for t_idx in range(len(j_seq)):
                    if j_seq[t_idx]:
                        cur_consec += 1
                        if cur_consec >= hold_steps_req:
                            found_time = (t_idx - hold_steps_req + 1) * 0.02
                            break
                    else:
                        cur_consec = 0
                if found_time is None:
                    failed_envs += 1

            env_pass_rate = float((args.num_envs - failed_envs) / args.num_envs * 100.0)
            surv_rate = float((args.num_envs - np.sum(done_env_flags)) / args.num_envs * 100.0)

            c_dict = {
                "steady_actual_h_m": round(float(np.mean(steady_actual_h)), 4),
                "h_steady_err_cm": round(float(np.mean(steady_h_abs) * 100.0), 3),
                "h_pass_rate": round(sample_pass_h, 1),
                "vy_rms_mps": round(float(np.sqrt(np.mean(steady_vy_sq))), 4),
                "vy_pass_rate": round(sample_pass_vy, 1),
                "wz_rms_radps": round(float(np.sqrt(np.mean(steady_wz_sq))), 4),
                "wz_pass_rate": round(sample_pass_wz, 1),
                "joint_pass_rate": round(sample_pass_joint, 1),
                "env_pass_rate": round(env_pass_rate, 1),
                "pitch_rms_deg": round(float(np.mean(steady_pitch_abs)), 2),
                "episode_surv_rate": round(surv_rate, 2),
            }
            model_res[case_name] = c_dict
            print(f"[{m_label}] {case_name:24s} | H_act:{c_dict['steady_actual_h_m']:.4f} H_err:{c_dict['h_steady_err_cm']:4.2f}cm | H%:{c_dict['h_pass_rate']:4.1f}% Vy%:{c_dict['vy_pass_rate']:4.1f}% Wz%:{c_dict['wz_pass_rate']:4.1f}% | Jnt%:{c_dict['joint_pass_rate']:4.1f}% Env%:{c_dict['env_pass_rate']:4.1f}%")

        results_all[m_label] = model_res

    out_file = "/home/xy/IsaacTu/evaluation_assets/phase8b/results/phase8b_mid_checkpoints_eval.json"
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 8B mid-training evaluation saved to {out_file}!\n")

if __name__ == '__main__':
    run_phase8b_mid_evaluation()
