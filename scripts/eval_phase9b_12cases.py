"""
Phase 9B 12-Scenario Comparative Evaluation Script (4 Models).
Evaluates:
A: Champion v4 m10
B: Phase 8C m13 (Control: 3 iter without sym loss)
C: Phase 9A m13 (Exp: lambda=0.01)
D: Phase 9B m13 (Exp: lambda=0.10)

Across the 12 key benchmark scenarios:
- G1 Squats: g1_fwd_left_squat, g1_bwd_left_squat, g1_fwd_right_squat, g1_bwd_right_squat
- G4 Squats: g4_exp_fwd_left_squat, g4_exp_bwd_left_squat, g4_exp_fwd_right_squat, g4_exp_bwd_right_squat
- Regressions & Rise: g3_inplace_squat, g3_cruise_squat, g3_nominal_compound, g2_bwd_right_rise
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

obs_indices = torch.tensor([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    15, 16, 17, 12, 13, 14,
    21, 22, 23, 18, 19, 20,
    27, 28, 29, 24, 25, 26,
    30
], dtype=torch.long)
obs_signs = torch.tensor([
    -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0
], dtype=torch.float)
act_indices = torch.tensor([3, 4, 5, 0, 1, 2], dtype=torch.long)

def S_obs(obs):
    device = obs.device
    return obs[..., obs_indices.to(device)] * obs_signs.to(device)

def S_act(act):
    device = act.device
    return act[..., act_indices.to(device)]

def run_evaluation():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    models_to_test = {
        "champ_v4_m10": "/home/xy/IsaacTu/evaluation_assets/phase8b/models/phase8b_m10.pt",
        "phase8c_m13":  "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct12_15-41-44_phase8c_curriculum_ext_resume_15iter/model_13.pt",
        "phase9a_m13":  "/home/xy/IsaacTu/evaluation_assets/phase9a/models/phase9a_m13.pt",
        "phase9b_m13":  "/home/xy/IsaacTu/evaluation_assets/phase9b/models/phase9b_m13.pt",
    }

    scenarios = [
        # G1 Squats
        ("g1_fwd_left_squat",      0.10,  0.20, 0.36),
        ("g1_bwd_left_squat",     -0.10,  0.20, 0.36),
        ("g1_fwd_right_squat",     0.10, -0.20, 0.36),
        ("g1_bwd_right_squat",    -0.10, -0.20, 0.36),
        # G4 Squats
        ("g4_exp_fwd_left_squat",  0.16,  0.30, 0.36),
        ("g4_exp_bwd_left_squat", -0.16,  0.30, 0.36),
        ("g4_exp_fwd_right_squat", 0.16, -0.30, 0.36),
        ("g4_exp_bwd_right_squat",-0.16, -0.30, 0.36),
        # Regressions & Rise
        ("g3_inplace_squat",       0.00,  0.00, 0.36),
        ("g3_cruise_squat",        0.15,  0.00, 0.36),
        ("g3_nominal_compound",    0.10,  0.20, 0.39774),
        ("g2_bwd_right_rise",     -0.10, -0.20, 0.44),
    ]

    tol_h = 0.010
    tol_vy = 0.030
    tol_wz = 0.050
    rad_to_deg = 180.0 / np.pi
    total_steps = 300
    step_cmd_start = 100
    steady_start = 200

    all_results = {}

    for model_name, ckpt_path in models_to_test.items():
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint {ckpt_path} not found! Skipping {model_name}...")
            continue
        print(f"\n================ Evaluating Model: {model_name} ================")
        loaded = torch.load(ckpt_path, map_location=env.device)
        ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
        ac.load_state_dict(loaded['model_state_dict'])
        ac.eval()
        actor = ac.actor

        model_results = {}

        for case_name, cmd_vy, cmd_wz, target_h in scenarios:
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
            steady_h_signed = []
            steady_vy = []
            steady_wz = []
            steady_pitch = []
            steady_roll = []
            steady_l_sym = []
            steady_err_q_Lknee = []
            steady_err_q_Rknee = []
            wheel_sat_steps = np.zeros(args.num_envs)

            for step in range(total_steps):
                curr_h_ctrl = 0.39774 if step < step_cmd_start else h_ctrl
                env.commands[:, 0] = 0.0
                env.commands[:, 1] = cmd_vy
                env.commands[:, 2] = cmd_wz
                env.height_cmd_buf[:] = curr_h_ctrl

                obs = env.get_observations()
                with torch.no_grad():
                    mu_orig = actor(obs)
                    obs_mirr = S_obs(obs)
                    mu_mirr = actor(obs_mirr)
                    mu_mirr_expected = S_act(mu_orig)
                    diff_sq = torch.square(mu_mirr - mu_mirr_expected)
                    l_sym_step = torch.mean(torch.sum(diff_sq, dim=-1) / 6.0)

                    actions = mu_orig

                obs, _, _, dones, _ = env.step(actions)

                d_arr = dones.cpu().numpy()
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

                    q1_ik, q2_ik = env.compute_leg_ik(env.height_cmd_buf)
                    q2_ik_val = q2_ik.cpu().numpy()
                    q_Lknee = env.dof_pos[:, env.left_knee_idx].cpu().numpy()
                    q_Rknee = env.dof_pos[:, env.right_knee_idx].cpu().numpy()

                    steady_actual_h.append(base_h)
                    steady_h_signed.append(base_h - h_ref)
                    steady_vy.append(base_vy)
                    steady_wz.append(base_wz)
                    steady_pitch.append(pitch)
                    steady_roll.append(roll)
                    steady_l_sym.append(l_sym_step.item())
                    steady_err_q_Lknee.append(q_Lknee - q2_ik_val)
                    steady_err_q_Rknee.append(q_Rknee - q2_ik_val)

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
            surv_rate = float(np.mean(~done_env_flags) * 100.0)
            sat_rate = float(np.mean(wheel_sat_steps / total_steps) * 100.0)

            model_results[case_name] = {
                "h_ref": h_ref,
                "h_ctrl": h_ctrl,
                "steady_h_mean": float(np.mean(steady_actual_h)),
                "h_signed_err_cm": float(np.mean(steady_h_signed) * 100.0),
                "h_pass_rate": sample_pass_h,
                "vy_pass_rate": sample_pass_vy,
                "wz_pass_rate": sample_pass_wz,
                "joint_pass_rate": sample_pass_joint,
                "env_pass_rate": env_pass_rate,
                "episode_surv_rate": surv_rate,
                "wheel_sat_rate": sat_rate,
                "l_sym_mean": float(np.mean(steady_l_sym)),
                "err_q_Lknee_rad": float(np.mean(steady_err_q_Lknee)),
                "err_q_Rknee_rad": float(np.mean(steady_err_q_Rknee)),
                "pitch_mean_deg": float(np.mean(steady_pitch)),
                "roll_mean_deg": float(np.mean(steady_roll)),
            }
            print(f"  {case_name:<24}: Joint%={sample_pass_joint:>5.1f}%, H_err={np.mean(steady_h_signed)*100:>+6.2f}cm, L_sym={np.mean(steady_l_sym):.5f}")

        all_results[model_name] = model_results

    out_path = "evaluation_assets/phase9b/results/phase9b_12cases_eval.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n12-scenario evaluation completed! Results saved to {out_path}.")

if __name__ == '__main__':
    run_evaluation()
