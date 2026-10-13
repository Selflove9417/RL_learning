"""
Phase 8F Symmetry & Bilateral Diagnostics Script.
Evaluates Champion v4 (Phase 8B m10) on the 4 mirrored scenario pairs (8 cases total)
to extract frame-level and steady-state detailed symmetry metrics:
1. Signed height error relative to h_ref.
2. Actual yaw rate and error.
3. Left and right wheel output torques, and differential torque tau_d.
4. Left and right hip/knee joint positions and tracking errors relative to IK targets.
5. Signed pitch and roll angles.
6. PPO action residuals (filtered_actions) across joints.
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

def run_symmetry_diagnostics():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    ckpt_path = "/home/xy/IsaacTu/evaluation_assets/phase8b/models/phase8b_m10.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found!")

    loaded = torch.load(ckpt_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    pol = ac.act_inference

    pairs = [
        ("g4_exp_fwd_left_squat",   "g4_exp_fwd_right_squat",   0.16,  0.30, 0.36),
        ("g4_exp_bwd_left_squat",   "g4_exp_bwd_right_squat",  -0.16,  0.30, 0.36),
        ("g1_fwd_left_squat",       "g1_fwd_right_squat",       0.10,  0.20, 0.36),
        ("g1_bwd_left_squat",       "g1_bwd_right_squat",      -0.10,  0.20, 0.36),
    ]

    total_steps = 300
    step_cmd_start = 100
    steady_start = 200
    rad_to_deg = 180.0 / np.pi

    results = {}

    for left_name, right_name, cmd_vy, wz_mag, h_ref in pairs:
        for case_name, cmd_wz in [(left_name, wz_mag), (right_name, -wz_mag)]:
            print(f"--> Evaluating {case_name} (vy={cmd_vy}, wz={cmd_wz}, h_ref={h_ref})...")

            delta_h = -0.005 * float(np.clip((0.39774 - h_ref) / (0.39774 - 0.360), 0.0, 1.0))
            h_ctrl = h_ref + delta_h

            torch.manual_seed(42)
            np.random.seed(42)
            env.reset()

            # Metric collectors for steady state
            steady_h = []
            steady_h_signed_err = []
            steady_vy = []
            steady_wz = []
            steady_pitch = []
            steady_roll = []

            # Torques
            steady_tau_L = []
            steady_tau_R = []
            steady_tau_d = []

            # Leg joint angles
            steady_q_Lhip = []
            steady_q_Rhip = []
            steady_q_Lknee = []
            steady_q_Rknee = []

            # IK targets
            steady_q1_ik = []
            steady_q2_ik = []

            # PPO filtered action residuals
            steady_act_Lhip = []
            steady_act_Rhip = []
            steady_act_Lknee = []
            steady_act_Rknee = []
            steady_act_Lwheel = []
            steady_act_Rwheel = []

            # Pass rates
            h_pass_count = 0
            vy_pass_count = 0
            wz_pass_count = 0
            joint_pass_count = 0
            total_steady_samples = 0

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

                if step >= steady_start:
                    base_h = env.root_states[:, 2].cpu().numpy()
                    base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                    base_wz = env.base_ang_vel[:, 2].cpu().numpy()

                    # Projected gravity for pitch & roll
                    gx = env.projected_gravity[:, 0].cpu().numpy()
                    gy = env.projected_gravity[:, 1].cpu().numpy()
                    pitch = np.arcsin(np.clip(gy, -1.0, 1.0)) * rad_to_deg
                    roll = np.arcsin(np.clip(gx, -1.0, 1.0)) * rad_to_deg

                    # Torques
                    tau_L = env.torques[:, env.left_wheel_idx].cpu().numpy()
                    tau_R = env.torques[:, env.right_wheel_idx].cpu().numpy()
                    tau_d = 0.5 * (tau_R - tau_L)

                    # DOF positions
                    q_Lhip = env.dof_pos[:, env.left_hip_idx].cpu().numpy()
                    q_Rhip = env.dof_pos[:, env.right_hip_idx].cpu().numpy()
                    q_Lknee = env.dof_pos[:, env.left_knee_idx].cpu().numpy()
                    q_Rknee = env.dof_pos[:, env.right_knee_idx].cpu().numpy()

                    # IK targets computed in env
                    q1_ik, q2_ik = env.compute_leg_ik(env.height_cmd_buf)
                    q1_ik = q1_ik.cpu().numpy()
                    q2_ik = q2_ik.cpu().numpy()

                    # Filtered actions (PPO policy output residuals)
                    act = env.filtered_actions.cpu().numpy()

                    steady_h.append(base_h)
                    steady_h_signed_err.append(base_h - h_ref)
                    steady_vy.append(base_vy)
                    steady_wz.append(base_wz)
                    steady_pitch.append(pitch)
                    steady_roll.append(roll)

                    steady_tau_L.append(tau_L)
                    steady_tau_R.append(tau_R)
                    steady_tau_d.append(tau_d)

                    steady_q_Lhip.append(q_Lhip)
                    steady_q_Rhip.append(q_Rhip)
                    steady_q_Lknee.append(q_Lknee)
                    steady_q_Rknee.append(q_Rknee)

                    steady_q1_ik.append(q1_ik)
                    steady_q2_ik.append(q2_ik)

                    steady_act_Lhip.append(act[:, env.left_hip_idx])
                    steady_act_Rhip.append(act[:, env.right_hip_idx])
                    steady_act_Lknee.append(act[:, env.left_knee_idx])
                    steady_act_Rknee.append(act[:, env.right_knee_idx])
                    steady_act_Lwheel.append(act[:, env.left_wheel_idx])
                    steady_act_Rwheel.append(act[:, env.right_wheel_idx])

                    # Step pass rates
                    h_pass = (np.abs(base_h - h_ref) <= 0.010)
                    vy_pass = (np.abs(base_vy - cmd_vy) <= 0.030)
                    wz_pass = (np.abs(base_wz - cmd_wz) <= 0.050)
                    joint_pass = h_pass & vy_pass & wz_pass

                    h_pass_count += np.sum(h_pass)
                    vy_pass_count += np.sum(vy_pass)
                    wz_pass_count += np.sum(wz_pass)
                    joint_pass_count += np.sum(joint_pass)
                    total_steady_samples += len(base_h)

            # Summaries
            results[case_name] = {
                "cmd_vy": cmd_vy,
                "cmd_wz": cmd_wz,
                "h_ref": h_ref,
                "h_ctrl": h_ctrl,
                "steady_h_mean": float(np.mean(steady_h)),
                "h_signed_err_cm": float(np.mean(steady_h_signed_err) * 100.0),
                "h_err_abs_cm": float(np.mean(np.abs(steady_h_signed_err)) * 100.0),
                "vy_mean": float(np.mean(steady_vy)),
                "vy_err_mean": float(np.mean(np.array(steady_vy) - cmd_vy)),
                "wz_mean": float(np.mean(steady_wz)),
                "wz_err_mean": float(np.mean(np.array(steady_wz) - cmd_wz)),
                "pitch_signed_deg": float(np.mean(steady_pitch)),
                "pitch_abs_deg": float(np.mean(np.abs(steady_pitch))),
                "roll_signed_deg": float(np.mean(steady_roll)),
                "roll_abs_deg": float(np.mean(np.abs(steady_roll))),
                # Wheel torques
                "tau_wheel_L_mean": float(np.mean(steady_tau_L)),
                "tau_wheel_R_mean": float(np.mean(steady_tau_R)),
                "tau_d_mean": float(np.mean(steady_tau_d)),
                # Leg joints vs IK
                "q_Lhip_mean": float(np.mean(steady_q_Lhip)),
                "q_Rhip_mean": float(np.mean(steady_q_Rhip)),
                "q_Lknee_mean": float(np.mean(steady_q_Lknee)),
                "q_Rknee_mean": float(np.mean(steady_q_Rknee)),
                "q1_ik_mean": float(np.mean(steady_q1_ik)),
                "q2_ik_mean": float(np.mean(steady_q2_ik)),
                "err_q_Lhip_rad": float(np.mean(np.array(steady_q_Lhip) - np.array(steady_q1_ik))),
                "err_q_Rhip_rad": float(np.mean(np.array(steady_q_Rhip) - np.array(steady_q1_ik))),
                "err_q_Lknee_rad": float(np.mean(np.array(steady_q_Lknee) - np.array(steady_q2_ik))),
                "err_q_Rknee_rad": float(np.mean(np.array(steady_q_Rknee) - np.array(steady_q2_ik))),
                # PPO action residuals
                "act_residual_Lhip": float(np.mean(steady_act_Lhip)),
                "act_residual_Rhip": float(np.mean(steady_act_Rhip)),
                "act_residual_Lknee": float(np.mean(steady_act_Lknee)),
                "act_residual_Rknee": float(np.mean(steady_act_Rknee)),
                "act_residual_Lwheel": float(np.mean(steady_act_Lwheel)),
                "act_residual_Rwheel": float(np.mean(steady_act_Rwheel)),
                # Pass rates (%)
                "h_pass_rate": float(h_pass_count / total_steady_samples * 100.0),
                "vy_pass_rate": float(vy_pass_count / total_steady_samples * 100.0),
                "wz_pass_rate": float(wz_pass_count / total_steady_samples * 100.0),
                "joint_pass_rate": float(joint_pass_count / total_steady_samples * 100.0),
            }

    out_file = "evaluation_assets/phase8f/results/symmetry_diagnostics_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll 8 diagnostics completed successfully! Results saved to {out_file}.")

if __name__ == '__main__':
    run_symmetry_diagnostics()
