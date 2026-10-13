"""
Phase 7G: Evaluation Consistency Audit and Key Degradation Diagnostic.
Tests:
1. Zero-compensation consistency test (Path A: disabled, Path B: forced delta_h=0).
   Checks frame-by-frame diff across 300 steps on:
   - g3_nominal_cruise
   - g4_exp_fwd_left_rise
   - g2_bwd_left_rise
   Compares Phase 7C command vs Phase 7F command.
2. G4 Key Degradation Audit:
   - g4_exp_bwd_right_squat under 7C command (-0.16, -0.30, 0.36) [uncomp vs -5mm comp]
   - g4_exp_bwd_right_squat under 7F command (-0.20, -0.20, 0.36) [uncomp vs -5mm comp]
"""

import isaacgym
import torch
import numpy as np
import os
import json
from isaacgym import gymtorch
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def run_phase7g_audit():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    # Rigid body states tensor
    body_states_tensor = env.gym.acquire_rigid_body_state_tensor(env.sim)
    bs = gymtorch.wrap_tensor(body_states_tensor).view(args.num_envs, env.num_bodies, 13)

    model_path = "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct12_11-16-16_phase7c_deadzone005_beta020_iter1/model_1.pt"
    loaded = torch.load(model_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    pol = ac.act_inference

    # Constants & tolerances
    tol_h = 0.010
    tol_vy = 0.030
    tol_wz = 0.050
    hold_steps_req = 25
    rad_to_deg = 180.0 / np.pi
    r_vec = np.array([0.0, 0.0, -0.070]) # 7C exact radius
    total_steps = 300
    step_cmd_start = 100
    steady_start = 200

    def evaluate_case(cmd_vy, cmd_wz, target_h, delta_h, seed=42, record_frames=False):
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.reset()

        h_ctrl = target_h + delta_h

        h_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
        vy_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
        wz_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
        done_env_flags = np.zeros(args.num_envs, dtype=bool)

        steady_actual_h = []
        steady_h_signed = []
        steady_h_abs = []
        steady_vy_sq = []
        steady_wz_sq = []
        steady_pitch_abs = []
        steady_pitch_max = []
        wheel_sat_steps = np.zeros(args.num_envs)
        knee_tau_list = []

        frame_data = [] if record_frames else None
        step_99_state = None

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

            # Wheel saturation
            tau_wheels = torch.abs(env.torques[:, env.wheel_indices]).cpu().numpy()
            is_sat = np.any(tau_wheels >= 6.95, axis=1)
            wheel_sat_steps += is_sat.astype(float)

            base_h = env.root_states[:, 2].cpu().numpy()
            base_vy = env.base_lin_vel[:, 1].cpu().numpy()
            base_wz = env.base_ang_vel[:, 2].cpu().numpy()

            if step == 99:
                step_99_state = {
                    "base_h_mean": float(np.mean(base_h)),
                    "base_vy_mean": float(np.mean(base_vy)),
                    "base_wz_mean": float(np.mean(base_wz)),
                    "actions_mean": float(torch.mean(actions).cpu().item())
                }

            if record_frames:
                frame_data.append({
                    "step": step,
                    "h_mean": float(np.mean(base_h)),
                    "vy_mean": float(np.mean(base_vy)),
                    "wz_mean": float(np.mean(base_wz)),
                    "act_mean": float(torch.mean(actions).cpu().item()),
                    "act_std": float(torch.std(actions).cpu().item()),
                })

            if step >= step_cmd_start:
                idx = step - step_cmd_start
                h_err_traj[:, idx] = np.abs(base_h - target_h)
                vy_err_traj[:, idx] = np.abs(base_vy - cmd_vy)
                wz_err_traj[:, idx] = np.abs(base_wz - cmd_wz)

            if step >= steady_start:
                h_signed_err = base_h - target_h
                h_err = np.abs(h_signed_err)
                steady_actual_h.append(base_h)
                steady_h_signed.append(h_signed_err)
                steady_h_abs.append(h_err)
                steady_vy_sq.append(np.square(base_vy - cmd_vy))
                steady_wz_sq.append(np.square(base_wz - cmd_wz))

                pitch = (torch.asin(torch.clamp(env.projected_gravity[:, 1], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
                steady_pitch_abs.append(np.abs(pitch))
                steady_pitch_max.append(np.max(np.abs(pitch)))

                tau_knee_L = np.abs(env.torques[:, env.left_knee_idx].cpu().numpy())
                tau_knee_R = np.abs(env.torques[:, env.right_knee_idx].cpu().numpy())
                knee_tau_list.append(0.5 * (tau_knee_L + tau_knee_R))

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

        res = {
            "h_ref": target_h,
            "h_ctrl": h_ctrl,
            "delta_h": delta_h,
            "steady_actual_h": float(np.mean(steady_actual_h)),
            "h_steady_err_cm": float(np.mean(steady_h_abs) * 100.0),
            "h_pass_rate": sample_pass_h,
            "vy_rms_mps": float(np.sqrt(np.mean(steady_vy_sq))),
            "vy_pass_rate": sample_pass_vy,
            "wz_rms_radps": float(np.sqrt(np.mean(steady_wz_sq))),
            "wz_pass_rate": sample_pass_wz,
            "joint_pass_rate": sample_pass_joint,
            "env_pass_rate": env_pass_rate,
            "pitch_rms_deg": float(np.mean(steady_pitch_abs)),
            "pitch_peak_deg": float(np.max(steady_pitch_max)),
            "knee_tau_mean_Nm": float(np.mean(knee_tau_list)),
            "wheel_sat_rate": float(np.mean(wheel_sat_steps) / total_steps * 100.0),
            "episode_surv_rate": surv_rate,
            "step_99_state": step_99_state,
        }
        return res, frame_data

    audit_results = {}

    print("\n" + "=" * 90)
    print("SECTION II: ZERO-COMPENSATION CONSISTENCY TEST")
    print("=" * 90)

    # 1. g3_nominal_cruise under 7C command (0.20, 0.00, 0.39774) vs 7F command (0.10, 0.00, 0.39774)
    print("\n--- Testing g3_nominal_cruise ---")
    res_7c_cmd_A, frames_7c_A = evaluate_case(0.20, 0.00, 0.39774, delta_h=0.0, record_frames=True)
    res_7c_cmd_B, frames_7c_B = evaluate_case(0.20, 0.00, 0.39774, delta_h=0.0, record_frames=True) # forced 0
    res_7f_cmd, frames_7f = evaluate_case(0.10, 0.00, 0.39774, delta_h=0.0, record_frames=True)

    print(f"7C Command (vy=0.20): Path A (Uncomp)    -> J%={res_7c_cmd_A['joint_pass_rate']:.1f}%, H%={res_7c_cmd_A['h_pass_rate']:.1f}%, Vy%={res_7c_cmd_A['vy_pass_rate']:.1f}%, Wz%={res_7c_cmd_A['wz_pass_rate']:.1f}%, Env%={res_7c_cmd_A['env_pass_rate']:.1f}%")
    print(f"7C Command (vy=0.20): Path B (Forced 0)  -> J%={res_7c_cmd_B['joint_pass_rate']:.1f}%, H%={res_7c_cmd_B['h_pass_rate']:.1f}%, Vy%={res_7c_cmd_B['vy_pass_rate']:.1f}%, Wz%={res_7c_cmd_B['wz_pass_rate']:.1f}%, Env%={res_7c_cmd_B['env_pass_rate']:.1f}%")
    print(f"7F Command (vy=0.10): (Mismatched Cmd!)   -> J%={res_7f_cmd['joint_pass_rate']:.1f}%, H%={res_7f_cmd['h_pass_rate']:.1f}%, Vy%={res_7f_cmd['vy_pass_rate']:.1f}%, Wz%={res_7f_cmd['wz_pass_rate']:.1f}%, Env%={res_7f_cmd['env_pass_rate']:.1f}%")

    # Frame diff between Path A and Path B for 7C command
    max_act_diff = max(abs(fa['act_mean'] - fb['act_mean']) for fa, fb in zip(frames_7c_A, frames_7c_B))
    max_h_diff = max(abs(fa['h_mean'] - fb['h_mean']) for fa, fb in zip(frames_7c_A, frames_7c_B))
    print(f"Path A vs Path B frame-by-frame max action diff: {max_act_diff:.8f}, max height diff: {max_h_diff:.8f}")

    # 2. g4_exp_fwd_left_rise under 7C command (0.16, 0.30, 0.44) vs 7F command (0.20, 0.20, 0.44)
    print("\n--- Testing g4_exp_fwd_left_rise ---")
    res_rise_7c_A, _ = evaluate_case(0.16, 0.30, 0.44, delta_h=0.0)
    res_rise_7c_B, _ = evaluate_case(0.16, 0.30, 0.44, delta_h=0.0)
    res_rise_7f, _ = evaluate_case(0.20, 0.20, 0.44, delta_h=0.0)
    print(f"7C Command (vy=0.16, wz=0.30): Path A -> J%={res_rise_7c_A['joint_pass_rate']:.1f}%, H%={res_rise_7c_A['h_pass_rate']:.1f}%, Vy%={res_rise_7c_A['vy_pass_rate']:.1f}%, Wz%={res_rise_7c_A['wz_pass_rate']:.1f}%, Env%={res_rise_7c_A['env_pass_rate']:.1f}%")
    print(f"7C Command (vy=0.16, wz=0.30): Path B -> J%={res_rise_7c_B['joint_pass_rate']:.1f}%, H%={res_rise_7c_B['h_pass_rate']:.1f}%, Vy%={res_rise_7c_B['vy_pass_rate']:.1f}%, Wz%={res_rise_7c_B['wz_pass_rate']:.1f}%, Env%={res_rise_7c_B['env_pass_rate']:.1f}%")
    print(f"7F Command (vy=0.20, wz=0.20): Mismatch -> J%={res_rise_7f['joint_pass_rate']:.1f}%, H%={res_rise_7f['h_pass_rate']:.1f}%, Vy%={res_rise_7f['vy_pass_rate']:.1f}%, Wz%={res_rise_7f['wz_pass_rate']:.1f}%, Env%={res_rise_7f['env_pass_rate']:.1f}%")

    # 3. g2_bwd_left_rise (-0.10, 0.20, 0.44)
    print("\n--- Testing g2_bwd_left_rise ---")
    res_g2_A, _ = evaluate_case(-0.10, 0.20, 0.44, delta_h=0.0)
    res_g2_B, _ = evaluate_case(-0.10, 0.20, 0.44, delta_h=0.0)
    print(f"(-0.10, 0.20, 0.44): Path A -> J%={res_g2_A['joint_pass_rate']:.1f}%, H%={res_g2_A['h_pass_rate']:.1f}%, Vy%={res_g2_A['vy_pass_rate']:.1f}%, Wz%={res_g2_A['wz_pass_rate']:.1f}%, Env%={res_g2_A['env_pass_rate']:.1f}%")
    print(f"(-0.10, 0.20, 0.44): Path B -> J%={res_g2_B['joint_pass_rate']:.1f}%, H%={res_g2_B['h_pass_rate']:.1f}%, Vy%={res_g2_B['vy_pass_rate']:.1f}%, Wz%={res_g2_B['wz_pass_rate']:.1f}%, Env%={res_g2_B['env_pass_rate']:.1f}%")

    print("\n" + "=" * 90)
    print("SECTION III: G4 KEY DEGRADATION AUDIT (g4_exp_bwd_right_squat)")
    print("=" * 90)

    # 1. Under Phase 7C Protocol: (-0.16, -0.30, 0.36)
    print("\n[Protocol 7C Command: vy=-0.16, wz=-0.30, target_h=0.360]")
    res_g4_7c_uncomp, _ = evaluate_case(-0.16, -0.30, 0.36, delta_h=0.0)
    res_g4_7c_comp, _   = evaluate_case(-0.16, -0.30, 0.36, delta_h=-0.005)
    print(f"Uncompensated (h_cmd=0.360): J%={res_g4_7c_uncomp['joint_pass_rate']:.1f}%, Env%={res_g4_7c_uncomp['env_pass_rate']:.1f}%, H%={res_g4_7c_uncomp['h_pass_rate']:.1f}%, Vy%={res_g4_7c_uncomp['vy_pass_rate']:.1f}%, Wz%={res_g4_7c_uncomp['wz_pass_rate']:.1f}%")
    print(f"  Steady H={res_g4_7c_uncomp['steady_actual_h']:.4f}m, H_err={res_g4_7c_uncomp['h_steady_err_cm']:.2f}cm, Vy_rms={res_g4_7c_uncomp['vy_rms_mps']:.4f}, Wz_rms={res_g4_7c_uncomp['wz_rms_radps']:.4f}, Pitch={res_g4_7c_uncomp['pitch_rms_deg']:.2f}deg, WheelSat={res_g4_7c_uncomp['wheel_sat_rate']:.2f}%")
    print(f"-5mm Compensated (h_cmd=0.355): J%={res_g4_7c_comp['joint_pass_rate']:.1f}%, Env%={res_g4_7c_comp['env_pass_rate']:.1f}%, H%={res_g4_7c_comp['h_pass_rate']:.1f}%, Vy%={res_g4_7c_comp['vy_pass_rate']:.1f}%, Wz%={res_g4_7c_comp['wz_pass_rate']:.1f}%")
    print(f"  Steady H={res_g4_7c_comp['steady_actual_h']:.4f}m, H_err={res_g4_7c_comp['h_steady_err_cm']:.2f}cm, Vy_rms={res_g4_7c_comp['vy_rms_mps']:.4f}, Wz_rms={res_g4_7c_comp['wz_rms_radps']:.4f}, Pitch={res_g4_7c_comp['pitch_rms_deg']:.2f}deg, WheelSat={res_g4_7c_comp['wheel_sat_rate']:.2f}%")

    # 2. Under Phase 7F Mismatched Command: (-0.20, -0.20, 0.36)
    print("\n[Protocol 7F Mismatched Command: vy=-0.20, wz=-0.20, target_h=0.360]")
    res_g4_7f_uncomp, _ = evaluate_case(-0.20, -0.20, 0.36, delta_h=0.0)
    res_g4_7f_comp, _   = evaluate_case(-0.20, -0.20, 0.36, delta_h=-0.005)
    print(f"Uncompensated (h_cmd=0.360): J%={res_g4_7f_uncomp['joint_pass_rate']:.1f}%, Env%={res_g4_7f_uncomp['env_pass_rate']:.1f}%, H%={res_g4_7f_uncomp['h_pass_rate']:.1f}%, Vy%={res_g4_7f_uncomp['vy_pass_rate']:.1f}%, Wz%={res_g4_7f_uncomp['wz_pass_rate']:.1f}%")
    print(f"  Steady H={res_g4_7f_uncomp['steady_actual_h']:.4f}m, H_err={res_g4_7f_uncomp['h_steady_err_cm']:.2f}cm, Vy_rms={res_g4_7f_uncomp['vy_rms_mps']:.4f}, Wz_rms={res_g4_7f_uncomp['wz_rms_radps']:.4f}, Pitch={res_g4_7f_uncomp['pitch_rms_deg']:.2f}deg, WheelSat={res_g4_7f_uncomp['wheel_sat_rate']:.2f}%")
    print(f"-5mm Compensated (h_cmd=0.355): J%={res_g4_7f_comp['joint_pass_rate']:.1f}%, Env%={res_g4_7f_comp['env_pass_rate']:.1f}%, H%={res_g4_7f_comp['h_pass_rate']:.1f}%, Vy%={res_g4_7f_comp['vy_pass_rate']:.1f}%, Wz%={res_g4_7f_comp['wz_pass_rate']:.1f}%")
    print(f"  Steady H={res_g4_7f_comp['steady_actual_h']:.4f}m, H_err={res_g4_7f_comp['h_steady_err_cm']:.2f}cm, Vy_rms={res_g4_7f_comp['vy_rms_mps']:.4f}, Wz_rms={res_g4_7f_comp['wz_rms_radps']:.4f}, Pitch={res_g4_7f_comp['pitch_rms_deg']:.2f}deg, WheelSat={res_g4_7f_comp['wheel_sat_rate']:.2f}%")

    out_diag = {
        "g3_nominal_cruise": {
            "7c_cmd_PathA": res_7c_cmd_A,
            "7c_cmd_PathB": res_7c_cmd_B,
            "7f_cmd": res_7f_cmd,
        },
        "g4_exp_fwd_left_rise": {
            "7c_cmd_PathA": res_rise_7c_A,
            "7c_cmd_PathB": res_rise_7c_B,
            "7f_cmd": res_rise_7f,
        },
        "g2_bwd_left_rise": {
            "PathA": res_g2_A,
            "PathB": res_g2_B,
        },
        "g4_exp_bwd_right_squat": {
            "7c_cmd_uncomp": res_g4_7c_uncomp,
            "7c_cmd_comp": res_g4_7c_comp,
            "7f_cmd_uncomp": res_g4_7f_uncomp,
            "7f_cmd_comp": res_g4_7f_comp,
        }
    }

    out_dir = "/home/xy/IsaacTu/evaluation_assets/phase7g/diagnosis"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "phase7g_consistency_diagnosis.json")
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(out_diag, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 7G diagnosis results saved to {out_file}!\n")

if __name__ == '__main__':
    run_phase7g_audit()
