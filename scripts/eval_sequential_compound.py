"""
Sequential Continuous Compound Motion Evaluation Script.
Evaluates multi-stage continuous command sequences within a SINGLE EPISODE without active resets:
1. Stand & Balance (0.0s ~ 2.4s)
2. Forward Cruise (2.4s ~ 5.4s)
3. Forward Left Turn (5.4s ~ 8.4s)
4. Forward Right Turn (8.4s ~ 11.4s)
5. In-Motion Squat (11.4s ~ 14.4s)
6. Restore Height / Rise (14.4s ~ 17.4s)
7. Backward & Turn (17.4s ~ 20.4s)
8. Brake / Stop & Balance (20.4s ~ 23.4s)

Measures:
- Attitude stability (pitch/roll RMS & peak)
- Velocity & height tracking (Joint%, Env%, H%, Vy%, Wz%, signed H err)
- Response time & overshoot
- Knee torque, wheel torque, wheel saturation rate (6.95 Nm)
- Action smoothness (action rate delta)
- Continuous sequence survival & task completion
"""

import isaacgym
import os
import json
import numpy as np
import torch
from isaacgym import gymtorch

from legged_gym.envs import *
import sys
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def parse_custom_args():
    model_path = None
    output_path = None
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

def run_sequential_evaluation():
    custom_model_path, custom_out_path = parse_custom_args()
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False
    env.max_episode_length = 2000

    if custom_model_path is not None:
        models_to_test = {os.path.basename(custom_model_path).replace('.pt', ''): custom_model_path}
    else:
        models_to_test = {
            "champ_v4_m10": "/home/xy/IsaacTu/evaluation_assets/phase8b/models/phase8b_m10.pt",
            "phase9b_m13":  "/home/xy/IsaacTu/evaluation_assets/phase9b/models/phase9b_m13.pt",
            "phase10_m20":  "/home/xy/IsaacTu/evaluation_assets/phase10/models/phase10_m20.pt",
            "phase11_m20":  "/home/xy/IsaacTu/evaluation_assets/phase11/models/phase11_m20.pt",
        }

    # Define the 8 sequence stages
    # (stage_name, duration_steps, target_vy, target_wz, target_h_ref)
    sequence_stages = [
        ("1_stand_balance",     120,  0.00,  0.00, 0.39774), # 0.0s ~ 2.4s
        ("2_fwd_cruise",        150,  0.20,  0.00, 0.39774), # 2.4s ~ 5.4s
        ("3_fwd_left_turn",     150,  0.15,  0.25, 0.39774), # 5.4s ~ 8.4s
        ("4_fwd_right_turn",    150,  0.15, -0.25, 0.39774), # 8.4s ~ 11.4s
        ("5_inmotion_squat",    150,  0.10,  0.00, 0.36000), # 11.4s ~ 14.4s
        ("6_restore_rise",      150,  0.10,  0.00, 0.44000), # 14.4s ~ 17.4s
        ("7_bwd_turn",          150, -0.10, -0.20, 0.39774), # 17.4s ~ 20.4s
        ("8_stop_balance",      150,  0.00,  0.00, 0.39774), # 20.4s ~ 23.4s
    ]

    tol_h = 0.010
    tol_vy = 0.030
    tol_wz = 0.050
    rad_to_deg = 180.0 / np.pi
    dt = env.dt

    all_results = {}

    for model_name, ckpt_path in models_to_test.items():
        if not os.path.exists(ckpt_path):
            print(f"Warning: {ckpt_path} not found. Skipping {model_name}...")
            continue

        print(f"\n================ Evaluating Sequential Compound Motion: {model_name} ================")
        loaded = torch.load(ckpt_path, map_location=env.device)
        ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
        ac.load_state_dict(loaded['model_state_dict'])
        ac.eval()
        pol = ac.act_inference

        eval_seed = args.seed if args.seed is not None else 42
        torch.manual_seed(eval_seed)
        np.random.seed(eval_seed)
        env.reset()

        prev_actions = torch.zeros(args.num_envs, 6, device=env.device)
        done_env_flags = np.zeros(args.num_envs, dtype=bool)

        model_stages_results = {}
        total_steps_executed = 0

        global_wheel_sat_steps = np.zeros(args.num_envs)
        global_action_rate_list = []

        for stage_idx, (st_name, st_steps, cmd_vy, cmd_wz, target_h) in enumerate(sequence_stages):
            h_ref = target_h
            delta_h = -0.005 * float(np.clip((0.39774 - h_ref) / (0.39774 - 0.360), 0.0, 1.0))
            h_ctrl = h_ref + delta_h

            st_h_err = []
            st_vy_err = []
            st_wz_err = []
            st_pitch_list = []
            st_roll_list = []
            st_h_signed = []
            st_knee_tau_list = []
            st_wheel_sat_steps = np.zeros(args.num_envs)
            st_act_rate_list = []

            steady_window_len = 75
            steady_start_step = st_steps - steady_window_len

            stage_joint_pass_mat = np.zeros((args.num_envs, st_steps), dtype=bool)
            stage_h_pass_mat = np.zeros((args.num_envs, st_steps), dtype=bool)
            stage_vy_pass_mat = np.zeros((args.num_envs, st_steps), dtype=bool)
            stage_wz_pass_mat = np.zeros((args.num_envs, st_steps), dtype=bool)

            for step_in_stage in range(st_steps):
                env.commands[:, 0] = 0.0
                env.commands[:, 1] = cmd_vy
                env.commands[:, 2] = cmd_wz
                env.height_cmd_buf[:] = h_ctrl

                obs = env.get_observations()
                with torch.no_grad():
                    actions = pol(obs)

                obs, _, _, dones, _ = env.step(actions)
                d_arr = dones.cpu().numpy()
                done_env_flags |= d_arr

                act_rate = torch.norm(actions - prev_actions, dim=-1).cpu().numpy()
                prev_actions = actions.clone()
                st_act_rate_list.append(np.mean(act_rate))
                global_action_rate_list.append(np.mean(act_rate))

                tau_wheels = torch.abs(env.torques[:, env.wheel_indices]).cpu().numpy()
                is_sat = np.any(tau_wheels >= 6.95, axis=1)
                st_wheel_sat_steps += is_sat.astype(float)
                global_wheel_sat_steps += is_sat.astype(float)

                tau_knees = torch.abs(env.torques[:, [1, 4]]).cpu().numpy()
                st_knee_tau_list.append(np.mean(tau_knees))

                base_h = env.root_states[:, 2].cpu().numpy()
                base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                base_wz = env.base_ang_vel[:, 2].cpu().numpy()
                gx = env.projected_gravity[:, 0].cpu().numpy()
                gy = env.projected_gravity[:, 1].cpu().numpy()
                pitch = np.arcsin(np.clip(gy, -1.0, 1.0)) * rad_to_deg
                roll = np.arcsin(np.clip(gx, -1.0, 1.0)) * rad_to_deg

                err_h = np.abs(base_h - h_ref)
                err_vy = np.abs(base_vy - cmd_vy)
                err_wz = np.abs(base_wz - cmd_wz)

                stage_h_pass_mat[:, step_in_stage] = (err_h <= tol_h)
                stage_vy_pass_mat[:, step_in_stage] = (err_vy <= tol_vy)
                stage_wz_pass_mat[:, step_in_stage] = (err_wz <= tol_wz)
                stage_joint_pass_mat[:, step_in_stage] = (err_h <= tol_h) & (err_vy <= tol_vy) & (err_wz <= tol_wz)

                if step_in_stage >= steady_start_step:
                    st_h_signed.append(base_h - h_ref)
                    st_pitch_list.append(pitch)
                    st_roll_list.append(roll)

            total_steps_executed += st_steps

            # Compute Stage Metrics in steady window
            steady_slice = slice(steady_start_step, st_steps)
            sample_joint_pass = float(np.mean(stage_joint_pass_mat[:, steady_slice]) * 100.0)
            sample_h_pass = float(np.mean(stage_h_pass_mat[:, steady_slice]) * 100.0)
            sample_vy_pass = float(np.mean(stage_vy_pass_mat[:, steady_slice]) * 100.0)
            sample_wz_pass = float(np.mean(stage_wz_pass_mat[:, steady_slice]) * 100.0)

            env_pass_count = 0
            settle_times = []
            for e in range(args.num_envs):
                if not done_env_flags[e] and np.mean(stage_joint_pass_mat[e, steady_slice]) >= 0.80:
                    env_pass_count += 1
                pass_idx = np.where(stage_joint_pass_mat[e, :])[0]
                if len(pass_idx) > 0:
                    settle_times.append((pass_idx[0] + 1) * dt)
                else:
                    settle_times.append(st_steps * dt)

            env_pass_rate = float(env_pass_count / args.num_envs * 100.0)
            settle_p50 = float(np.percentile(settle_times, 50))
            surv_rate = float(np.mean(~done_env_flags) * 100.0)
            st_sat_rate = float(np.mean(st_wheel_sat_steps / st_steps) * 100.0)

            # Overshoot computation
            if "squat" in st_name:
                # Under-squat or over-squat: min height reached
                min_h = float(np.min(base_h))
                overshoot_cm = float((target_h - min_h) * 100.0)
            elif "rise" in st_name:
                max_h = float(np.max(base_h))
                overshoot_cm = float((max_h - target_h) * 100.0)
            else:
                overshoot_cm = float(np.max(err_h) * 100.0)

            stage_info = {
                "cmd_vy": cmd_vy,
                "cmd_wz": cmd_wz,
                "target_h": target_h,
                "joint_pass_rate": sample_joint_pass,
                "env_pass_rate": env_pass_rate,
                "h_pass_rate": sample_h_pass,
                "vy_pass_rate": sample_vy_pass,
                "wz_pass_rate": sample_wz_pass,
                "h_signed_err_cm": float(np.mean(st_h_signed) * 100.0),
                "pitch_rms_deg": float(np.sqrt(np.mean(np.array(st_pitch_list)**2))),
                "roll_rms_deg": float(np.sqrt(np.mean(np.array(st_roll_list)**2))),
                "pitch_peak_deg": float(np.max(np.abs(st_pitch_list))),
                "roll_peak_deg": float(np.max(np.abs(st_roll_list))),
                "settle_time_p50_s": settle_p50,
                "overshoot_cm": overshoot_cm,
                "knee_tau_mean_Nm": float(np.mean(st_knee_tau_list)),
                "wheel_sat_rate": st_sat_rate,
                "action_rate_rms": float(np.mean(st_act_rate_list)),
                "survival_rate": surv_rate,
            }
            model_stages_results[st_name] = stage_info
            print(f"  Stage {st_name:<18}: Joint%={sample_joint_pass:>5.1f}%, Env%={env_pass_rate:>5.1f}%, Herr={stage_info['h_signed_err_cm']:>+6.2f}cm, PitchRMS={stage_info['pitch_rms_deg']:>4.1f}°, Settle={settle_p50:>4.2f}s, Surv%={surv_rate:>5.1f}%")

        # Global summary for model across entire sequence
        mean_seq_joint = float(np.mean([s["joint_pass_rate"] for s in model_stages_results.values()]))
        mean_seq_env = float(np.mean([s["env_pass_rate"] for s in model_stages_results.values()]))
        final_surv = float(np.mean(~done_env_flags) * 100.0)
        overall_sat = float(np.mean(global_wheel_sat_steps / total_steps_executed) * 100.0)
        overall_smooth = float(np.mean(global_action_rate_list))
        final_completed = float(np.sum((~done_env_flags) & (stage_joint_pass_mat[:, -1])) / args.num_envs * 100.0)

        model_summary = {
            "mean_sequence_joint_rate": mean_seq_joint,
            "mean_sequence_env_rate": mean_seq_env,
            "final_episode_survival_rate": final_surv,
            "sequence_task_completion_rate": final_completed,
            "overall_wheel_sat_rate": overall_sat,
            "overall_action_rate_rms": overall_smooth,
        }
        model_stages_results["_summary"] = model_summary
        print(f"\n--- {model_name} Sequence Summary: Mean Joint%={mean_seq_joint:.2f}%, Mean Env%={mean_seq_env:.2f}%, Survival={final_surv:.1f}%, ActionRate={overall_smooth:.4f} ---\n")

        all_results[model_name] = model_stages_results

    out_path = custom_out_path if custom_out_path is not None else "evaluation_assets/phase12/sequential_compound_motion_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Sequential continuous compound evaluation completed! Saved to {out_path}.")

if __name__ == '__main__':
    run_sequential_evaluation()
