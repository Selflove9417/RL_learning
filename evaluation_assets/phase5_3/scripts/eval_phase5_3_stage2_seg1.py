import isaacgym
import torch
import os
import json
import glob
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def run_evaluation():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    # 查找最新训练的 stage2_seg1 目录
    stage2_dirs = sorted(glob.glob("/home/xy/IsaacTu/legged_gym/logs/myrobot/*turning_height_stage5_3_stage2_seg1"))
    if not stage2_dirs:
        raise RuntimeError("No stage2_seg1 training directory found!")
    latest_stage2_dir = stage2_dirs[-1]
    print(f"Target Stage 2 Seg 1 Run Dir: {latest_stage2_dir}")

    models_to_test = {
        "stage1_model25": "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_15-48-32_turning_height_stage5_3_formal/model_25.pt",
        "stage2_model10": os.path.join(latest_stage2_dir, "model_10.pt"),
        "stage2_model25": os.path.join(latest_stage2_dir, "model_25.pt"),
    }

    # 4 个测试套件
    scenarios = {
        # Suite 1: Stage 1 训练分布内复合工况 (vy=0.10, wz=0.20, H in [0.36, 0.44])
        "suite1_stage1_indist": [
            ("s1_fwd_left_squat",     0.10,  0.20, 0.36),
            ("s1_fwd_right_squat",    0.10, -0.20, 0.36),
            ("s1_fwd_straight_squat", 0.10,  0.00, 0.36),
            ("s1_bwd_left_squat",    -0.10,  0.20, 0.36),
            ("s1_bwd_right_squat",   -0.10, -0.20, 0.36),
            ("s1_fwd_left_rise",      0.10,  0.20, 0.44),
            ("s1_fwd_right_rise",     0.10, -0.20, 0.44),
            ("s1_bwd_left_rise",     -0.10,  0.20, 0.44),
            ("s1_bwd_right_rise",    -0.10, -0.20, 0.44),
        ],
        # Suite 2: Stage 2 扩展范围工况 (vy=0.16, wz=0.30, H in [0.36, 0.44])
        "suite2_stage2_expanded": [
            ("s2_exp_fwd_left_squat",   0.16,  0.30, 0.36),
            ("s2_exp_fwd_right_squat",  0.16, -0.30, 0.36),
            ("s2_exp_bwd_left_squat",  -0.16,  0.30, 0.36),
            ("s2_exp_bwd_right_squat", -0.16, -0.30, 0.36),
            ("s2_exp_fwd_left_rise",    0.16,  0.30, 0.44),
            ("s2_exp_fwd_right_rise",   0.16, -0.30, 0.44),
            ("s2_exp_bwd_left_rise",   -0.16,  0.30, 0.44),
            ("s2_exp_bwd_right_rise",  -0.16, -0.30, 0.44),
        ],
        # Suite 3: 极限高度泛化应力测试 (H=0.35m, H=0.45m)
        "suite3_generalization": [
            ("s3_ext_fwd_left_squat_035",   0.12,  0.25, 0.35),
            ("s3_ext_bwd_right_squat_035", -0.12, -0.25, 0.35),
            ("s3_ext_fwd_right_rise_045",   0.12, -0.25, 0.45),
            ("s3_ext_bwd_left_rise_045",   -0.12,  0.25, 0.45),
        ],
        # Suite 4: 历史技能防遗忘回归测试
        "suite4_regression": [
            ("s4_nominal_stand",         0.00,  0.00, 0.39774),
            ("s4_nominal_inplace_turn",  0.00,  0.35, 0.39774),
            ("s4_nominal_cruise_fwd",    0.20,  0.00, 0.39774),
        ]
    }

    rad_to_deg = 180.0 / np.pi
    R_wheel = 0.070
    d_track = 0.285 / 2.0
    total_steps = 300
    steady_start = 200 # 稳态统计区间 200~300 (4.0s ~ 6.0s)，剔除切换暂态

    full_results = {}

    for model_name, ckpt_path in models_to_test.items():
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint {ckpt_path} not found! Skipping...")
            continue
        print(f"\n================ Evaluating Model: {model_name} ================")
        loaded = torch.load(ckpt_path, map_location=env.device)
        ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
        ac.load_state_dict(loaded['model_state_dict'])
        ac.eval()
        pol = ac.act_inference

        model_eval_data = {}

        for suite_name, case_list in scenarios.items():
            suite_data = {}
            for name, cmd_vy, cmd_wz, target_h in case_list:
                torch.manual_seed(42)
                env.reset()

                h_err_steady, deadband_steady = [], []
                vy_err_steady, wz_err_steady = [], []
                pitch_steady = []
                knee_tau_max, knee_tau_mean = [], []
                fn_L_list, fn_R_list = [], []
                slip_L_list, slip_R_list = [], []
                dones_accum = 0

                for step in range(total_steps):
                    curr_h = 0.39774 if step < 100 else target_h
                    env.commands[:, 0] = 0.0
                    env.commands[:, 1] = cmd_vy
                    env.commands[:, 2] = cmd_wz
                    env.height_cmd_buf[:] = curr_h

                    obs = env.get_observations()
                    with torch.no_grad():
                        actions = pol(obs)
                    obs, _, _, dones, _ = env.step(actions)
                    dones_accum += dones.sum().item()

                    if step >= steady_start:
                        h_act = env.root_states[:, 2].cpu().numpy()
                        h_err = np.abs(h_act - target_h)
                        deadband = np.maximum(0.0, h_err - 0.015)
                        h_err_steady.append(h_err)
                        deadband_steady.append(deadband)

                        base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                        base_wz = env.base_ang_vel[:, 2].cpu().numpy()
                        vy_err_steady.append(np.square(base_vy - cmd_vy))
                        wz_err_steady.append(np.square(base_wz - cmd_wz))

                        pitch = (torch.asin(torch.clamp(env.projected_gravity[:, 1], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
                        pitch_steady.append(np.abs(pitch))

                        tau_knee_L = np.abs(env.torques[:, env.left_knee_idx].cpu().numpy())
                        tau_knee_R = np.abs(env.torques[:, env.right_knee_idx].cpu().numpy())
                        knee_tau_mean.append(0.5 * (tau_knee_L + tau_knee_R))
                        knee_tau_max.append(np.maximum(tau_knee_L, tau_knee_R))

                        fn_L = env.contact_forces[:, env.feet_indices[0], 2].cpu().numpy()
                        fn_R = env.contact_forces[:, env.feet_indices[1], 2].cpu().numpy()
                        fn_L_list.append(fn_L)
                        fn_R_list.append(fn_R)

                        w_vel_L = env.dof_vel[:, env.left_wheel_idx].cpu().numpy()
                        w_vel_R = env.dof_vel[:, env.right_wheel_idx].cpu().numpy()
                        v_wheel_L = -w_vel_L * R_wheel
                        v_wheel_R = w_vel_R * R_wheel
                        v_robot_L = base_vy - base_wz * d_track
                        v_robot_R = base_vy + base_wz * d_track
                        slip_L_list.append(np.abs(v_wheel_L - v_robot_L))
                        slip_R_list.append(np.abs(v_wheel_R - v_robot_R))

                mean_fn_L = float(np.mean(fn_L_list))
                mean_fn_R = float(np.mean(fn_R_list))
                asym_ratio = mean_fn_L / max(1e-3, mean_fn_R)

                case_metrics = {
                    "h_steady_err_cm": round(float(np.mean(h_err_steady) * 100.0), 3),
                    "deadband_pen_cm": round(float(np.mean(deadband_steady) * 100.0), 3),
                    "vy_rms_mps": round(float(np.sqrt(np.mean(vy_err_steady))), 4),
                    "wz_rms_radps": round(float(np.sqrt(np.mean(wz_err_steady))), 4),
                    "pitch_abs_deg": round(float(np.mean(pitch_steady)), 2),
                    "knee_tau_mean_Nm": round(float(np.mean(knee_tau_mean)), 2),
                    "knee_tau_max_Nm": round(float(np.max(knee_tau_max)), 2),
                    "wheel_slip_mean_mps": round(float(0.5 * (np.mean(slip_L_list) + np.mean(slip_R_list))), 4),
                    "wheel_slip_max_mps": round(float(max(np.max(slip_L_list), np.max(slip_R_list))), 4),
                    "fn_L_N": round(mean_fn_L, 2),
                    "fn_R_N": round(mean_fn_R, 2),
                    "fn_asym_ratio": round(asym_ratio, 3),
                    "falls": dones_accum,
                    "survival_rate": round(float((1.0 - dones_accum / (args.num_envs * total_steps)) * 100.0), 2)
                }
                suite_data[name] = case_metrics
                print(f"[{suite_name}] {name:28s} | H_err: {case_metrics['h_steady_err_cm']:5.2f}cm | Vy_rms: {case_metrics['vy_rms_mps']:6.4f} | Wz_rms: {case_metrics['wz_rms_radps']:6.4f} | KneeTauMax: {case_metrics['knee_tau_max_Nm']:5.1f}Nm | SlipMax: {case_metrics['wheel_slip_max_mps']:5.3f} | Surv: {case_metrics['survival_rate']}%")

            model_eval_data[suite_name] = suite_data
        full_results[model_name] = model_eval_data

    out_file = "/home/xy/.gemini/antigravity/brain/853aaed2-8860-4930-be3f-1bb608c17795/scratch/eval_stage2_seg1_results.json"
    with open(out_file, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nComprehensive Stage 2 Seg 1 evaluation saved to: {out_file}\n")

if __name__ == '__main__':
    run_evaluation()
