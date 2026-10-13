"""
Phase 7D: Detailed Mechanism Diagnosis for In-place Squat (g3_inplace_squat).
Compares Phase 6B Champion v2 vs Phase 7C Champion v3 under identical 32 envs and seed 42.

Key Diagnostic Analyses:
1. Exact Venn diagram / intersection decomposition of pass criteria (H, Vy, Wz).
2. Detailed distribution of Vy (mean, std, RMS, p50, p95, max, tail behavior).
3. Height trajectory breakdown: Transition Phase (steps 100-200) vs Steady-State Phase (steps 200-300).
4. Signed height bias analysis (percentage of z > 0.36 vs z < 0.36, signed mean/median bias).
5. Wheel kinematics and torques.
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

def run_phase7d_diagnosis():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    models = {
        "Phase6B_Champ_v2": "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct11_21-24-38_phase6b_pilot_deadzone010_beta020_iter1/model_1.pt",
        "Phase7C_Champ_v3": "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct12_11-16-16_phase7c_deadzone005_beta020_iter1/model_1.pt",
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

        torch.manual_seed(42)
        np.random.seed(42)
        env.reset()

        ts_base_h = np.zeros((total_steps, num_envs))
        ts_signed_h_err = np.zeros((total_steps, num_envs)) # z - target_h
        ts_abs_h_err = np.zeros((total_steps, num_envs))
        ts_vy = np.zeros((total_steps, num_envs))
        ts_wz = np.zeros((total_steps, num_envs))
        ts_pitch = np.zeros((total_steps, num_envs))
        ts_pitch_rate = np.zeros((total_steps, num_envs))

        ts_wheel_vel_L = np.zeros((total_steps, num_envs))
        ts_wheel_vel_R = np.zeros((total_steps, num_envs))
        ts_wheel_tau_L = np.zeros((total_steps, num_envs))
        ts_wheel_tau_R = np.zeros((total_steps, num_envs))

        done_flags = np.zeros(num_envs, dtype=bool)

        for step in range(total_steps):
            curr_h = 0.39774 if step < step_cmd_start else target_h
            env.commands[:, 0] = 0.0
            env.commands[:, 1] = v_target
            env.commands[:, 2] = w_target
            env.height_cmd_buf[:] = curr_h

            obs = env.get_observations()
            with torch.no_grad():
                actions = policy.act_inference(obs)
            obs, _, _, dones, _ = env.step(actions)

            base_pos = env.root_states[:, 0:3].cpu().numpy()
            base_h = base_pos[:, 2]
            base_vy = env.base_lin_vel[:, 1].cpu().numpy()
            base_wz = env.base_ang_vel[:, 2].cpu().numpy()

            pitch = (torch.asin(torch.clamp(env.projected_gravity[:, 1], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
            pitch_rate = (env.base_ang_vel[:, 0] * rad_to_deg).cpu().numpy()

            wheel_vel_L = env.dof_vel[:, env.left_wheel_idx].cpu().numpy()
            wheel_vel_R = env.dof_vel[:, env.right_wheel_idx].cpu().numpy()
            wheel_tau_L = env.torques[:, env.left_wheel_idx].cpu().numpy()
            wheel_tau_R = env.torques[:, env.right_wheel_idx].cpu().numpy()

            ts_base_h[step] = base_h
            ts_signed_h_err[step] = base_h - curr_h
            ts_abs_h_err[step] = np.abs(base_h - curr_h)
            ts_vy[step] = base_vy
            ts_wz[step] = base_wz
            ts_pitch[step] = pitch
            ts_pitch_rate[step] = pitch_rate

            ts_wheel_vel_L[step] = wheel_vel_L
            ts_wheel_vel_R[step] = wheel_vel_R
            ts_wheel_tau_L[step] = wheel_tau_L
            ts_wheel_tau_R[step] = wheel_tau_R

            d_arr = dones.cpu().numpy()
            if np.any(d_arr):
                done_flags |= d_arr

        # ========================================================
        # 1. 稳态切片 (steps 200..300, 100 步 x 32 envs = 3200 样本)
        # ========================================================
        s_slice = slice(steady_start, total_steps)
        steady_signed_h = ts_signed_h_err[s_slice].flatten() # [3200]
        steady_abs_h = ts_abs_h_err[s_slice].flatten()
        steady_base_h = ts_base_h[s_slice].flatten()
        steady_vy = ts_vy[s_slice].flatten()
        steady_abs_vy = np.abs(steady_vy)
        steady_wz = ts_wz[s_slice].flatten()
        steady_abs_wz = np.abs(steady_wz)

        # 门槛掩码 (Tol: H<=1cm, Vy<=0.03m/s, Wz<=0.05rad/s)
        pass_h_mask = (steady_abs_h <= 0.010)
        pass_vy_mask = (steady_abs_vy <= 0.030)
        pass_wz_mask = (steady_abs_wz <= 0.050)

        # 集合交集分解
        total_samples = len(steady_signed_h)
        n_joint = np.sum(pass_h_mask & pass_vy_mask & pass_wz_mask)
        n_h_pass = np.sum(pass_h_mask)
        n_vy_pass = np.sum(pass_vy_mask)
        n_wz_pass = np.sum(pass_wz_mask)

        # 重点排查：在高度达标的样本中，线速度是否达标？
        # P(Vy pass | H pass)
        p_vy_given_h = np.sum(pass_h_mask & pass_vy_mask) / (n_h_pass + 1e-9)
        # 高度达标但线速度超差 (Bottleneck 1)
        n_h_pass_vy_fail = np.sum(pass_h_mask & ~pass_vy_mask & pass_wz_mask)
        # 线速度达标但高度超差 (Bottleneck 2)
        n_vy_pass_h_fail = np.sum(~pass_h_mask & pass_vy_mask & pass_wz_mask)
        # 两者均超差
        n_both_fail = np.sum(~pass_h_mask & ~pass_vy_mask & pass_wz_mask)

        # ========================================================
        # 2. 线速度波动详细统计
        # ========================================================
        vy_mean = float(np.mean(steady_vy))
        vy_std = float(np.std(steady_vy))
        vy_rms = float(np.sqrt(np.mean(steady_vy**2)))
        vy_p50 = float(np.percentile(steady_abs_vy, 50))
        vy_p75 = float(np.percentile(steady_abs_vy, 75))
        vy_p90 = float(np.percentile(steady_abs_vy, 90))
        vy_p95 = float(np.percentile(steady_abs_vy, 95))
        vy_p99 = float(np.percentile(steady_abs_vy, 99))
        vy_max = float(np.max(steady_abs_vy))

        # ========================================================
        # 3. 动态过渡阶段 (steps 100..200) vs 稳态阶段 (steps 200..300)
        # ========================================================
        t_slice = slice(step_cmd_start, steady_start)
        trans_signed_h = ts_signed_h_err[t_slice].flatten()
        trans_abs_h = ts_abs_h_err[t_slice].flatten()
        trans_base_h = ts_base_h[t_slice].flatten()

        trans_h_mean = float(np.mean(trans_base_h))
        trans_h_err_mean_cm = float(np.mean(trans_abs_h) * 100.0)

        steady_h_mean = float(np.mean(steady_base_h))
        steady_h_err_mean_cm = float(np.mean(steady_abs_h) * 100.0)

        # 检查阶跃下沉速度：第 100 步到第 150 步
        # step 100: 0.3977 -> step 150: height
        h_at_120 = float(np.mean(ts_base_h[120]))
        h_at_140 = float(np.mean(ts_base_h[140]))
        h_at_160 = float(np.mean(ts_base_h[160]))
        h_at_180 = float(np.mean(ts_base_h[180]))
        h_at_200 = float(np.mean(ts_base_h[200]))

        # ========================================================
        # 4. 高度偏置分析 (稳态阶段 z 与 0.360 的正负性)
        # ========================================================
        pct_positive_bias = float(np.mean(steady_signed_h > 0) * 100.0) # z > 0.360
        pct_negative_bias = float(np.mean(steady_signed_h < 0) * 100.0) # z < 0.360
        pct_in_tolerance = float(np.mean(steady_abs_h <= 0.010) * 100.0)
        pct_above_tolerance = float(np.mean(steady_signed_h > 0.010) * 100.0) # z > 0.370
        pct_below_tolerance = float(np.mean(steady_signed_h < -0.010) * 100.0) # z < 0.350

        signed_h_mean_cm = float(np.mean(steady_signed_h) * 100.0)
        signed_h_median_cm = float(np.median(steady_signed_h) * 100.0)

        # 轮速与力矩
        wheel_vel_mag = 0.5 * (np.abs(ts_wheel_vel_L[s_slice]) + np.abs(ts_wheel_vel_R[s_slice]))
        wheel_tau_mag = 0.5 * (np.abs(ts_wheel_tau_L[s_slice]) + np.abs(ts_wheel_tau_R[s_slice]))

        model_diag = {
            "joint_pass_rate_pct": round(float(n_joint / total_samples * 100.0), 2),
            "h_pass_rate_pct": round(float(n_h_pass / total_samples * 100.0), 2),
            "vy_pass_rate_pct": round(float(n_vy_pass / total_samples * 100.0), 2),
            "wz_pass_rate_pct": round(float(n_wz_pass / total_samples * 100.0), 2),

            # 交集分解
            "intersection": {
                "joint_pass_all_3": round(float(n_joint / total_samples * 100.0), 2),
                "h_pass_but_vy_fail": round(float(n_h_pass_vy_fail / total_samples * 100.0), 2),
                "vy_pass_but_h_fail": round(float(n_vy_pass_h_fail / total_samples * 100.0), 2),
                "both_h_and_vy_fail": round(float(n_both_fail / total_samples * 100.0), 2),
                "cond_p_vy_pass_given_h_pass_pct": round(float(p_vy_given_h * 100.0), 2),
            },

            # 速度波动
            "vy_stats": {
                "vy_mean_mps": round(vy_mean, 5),
                "vy_std_mps": round(vy_std, 5),
                "vy_rms_mps": round(vy_rms, 5),
                "abs_vy_p50_mps": round(vy_p50, 4),
                "abs_vy_p75_mps": round(vy_p75, 4),
                "abs_vy_p90_mps": round(vy_p90, 4),
                "abs_vy_p95_mps": round(vy_p95, 4),
                "abs_vy_p99_mps": round(vy_p99, 4),
                "abs_vy_max_mps": round(vy_max, 4),
            },

            # 阶段误差对比
            "phase_comparison": {
                "trans_step100_200": {
                    "base_h_mean_m": round(trans_h_mean, 4),
                    "h_err_mean_cm": round(trans_h_err_mean_cm, 3),
                },
                "steady_step200_300": {
                    "base_h_mean_m": round(steady_h_mean, 4),
                    "h_err_mean_cm": round(steady_h_err_mean_cm, 3),
                },
                "step_trajectory": {
                    "step_120": round(h_at_120, 4),
                    "step_140": round(h_at_140, 4),
                    "step_160": round(h_at_160, 4),
                    "step_180": round(h_at_180, 4),
                    "step_200": round(h_at_200, 4),
                }
            },

            # 高度偏置
            "height_bias": {
                "signed_h_mean_cm": round(signed_h_mean_cm, 3),
                "signed_h_median_cm": round(signed_h_median_cm, 3),
                "pct_z_greater_than_target": round(pct_positive_bias, 2),
                "pct_z_less_than_target": round(pct_negative_bias, 2),
                "pct_in_tol_range_35_37": round(pct_in_tolerance, 2),
                "pct_above_tol_z_gt_37": round(pct_above_tolerance, 2),
                "pct_below_tol_z_lt_35": round(pct_below_tolerance, 2),
            },

            # 轮端
            "wheel_stats": {
                "wheel_vel_mean_rad_s": round(float(np.mean(wheel_vel_mag)), 4),
                "wheel_vel_rms_rad_s": round(float(np.sqrt(np.mean(wheel_vel_mag**2))), 4),
                "wheel_tau_mean_Nm": round(float(np.mean(wheel_tau_mag)), 3),
            }
        }

        results[m_name] = model_diag

    out_dir = "evaluation_assets/phase7d/diagnosis"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "phase7d_mechanism_diagnosis.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 7D Diagnosis Results Saved to: {out_file}\n")

if __name__ == '__main__':
    run_phase7d_diagnosis()
