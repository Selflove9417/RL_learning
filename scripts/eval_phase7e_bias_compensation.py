"""
Phase 7E: Low Height Command Bias Compensation Verification on g3_inplace_squat.
Model: Phase 7C Champion v3
Compares:
1. Nominal Control Input: H_cmd = 0.360 m
2. Compensated Control Input: H_cmd = 0.355 m (-5 mm compensation)

Both evaluated against original benchmark reference: H_ref = 0.360 m.
Also separately records tracking error relative to H_cmd.
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

def run_phase7e_experiment():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    ckpt_path = "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct12_11-16-16_phase7c_deadzone005_beta020_iter1/model_1.pt"
    device = env.device
    num_envs = args.num_envs
    total_steps = 300
    step_cmd_start = 100
    steady_start = 200

    h_ref = 0.360 # 评测基准参考高度
    tol_h = 0.010
    tol_vy = 0.030
    tol_wz = 0.050
    hold_steps_req = 25 # 0.5s at 50Hz

    rad_to_deg = 180.0 / np.pi

    test_configs = [
        {"name": "Nominal_Hcmd_0360", "h_cmd": 0.360},
        {"name": "Compensated_Hcmd_0355", "h_cmd": 0.355},
    ]

    results = {}

    policy = ActorCritic(
        num_actor_obs=env.cfg.env.num_observations,
        num_critic_obs=env.cfg.env.num_observations,
        num_actions=env.cfg.env.num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation='elu',
        init_noise_std=1.0
    ).to(device)

    loaded = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(loaded['model_state_dict'])
    policy.eval()

    init_state_snapshots = {}

    for cfg in test_configs:
        c_name = cfg["name"]
        h_cmd_target = cfg["h_cmd"]
        print(f"\n=======================================================")
        print(f"Running Test: {c_name} (H_cmd = {h_cmd_target:.4f} m, H_ref = {h_ref:.4f} m)")
        print(f"=======================================================")

        torch.manual_seed(42)
        np.random.seed(42)
        env.reset()

        ts_base_h = np.zeros((total_steps, num_envs))
        ts_vy = np.zeros((total_steps, num_envs))
        ts_wz = np.zeros((total_steps, num_envs))
        ts_pitch = np.zeros((total_steps, num_envs))
        ts_pitch_rate = np.zeros((total_steps, num_envs))
        ts_wheel_vel = np.zeros((total_steps, num_envs))
        ts_wheel_tau = np.zeros((total_steps, num_envs))
        ts_wheel_sat = np.zeros((total_steps, num_envs))

        h_err_ref_traj = np.zeros((num_envs, total_steps - step_cmd_start))
        vy_err_traj = np.zeros((num_envs, total_steps - step_cmd_start))
        wz_err_traj = np.zeros((num_envs, total_steps - step_cmd_start))

        done_flags = np.zeros(num_envs, dtype=bool)

        for step in range(total_steps):
            # 第 100 步前维持 0.39774 标称高度
            curr_h_cmd = 0.39774 if step < step_cmd_start else h_cmd_target
            env.commands[:, 0] = 0.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            env.height_cmd_buf[:] = curr_h_cmd

            if step == step_cmd_start - 1:
                # 记录第 99 步下发阶跃前的环境状态快照，用于核验初始状态严格一致
                init_state_snapshots[c_name] = {
                    "base_h": float(np.mean(env.root_states[:, 2].cpu().numpy())),
                    "base_vy": float(np.mean(env.base_lin_vel[:, 1].cpu().numpy())),
                    "pitch": float(np.mean((torch.asin(torch.clamp(env.projected_gravity[:, 1], -1.0, 1.0)) * rad_to_deg).cpu().numpy()))
                }

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

            wheel_vel = 0.5 * (np.abs(env.dof_vel[:, env.left_wheel_idx].cpu().numpy()) + np.abs(env.dof_vel[:, env.right_wheel_idx].cpu().numpy()))
            wheel_tau = 0.5 * (np.abs(env.torques[:, env.left_wheel_idx].cpu().numpy()) + np.abs(env.torques[:, env.right_wheel_idx].cpu().numpy()))
            wheel_sat = ((np.abs(env.torques[:, env.left_wheel_idx].cpu().numpy()) >= 6.95) | (np.abs(env.torques[:, env.right_wheel_idx].cpu().numpy()) >= 6.95)).astype(float)

            ts_base_h[step] = base_h
            ts_vy[step] = base_vy
            ts_wz[step] = base_wz
            ts_pitch[step] = pitch
            ts_pitch_rate[step] = pitch_rate
            ts_wheel_vel[step] = wheel_vel
            ts_wheel_tau[step] = wheel_tau
            ts_wheel_sat[step] = wheel_sat

            if step >= step_cmd_start:
                idx = step - step_cmd_start
                h_err_ref_traj[:, idx] = np.abs(base_h - h_ref) # 相对 H_ref = 0.360m
                vy_err_traj[:, idx] = np.abs(base_vy)
                wz_err_traj[:, idx] = np.abs(base_wz)

            d_arr = dones.cpu().numpy()
            if np.any(d_arr):
                done_flags |= d_arr

        # ========================================================
        # 统计分析
        # ========================================================
        s_slice = slice(steady_start, total_steps)
        steady_h_samples = ts_base_h[s_slice].flatten() # [3200]
        steady_vy_samples = ts_vy[s_slice].flatten()
        steady_wz_samples = ts_wz[s_slice].flatten()
        steady_pitch_samples = ts_pitch[s_slice].flatten()
        steady_pitch_rate_samples = ts_pitch_rate[s_slice].flatten()
        steady_wheel_vel_samples = ts_wheel_vel[s_slice].flatten()
        steady_wheel_tau_samples = ts_wheel_tau[s_slice].flatten()
        steady_wheel_sat_samples = ts_wheel_sat[s_slice].flatten()

        # 1. 实际高度与双参考系误差
        steady_actual_h_mean = float(np.mean(steady_h_samples))
        steady_actual_h_std = float(np.std(steady_h_samples))

        # (a) 相对原始任务参考 H_ref = 0.360 m
        signed_err_href = steady_h_samples - h_ref
        steady_err_href_mean_cm = float(np.mean(np.abs(signed_err_href)) * 100.0)
        signed_err_href_mean_cm = float(np.mean(signed_err_href) * 100.0)
        signed_err_href_median_cm = float(np.median(signed_err_href) * 100.0)

        # (b) 相对输入控制指令 H_cmd
        signed_err_hcmd = steady_h_samples - h_cmd_target
        steady_err_hcmd_mean_cm = float(np.mean(np.abs(signed_err_hcmd)) * 100.0)
        signed_err_hcmd_mean_cm = float(np.mean(signed_err_hcmd) * 100.0)

        # 2. 达标率计算 (严格按 H_ref = 0.360m 标准协议)
        pass_h_mat = (h_err_ref_traj <= tol_h)
        pass_vy_mat = (vy_err_traj <= tol_vy)
        pass_wz_mat = (wz_err_traj <= tol_wz)
        pass_joint_mat = pass_h_mat & pass_vy_mat & pass_wz_mat

        steady_slice_traj = slice(100, 200) # steps 200..300 in traj
        sample_pass_h = float(np.mean(pass_h_mat[:, steady_slice_traj]) * 100.0)
        sample_pass_vy = float(np.mean(pass_vy_mat[:, steady_slice_traj]) * 100.0)
        sample_pass_wz = float(np.mean(pass_wz_mat[:, steady_slice_traj]) * 100.0)
        sample_pass_joint = float(np.mean(pass_joint_mat[:, steady_slice_traj]) * 100.0)

        # 环境维度达标率 (连续 25 步保持且稳态窗达标)
        settling_times = []
        failed_envs = 0
        for e in range(num_envs):
            if done_flags[e]:
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
            if found_time is not None:
                settling_times.append(found_time)
            else:
                failed_envs += 1

        env_pass_rate = float((num_envs - failed_envs) / num_envs * 100.0)
        p50_settle = float(np.percentile(settling_times, 50)) if len(settling_times) > 0 else -1.0
        p95_settle = float(np.percentile(settling_times, 95)) if len(settling_times) > 0 else -1.0
        surv_rate = float((num_envs - np.sum(done_flags)) / num_envs * 100.0)

        # 3. 动力学与姿态
        vy_rms = float(np.sqrt(np.mean(steady_vy_samples**2)))
        wz_rms = float(np.sqrt(np.mean(steady_wz_samples**2)))
        pitch_mean_deg = float(np.mean(steady_pitch_samples))
        pitch_abs_deg = float(np.mean(np.abs(steady_pitch_samples)))
        pitch_max_deg = float(np.max(np.abs(steady_pitch_samples)))
        pitch_rate_rms_deg_s = float(np.sqrt(np.mean(steady_pitch_rate_samples**2)))

        wheel_vel_mean = float(np.mean(steady_wheel_vel_samples))
        wheel_vel_rms = float(np.sqrt(np.mean(steady_wheel_vel_samples**2)))
        wheel_tau_mean = float(np.mean(steady_wheel_tau_samples))
        wheel_tau_max = float(np.max(steady_wheel_tau_samples))
        wheel_sat_rate = float(np.mean(steady_wheel_sat_samples) * 100.0)

        # 高度分布区间统计
        pct_in_href_tol = float(np.mean(np.abs(steady_h_samples - h_ref) <= tol_h) * 100.0) # [0.350, 0.370]
        pct_above_href_tol = float(np.mean((steady_h_samples - h_ref) > tol_h) * 100.0)     # > 0.370
        pct_below_href_tol = float(np.mean((steady_h_samples - h_ref) < -tol_h) * 100.0)    # < 0.350

        results[c_name] = {
            "h_cmd_target_m": h_cmd_target,
            "h_ref_eval_m": h_ref,
            
            # 高度响应
            "steady_actual_h_mean_m": round(steady_actual_h_mean, 5),
            "steady_actual_h_std_m": round(steady_actual_h_std, 5),
            "signed_err_href_mean_cm": round(signed_err_href_mean_cm, 3),
            "signed_err_href_median_cm": round(signed_err_href_median_cm, 3),
            "abs_err_href_mean_cm": round(steady_err_href_mean_cm, 3),
            "signed_err_hcmd_mean_cm": round(signed_err_hcmd_mean_cm, 3),
            "abs_err_hcmd_mean_cm": round(steady_err_hcmd_mean_cm, 3),

            # 高度区间分布
            "pct_in_href_tol_35_37": round(pct_in_href_tol, 2),
            "pct_above_href_tol_gt_37": round(pct_above_href_tol, 2),
            "pct_below_href_tol_lt_35": round(pct_below_href_tol, 2),

            # 核心达标率 (对齐 Phase 5.3 ~ 7D 协议)
            "joint_pass_rate_pct": round(sample_pass_joint, 2),
            "env_pass_rate_pct": round(env_pass_rate, 2),
            "height_pass_rate_pct": round(sample_pass_h, 2),
            "vy_pass_rate_pct": round(sample_pass_vy, 2),
            "wz_pass_rate_pct": round(sample_pass_wz, 2),
            "settle_p50_s": round(p50_settle, 2),
            "settle_p95_s": round(p95_settle, 2),
            "survival_rate_pct": round(surv_rate, 2),

            # 速度与姿态
            "vy_rms_mps": round(vy_rms, 5),
            "wz_rms_radps": round(wz_rms, 5),
            "pitch_mean_deg": round(pitch_mean_deg, 3),
            "pitch_abs_deg": round(pitch_abs_deg, 3),
            "pitch_max_deg": round(pitch_max_deg, 3),
            "pitch_rate_rms_deg_s": round(pitch_rate_rms_deg_s, 3),

            # 轮端
            "wheel_vel_mean_radps": round(wheel_vel_mean, 4),
            "wheel_vel_rms_radps": round(wheel_vel_rms, 4),
            "wheel_tau_mean_Nm": round(wheel_tau_mean, 3),
            "wheel_tau_max_Nm": round(wheel_tau_max, 3),
            "wheel_sat_rate_pct": round(wheel_sat_rate, 2),
        }

    # 验证初始状态严格一致性
    snap1 = init_state_snapshots["Nominal_Hcmd_0360"]
    snap2 = init_state_snapshots["Compensated_Hcmd_0355"]
    is_initial_state_identical = (
        abs(snap1["base_h"] - snap2["base_h"]) < 1e-6 and
        abs(snap1["base_vy"] - snap2["base_vy"]) < 1e-6 and
        abs(snap1["pitch"] - snap2["pitch"]) < 1e-6
    )

    out_data = {
        "model_checkpoint": ckpt_path,
        "is_initial_state_identical": is_initial_state_identical,
        "init_state_step99": init_state_snapshots,
        "results": results
    }

    out_dir = "evaluation_assets/phase7e/results"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "eval_phase7e_bias_compensation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 7E Bias Compensation Results Saved: {out_file}\n")

    # 打印对比分析表
    res_nom = results["Nominal_Hcmd_0360"]
    res_comp = results["Compensated_Hcmd_0355"]

    print("==================================================================================================")
    print(f"{'Metric':<38} | {'Nominal (Hcmd=0.360)':<22} | {'Compensated (Hcmd=0.355)':<22} | {'Delta':<12}")
    print("--------------------------------------------------------------------------------------------------")
    print(f"{'Initial State Match (step 99)':<38} | {str(is_initial_state_identical):<22} | {str(is_initial_state_identical):<22} | {'Identical'}")
    
    metrics = [
        ("Actual Steady Height (m)", "steady_actual_h_mean_m", "m"),
        ("Signed Error vs H_ref=0.360m (cm)", "signed_err_href_mean_cm", "cm"),
        ("Abs Error vs H_ref=0.360m (cm)", "abs_err_href_mean_cm", "cm"),
        ("Signed Error vs H_cmd (cm)", "signed_err_hcmd_mean_cm", "cm"),
        ("Joint Pass Rate Joint% (vs H_ref)", "joint_pass_rate_pct", "%"),
        ("Env Pass Rate Env% (vs H_ref)", "env_pass_rate_pct", "%"),
        ("Height Pass Rate Height% (vs H_ref)", "height_pass_rate_pct", "%"),
        ("Samples in [0.350, 0.370]m (%)", "pct_in_href_tol_35_37", "%"),
        ("Samples > 0.370m (over-height) (%)", "pct_above_href_tol_gt_37", "%"),
        ("Samples < 0.350m (under-height) (%)", "pct_below_href_tol_lt_35", "%"),
        ("Vy Pass Rate (%)", "vy_pass_rate_pct", "%"),
        ("Vy RMS (m/s)", "vy_rms_mps", "m/s"),
        ("Wz Pass Rate (%)", "wz_pass_rate_pct", "%"),
        ("Wz RMS (rad/s)", "wz_rms_radps", "rad/s"),
        ("Pitch Mean (deg)", "pitch_mean_deg", "deg"),
        ("Pitch Rate RMS (deg/s)", "pitch_rate_rms_deg_s", "deg/s"),
        ("Wheel Torque Mean (Nm)", "wheel_tau_mean_Nm", "Nm"),
        ("Wheel Saturation Rate (%)", "wheel_sat_rate_pct", "%"),
        ("Survival Rate (%)", "survival_rate_pct", "%"),
    ]

    for label, key, unit in metrics:
        v1 = res_nom[key]
        v2 = res_comp[key]
        delta = v2 - v1
        print(f"{label:<38} | {v1:10.4f} {unit:<9} | {v2:10.4f} {unit:<9} | {delta:+10.4f} {unit}")
    print("==================================================================================================")

if __name__ == '__main__':
    run_phase7e_experiment()
