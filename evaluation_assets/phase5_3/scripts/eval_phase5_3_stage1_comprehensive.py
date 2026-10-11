import isaacgym
import torch
import os
import json
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def calculate_single_target_settling(err_arr, step_idx, total_steps, threshold, hold_steps=25, dt=0.02):
    """
    逐环境计算单目标的调节时间统计。
    err_arr: [total_steps, num_envs]
    """
    num_envs = err_arr.shape[1]
    settling_times = []
    settled_flags = []
    unheld_counts = []

    for e in range(num_envs):
        traj = err_arr[step_idx:, e]
        settled_step = None
        max_check = len(traj) - hold_steps
        unheld = 0

        for t in range(max(1, max_check)):
            if traj[t] <= threshold:
                if np.all(traj[t : t + hold_steps] <= threshold):
                    settled_step = t
                    break
                else:
                    unheld += 1

        if settled_step is not None:
            settling_times.append(settled_step * dt)
            settled_flags.append(True)
        else:
            settled_flags.append(False)
        unheld_counts.append(unheld)

    pass_count = sum(settled_flags)
    pass_rate = pass_count / num_envs * 100.0
    fail_count = num_envs - pass_count

    if pass_count > 0:
        p50 = float(np.median(settling_times))
        p95 = float(np.percentile(settling_times, 95))
        mean_s = float(np.mean(settling_times))
    else:
        p50, p95, mean_s = None, None, None

    return {
        "pass_rate_pct": round(pass_rate, 1),
        "pass_count": int(pass_count),
        "fail_count": int(fail_count),
        "median_settle_s": round(p50, 3) if p50 is not None else None,
        "p95_settle_s": round(p95, 3) if p95 is not None else None,
        "mean_settle_s": round(mean_s, 3) if mean_s is not None else None,
        "unheld_entries_avg": round(float(np.mean(unheld_counts)), 1),
    }

def calculate_joint_3target_settling(h_err_arr, vy_err_arr, wz_err_arr, step_idx, total_steps, 
                                     h_thresh=0.010, vy_thresh=0.030, wz_thresh=0.050, hold_steps=25, dt=0.02):
    """
    逐环境计算高度、纵向速度、偏航角速度三目标联合达标率与调节时间。
    """
    num_envs = h_err_arr.shape[1]
    joint_settling_times = []
    joint_settled_flags = []

    for e in range(num_envs):
        h_t = h_err_arr[step_idx:, e]
        vy_t = vy_err_arr[step_idx:, e]
        wz_t = wz_err_arr[step_idx:, e]

        joint_ok = (h_t <= h_thresh) & (vy_t <= vy_thresh) & (wz_t <= wz_thresh)

        settled_step = None
        max_check = len(joint_ok) - hold_steps
        for t in range(max(1, max_check)):
            if np.all(joint_ok[t : t + hold_steps]):
                settled_step = t
                break

        if settled_step is not None:
            joint_settling_times.append(settled_step * dt)
            joint_settled_flags.append(True)
        else:
            joint_settled_flags.append(False)

    pass_count = sum(joint_settled_flags)
    pass_rate = pass_count / num_envs * 100.0
    fail_count = num_envs - pass_count

    if pass_count > 0:
        p50 = float(np.median(joint_settling_times))
        p95 = float(np.percentile(joint_settling_times, 95))
        mean_s = float(np.mean(joint_settling_times))
    else:
        p50, p95, mean_s = None, None, None

    return {
        "joint_pass_rate_pct": round(pass_rate, 1),
        "joint_pass_count": int(pass_count),
        "joint_fail_count": int(fail_count),
        "joint_median_settle_s": round(p50, 3) if p50 is not None else None,
        "joint_p95_settle_s": round(p95, 3) if p95 is not None else None,
        "joint_mean_settle_s": round(mean_s, 3) if mean_s is not None else None,
    }

def calculate_max_non_dual(contact_dual_arr, dt=0.02):
    num_envs = contact_dual_arr.shape[1]
    max_runs = []
    for e in range(num_envs):
        c = contact_dual_arr[:, e]
        max_run = 0
        curr_run = 0
        for val in c:
            if val < 0.5:
                curr_run += 1
                if curr_run > max_run:
                    max_run = curr_run
            else:
                curr_run = 0
        max_runs.append(max_run * dt)
    return float(np.max(max_runs))

def run_comprehensive_eval():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 20

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    models_to_eval = [
        ("Phase5_2_m25", "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_11-37-33_cruising_height_stage5_2_formal/model_25.pt"),
        ("Phase5_3_short_m10", "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_15-32-04_turning_height_stage5_3_short/model_10.pt"),
        ("Phase5_3_formal_m25", "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_15-48-32_turning_height_stage5_3_formal/model_25.pt"),
        ("Phase5_3_formal_m50", "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_15-48-32_turning_height_stage5_3_formal/model_50.pt"),
    ]

    rad_to_deg = 180.0 / np.pi

    # 两大评测类别场景定义
    # Category 1: Stage 1 训练范围内测试 (H in [0.36, 0.44] m)
    # Category 2: 完整工作区间泛化测试 (H = 0.35, 0.45 m 及旧技能保持)
    scenarios = [
        # --- Category 1: Stage 1 范围内基准测试 ---
        ("cat1_in_range", "in_fwd_left_squat", 
         [(0, 100, 0.10, 0.20, 0.39774), (100, 300, 0.10, 0.20, 0.36000)], 300),
        ("cat1_in_range", "in_fwd_right_raise", 
         [(0, 100, 0.10, -0.20, 0.39774), (100, 300, 0.10, -0.20, 0.44000)], 300),
        ("cat1_in_range", "in_bwd_left_squat", 
         [(0, 100, -0.10, 0.20, 0.39774), (100, 300, -0.10, 0.20, 0.36000)], 300),
        ("cat1_in_range", "in_bwd_right_raise", 
         [(0, 100, -0.10, -0.20, 0.39774), (100, 300, -0.10, -0.20, 0.44000)], 300),
        ("cat1_in_range", "in_fwd_right_squat", 
         [(0, 100, 0.10, -0.20, 0.39774), (100, 300, 0.10, -0.20, 0.36000)], 300),
        ("cat1_in_range", "in_bwd_left_raise", 
         [(0, 100, -0.10, 0.20, 0.39774), (100, 300, -0.10, 0.20, 0.44000)], 300),
        ("cat1_in_range", "in_fwd_left_restore", 
         [(0, 100, 0.10, 0.20, 0.36000), (100, 300, 0.10, 0.20, 0.39774)], 300),
        ("cat1_in_range", "in_bwd_right_restore", 
         [(0, 100, -0.10, -0.20, 0.44000), (100, 300, -0.10, -0.20, 0.39774)], 300),

        # --- Category 2: 完整工作区间泛化与旧技能保持 ---
        ("cat2_generalization", "gen_fwd_left_squat_0.35", 
         [(0, 100, 0.10, 0.20, 0.39774), (100, 300, 0.10, 0.20, 0.35000)], 300),
        ("cat2_generalization", "gen_fwd_right_raise_0.45", 
         [(0, 100, 0.10, -0.20, 0.39774), (100, 300, 0.10, -0.20, 0.45000)], 300),
        ("cat2_generalization", "gen_bwd_left_squat_0.35", 
         [(0, 100, -0.10, 0.20, 0.39774), (100, 300, -0.10, 0.20, 0.35000)], 300),
        ("cat2_generalization", "gen_bwd_right_raise_0.45", 
         [(0, 100, -0.10, -0.20, 0.39774), (100, 300, -0.10, -0.20, 0.45000)], 300),
        ("cat2_retention", "ret_inplace_yaw_dyn_h", 
         [(0, 100, 0.0, 0.20, 0.39774), (100, 200, 0.0, 0.20, 0.35000), (200, 300, 0.0, 0.20, 0.45000)], 300),
        ("cat2_retention", "ret_straight_cruise_dyn_h", 
         [(0, 100, 0.10, 0.0, 0.39774), (100, 200, 0.10, 0.0, 0.35000), (200, 300, 0.10, 0.0, 0.45000)], 300),
        ("cat2_retention", "ret_nominal_fwd_turn", 
         [(0, 200, 0.20, -0.35, 0.39774)], 200),
        ("cat2_retention", "ret_nominal_stand_park", 
         [(0, 200, 0.0, 0.0, 0.39774)], 200),
    ]

    all_results = {}

    for m_label, ckpt_p in models_to_eval:
        print(f"\n=======================================================")
        print(f"=== Evaluating Candidate: {m_label} ===")
        print(f"=======================================================")
        loaded = torch.load(ckpt_p, map_location=env.device)
        ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
        ac.load_state_dict(loaded['model_state_dict'])
        ac.eval()
        pol = ac.act_inference

        all_results[m_label] = {}

        for cat_name, s_name, stages, total_steps in scenarios:
            torch.manual_seed(42)
            env.reset()

            early_dones = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            h_act_ts, h_tar_ts = [], []
            cmd_vy_ts, cmd_wz_ts = [], []
            vy_ts, wz_ts = [], []
            pitch_ts, roll_ts = [], []
            tau_L_ts, tau_R_ts = [], []
            knee_tau_ts, knee_pos_ts = [], []
            cL_ts, cR_ts = [], []
            wheel_sat_count = 0
            max_consec_sat = 0
            curr_consec_sat = np.zeros(env.num_envs, dtype=int)

            for step in range(total_steps):
                for s_start, s_end, c_vy, c_wz, s_h in stages:
                    if s_start <= step < s_end:
                        curr_vy = c_vy
                        curr_wz = c_wz
                        curr_h = s_h
                        break

                env.commands[:, 0] = 0.0
                env.commands[:, 1] = curr_vy
                env.commands[:, 2] = curr_wz
                env.height_cmd_buf[:] = curr_h

                obs = env.get_observations()
                with torch.no_grad():
                    actions = pol(obs)
                obs, _, _, dones, _ = env.step(actions)
                if dones.any(): early_dones = early_dones | dones

                h_act = env.root_states[:, 2].clone().cpu().numpy()
                vy = env.base_lin_vel[:, 1].cpu().numpy()
                wz = env.base_ang_vel[:, 2].cpu().numpy()
                pitch = (torch.asin(torch.clamp(env.projected_gravity[:, 1], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
                roll = (torch.asin(torch.clamp(env.projected_gravity[:, 0], -1.0, 1.0)) * rad_to_deg).cpu().numpy()

                cL = (env.contact_forces[:, env.feet_indices[0], 2] > 1.0).float().cpu().numpy()
                cR = (env.contact_forces[:, env.feet_indices[1], 2] > 1.0).float().cpu().numpy()

                tau_L = torch.abs(env.torques[:, env.left_wheel_idx]).cpu().numpy()
                tau_R = torch.abs(env.torques[:, env.right_wheel_idx]).cpu().numpy()
                knee_tau = torch.abs(env.torques[:, env.left_knee_idx]).cpu().numpy()
                knee_pos = env.dof_pos[:, env.left_knee_idx].cpu().numpy()

                is_sat = (tau_L >= 6.99) | (tau_R >= 6.99)
                wheel_sat_count += np.sum(is_sat)

                # 连续饱和追踪
                curr_consec_sat = np.where(is_sat, curr_consec_sat + 1, 0)
                max_consec_sat = max(max_consec_sat, int(np.max(curr_consec_sat)))

                h_act_ts.append(h_act)
                h_tar_ts.append(np.full_like(h_act, curr_h))
                cmd_vy_ts.append(np.full_like(vy, curr_vy))
                cmd_wz_ts.append(np.full_like(wz, curr_wz))
                vy_ts.append(vy)
                wz_ts.append(wz)
                pitch_ts.append(pitch)
                roll_ts.append(roll)
                cL_ts.append(cL)
                cR_ts.append(cR)
                tau_L_ts.append(tau_L)
                tau_R_ts.append(tau_R)
                knee_tau_ts.append(knee_tau)
                knee_pos_ts.append(knee_pos)

            h_act_arr = np.array(h_act_ts) # [steps, envs]
            h_tar_arr = np.array(h_tar_ts)
            h_err_arr = np.abs(h_act_arr - h_tar_arr)

            cmd_vy_arr = np.array(cmd_vy_ts)
            cmd_wz_arr = np.array(cmd_wz_ts)
            vy_arr = np.array(vy_ts)
            wz_arr = np.array(wz_ts)
            err_vy_arr = np.abs(vy_arr - cmd_vy_arr)
            err_wz_arr = np.abs(wz_arr - cmd_wz_arr)

            pitch_arr = np.array(pitch_ts)
            roll_arr = np.array(roll_ts)
            c_dual_arr = np.array(cL_ts) * np.array(cR_ts)
            tau_max_arr = np.maximum(np.array(tau_L_ts), np.array(tau_R_ts))
            knee_tau_arr = np.array(knee_tau_ts)
            knee_pos_arr = np.array(knee_pos_ts)

            total_env_steps = total_steps * env.num_envs
            wheel_sat_pct = (wheel_sat_count / total_env_steps) * 100.0

            steady_slice = slice(-80, None)
            h_steady_err = np.mean(h_err_arr[steady_slice])
            vy_steady_rms = np.sqrt(np.mean(err_vy_arr[steady_slice]**2))
            wz_steady_rms = np.sqrt(np.mean(err_wz_arr[steady_slice]**2))

            has_step = (len(stages) >= 2) and (stages[1][0] == 100)
            if has_step:
                # 1. 高度单目标达标统计
                h_stats = calculate_single_target_settling(h_err_arr, step_idx=100, total_steps=total_steps, threshold=0.010, hold_steps=25)
                # 2. 速度单目标达标统计
                vy_stats = calculate_single_target_settling(err_vy_arr, step_idx=100, total_steps=total_steps, threshold=0.030, hold_steps=25)
                # 3. 偏航单目标达标统计
                wz_stats = calculate_single_target_settling(err_wz_arr, step_idx=100, total_steps=total_steps, threshold=0.050, hold_steps=25)
                # 4. 三目标联合达标统计
                joint_stats = calculate_joint_3target_settling(h_err_arr, err_vy_arr, err_wz_arr, step_idx=100, total_steps=total_steps,
                                                              h_thresh=0.010, vy_thresh=0.030, wz_thresh=0.050, hold_steps=25)
            else:
                h_stats = {"pass_rate_pct": 100.0, "pass_count": 20, "fail_count": 0, "median_settle_s": 0.0, "p95_settle_s": 0.0, "mean_settle_s": 0.0, "unheld_entries_avg": 0.0}
                vy_stats = {"pass_rate_pct": 100.0, "pass_count": 20, "fail_count": 0, "median_settle_s": 0.0, "p95_settle_s": 0.0, "mean_settle_s": 0.0, "unheld_entries_avg": 0.0}
                wz_stats = {"pass_rate_pct": 100.0, "pass_count": 20, "fail_count": 0, "median_settle_s": 0.0, "p95_settle_s": 0.0, "mean_settle_s": 0.0, "unheld_entries_avg": 0.0}
                joint_stats = {"joint_pass_rate_pct": 100.0, "joint_pass_count": 20, "joint_fail_count": 0, "joint_median_settle_s": 0.0, "joint_p95_settle_s": 0.0, "joint_mean_settle_s": 0.0}

            max_non_dual_sec = calculate_max_non_dual(c_dual_arr, dt=0.02)

            res = {
                "category": cat_name,
                "h_err_mean_cm": round(float(np.mean(h_err_arr) * 100.0), 2),
                "h_err_rms_cm": round(float(np.sqrt(np.mean(h_err_arr**2)) * 100.0), 2),
                "h_err_steady_cm": round(float(h_steady_err * 100.0), 2),
                "h_settle_stats": h_stats,
                "vy_settle_stats": vy_stats,
                "wz_settle_stats": wz_stats,
                "joint_settle_stats": joint_stats,
                "vy_rms_err_mps": round(float(np.sqrt(np.mean(err_vy_arr**2))), 4),
                "vy_steady_rms_mps": round(float(vy_steady_rms), 4),
                "wz_rms_err_radps": round(float(np.sqrt(np.mean(err_wz_arr**2))), 4),
                "wz_steady_rms_radps": round(float(wz_steady_rms), 4),
                "pitch_rms_deg": round(float(np.sqrt(np.mean(pitch_arr**2))), 2),
                "pitch_peak_deg": round(float(np.max(np.abs(pitch_arr))), 2),
                "roll_rms_deg": round(float(np.sqrt(np.mean(roll_arr**2))), 2),
                "roll_peak_deg": round(float(np.max(np.abs(roll_arr))), 2),
                "contact_dual_pct": round(float(np.mean(c_dual_arr) * 100.0), 1),
                "max_consecutive_non_dual_s": round(float(max_non_dual_sec), 3),
                "wheel_sat_pct": round(float(wheel_sat_pct), 3),
                "max_consecutive_sat_steps": int(max_consec_sat),
                "wheel_tau_rms_Nm": round(float(np.sqrt(np.mean(tau_max_arr**2))), 2),
                "wheel_tau_peak_Nm": round(float(np.max(tau_max_arr)), 2),
                "knee_tau_mean_Nm": round(float(np.mean(knee_tau_arr)), 2),
                "knee_tau_peak_Nm": round(float(np.max(knee_tau_arr)), 2),
                "knee_pos_mean_rad": round(float(np.mean(knee_pos_arr)), 3),
                "early_dones": int(early_dones.sum().item()),
            }

            all_results[m_label][s_name] = res
            p_h = h_stats['pass_rate_pct']
            p_j = joint_stats['joint_pass_rate_pct']
            med_j = joint_stats['joint_median_settle_s']
            print(f"[{m_label} | {s_name}] H_err: {res['h_err_mean_cm']}cm (std: {res['h_err_steady_cm']}cm) | Wz_err: {res['wz_rms_err_radps']}rad/s | H_pass: {p_h}% | JointPass: {p_j}% (med: {med_j}s) | Pitch: {res['pitch_rms_deg']}° | Dones: {res['early_dones']}")

    out_file = "/home/xy/.gemini/antigravity/brain/853aaed2-8860-4930-be3f-1bb608c17795/scratch/phase5_3_stage1_comprehensive_eval.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll comprehensive evaluation results saved to {out_file}!\n")

if __name__ == '__main__':
    run_comprehensive_eval()
