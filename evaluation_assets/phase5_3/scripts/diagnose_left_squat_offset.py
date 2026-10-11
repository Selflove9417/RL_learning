import isaacgym
import torch
import os
import json
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def run_diagnostics():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 20

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    ckpt_path = "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_15-48-32_turning_height_stage5_3_formal/model_25.pt"
    loaded = torch.load(ckpt_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    pol = ac.act_inference

    rad_to_deg = 180.0 / np.pi
    R_wheel = 0.070

    # 5 个对比工况，高度均为 0.36m 下蹲
    cases = [
        ("fwd_left_squat",     0.10,  0.20, 0.36), # 前进左转下蹲
        ("fwd_right_squat",    0.10, -0.20, 0.36), # 前进右转下蹲
        ("fwd_straight_squat", 0.10,  0.00, 0.36), # 前进直行下蹲
        ("bwd_left_squat",    -0.10,  0.20, 0.36), # 后退左转下蹲
        ("bwd_right_squat",   -0.10, -0.20, 0.36), # 后退右转下蹲
    ]

    total_steps = 300
    steady_start = 200 # 稳态区间 200~300 (4.0s ~ 6.0s)

    diag_summary = {}

    for name, cmd_vy, cmd_wz, target_h in cases:
        torch.manual_seed(42)
        env.reset()

        # 数据收集容器 (稳态步)
        h_err_list, deadband_pen_list = [], []
        q_ik_hip_list, q_ik_knee_list = [], []
        dq_ppo_hip_L_list, dq_ppo_hip_R_list = [], []
        dq_ppo_knee_L_list, dq_ppo_knee_R_list = [], []
        q_des_hip_L_list, q_des_hip_R_list = [], []
        q_des_knee_L_list, q_des_knee_R_list = [], []
        q_act_hip_L_list, q_act_hip_R_list = [], []
        q_act_knee_L_list, q_act_knee_R_list = [], []
        tau_hip_L_list, tau_hip_R_list = [], []
        tau_knee_L_list, tau_knee_R_list = [], []
        fn_L_list, fn_R_list = [], []
        slip_L_list, slip_R_list = [], []
        vy_err_list, wz_err_list = [], []
        pitch_list = []

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

            if step >= steady_start:
                # 1. 高度与奖励死区
                h_act = env.root_states[:, 2].cpu().numpy()
                h_err = np.abs(h_act - target_h)
                deadband_pen = np.maximum(0.0, h_err - 0.015)

                # 2. IK 目标
                q1_ik, q2_ik = env.compute_leg_ik(env.height_cmd_buf)
                q1_ik_val = q1_ik.cpu().numpy()
                q2_ik_val = q2_ik.cpu().numpy()

                # 3. PPO 残差 (经 filtered_actions 与 scale)
                actions_scaled = (env.filtered_actions * env.cfg.control.action_scale).clone()
                dq_hip_L = (actions_scaled[:, env.left_hip_idx] * env.hip_scale_reduction).cpu().numpy()
                dq_hip_R = (actions_scaled[:, env.right_hip_idx] * env.hip_scale_reduction).cpu().numpy()
                dq_knee_L = (actions_scaled[:, env.left_knee_idx] * env.knee_scale_reduction).cpu().numpy()
                dq_knee_R = (actions_scaled[:, env.right_knee_idx] * env.knee_scale_reduction).cpu().numpy()

                # 4. PD 期望位置
                q_des_hip_L = env.position_targets[:, env.left_hip_idx].cpu().numpy()
                q_des_hip_R = env.position_targets[:, env.right_hip_idx].cpu().numpy()
                q_des_knee_L = env.position_targets[:, env.left_knee_idx].cpu().numpy()
                q_des_knee_R = env.position_targets[:, env.right_knee_idx].cpu().numpy()

                # 5. 实际关节位置
                q_act_hip_L = env.dof_pos[:, env.left_hip_idx].cpu().numpy()
                q_act_hip_R = env.dof_pos[:, env.right_hip_idx].cpu().numpy()
                q_act_knee_L = env.dof_pos[:, env.left_knee_idx].cpu().numpy()
                q_act_knee_R = env.dof_pos[:, env.right_knee_idx].cpu().numpy()

                # 6. 关节力矩
                tau_hip_L = env.torques[:, env.left_hip_idx].cpu().numpy()
                tau_hip_R = env.torques[:, env.right_hip_idx].cpu().numpy()
                tau_knee_L = env.torques[:, env.left_knee_idx].cpu().numpy()
                tau_knee_R = env.torques[:, env.right_knee_idx].cpu().numpy()

                # 7. 接触力
                fn_L = env.contact_forces[:, env.feet_indices[0], 2].cpu().numpy()
                fn_R = env.contact_forces[:, env.feet_indices[1], 2].cpu().numpy()

                # 8. 轮地滑动计算
                # 轮线速度 = dof_vel * R
                w_vel_L = env.dof_vel[:, env.left_wheel_idx].cpu().numpy()
                w_vel_R = env.dof_vel[:, env.right_wheel_idx].cpu().numpy()
                v_wheel_L = -w_vel_L * R_wheel # 注意轮子转轴坐标系极性
                v_wheel_R = w_vel_R * R_wheel
                base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                base_wz = env.base_ang_vel[:, 2].cpu().numpy()
                # 左右轮在机器人坐标系下的理论线速度
                d_track = 0.285 / 2.0 # 半轮距约 0.1425m
                v_robot_L = base_vy - base_wz * d_track
                v_robot_R = base_vy + base_wz * d_track
                slip_L = np.abs(v_wheel_L - v_robot_L)
                slip_R = np.abs(v_wheel_R - v_robot_R)

                # 9. 速度与俯仰
                vy_err = np.abs(base_vy - cmd_vy)
                wz_err = np.abs(base_wz - cmd_wz)
                pitch = (torch.asin(torch.clamp(env.projected_gravity[:, 1], -1.0, 1.0)) * rad_to_deg).cpu().numpy()

                h_err_list.append(h_err)
                deadband_pen_list.append(deadband_pen)
                q_ik_hip_list.append(q1_ik_val)
                q_ik_knee_list.append(q2_ik_val)
                dq_ppo_hip_L_list.append(dq_hip_L)
                dq_ppo_hip_R_list.append(dq_hip_R)
                dq_ppo_knee_L_list.append(dq_knee_L)
                dq_ppo_knee_R_list.append(dq_knee_R)
                q_des_hip_L_list.append(q_des_hip_L)
                q_des_hip_R_list.append(q_des_hip_R)
                q_des_knee_L_list.append(q_des_knee_L)
                q_des_knee_R_list.append(q_des_knee_R)
                q_act_hip_L_list.append(q_act_hip_L)
                q_act_hip_R_list.append(q_act_hip_R)
                q_act_knee_L_list.append(q_act_knee_L)
                q_act_knee_R_list.append(q_act_knee_R)
                tau_hip_L_list.append(tau_hip_L)
                tau_hip_R_list.append(tau_hip_R)
                tau_knee_L_list.append(tau_knee_L)
                tau_knee_R_list.append(tau_knee_R)
                fn_L_list.append(fn_L)
                fn_R_list.append(fn_R)
                slip_L_list.append(slip_L)
                slip_R_list.append(slip_R)
                vy_err_list.append(vy_err)
                wz_err_list.append(wz_err)
                pitch_list.append(pitch)

        res = {
            "h_steady_err_cm": round(float(np.mean(h_err_list) * 100.0), 3),
            "deadband_pen_cm": round(float(np.mean(deadband_pen_list) * 100.0), 3),
            # IK 输出
            "q_ik_hip_rad": round(float(np.mean(q_ik_hip_list)), 4),
            "q_ik_knee_rad": round(float(np.mean(q_ik_knee_list)), 4),
            # PPO 残差
            "dq_ppo_hip_L_rad": round(float(np.mean(dq_ppo_hip_L_list)), 4),
            "dq_ppo_hip_R_rad": round(float(np.mean(dq_ppo_hip_R_list)), 4),
            "dq_ppo_knee_L_rad": round(float(np.mean(dq_ppo_knee_L_list)), 4),
            "dq_ppo_knee_R_rad": round(float(np.mean(dq_ppo_knee_R_list)), 4),
            # PD 目标与实际跟踪误差
            "pd_track_err_hip_L_rad": round(float(np.mean(np.array(q_act_hip_L_list) - np.array(q_des_hip_L_list))), 4),
            "pd_track_err_hip_R_rad": round(float(np.mean(np.array(q_act_hip_R_list) - np.array(q_des_hip_R_list))), 4),
            "pd_track_err_knee_L_rad": round(float(np.mean(np.array(q_act_knee_L_list) - np.array(q_des_knee_L_list))), 4),
            "pd_track_err_knee_R_rad": round(float(np.mean(np.array(q_act_knee_R_list) - np.array(q_des_knee_R_list))), 4),
            # 力矩输出 (vs 限幅: hip 75Nm, knee 50Nm)
            "tau_hip_L_Nm": round(float(np.mean(np.abs(tau_hip_L_list))), 2),
            "tau_hip_R_Nm": round(float(np.mean(np.abs(tau_hip_R_list))), 2),
            "tau_knee_L_Nm": round(float(np.mean(np.abs(tau_knee_L_list))), 2),
            "tau_knee_R_Nm": round(float(np.mean(np.abs(tau_knee_R_list))), 2),
            # 接触力与不平衡度
            "fn_L_N": round(float(np.mean(fn_L_list)), 2),
            "fn_R_N": round(float(np.mean(fn_R_list)), 2),
            "fn_imbalance_ratio": round(float(np.mean(fn_L_list) / max(1e-3, np.mean(fn_R_list))), 3),
            # 轮地滑动
            "wheel_slip_L_mps": round(float(np.mean(slip_L_list)), 4),
            "wheel_slip_R_mps": round(float(np.mean(slip_R_list)), 4),
            # 姿态与跟踪
            "vy_err_mps": round(float(np.mean(vy_err_list)), 4),
            "wz_err_radps": round(float(np.mean(wz_err_list)), 4),
            "pitch_mean_deg": round(float(np.mean(pitch_list)), 2),
        }
        diag_summary[name] = res
        print(f"\n[{name}] H_err: {res['h_steady_err_cm']}cm | DeadbandPen: {res['deadband_pen_cm']}cm")
        print(f"  IK q1={res['q_ik_hip_rad']}, q2={res['q_ik_knee_rad']}")
        print(f"  PPO Residuals: Hip L={res['dq_ppo_hip_L_rad']}, R={res['dq_ppo_hip_R_rad']} | Knee L={res['dq_ppo_knee_L_rad']}, R={res['dq_ppo_knee_R_rad']}")
        print(f"  PD Track Error: Hip L={res['pd_track_err_hip_L_rad']}, Knee L={res['pd_track_err_knee_L_rad']}")
        print(f"  Torques: Hip L={res['tau_hip_L_Nm']}Nm, Knee L={res['tau_knee_L_Nm']}Nm (Limits: 75/50)")
        print(f"  Contact Fn: L={res['fn_L_N']}N, R={res['fn_R_N']}N (Ratio L/R={res['fn_imbalance_ratio']})")
        print(f"  Wheel Slip: L={res['wheel_slip_L_mps']}m/s, R={res['wheel_slip_R_mps']}m/s | Wz_err: {res['wz_err_radps']}rad/s | Pitch: {res['pitch_mean_deg']}°")

    out_file = "/home/xy/.gemini/antigravity/brain/853aaed2-8860-4930-be3f-1bb608c17795/scratch/diagnose_left_squat_offset_results.json"
    with open(out_file, "w") as f:
        json.dump(diag_summary, f, indent=2)
    print(f"\nAll diagnostics saved to {out_file}!\n")

if __name__ == '__main__':
    run_diagnostics()
