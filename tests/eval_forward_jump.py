#!/usr/bin/env python3
# ==============================================================================
# MyRobot Forward Jump: Stage 1 Evaluation & Comparative Benchmark
# ==============================================================================

import os
import sys
import csv
import json
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz
import torch
import imageio

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.envs.myrobot.myrobot_dual_env import compute_exact_cylinder_clearance


def run_evaluation():
    output_dir = os.path.join(workspace_root, "evaluation_assets", "phase1_forward_jump")
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(output_dir, "forward_jump_demonstration.mp4")
    csv_path = os.path.join(output_dir, "forward_jump_telemetry.csv")
    summary_path = os.path.join(output_dir, "forward_jump_evaluation_summary.json")

    print("=" * 80)
    print("🚀 MyRobot 前向跳跃第一阶段 (Phase 1) 严谨定量评测与基准对照")
    print("=" * 80)

    # 1. 确定被评测的模型路径
    baseline_model_path = os.path.join(workspace_root, "evaluation_assets", "frozen_stage8", "model_40.pt")
    trial_run_dir1 = os.path.join(workspace_root, "legged_gym", "logs", "myrobot_forward_jump", "Oct16_10-39-46_fwd_jump_phase1_trial")
    trial_model_path1 = os.path.join(trial_run_dir1, "model_30.pt")

    trial_run_dir2 = os.path.join(workspace_root, "legged_gym", "logs", "myrobot_forward_jump", "Oct16_10-44-22_fwd_jump_phase1_trial_yaw_aligned")
    trial_model_path2 = os.path.join(trial_run_dir2, "model_30.pt")

    assert os.path.isfile(baseline_model_path), f"基准模型未找到: {baseline_model_path}"
    assert os.path.isfile(trial_model_path1), f"试训模型1未找到: {trial_model_path1}"
    assert os.path.isfile(trial_model_path2), f"试训模型2未找到: {trial_model_path2}"

    models_to_evaluate = [
        {"name": "Stage 8 Frozen In-Place Jump", "path": baseline_model_path, "is_forward_target": False, "record": False},
        {"name": "Phase 1 Forward Jump (Trial 1)", "path": trial_model_path1, "is_forward_target": True, "record": False},
        {"name": "Phase 1 Forward Jump (Trial 2 Yaw-Aligned)", "path": trial_model_path2, "is_forward_target": True, "record": True},
    ]

    results_all = {}

    # 创建仿真环境 (单进程只初始化一次 PhysX)
    from legged_gym.utils import get_args
    sys.argv = ['eval_fwd', '--task', 'myrobot_forward_jump', '--num_envs', '1']
    args = get_args()
    args.headless = False
    args.seed = 42

    env_cfg, train_cfg = task_registry.get_cfgs(name='myrobot_forward_jump')
    env_cfg.env.num_envs = 1
    env_cfg.seed = 42
    env_cfg.env.episode_length_s = 10.0
    env_cfg.noise.add_noise = False
    env_cfg.env.enable_camera_sensors = True

    env, _ = task_registry.make_env(name='myrobot_forward_jump', args=args, env_cfg=env_cfg)
    num_obs = env_cfg.env.num_observations
    num_actions = env_cfg.env.num_actions

    camera_props = gymapi.CameraProperties()
    camera_props.width = 1280
    camera_props.height = 720
    camera_props.enable_tensors = False
    camera_props.horizontal_fov = 65.0
    camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    print(f"📷 离线摄像机创建状态: handle={camera_handle}")
    cam_offset = np.array([-1.5, -2.0, 0.85])

    ac = ActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0
    ).to(env.device)

    for m_info in models_to_evaluate:
        m_name = m_info["name"]
        m_path = m_info["path"]
        is_forward = m_info["is_forward_target"]

        print(f"\n" + "-" * 70)
        print(f"📊 开始评测策略: {m_name}")
        print(f"📦 模型路径: {m_path}")
        print("-" * 70)

        # 加载策略权重
        ckpt = torch.load(m_path, map_location=env.device)
        ac.load_state_dict(ckpt["model_state_dict"])
        ac.eval()
        policy = ac.act_inference

        record_video = m_info.get("record", False)
        record_csv = m_info.get("record", False)

        frames = []
        csv_writer = None
        csv_file_obj = None

        if record_csv:
            csv_file_obj = open(csv_path, "w", newline="")
            csv_writer = csv.writer(csv_file_obj)
            csv_writer.writerow([
                "time_s", "step", "jump_phase",
                "root_x", "root_y", "root_z",
                "vel_x", "vel_y", "vel_z",
                "roll_deg", "pitch_deg", "yaw_deg",
                "is_airborne", "has_taken_off",
                "clr_min_cm", "clr_L_cm", "clr_R_cm",
                "fn_L_N", "fn_R_N",
                "flight_distance_y_m", "final_flight_distance_m",
                "tau_hip_L", "tau_knee_L", "tau_hip_R", "tau_knee_R",
                "tau_wheel_L", "tau_wheel_R"
            ])

        # 重置环境
        env.reset()
        if not is_forward:
            env.commands[:, 1] = 0.0 # stage 8 in-place jump command

        # 跟踪统计变量 (评测 1 个完整单跳周期 2.0s = 100 步，并延续至 3.0s 充分观察平稳恢复)
        duration_s = 3.0
        total_steps = int(duration_s / env.dt)

        max_clearance = 0.0
        takeoff_vy = 0.0
        takeoff_vz = 0.0
        touchdown_pitch = 0.0
        touchdown_roll = 0.0
        peak_hip_tau = 0.0
        peak_knee_tau = 0.0
        peak_wheel_tau = 0.0
        airborne_steps = 0
        abnormal_resets = 0

        initial_pos_y = env.root_states[0, 1].item()
        initial_yaw = 0.0
        final_yaw = 0.0
        recovery_vel_y = []
        recovery_vel_x = []

        took_off_event = False
        touched_down_event = False

        for step in range(total_steps):
            t_sim = step * env.dt

            # 读取状态
            r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
            roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
            pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
            yaw = torch.atan2(torch.sin(y_raw[0]), torch.cos(y_raw[0])).item() * 180.0 / np.pi

            if step == 0:
                initial_yaw = yaw
            final_yaw = yaw

            root_pos = env.root_states[0, 0:3].cpu().numpy()
            base_lin_vel = env.base_lin_vel[0].cpu().numpy()
            is_air = bool(env.is_airborne[0].item())
            has_took_off = bool(env.has_taken_off[0].item())

            clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(env)
            clr_min_cm = clr_min[0].item() * 100.0
            clr_L_cm = clr_L[0].item() * 100.0
            clr_R_cm = clr_R[0].item() * 100.0

            fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
            fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()

            cur_hip_tau = max(abs(env.torques[0, env.left_hip_idx].item()), abs(env.torques[0, env.right_hip_idx].item()))
            cur_knee_tau = max(abs(env.torques[0, env.left_knee_idx].item()), abs(env.torques[0, env.right_knee_idx].item()))
            cur_wheel_tau = max(abs(env.torques[0, env.left_wheel_idx].item()), abs(env.torques[0, env.right_wheel_idx].item()))

            peak_hip_tau = max(peak_hip_tau, cur_hip_tau)
            peak_knee_tau = max(peak_knee_tau, cur_knee_tau)
            peak_wheel_tau = max(peak_wheel_tau, cur_wheel_tau)

            if is_air and has_took_off:
                airborne_steps += 1
                max_clearance = max(max_clearance, clr_min_cm)

            # 真实爆发起跳瞬间事件 (由环境物理流判定 has_taken_off)
            if is_air and has_took_off and not took_off_event:
                took_off_event = True
                takeoff_vy = env.takeoff_vy[0].item() if hasattr(env, 'takeoff_vy') else base_lin_vel[1]
                takeoff_vz = env.takeoff_vz[0].item() if hasattr(env, 'takeoff_vz') else base_lin_vel[2]

            # 真实触地瞬间事件
            if took_off_event and (not is_air) and (not touched_down_event):
                touched_down_event = True
                touchdown_pitch = pitch
                touchdown_roll = roll

            # 落地平稳恢复期 (phase in [0.45, 0.95])
            cur_phase = env.jump_phase[0].item()
            if cur_phase >= 0.45 and cur_phase <= 0.95 and touched_down_event:
                recovery_vel_y.append(base_lin_vel[1])
                recovery_vel_x.append(base_lin_vel[0])

            # 记录 CSV
            if record_csv:
                csv_writer.writerow([
                    f"{t_sim:.3f}", step, f"{cur_phase:.4f}",
                    f"{root_pos[0]:.4f}", f"{root_pos[1]:.4f}", f"{root_pos[2]:.4f}",
                    f"{base_lin_vel[0]:.4f}", f"{base_lin_vel[1]:.4f}", f"{base_lin_vel[2]:.4f}",
                    f"{roll:.2f}", f"{pitch:.2f}", f"{yaw:.2f}",
                    int(is_air), int(has_took_off),
                    f"{clr_min_cm:.2f}", f"{clr_L_cm:.2f}", f"{clr_R_cm:.2f}",
                    f"{fn_L:.2f}", f"{fn_R:.2f}",
                    f"{env.flight_distance_y[0].item():.4f}", f"{env.final_flight_distance[0].item():.4f}",
                    f"{env.torques[0, env.left_hip_idx].item():.2f}", f"{env.torques[0, env.left_knee_idx].item():.2f}",
                    f"{env.torques[0, env.right_hip_idx].item():.2f}", f"{env.torques[0, env.right_knee_idx].item():.2f}",
                    f"{env.torques[0, env.left_wheel_idx].item():.2f}", f"{env.torques[0, env.right_wheel_idx].item():.2f}"
                ])

            # 视频帧截取
            if record_video:
                cam_target = gymapi.Vec3(root_pos[0], root_pos[1], root_pos[2] + 0.15)
                cam_eye = gymapi.Vec3(root_pos[0] + cam_offset[0], root_pos[1] + cam_offset[1], root_pos[2] + cam_offset[2])
                env.gym.set_camera_location(camera_handle, env.envs[0], cam_eye, cam_target)
                env.gym.render_all_camera_sensors(env.sim)
                img = env.gym.get_camera_image(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
                img = img.reshape((camera_props.height, camera_props.width, 4))[:, :, :3]
                frames.append(img)

            # 策略单步推进
            obs = env.get_observations()
            with torch.no_grad():
                actions = policy(obs)
            env.step(actions)

            if env.reset_buf[0].item() != 0 and step < total_steps - 5:
                abnormal_resets += 1

        if record_csv:
            csv_file_obj.close()
            print(f"✅ 遥测数据已保存至: {csv_path}")

        if record_video:
            print(f"🎥 正在将 {len(frames)} 帧合成高清演示视频: {video_path}...")
            imageio.mimsave(video_path, frames, fps=50, quality=8)
            print(f"✅ 视频生成完毕: {video_path} ({os.path.getsize(video_path)/1024/1024:.2f} MB)")

        # 统计指标
        authentic_flight_distance = env.final_flight_distance[0].item()
        mean_recovery_vy = float(np.mean(recovery_vel_y)) if len(recovery_vel_y) > 0 else 0.0
        mean_recovery_vx = float(np.mean(recovery_vel_x)) if len(recovery_vel_x) > 0 else 0.0
        yaw_drift = abs(final_yaw - initial_yaw)

        res_dict = {
            "model_path": m_path,
            "max_bilateral_clearance_cm": round(max_clearance, 2),
            "authentic_airborne_forward_distance_m": round(authentic_flight_distance, 4),
            "authentic_airborne_forward_distance_cm": round(authentic_flight_distance * 100.0, 2),
            "takeoff_horizontal_velocity_vy_mps": round(takeoff_vy, 3),
            "takeoff_vertical_velocity_vz_mps": round(takeoff_vz, 3),
            "landing_pitch_deg": round(touchdown_pitch, 2),
            "landing_roll_deg": round(touchdown_roll, 2),
            "post_landing_residual_vy_mps": round(mean_recovery_vy, 3),
            "post_landing_residual_vx_mps": round(mean_recovery_vx, 3),
            "heading_yaw_drift_deg": round(yaw_drift, 2),
            "peak_hip_torque_nm": round(peak_hip_tau, 2),
            "peak_knee_torque_nm": round(peak_knee_tau, 2),
            "peak_wheel_torque_nm": round(peak_wheel_tau, 2),
            "airborne_duration_s": round(airborne_steps * 0.02, 3),
            "airborne_steps": airborne_steps,
            "abnormal_resets": abnormal_resets
        }
        results_all[m_name] = res_dict

        print("\n" + "=" * 50)
        print(f"📈 {m_name} 关键定量评测成绩:")
        print(f"  * 真实双轮最高净空   : {res_dict['max_bilateral_clearance_cm']} cm")
        print(f"  * 离地到触地前向距离 : {res_dict['authentic_airborne_forward_distance_cm']} cm ({res_dict['authentic_airborne_forward_distance_m']} m)")
        print(f"  * 起跳瞬间水平速度   : {res_dict['takeoff_horizontal_velocity_vy_mps']} m/s")
        print(f"  * 起跳瞬间竖直速度   : {res_dict['takeoff_vertical_velocity_vz_mps']} m/s")
        print(f"  * 触地俯仰角 (Pitch) : {res_dict['landing_pitch_deg']} deg")
        print(f"  * 触地翻滚角 (Roll)  : {res_dict['landing_roll_deg']} deg")
        print(f"  * 落地后残余水平速度 : {res_dict['post_landing_residual_vy_mps']} m/s (侧向 {res_dict['post_landing_residual_vx_mps']} m/s)")
        print(f"  * 航向偏差 (Yaw drift): {res_dict['heading_yaw_drift_deg']} deg")
        print(f"  * 关节峰值力矩       : Hip={res_dict['peak_hip_torque_nm']}Nm, Knee={res_dict['peak_knee_torque_nm']}Nm, Wheel={res_dict['peak_wheel_torque_nm']}Nm")
        print(f"  * 异常重置次数       : {res_dict['abnormal_resets']}")
        print("=" * 50)

    # 汇总写入 JSON
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 完整对比评测汇总已保存至: {summary_path}")

    return results_all


if __name__ == "__main__":
    run_evaluation()
