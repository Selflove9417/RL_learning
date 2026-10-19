#!/usr/bin/env python3
"""
MyRobot Phase 4.9: 3D GUI Visualizer & Player for Pilot C Takeoff + Pilot B Recovery (C/B)
========================================================================================
Interactive 3D simulation player running the dual-policy forward jump pipeline:
  - Phase 15 Stationary Balance Settling (phase15_m130.pt)
  - Pilot C Takeoff & Airborne Propulsion (model_pilot_c_step1.pt)
  - Airborne Hip Compensation (+0.10 rad)
  - Pilot B Touchdown Recovery with ZeroCmd (model_pilot_b_step1.pt, obs[9] = 0.0)

Keybindings (in 3D Viewer Window):
  [Space] / [R] : Re-trigger / Reset Jump Cycle
  [C]           : Switch Camera Angle (Side / Isometric / Rear / Free View)
  [P]           : Pause / Resume Simulation
  [ESC]         : Exit Visualizer
"""

import os
import sys
import time
import argparse
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse
import torch

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.envs.myrobot.myrobot_dual_env import compute_exact_cylinder_clearance, compute_leg_ik


def create_player(seed=3001, mode="C/B", auto_replay=True, hip_bias=0.10, phase_threshold=0.22, zero_cmd=True, headless=False, max_episodes=None):
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["play_pilot_c_b", "--task", task_name, "--num_envs", "1"]
    args = get_args()
    args.headless = headless
    args.seed = seed

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.normalization.clip_actions = 100.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.noise.add_noise = False

    print(f"\n==================================================================")
    print(f" 🚀 MyRobot 3D GUI Visualizer: Pilot C Takeoff + Pilot B Recovery")
    print(f"==================================================================")
    print(f" Policy Mode    : {mode}")
    print(f" Simulation Seed: {seed}")
    print(f" Auto-Replay    : {auto_replay}")
    print(f" Hip Bias       : {hip_bias:+.2f} rad (phase >= {phase_threshold})")
    print(f" ZeroCmd Mask   : {zero_cmd} (Slot 9 = 0.0 post-touchdown)")
    print(f"==================================================================\n")

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # Subscribe Viewer Keyboard Events
    if env.viewer is not None:
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_SPACE, "JUMP")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R, "RESET")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_C, "CAMERA")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_P, "PAUSE")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_ESCAPE, "QUIT")

    # 1. Load Phase 15 Balance Policy
    bal_model_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase15/models/phase15_m130.pt"
    ac_balance = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ckpt_bal = torch.load(bal_model_path, map_location=env.device)
    ac_balance.load_state_dict(ckpt_bal["model_state_dict"])
    ac_balance.eval()

    # 2. Load Pilot B and Pilot C Models
    pilot_b_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase4_8_pilot_b/model_pilot_b_step1.pt"
    pilot_c_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase4_9_pilot_c/model_pilot_c_step1.pt"

    ac_pilot_b = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ckpt_b = torch.load(pilot_b_path, map_location=env.device)
    ac_pilot_b.load_state_dict(ckpt_b["model_state_dict"])
    ac_pilot_b.eval()

    ac_pilot_c = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ckpt_c = torch.load(pilot_c_path, map_location=env.device)
    ac_pilot_c.load_state_dict(ckpt_c["model_state_dict"])
    ac_pilot_c.eval()

    # Select Active Takeoff & Recovery Policies
    mode_upper = mode.upper()
    if mode_upper == "C/B":
        ac_takeoff = ac_pilot_c
        ac_recovery = ac_pilot_b
        takeoff_name = "Pilot C"
        recovery_name = "Pilot B"
    elif mode_upper == "C/C":
        ac_takeoff = ac_pilot_c
        ac_recovery = ac_pilot_c
        takeoff_name = "Pilot C"
        recovery_name = "Pilot C"
    elif mode_upper == "B/B":
        ac_takeoff = ac_pilot_b
        ac_recovery = ac_pilot_b
        takeoff_name = "Pilot B"
        recovery_name = "Pilot B"
    elif mode_upper == "B/C":
        ac_takeoff = ac_pilot_b
        ac_recovery = ac_pilot_c
        takeoff_name = "Pilot B"
        recovery_name = "Pilot C"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Camera Modes
    cam_modes = ["SIDE_VIEW", "ISOMETRIC", "REAR_VIEW", "FREE_CAMERA"]
    cam_mode_idx = 0

    def update_camera():
        if env.viewer is None or cam_modes[cam_mode_idx] == "FREE_CAMERA":
            return
        root_pos = env.root_states[0, 0:3].cpu().numpy()
        cam_target = gymapi.Vec3(root_pos[0], root_pos[1], root_pos[2] + 0.15)
        if cam_modes[cam_mode_idx] == "SIDE_VIEW":
            cam_eye = gymapi.Vec3(root_pos[0] - 2.2, root_pos[1], root_pos[2] + 0.35)
        elif cam_modes[cam_mode_idx] == "ISOMETRIC":
            cam_eye = gymapi.Vec3(root_pos[0] - 1.6, root_pos[1] - 1.8, root_pos[2] + 0.70)
        elif cam_modes[cam_mode_idx] == "REAR_VIEW":
            cam_eye = gymapi.Vec3(root_pos[0], root_pos[1] - 2.2, root_pos[2] + 0.50)
        env.gym.viewer_camera_look_at(env.viewer, env.envs[0], cam_eye, cam_target)

    def render_graphics():
        if env.viewer is None:
            return
        if env.device != 'cpu':
            env.gym.fetch_results(env.sim, True)
        update_camera()
        env.gym.step_graphics(env.sim)
        env.gym.draw_viewer(env.viewer, env.sim, True)
        env.gym.sync_frame_time(env.sim)

    def is_viewer_closed():
        if env.viewer is None:
            return False
        return env.gym.query_viewer_has_closed(env.viewer)

    def query_action_events():
        if env.viewer is None:
            return []
        return env.gym.query_viewer_action_events(env.viewer)

    paused = False
    episode_num = 1

    print(" 🕹️  控制指南:")
    print("   [Space] / [R] : 立即重新执行跳跃")
    print("   [C]           : 切换摄像机视点 (侧视 / 等轴 / 后视 / 自由鼠标视角)")
    print("   [P]           : 暂停 / 继续仿真")
    print("   [ESC]         : 退出窗口")
    print("------------------------------------------------------------------\n")

    while not is_viewer_closed():
        if max_episodes is not None and episode_num > max_episodes:
            break
        print(f"\n>>> [Episode {episode_num}] 开始阶段 1: Phase 15 原地平稳站立沉降...")

        # -------------------------------------------------------------
        # Stage 1: Phase 15 Stationary Balance Settling
        # -------------------------------------------------------------
        env.reset_idx(torch.arange(env.num_envs, device=env.device))
        env.root_states[:, 2] = 0.397739
        env.root_states[:, 7:13] = 0.0
        env.dof_vel[:, :] = 0.0
        env.rb_states[:, :, 7:13] = 0.0
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)

        bal_filtered_actions = torch.zeros(env.num_envs, 6, device=env.device)
        pi_integral = torch.zeros(env.num_envs, device=env.device)
        pi_v_filt = torch.zeros(env.num_envs, device=env.device)

        max_settle_steps = int(1.2 / env.dt)
        stable_consecutive_steps = 0
        h_ctrl = 0.39774
        norm_h = ((h_ctrl - 0.40) / 0.15)
        norm_h_tensor = torch.tensor([[norm_h]], device=env.device)

        user_requested_quit = False
        user_requested_reset = False

        for step in range(max_settle_steps):
            if is_viewer_closed():
                user_requested_quit = True
                break

            # Handle UI Events
            for evt in query_action_events():
                if evt.value > 0.5:
                    if evt.action == "QUIT":
                        user_requested_quit = True
                    elif evt.action == "CAMERA":
                        cam_mode_idx = (cam_mode_idx + 1) % len(cam_modes)
                        print(f" 🎥 切换视角: {cam_modes[cam_mode_idx]}")
                    elif evt.action == "PAUSE":
                        paused = not paused
                        print(f" ⏸️  仿真状态: {'已暂停' if paused else '继续运行'}")
                    elif evt.action in ["RESET", "JUMP"]:
                        user_requested_reset = True

            if user_requested_quit:
                break
            if user_requested_reset:
                break

            while paused:
                for evt in query_action_events():
                    if evt.value > 0.5:
                        if evt.action == "PAUSE":
                            paused = False
                            print(" ▶️  继续仿真")
                        elif evt.action == "CAMERA":
                            cam_mode_idx = (cam_mode_idx + 1) % len(cam_modes)
                            print(f" 🎥 切换视角: {cam_modes[cam_mode_idx]}")
                        elif evt.action == "QUIT":
                            user_requested_quit = True
                            paused = False
                render_graphics()
                time.sleep(0.02)
                if user_requested_quit:
                    break

            if user_requested_quit:
                break

            # Balance Policy Step
            dof_pos_bal = (env.dof_pos - env.default_dof_pos).clone()
            dof_pos_bal[:, env.wheel_indices] = 0.0

            v_fwd = env.base_lin_vel[:, 1]
            pi_v_filt = 0.85 * pi_v_filt + 0.15 * v_fwd
            err_v = 0.0 - pi_v_filt
            pi_integral = torch.clamp(pi_integral + err_v * env.dt, -0.075, 0.075)
            cmd_trim = torch.clamp(0.35 * err_v + 0.60 * pi_integral, -0.045, 0.045)

            bal_cmds = torch.zeros(env.num_envs, 3, device=env.device)
            bal_cmds[:, 1] = cmd_trim

            bal_obs = torch.cat([
                env.base_lin_vel * 2.0,
                env.base_ang_vel * 0.25,
                env.projected_gravity,
                bal_cmds * torch.tensor([2.0, 2.0, 0.25], device=env.device),
                dof_pos_bal * 1.0,
                env.dof_vel * 0.05,
                bal_filtered_actions,
                norm_h_tensor
            ], dim=-1)

            with torch.no_grad():
                act_bal_raw = ac_balance.act_inference(bal_obs)
            bal_filtered_actions = 0.80 * act_bal_raw + 0.20 * bal_filtered_actions

            q1_ik, q2_ik = compute_leg_ik(torch.tensor([h_ctrl], device=env.device))
            tgt_hip_bal_L = env.default_dof_pos[:, env.left_hip_idx] + q1_ik + 0.25 * 0.7 * bal_filtered_actions[:, env.left_hip_idx]
            tgt_knee_bal_L = env.default_dof_pos[:, env.left_knee_idx] + q2_ik + 0.25 * 0.5 * bal_filtered_actions[:, env.left_knee_idx]
            tgt_hip_bal_R = env.default_dof_pos[:, env.right_hip_idx] + q1_ik + 0.25 * 0.7 * bal_filtered_actions[:, env.right_hip_idx]
            tgt_knee_bal_R = env.default_dof_pos[:, env.right_knee_idx] + q2_ik + 0.25 * 0.5 * bal_filtered_actions[:, env.right_knee_idx]

            base_tau_L = 7.0 * torch.clamp(bal_filtered_actions[:, env.left_wheel_idx], -1.0, 1.0) - 0.3 * env.dof_vel[:, env.left_wheel_idx]
            base_tau_R = 7.0 * torch.clamp(bal_filtered_actions[:, env.right_wheel_idx], -1.0, 1.0) - 0.3 * env.dof_vel[:, env.right_wheel_idx]
            wheel_speed_error = env.dof_vel[:, env.left_wheel_idx] - env.dof_vel[:, env.right_wheel_idx]
            delta_tau_omega = torch.clamp(0.10 * wheel_speed_error, -1.5, 1.5)

            tau_limit_w = env.torque_limits[env.left_wheel_idx].item()
            tau_c = 0.5 * (base_tau_L + base_tau_R)
            tau_d = 0.5 * (base_tau_R - base_tau_L) + delta_tau_omega

            tau_d_clamped = torch.clamp(tau_d, -tau_limit_w, tau_limit_w)
            tau_c_limit = torch.clamp(torch.tensor([tau_limit_w], device=env.device) - torch.abs(tau_d_clamped), min=0.0)
            tau_c_allocated = torch.clamp(tau_c, -tau_c_limit, tau_c_limit)

            tau_w_L = tau_c_allocated - tau_d_clamped
            tau_w_R = tau_c_allocated + tau_d_clamped

            kp_leg = 40.0
            kd_leg = 2.0

            for _ in range(env.cfg.control.decimation):
                torques = torch.zeros(env.num_envs, env.num_actions, device=env.device)
                torques[:, env.left_hip_idx] = kp_leg * (tgt_hip_bal_L - env.dof_pos[:, env.left_hip_idx]) - kd_leg * env.dof_vel[:, env.left_hip_idx]
                torques[:, env.right_hip_idx] = kp_leg * (tgt_hip_bal_R - env.dof_pos[:, env.right_hip_idx]) - kd_leg * env.dof_vel[:, env.right_hip_idx]
                torques[:, env.left_knee_idx] = kp_leg * (tgt_knee_bal_L - env.dof_pos[:, env.left_knee_idx]) - kd_leg * env.dof_vel[:, env.left_knee_idx]
                torques[:, env.right_knee_idx] = kp_leg * (tgt_knee_bal_R - env.dof_pos[:, env.right_knee_idx]) - kd_leg * env.dof_vel[:, env.right_knee_idx]
                torques[:, env.left_wheel_idx] = tau_w_L
                torques[:, env.right_wheel_idx] = tau_w_R
                torques = torch.clip(torques, -env.torque_limits, env.torque_limits)

                env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(torques))
                env.gym.simulate(env.sim)
                env.gym.fetch_results(env.sim, True)
                env.gym.refresh_dof_state_tensor(env.sim)

            env.gym.refresh_actor_root_state_tensor(env.sim)
            env.gym.refresh_net_contact_force_tensor(env.sim)
            env.gym.refresh_rigid_body_state_tensor(env.sim)

            env.base_quat[:] = env.root_states[:, 3:7]
            env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
            env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
            env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)

            render_graphics()

            b_vy = env.base_lin_vel[0, 1].item()
            b_vz = env.base_lin_vel[0, 2].item()
            _, true_com_vel = env._compute_true_com()
            c_vy = true_com_vel[0, 1].item()
            fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
            fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()
            r_raw, p_raw, _ = get_euler_xyz(env.base_quat)
            pitch_deg = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
            roll_deg = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi

            is_stable_now = (abs(b_vy) <= 0.03) and (abs(c_vy) <= 0.03) and (abs(b_vz) <= 0.05) and (fn_L >= 20.0 and fn_L <= 180.0) and (fn_R >= 20.0 and fn_R <= 180.0) and (abs(pitch_deg) <= 2.5) and (abs(roll_deg) <= 3.5)
            if is_stable_now:
                stable_consecutive_steps += 1
            else:
                stable_consecutive_steps = 0

            if stable_consecutive_steps >= 5 and step >= 15:
                break

        if user_requested_quit:
            break
        if user_requested_reset:
            continue

        # -------------------------------------------------------------
        # Stage 2: Jump & Recovery Execution
        # -------------------------------------------------------------
        print(f">>> [Episode {episode_num}] 开始阶段 2: 前向跳跃启动 (起跳: {takeoff_name} -> 落地: {recovery_name})")

        env.jump_timer[:] = 0.0
        env.jump_phase[:] = 0.0
        env.filtered_actions[:] = 0.0
        env.last_actions[:] = 0.0
        env.episode_length_buf[:] = 0
        env.jump_latched[:] = False
        env.has_taken_off[:] = False
        env.has_touched_down[:] = False
        env.one_shot_touchdown_event[:] = False

        env.compute_observations()
        obs = env.obs_buf.clone()

        total_jump_steps = int(2.8 / env.dt)  # 140 steps
        took_off = False
        landed = False
        takeoff_step = None
        touchdown_step = None
        max_clearance_cm = 0.0
        takeoff_com_y = 0.0
        touchdown_com_y = 0.0
        post_rec_vys = []
        td_pitch = 0.0
        td_roll = 0.0

        for step in range(total_jump_steps):
            if is_viewer_closed():
                user_requested_quit = True
                break

            # Handle UI Events
            for evt in query_action_events():
                if evt.value > 0.5:
                    if evt.action == "QUIT":
                        user_requested_quit = True
                    elif evt.action == "CAMERA":
                        cam_mode_idx = (cam_mode_idx + 1) % len(cam_modes)
                        print(f" 🎥 切换视角: {cam_modes[cam_mode_idx]}")
                    elif evt.action == "PAUSE":
                        paused = not paused
                        print(f" ⏸️  仿真状态: {'已暂停' if paused else '继续运行'}")
                    elif evt.action in ["RESET", "JUMP"]:
                        user_requested_reset = True

            if user_requested_quit or user_requested_reset:
                break

            while paused:
                for evt in query_action_events():
                    if evt.value > 0.5:
                        if evt.action == "PAUSE":
                            paused = False
                            print(" ▶️  继续仿真")
                        elif evt.action == "CAMERA":
                            cam_mode_idx = (cam_mode_idx + 1) % len(cam_modes)
                            print(f" 🎥 切换视角: {cam_modes[cam_mode_idx]}")
                        elif evt.action == "QUIT":
                            user_requested_quit = True
                            paused = False
                render_graphics()
                time.sleep(0.02)
                if user_requested_quit:
                    break

            if user_requested_quit or user_requested_reset:
                break

            true_com_pos, true_com_vel = env._compute_true_com()
            b_vel = env.base_lin_vel[0].cpu().numpy()
            r_raw, p_raw, _ = get_euler_xyz(env.base_quat)
            cur_pitch_deg = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
            cur_roll_deg = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi

            clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(env)
            clr_min_cm = clr_min[0].item() * 100.0
            is_air = bool(env.is_airborne[0].item())
            has_took = bool(env.has_taken_off[0].item())

            if is_air and has_took:
                max_clearance_cm = max(max_clearance_cm, clr_min_cm)

            # Takeoff detection
            if is_air and has_took and not took_off:
                took_off = True
                takeoff_step = step
                takeoff_com_y = float(true_com_pos[0, 1].item())
                print(f"   🛫 [Step {step:3d}] 起飞离地! (Takeoff via {takeoff_name})")

            # Touchdown detection
            if took_off and (not is_air) and (not landed):
                landed = True
                touchdown_step = step
                touchdown_com_y = float(true_com_pos[0, 1].item())
                td_pitch = cur_pitch_deg
                td_roll = cur_roll_deg
                print(f"   🛬 [Step {step:3d}] 触地落地! (Touchdown -> 切换至 {recovery_name} + ZeroCmd)")

            # Policy Selection & Observation Masking
            obs_for_actor = obs.clone()
            if zero_cmd and landed:
                obs_for_actor[:, 9] = 0.0

            if not landed:
                current_ac = ac_takeoff
                policy_tag = takeoff_name
            else:
                current_ac = ac_recovery
                policy_tag = f"{recovery_name}+ZeroCmd"

            with torch.no_grad():
                actions = current_ac.act_inference(obs_for_actor)

            # Airborne Descent Hip Compensation
            if is_air and (env.jump_phase[0].item() >= phase_threshold):
                actions[:, env.left_hip_idx] += hip_bias
                actions[:, env.right_hip_idx] += hip_bias

            obs, _, _, dones, _ = env.step(actions)
            render_graphics()

            # Record post-touchdown speed
            if landed and step > touchdown_step + 5:
                post_rec_vys.append(float(b_vel[1]))

            # Real-time HUD print
            flight_dist_now = (float(true_com_pos[0, 1].item()) - takeoff_com_y) * 100.0 if took_off else 0.0
            tau_c = 0.5 * (env.torques[0, env.left_wheel_idx].item() + env.torques[0, env.right_wheel_idx].item())
            print(f"\r   [{step*env.dt:4.2f}s | Step {step:3d}] 策略: {policy_tag:<18} | Clr: {clr_min_cm:5.1f}cm (Max: {max_clearance_cm:5.1f}cm) | Dist: {flight_dist_now:5.1f}cm | Vy: {b_vel[1]:+5.3f}m/s | Pitch: {cur_pitch_deg:+5.1f}° | Tau_c: {tau_c:+5.1f}Nm", end="", flush=True)

        print()

        if user_requested_quit:
            break
        if user_requested_reset:
            continue

        # -------------------------------------------------------------
        # Stage 3: Scorecard & Verification
        # -------------------------------------------------------------
        final_flight_dist_cm = (touchdown_com_y - takeoff_com_y) * 100.0 if (took_off and landed) else 0.0
        c4_vel = float(np.mean([abs(v) for v in post_rec_vys[-10:]])) if len(post_rec_vys) >= 10 else float(abs(env.base_lin_vel[0, 1].item()))

        pass_c0 = bool(took_off and landed)
        pass_c1 = bool(final_flight_dist_cm >= 12.0)
        pass_c2 = bool(max_clearance_cm >= 20.0)
        pass_c3 = bool(abs(td_pitch) <= 15.0 and abs(td_roll) <= 15.0)
        pass_c4 = bool(c4_vel <= 0.030)
        all_pass = pass_c0 and pass_c1 and pass_c2 and pass_c3 and pass_c4

        print("\n" + "=" * 68)
        print(f" 📊 [Episode {episode_num} 结果总评: 模式 {mode} | Seed {seed}]")
        print("=" * 68)
        print(f" C0 (离地与着陆)   : {'✅ PASS' if pass_c0 else '❌ FAIL'} (Takeoff step: {takeoff_step}, Touchdown step: {touchdown_step})")
        print(f" C1 (前向腾空距离) : {'✅ PASS' if pass_c1 else '❌ FAIL'} -> {final_flight_dist_cm:6.2f} cm  (门限: >= 12.0 cm)")
        print(f" C2 (双轮最大净空) : {'✅ PASS' if pass_c2 else '❌ FAIL'} -> {max_clearance_cm:6.2f} cm  (门限: >= 20.0 cm)")
        print(f" C3 (触地俯仰/横滚): {'✅ PASS' if pass_c3 else '❌ FAIL'} -> Pitch: {td_pitch:+5.1f}°, Roll: {td_roll:+5.1f}° (门限: <= 15.0°)")
        print(f" C4 (末端平稳刹停) : {'✅ PASS' if pass_c4 else '❌ FAIL'} -> {c4_vel:6.4f} m/s (门限: <= 0.030 m/s)")
        print(f" 全指标联合达标    : {'🏆 100% 联合达标 (JOINT PASS)' if all_pass else '⚠️ 未达标'}")
        print("=" * 68)

        episode_num += 1

        if not auto_replay:
            print(" 按 [Space] 或 [R] 开始下一次跳跃，按 [ESC] 退出...")
            wait_for_next = True
            while wait_for_next and not is_viewer_closed():
                for evt in query_action_events():
                    if evt.value > 0.5:
                        if evt.action in ["RESET", "JUMP"]:
                            wait_for_next = False
                        elif evt.action == "QUIT":
                            user_requested_quit = True
                            wait_for_next = False
                        elif evt.action == "CAMERA":
                            cam_mode_idx = (cam_mode_idx + 1) % len(cam_modes)
                            print(f" 🎥 切换视角: {cam_modes[cam_mode_idx]}")
                render_graphics()
                time.sleep(0.02)
            if user_requested_quit:
                break
        else:
            print(" ⏳ 保持驻车观察 1.5 秒后自动回放下一轮... (按 [Space]/[R] 立即回放, [P] 暂停, [ESC] 退出)")
            t_pause_start = time.time()
            replaying_now = False
            while (time.time() - t_pause_start < 1.5) and not is_viewer_closed():
                for evt in query_action_events():
                    if evt.value > 0.5:
                        if evt.action in ["RESET", "JUMP"]:
                            replaying_now = True
                            break
                        elif evt.action == "QUIT":
                            user_requested_quit = True
                            break
                        elif evt.action == "CAMERA":
                            cam_mode_idx = (cam_mode_idx + 1) % len(cam_modes)
                            print(f" 🎥 切换视角: {cam_modes[cam_mode_idx]}")
                if replaying_now or user_requested_quit:
                    break
                render_graphics()
                time.sleep(0.02)
            if user_requested_quit:
                break

    print("\n👋 3D GUI Visualizer 已退出。")


def main():
    parser = argparse.ArgumentParser(description="MyRobot 3D GUI Visualizer for Pilot C Takeoff + Pilot B Recovery")
    parser.add_argument("--seed", type=int, default=3001, help="Simulation seed (default: 3001)")
    parser.add_argument("--mode", type=str, default="C/B", choices=["C/B", "C/C", "B/B", "B/C"], help="Policy combination (default: C/B)")
    parser.add_argument("--no_loop", action="store_true", help="Disable automatic replay loop")
    parser.add_argument("--hip_bias", type=float, default=0.10, help="Aerial descent hip bias (default: +0.10)")
    parser.add_argument("--phase_threshold", type=float, default=0.22, help="Phase threshold for hip bias (default: 0.22)")
    parser.add_argument("--no_zero_cmd", action="store_true", help="Disable ZeroCmd post-touchdown")
    parser.add_argument("--headless", action="store_true", help="Run without 3D window (for batch testing)")
    parser.add_argument("--max_episodes", type=int, default=None, help="Maximum jump episodes to play")
    args = parser.parse_args()

    create_player(
        seed=args.seed,
        mode=args.mode,
        auto_replay=not args.no_loop,
        hip_bias=args.hip_bias,
        phase_threshold=args.phase_threshold,
        zero_cmd=not args.no_zero_cmd,
        headless=args.headless,
        max_episodes=args.max_episodes
    )


if __name__ == "__main__":
    main()
