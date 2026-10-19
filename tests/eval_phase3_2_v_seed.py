#!/usr/bin/env python3
# ==============================================================================
# Phase 3.2-V: Single Seed Comprehensive Evaluator for Candidate Freeze & Unseen Seeds
# Fully records all 12 telemetry dimensions + handoff transient + compensation apex check
# ==============================================================================

import os
import sys
import json
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

def eval_seed_full(seed, hip_bias=0.10, phase_threshold=0.22, out_path=""):
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["eval_candidate", "--task", task_name, "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = seed

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.normalization.clip_actions = 100.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.noise.add_noise = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # 1. Load Phase 15 Balance Policy
    bal_model_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase15/models/phase15_m130.pt"
    ac_balance = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ckpt_bal = torch.load(bal_model_path, map_location=env.device)
    ac_balance.load_state_dict(ckpt_bal["model_state_dict"])
    ac_balance.eval()

    # 2. Load Champion Forward Jump Policy
    jump_model_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    ac_jump = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ckpt_jump = torch.load(jump_model_path, map_location=env.device)
    ac_jump.load_state_dict(ckpt_jump["model_state_dict"])
    ac_jump.eval()

    # Place robot at tangent ground contact pose z = 0.397739
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

    # Balance buffers
    bal_filtered_actions = torch.zeros(env.num_envs, 6, device=env.device)
    pi_integral = torch.zeros(env.num_envs, device=env.device)
    pi_v_filt = torch.zeros(env.num_envs, device=env.device)

    max_settle_steps = int(1.5 / env.dt)
    stable_consecutive_steps = 0
    stable_reached = False
    settle_step_reached = 0

    h_ctrl = 0.39774
    norm_h = ((h_ctrl - 0.40) / 0.15)
    norm_h_tensor = torch.tensor([[norm_h]], device=env.device)

    last_bal_targets = torch.zeros(env.num_envs, 6, device=env.device)
    last_bal_torques = torch.zeros(env.num_envs, 6, device=env.device)

    init_com_y = None
    init_wheel_y = None

    # Balance Settling Phase
    for step in range(max_settle_steps):
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

        last_bal_targets[0, env.left_hip_idx] = tgt_hip_bal_L
        last_bal_targets[0, env.left_knee_idx] = tgt_knee_bal_L
        last_bal_targets[0, env.right_hip_idx] = tgt_hip_bal_R
        last_bal_targets[0, env.right_knee_idx] = tgt_knee_bal_R

        base_tau_L = 7.0 * torch.clamp(bal_filtered_actions[:, env.left_wheel_idx], -1.0, 1.0) - 0.3 * env.dof_vel[:, env.left_wheel_idx]
        base_tau_R = 7.0 * torch.clamp(bal_filtered_actions[:, env.right_wheel_idx], -1.0, 1.0) - 0.3 * env.dof_vel[:, env.right_wheel_idx]
        wheel_speed_error = env.dof_vel[:, env.left_wheel_idx] - env.dof_vel[:, env.right_wheel_idx]
        delta_tau_omega = torch.clamp(0.10 * wheel_speed_error, -1.5, 1.5)

        tau_c = 0.5 * (base_tau_L + base_tau_R)
        tau_d = 0.5 * (base_tau_R - base_tau_L) + delta_tau_omega

        tau_limit_w = env.torque_limits[env.left_wheel_idx]
        tau_d_clamped = torch.clamp(tau_d, -tau_limit_w, tau_limit_w)
        tau_c_limit = torch.clamp(tau_limit_w - torch.abs(tau_d_clamped), min=0.0)
        tau_c_allocated = torch.clamp(tau_c, -tau_c_limit, tau_c_limit)

        tau_w_L = tau_c_allocated - tau_d_clamped
        tau_w_R = tau_c_allocated + tau_d_clamped

        for _ in range(env.cfg.control.decimation):
            torques = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            kp_leg = 40.0
            kd_leg = 2.0
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

        last_bal_torques[:] = torques
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        if step == 0:
            init_com_pos, _ = env._compute_true_com()
            init_com_y = init_com_pos[0, 1].item()
            init_wL_y = env.rb_states[0, env.left_wheel_body, 1].item()
            init_wR_y = env.rb_states[0, env.right_wheel_body, 1].item()
            init_wheel_y = 0.5 * (init_wL_y + init_wR_y)

        env.base_quat[:] = env.root_states[:, 3:7]
        env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
        env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
        env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)

        b_vy = env.base_lin_vel[0, 1].item()
        b_vz = env.base_lin_vel[0, 2].item()
        _, true_com_vel = env._compute_true_com()
        c_vy = true_com_vel[0, 1].item()
        c_vz = true_com_vel[0, 2].item()
        fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
        fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()
        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        pitch_deg = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        roll_deg = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi

        is_stable_now = (abs(b_vy) <= 0.03) and (abs(c_vy) <= 0.03) and (abs(b_vz) <= 0.05) and (fn_L >= 20.0 and fn_L <= 180.0) and (fn_R >= 20.0 and fn_R <= 180.0) and (abs(pitch_deg) <= 2.5) and (abs(roll_deg) <= 3.5)
        if is_stable_now:
            stable_consecutive_steps += 1
        else:
            stable_consecutive_steps = 0

        if stable_consecutive_steps >= 5 and step >= 15:
            stable_reached = True
            settle_step_reached = step + 1
            break

    if not stable_reached:
        settle_step_reached = max_settle_steps

    # Record Handoff State
    handoff_com_pos, handoff_com_vel = env._compute_true_com()
    handoff_com_y = handoff_com_pos[0, 1].item()
    dwell_com_disp_cm = (handoff_com_y - init_com_y) * 100.0

    handoff_wheel_L_y = env.rb_states[0, env.left_wheel_body, 1].item()
    handoff_wheel_R_y = env.rb_states[0, env.right_wheel_body, 1].item()
    handoff_wheel_y = 0.5 * (handoff_wheel_L_y + handoff_wheel_R_y)
    dwell_wheel_disp_cm = (handoff_wheel_y - init_wheel_y) * 100.0

    handoff_base_vy = env.base_lin_vel[0, 1].item()
    handoff_com_vy = handoff_com_vel[0, 1].item()
    handoff_com_vz = handoff_com_vel[0, 2].item()
    handoff_base_ang_vel = env.base_ang_vel[0].cpu().numpy().tolist()
    handoff_fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
    handoff_fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()

    # Step 2: Jump Phase Initialization
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

    total_jump_steps = int(2.8 / env.dt)
    takeoff_step = None
    touchdown_step = None
    took_off = False
    landed = False
    max_clearance_cm = 0.0
    takeoff_com_y = 0.0
    touchdown_com_y = 0.0
    touchdown_pitch = 0.0
    touchdown_roll = 0.0
    touchdown_yaw = 0.0
    takeoff_com_vy = 0.0
    takeoff_com_vz = 0.0

    # Apex tracking
    apex_z = -999.0
    apex_step = None
    apex_t = None

    # Compensation activation tracking
    comp_activated = False
    comp_first_step = None
    comp_first_t = None
    comp_first_com_vz = None
    comp_is_after_apex = None

    # Pre-takeoff runup tracking
    pre_takeoff_vys = []
    pre_takeoff_wheel_ys = []
    pre_takeoff_com_ys = []
    prep_base_vys = []
    prep_com_vys = []
    prep_wheel_ys = []
    prep_com_ys = []

    # Post-touchdown recovery tracking
    post_rec_vys = []

    # Airborne attitude peaks
    airborne_pitches = []
    airborne_rolls = []
    airborne_yaws = []

    # Torques & saturation tracking
    peak_torques = torch.zeros(6, device=env.device)
    torque_saturation_counts = torch.zeros(6, device=env.device)
    torque_limits = env.torque_limits.clone()

    # Handoff transient checking
    step0_jump_targets = None
    step0_jump_torques = None
    handoff_delta_targets = None
    handoff_delta_torques = None

    abnormal_resets = 0

    for step in range(total_jump_steps):
        t_sim = step * env.dt
        true_com_pos, true_com_vel = env._compute_true_com()
        b_vel = env.base_lin_vel[0].cpu().numpy()
        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        cur_pitch_deg = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        cur_roll_deg = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
        cur_yaw_deg = torch.atan2(torch.sin(y_raw[0]), torch.cos(y_raw[0])).item() * 180.0 / np.pi

        clr_min, _, _ = compute_exact_cylinder_clearance(env)
        clr_min_cm = clr_min[0].item() * 100.0
        is_air = bool(env.is_airborne[0].item())
        has_took = bool(env.has_taken_off[0].item())

        # Pre-takeoff tracking
        cur_wL_y = env.rb_states[0, env.left_wheel_body, 1].item()
        cur_wR_y = env.rb_states[0, env.right_wheel_body, 1].item()
        cur_w_y = 0.5 * (cur_wL_y + cur_wR_y)

        if not took_off:
            pre_takeoff_vys.append(b_vel[1])
            pre_takeoff_wheel_ys.append(cur_w_y)
            pre_takeoff_com_ys.append(true_com_pos[0, 1].item())
            if step <= 5:
                prep_base_vys.append(abs(b_vel[1]))
                prep_com_vys.append(abs(true_com_vel[0, 1].item()))
                prep_wheel_ys.append(cur_w_y)
                prep_com_ys.append(true_com_pos[0, 1].item())

        # Apex tracking
        com_z = true_com_pos[0, 2].item()
        if is_air and has_took:
            if com_z > apex_z:
                apex_z = com_z
                apex_step = step
                apex_t = t_sim
            airborne_pitches.append(cur_pitch_deg)
            airborne_rolls.append(cur_roll_deg)
            airborne_yaws.append(cur_yaw_deg)
            max_clearance_cm = max(max_clearance_cm, clr_min_cm)

        # Takeoff detection
        if is_air and has_took and not took_off:
            took_off = True
            takeoff_step = step
            takeoff_t = t_sim
            takeoff_com_y = float(true_com_pos[0, 1].item())
            takeoff_com_vy = float(true_com_vel[0, 1].item())
            takeoff_com_vz = float(true_com_vel[0, 2].item())

        # Touchdown detection
        if took_off and (not is_air) and (not landed):
            landed = True
            touchdown_step = step
            touchdown_t = t_sim
            touchdown_com_y = float(true_com_pos[0, 1].item())
            touchdown_pitch = cur_pitch_deg
            touchdown_roll = cur_roll_deg
            touchdown_yaw = cur_yaw_deg

        # Post-touchdown recovery tracking (2 seconds window post touchdown)
        if landed and step > touchdown_step + 5:
            post_rec_vys.append(b_vel[1])

        # Step 0 Handoff transient capture
        with torch.no_grad():
            actions = ac_jump.act_inference(obs)

        if step == 0:
            step0_jump_targets = env.default_dof_pos + actions * env.cfg.control.action_scale
            handoff_delta_targets = (step0_jump_targets - last_bal_targets).abs().cpu().numpy().tolist()[0]

        # Single Minimal Intervention: Hip extension trim during descent
        comp_cond = is_air and (env.jump_phase[0].item() >= phase_threshold)
        if comp_cond and (hip_bias != 0.0):
            actions[:, env.left_hip_idx] += hip_bias
            actions[:, env.right_hip_idx] += hip_bias
            if not comp_activated:
                comp_activated = True
                comp_first_step = step
                comp_first_t = t_sim
                comp_first_com_vz = float(true_com_vel[0, 2].item())
                comp_is_after_apex = (comp_first_com_vz <= 0.0) or (apex_step is not None and step >= apex_step)

        obs, _, _, _, _ = env.step(actions)

        # Capture torques applied
        cur_torques = env.torques[0].abs()
        peak_torques = torch.maximum(peak_torques, cur_torques)
        saturated = (cur_torques >= 0.98 * torque_limits)
        torque_saturation_counts += saturated.float()

        if step == 0:
            step0_jump_torques = env.torques[0].clone()
            handoff_delta_torques = (step0_jump_torques - last_bal_torques[0]).abs().cpu().numpy().tolist()

        if env.reset_buf[0].item() != 0 and step < total_jump_steps - 5:
            abnormal_resets += 1

    # Metrics computation
    fwd_dist_cm = (touchdown_com_y - takeoff_com_y) * 100.0 if (took_off and landed) else 0.0
    rec_vy_last10 = float(np.mean(post_rec_vys[-10:])) if len(post_rec_vys) >= 10 else 0.0
    rec_vy_2s_mean = float(np.mean(post_rec_vys)) if len(post_rec_vys) > 0 else 0.0
    rec_vy_2s_std = float(np.std(post_rec_vys)) if len(post_rec_vys) > 0 else 0.0
    rec_vy_2s_max = float(np.max(np.abs(post_rec_vys))) if len(post_rec_vys) > 0 else 0.0

    pre_takeoff_max_vy = float(np.max(np.abs(pre_takeoff_vys))) if len(pre_takeoff_vys) > 0 else 0.0
    pre_takeoff_wheel_disp_cm = (pre_takeoff_wheel_ys[-1] - pre_takeoff_wheel_ys[0]) * 100.0 if len(pre_takeoff_wheel_ys) > 1 else 0.0
    pre_takeoff_com_disp_cm = (pre_takeoff_com_ys[-1] - pre_takeoff_com_ys[0]) * 100.0 if len(pre_takeoff_com_ys) > 1 else 0.0

    airborne_pitch_max = float(np.max(airborne_pitches)) if len(airborne_pitches) > 0 else 0.0
    airborne_pitch_min = float(np.min(airborne_pitches)) if len(airborne_pitches) > 0 else 0.0
    airborne_roll_max = float(np.max(np.abs(airborne_rolls))) if len(airborne_rolls) > 0 else 0.0
    airborne_yaw_drift = float(np.max(np.abs(airborne_yaws)) - np.abs(airborne_yaws[0])) if len(airborne_yaws) > 0 else 0.0

    prep_max_base_vy = float(np.max(prep_base_vys)) if len(prep_base_vys) > 0 else 0.0
    prep_max_com_vy = float(np.max(prep_com_vys)) if len(prep_com_vys) > 0 else 0.0
    prep_wheel_slip_cm = abs(prep_wheel_ys[-1] - prep_wheel_ys[0]) * 100.0 if len(prep_wheel_ys) > 1 else 0.0
    prep_com_disp_cm = abs(prep_com_ys[-1] - prep_com_ys[0]) * 100.0 if len(prep_com_ys) > 1 else 0.0

    # Criteria Verification (C1-C6 Strict Joint Acceptance)
    # C0: No rolling run-up during preparation (step <= 5, t <= 0.10s)
    passed_pre_runup = (prep_max_base_vy <= 0.20) and (prep_wheel_slip_cm <= 2.0)
    # C1: Airborne forward distance >= 8 cm
    passed_fwd_8cm = fwd_dist_cm >= 8.0
    # C2: Clearance >= 20 cm
    passed_clr_20cm = max_clearance_cm >= 20.0
    # C3: Attitude at touchdown <= 10 deg (Pitch & Roll)
    passed_pitch_10deg = abs(touchdown_pitch) <= 10.0
    passed_roll_10deg = abs(touchdown_roll) <= 10.0
    # C4: Recovery velocity <= 0.03 m/s
    passed_rec_vy_003 = abs(rec_vy_last10) <= 0.03
    # C5: Stability: zero abnormal resets
    passed_no_reset = abnormal_resets == 0

    joint_passed_strict = (
        passed_pre_runup and
        passed_fwd_8cm and
        passed_clr_20cm and
        passed_pitch_10deg and
        passed_roll_10deg and
        passed_rec_vy_003 and
        passed_no_reset
    )

    result = {
        "seed": seed,
        "hip_bias": hip_bias,
        "phase_threshold": phase_threshold,

        # 1. Balance dwell
        "stable_reached": stable_reached,
        "settle_time_s": settle_step_reached * env.dt,
        "dwell_timeout": not stable_reached,

        # 2. Dwell displacement
        "dwell_com_disp_cm": dwell_com_disp_cm,
        "dwell_wheel_disp_cm": dwell_wheel_disp_cm,

        # 3. Handoff velocities & forces
        "handoff_com_vy": handoff_com_vy,
        "handoff_com_vz": handoff_com_vz,
        "handoff_base_vy": handoff_base_vy,
        "handoff_base_ang_vel": handoff_base_ang_vel,
        "handoff_fn_L": handoff_fn_L,
        "handoff_fn_R": handoff_fn_R,

        # Handoff transient delta
        "handoff_delta_targets": handoff_delta_targets,
        "handoff_delta_torques": handoff_delta_torques,

        # 4. Pre-takeoff runup & prep compliance
        "prep_max_base_vy": prep_max_base_vy,
        "prep_max_com_vy": prep_max_com_vy,
        "prep_wheel_slip_cm": prep_wheel_slip_cm,
        "prep_com_disp_cm": prep_com_disp_cm,
        "pre_takeoff_max_vy": pre_takeoff_max_vy,
        "pre_takeoff_wheel_disp_cm": pre_takeoff_wheel_disp_cm,
        "pre_takeoff_com_disp_cm": pre_takeoff_com_disp_cm,

        # 5. Takeoff velocities
        "took_off": took_off,
        "takeoff_step": takeoff_step,
        "takeoff_t": takeoff_t,
        "takeoff_com_vy": takeoff_com_vy,
        "takeoff_com_vz": takeoff_com_vz,

        # Apex and compensation trigger
        "apex_z": apex_z,
        "apex_step": apex_step,
        "apex_t": apex_t,
        "comp_activated": comp_activated,
        "comp_first_step": comp_first_step,
        "comp_first_t": comp_first_t,
        "comp_first_com_vz": comp_first_com_vz,
        "comp_is_after_apex": comp_is_after_apex,

        # 6. Airborne forward displacement
        "landed": landed,
        "touchdown_step": touchdown_step,
        "touchdown_t": touchdown_t,
        "fwd_dist_cm": fwd_dist_cm,

        # 7. Dual wheel peak clearance
        "max_clearance_cm": max_clearance_cm,

        # 8. Airborne attitudes
        "airborne_pitch_max_deg": airborne_pitch_max,
        "airborne_pitch_min_deg": airborne_pitch_min,
        "airborne_roll_max_deg": airborne_roll_max,
        "airborne_yaw_drift_deg": airborne_yaw_drift,

        # 9. Touchdown attitudes (Physical: Pitch=r_raw, Roll=p_raw)
        "touchdown_pitch_deg": touchdown_pitch,
        "touchdown_roll_deg": touchdown_roll,
        "touchdown_yaw_deg": touchdown_yaw,

        # 10. Post-touchdown recovery (2s window)
        "post_rec_vy_last10": rec_vy_last10,
        "post_rec_vy_2s_mean": rec_vy_2s_mean,
        "post_rec_vy_2s_std": rec_vy_2s_std,
        "post_rec_vy_2s_max": rec_vy_2s_max,

        # 11. Torques and saturation
        "peak_torques": peak_torques.cpu().numpy().tolist(),
        "torque_saturation_counts": torque_saturation_counts.cpu().numpy().tolist(),

        # 12. Abnormal resets
        "abnormal_resets": abnormal_resets,

        # Acceptance Criteria
        "passed_pre_runup": passed_pre_runup,
        "passed_fwd_8cm": passed_fwd_8cm,
        "passed_clr_20cm": passed_clr_20cm,
        "passed_pitch_10deg": passed_pitch_10deg,
        "passed_roll_10deg": passed_roll_10deg,
        "passed_rec_vy_003": passed_rec_vy_003,
        "passed_no_reset": passed_no_reset,
        "joint_passed_strict": joint_passed_strict
    }

    print("===JSON_START===")
    print(json.dumps(result, indent=2))
    print("===JSON_END===")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--hip_bias", type=float, default=0.10)
    parser.add_argument("--phase_threshold", type=float, default=0.22)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()
    eval_seed_full(args.seed, args.hip_bias, args.phase_threshold, args.out)
