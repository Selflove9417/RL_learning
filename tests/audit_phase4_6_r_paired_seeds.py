#!/usr/bin/env python3
# ==============================================================================
# Phase 4.6-R: Single Seed Paired Trajectory & Value Target Audit
# Evaluates exact benchmark configuration matching eval_phase4_seed.py:
# - Task: myrobot_forward_jump_static
# - num_envs: 1
# - seed: seed
# - push_robots: False, noise: False
# - deterministic target_forward_vel: 0.22 m/s
# - hip_bias: +0.10 normalized action for airborne & phase >= 0.22
# Captures:
# 1. Exact step-by-step physical trajectory & rewards
# 2. Both Model A (Champion) & Model B (Calibrated) Critic values and Actor outputs
# 3. True Pre-Reset terminal state s_140 vs Post-Reset initial state s_0'
# 4. Target A (Finite terminal=0) vs Target B (Pre-reset bootstrap) vs Target C (Post-reset bootstrap)
# ==============================================================================

import os
import sys
import json
import argparse
import math
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


def run_audit(seed, out_path):
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["audit_seed", "--task", task_name, "--num_envs", "1", "--headless"]
    args = get_args()
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

    # 2. Load Model A (Champion) & Model B (Calibrated)
    champ_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    calib_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase4_6_critic_calibration/model_critic_calibrated.pt"

    ac_champ = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_champ.load_state_dict(torch.load(champ_path, map_location=env.device)["model_state_dict"])
    ac_champ.eval()

    ac_calib = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_calib.load_state_dict(torch.load(calib_path, map_location=env.device)["model_state_dict"])
    ac_calib.eval()

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

    # Balance Settling Phase
    for step in range(max_settle_steps):
        dof_pos_bal = (env.dof_pos - env.default_dof_pos).clone()
        dof_pos_bal[:, env.wheel_indices] = 0.0

        v_fwd = env.base_lin_vel[:, 1]
        pi_v_filt = 0.85 * pi_v_filt + 0.15 * v_fwd
        v_err = 0.0 - pi_v_filt
        pi_integral = torch.clamp(pi_integral + v_err * env.dt, -0.3, 0.3)
        pi_target = 0.8 * v_err + 0.2 * pi_integral

        bal_obs = torch.cat([
            env.base_lin_vel * 2.0,
            env.base_ang_vel * 0.25,
            env.projected_gravity,
            torch.tensor([[0.0, 0.0, 0.0]], device=env.device),
            dof_pos_bal * 1.0,
            env.dof_vel * 0.05,
            bal_filtered_actions,
            norm_h_tensor
        ], dim=-1)

        with torch.no_grad():
            bal_raw_actions = ac_balance.act_inference(bal_obs)

        bal_filtered_actions = 0.8 * torch.clamp(bal_raw_actions, -1.0, 1.0) + 0.2 * bal_filtered_actions

        h_val = 0.40 + 0.15 * norm_h
        q1_ik, q2_ik = compute_leg_ik(torch.tensor([h_val], device=env.device))
        q1_ik = q1_ik[0].item()
        q2_ik = q2_ik[0].item()

        tgt_hip_bal_L = q1_ik + 0.6 * bal_filtered_actions[0, env.left_hip_idx].item()
        tgt_knee_bal_L = q2_ik + 0.8 * bal_filtered_actions[0, env.left_knee_idx].item()
        tgt_hip_bal_R = q1_ik + 0.6 * bal_filtered_actions[0, env.right_hip_idx].item()
        tgt_knee_bal_R = q2_ik + 0.8 * bal_filtered_actions[0, env.right_knee_idx].item()

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

        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)

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
        r_raw, p_raw, _ = get_euler_xyz(env.base_quat)
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

    # 3. Clean Jump Phase Handoff at Step 0
    env.jump_timer[:] = 0.0
    env.jump_phase[:] = 0.0
    env.filtered_actions[:] = 0.0
    env.last_actions[:] = 0.0
    env.episode_length_buf[:] = 0
    env.jump_latched[:] = False
    env.has_taken_off[:] = False
    env.has_touched_down[:] = False
    env.one_shot_touchdown_event[:] = False
    env.commands[:, 1] = 0.22

    env.compute_observations()
    obs = env.obs_buf.clone()

    T = 140
    obs_list = []
    val_a_list = []
    val_b_list = []
    act_a_list = []
    act_b_list = []
    rew_list = []
    rew_breakdown = {}

    fwd_dist_cm = 0.0
    max_clearance_cm = 0.0
    takeoff_step = -1
    touchdown_step = -1
    takeoff_com_y = 0.0
    touchdown_com_y = 0.0

    took_off = False
    landed = False

    actor_diffs = []

    # Store reward scale mapping
    reward_keys = list(env.reward_names)

    for step in range(T):
        obs_list.append(obs.clone())

        # Evaluate both models on exact same observation
        with torch.no_grad():
            va = ac_champ.evaluate(obs).item()
            vb = ac_calib.evaluate(obs).item()
            val_a_list.append(va)
            val_b_list.append(vb)

            # Actor outputs verification
            act_a = ac_champ.act_inference(obs)
            act_b = ac_calib.act_inference(obs)
            act_diff = (act_a - act_b).abs().max().item()
            actor_diffs.append(act_diff)

            act_a_list.append(act_a.clone())
            act_b_list.append(act_b.clone())

        actions = act_a.clone()

        # Check telemetry before stepping
        true_com_pos, true_com_vel = env._compute_true_com()
        clr_min, _, _ = compute_exact_cylinder_clearance(env)
        clr_cm = clr_min[0].item() * 100.0
        is_air = bool(env.is_airborne[0].item())
        has_took = bool(env.has_taken_off[0].item())

        if is_air and has_took:
            max_clearance_cm = max(max_clearance_cm, clr_cm)

        if is_air and has_took and not took_off:
            took_off = True
            takeoff_step = step
            takeoff_com_y = float(true_com_pos[0, 1].item())

        if took_off and (not is_air) and (not landed):
            landed = True
            touchdown_step = step
            touchdown_com_y = float(true_com_pos[0, 1].item())

        # Aerial hip compensation
        comp_cond = is_air and (env.jump_phase[0].item() >= 0.22)
        if comp_cond:
            actions[:, env.left_hip_idx] += 0.10
            actions[:, env.right_hip_idx] += 0.10

        # Step physics manually to intercept pre-reset state at step 139!
        # Step logic from MyRobotForwardJumpStatic:
        actions_to_step = actions.clone()
        clip_actions = env.cfg.normalization.clip_actions
        clipped = torch.clip(actions_to_step, -clip_actions, clip_actions).to(env.device)
        env.last_actions[:] = env.filtered_actions
        env.filtered_actions[:] = 0.80 * clipped + 0.20 * env.last_actions

        env.jump_timer += env.dt
        landing_completed = env.has_taken_off & env.has_touched_down & (env.jump_timer >= 0.50 * env.jump_cycle_time)
        env.jump_latched |= landing_completed
        clamped_timer = torch.where(
            env.jump_latched,
            torch.clamp(env.jump_timer, max=0.85 * env.jump_cycle_time),
            env.jump_timer
        )
        env.jump_phase[:] = clamped_timer / env.jump_cycle_time

        # Step physics via legged_robot physics loop
        for _ in range(env.cfg.control.decimation):
            env.torques = env._compute_torques(env.filtered_actions).view(env.torques.shape)
            env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env.gym.refresh_dof_state_tensor(env.sim)

        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        env.episode_length_buf += 1
        env.common_step_counter += 1

        env.base_quat[:] = env.root_states[:, 3:7]
        env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
        env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
        env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)

        env._post_physics_step_callback()

        # Compute rewards for this transition
        env.compute_reward()
        step_reward = env.rew_buf[0].item()
        rew_list.append(step_reward)

        # Record reward breakdown
        for name in reward_keys:
            if name not in rew_breakdown:
                rew_breakdown[name] = []
            idx = env.reward_names.index(name)
            scale = env.reward_scales[name]
            r_val = env.reward_functions[idx]()[0].item() * scale
            rew_breakdown[name].append(r_val)

        # Intercept before reset at step 139!
        if step == T - 1:
            # Capture TRUE pre-reset observation s_140!
            env.compute_observations()
            s_140_pre_reset = env.obs_buf.clone()

            # Now perform explicit reset to capture post-reset initial observation s_0'!
            env.reset_idx(torch.tensor([0], device=env.device))
            env.compute_observations()
            s_0_post_reset = env.obs_buf.clone()
            obs = s_0_post_reset
        else:
            env.check_termination()
            env_ids = env.reset_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                print(f"[WARNING] Early reset triggered at step {step}!")
                env.reset_idx(env_ids)
            env.compute_observations()
            obs = env.obs_buf.clone()

    if took_off and landed:
        fwd_dist_cm = (touchdown_com_y - takeoff_com_y) * 100.0
    else:
        fwd_dist_cm = 0.0

    # Evaluate terminal states
    with torch.no_grad():
        va_pre = ac_champ.evaluate(s_140_pre_reset).item()
        vb_pre = ac_calib.evaluate(s_140_pre_reset).item()
        va_post = ac_champ.evaluate(s_0_post_reset).item()
        vb_post = ac_calib.evaluate(s_0_post_reset).item()

    rews = np.array(rew_list)
    vals_a = np.array(val_a_list)
    vals_b = np.array(val_b_list)

    gamma = 0.99
    lam = 0.95

    # 1. Target A: True Finite Horizon (Terminal value = 0)
    ret_fin = np.zeros(T)
    running_g = 0.0
    for t in reversed(range(T)):
        running_g = rews[t] + gamma * running_g
        ret_fin[t] = running_g

    # 2. Target B: Correct Pre-Reset Bootstrap (V(s_140_pre))
    ret_boot_pre_a = np.zeros(T)
    running_g = va_pre
    for t in reversed(range(T)):
        running_g = rews[t] + gamma * running_g
        ret_boot_pre_a[t] = running_g

    ret_boot_pre_b = np.zeros(T)
    running_g = vb_pre
    for t in reversed(range(T)):
        running_g = rews[t] + gamma * running_g
        ret_boot_pre_b[t] = running_g

    # 3. Target C: Phase 4.6 Post-Reset Bootstrap Error (V(s_0_post))
    ret_boot_post_a = np.zeros(T)
    running_g = va_post
    for t in reversed(range(T)):
        running_g = rews[t] + gamma * running_g
        ret_boot_post_a[t] = running_g

    ret_boot_post_b = np.zeros(T)
    running_g = vb_post
    for t in reversed(range(T)):
        running_g = rews[t] + gamma * running_g
        ret_boot_post_b[t] = running_g

    # Compute GAE advantages for all targets
    def compute_gae(vals, final_val, targets):
        adv = np.zeros(T)
        gae = 0.0
        td_res = np.zeros(T)
        for t in reversed(range(T)):
            next_v = vals[t+1] if t < T - 1 else final_val
            delta = rews[t] + gamma * next_v - vals[t]
            td_res[t] = delta
            gae = delta + gamma * lam * gae
            adv[t] = gae
        return adv, td_res

    # For Target A (final_val = 0)
    adv_a_fin, td_a_fin = compute_gae(vals_a, 0.0, ret_fin)
    adv_b_fin, td_b_fin = compute_gae(vals_b, 0.0, ret_fin)

    # For Target B (final_val = v_pre)
    adv_a_pre, td_a_pre = compute_gae(vals_a, va_pre, ret_boot_pre_a)
    adv_b_pre, td_b_pre = compute_gae(vals_b, vb_pre, ret_boot_pre_b)

    # For Target C (final_val = v_post)
    adv_a_post, td_a_post = compute_gae(vals_a, va_post, ret_boot_post_a)
    adv_b_post, td_b_post = compute_gae(vals_b, vb_post, ret_boot_post_b)

    def calc_stats(targets, preds):
        err = targets - preds
        mse = float(np.mean(err**2))
        mae = float(np.mean(np.abs(err)))
        var_y = float(np.var(targets))
        var_diff = float(np.var(err))
        ev = 1.0 - (var_diff / (var_y + 1e-8))
        return {"mse": mse, "mae": mae, "ev": ev}

    stats = {
        "Target_A_finite": {
            "model_a": calc_stats(ret_fin, vals_a),
            "model_b": calc_stats(ret_fin, vals_b),
            "td_mean_a": float(np.mean(td_a_fin)),
            "td_mean_b": float(np.mean(td_b_fin)),
            "td_step139_a": float(td_a_fin[139]),
            "td_step139_b": float(td_b_fin[139]),
            "adv_launch_a": float(np.mean(adv_a_fin[10:14])),
            "adv_launch_b": float(np.mean(adv_b_fin[10:14])),
            "adv_flight_a": float(np.mean(adv_a_fin[14:32])),
            "adv_flight_b": float(np.mean(adv_b_fin[14:32])),
            "adv_touchdown_a": float(np.mean(adv_a_fin[32:40])),
            "adv_touchdown_b": float(np.mean(adv_b_fin[32:40])),
            "adv_recovery_a": float(np.mean(adv_a_fin[40:140])),
            "adv_recovery_b": float(np.mean(adv_b_fin[40:140])),
        },
        "Target_B_pre_reset_bootstrap": {
            "model_a": calc_stats(ret_boot_pre_a, vals_a),
            "model_b": calc_stats(ret_boot_pre_b, vals_b),
            "td_mean_a": float(np.mean(td_a_pre)),
            "td_mean_b": float(np.mean(td_b_pre)),
            "td_step139_a": float(td_a_pre[139]),
            "td_step139_b": float(td_b_pre[139]),
            "adv_launch_a": float(np.mean(adv_a_pre[10:14])),
            "adv_launch_b": float(np.mean(adv_b_pre[10:14])),
            "adv_flight_a": float(np.mean(adv_a_pre[14:32])),
            "adv_flight_b": float(np.mean(adv_b_pre[14:32])),
            "adv_touchdown_a": float(np.mean(adv_a_pre[32:40])),
            "adv_touchdown_b": float(np.mean(adv_b_pre[32:40])),
            "adv_recovery_a": float(np.mean(adv_a_pre[40:140])),
            "adv_recovery_b": float(np.mean(adv_b_pre[40:140])),
        },
        "Target_C_post_reset_bootstrap": {
            "model_a": calc_stats(ret_boot_post_a, vals_a),
            "model_b": calc_stats(ret_boot_post_b, vals_b),
            "td_mean_a": float(np.mean(td_a_post)),
            "td_mean_b": float(np.mean(td_b_post)),
            "td_step139_a": float(td_a_post[139]),
            "td_step139_b": float(td_b_post[139]),
            "adv_launch_a": float(np.mean(adv_a_post[10:14])),
            "adv_launch_b": float(np.mean(adv_b_post[10:14])),
            "adv_flight_a": float(np.mean(adv_a_post[14:32])),
            "adv_flight_b": float(np.mean(adv_b_post[14:32])),
            "adv_touchdown_a": float(np.mean(adv_a_post[32:40])),
            "adv_touchdown_b": float(np.mean(adv_b_post[32:40])),
            "adv_recovery_a": float(np.mean(adv_a_post[40:140])),
            "adv_recovery_b": float(np.mean(adv_b_post[40:140])),
        }
    }

    # Summary dictionary
    result = {
        "seed": seed,
        "actor_max_diff": float(max(actor_diffs)),
        "settle_step_reached": settle_step_reached,
        "fwd_dist_cm": fwd_dist_cm,
        "max_clearance_cm": max_clearance_cm,
        "takeoff_step": takeoff_step,
        "touchdown_step": touchdown_step,
        "undiscounted_return": float(np.sum(rews)),
        "terminal_values": {
            "va_pre_reset": va_pre,
            "vb_pre_reset": vb_pre,
            "va_post_reset": va_post,
            "vb_post_reset": vb_post,
            "va_step0": float(vals_a[0]),
            "vb_step0": float(vals_b[0]),
            "va_step139": float(vals_a[139]),
            "vb_step139": float(vals_b[139]),
        },
        "stats": stats,
        "reward_totals": {k: float(np.sum(v)) for k, v in rew_breakdown.items()}
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n[Audit Seed {seed} Complete]")
    print(f"  Fwd Dist: {fwd_dist_cm:.2f} cm | Clr: {max_clearance_cm:.2f} cm | Actor Diff: {max(actor_diffs):.2e}")
    print(f"  V(s_140_pre): Champ = {va_pre:.2f}, Calib = {vb_pre:.2f}")
    print(f"  V(s_0_post) : Champ = {va_post:.2f}, Calib = {vb_post:.2f}")
    print(f"  Target B (Pre-Reset Bootstrap): Model A MSE = {stats['Target_B_pre_reset_bootstrap']['model_a']['mse']:.4f}, Model B MSE = {stats['Target_B_pre_reset_bootstrap']['model_b']['mse']:.4f}")
    print(f"  Target A (Finite Horizon)    : Model A MSE = {stats['Target_A_finite']['model_a']['mse']:.4f}, Model B MSE = {stats['Target_A_finite']['model_b']['mse']:.4f}")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    run_audit(args.seed, args.out)
