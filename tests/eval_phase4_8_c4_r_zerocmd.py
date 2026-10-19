#!/usr/bin/env python3
# ==============================================================================
# Phase 4.8-C4-R: ZeroCmd Rigorous 5-Dev-Seed Physical Verification
# Exactly matches eval_phase4_seed.py, with single-variable observation zeroing
# Compares:
#   Group A: Baseline Pilot B (zero_cmd = False, obs[:, 9] intact)
#   Group B: ZeroCmd Pilot B (zero_cmd = True, obs[:, 9] = 0.0 strictly post-touchdown)
# Across Seeds: 3001, 3002, 3003, 3004, 3005
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


def eval_zerocmd_seed(seed, jump_model_path=None, hip_bias=0.10, phase_threshold=0.22, out_path="", zero_cmd=False):
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

    # 2. Load Jump Policy
    if jump_model_path is None or not os.path.exists(jump_model_path):
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

    # Step 1: Balance Settling Phase
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

        last_bal_torques[0] = torques[0].clone()
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
    takeoff_pitch = 0.0
    touchdown_pitch = 0.0
    touchdown_roll = 0.0
    touchdown_yaw = 0.0

    post_rec_vys = []
    prep_base_vys = []
    prep_com_vys = []
    prep_wheel_ys = []
    prep_com_ys = []

    obs_slot9_pre_td = None
    obs_slot9_post_td = None
    torque_wL_rec = []
    torque_wR_rec = []
    torque_fwd_rec = []

    abnormal_resets = 0

    # Step 3: Jump & Recovery Execution
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

        cur_wL_y = env.rb_states[0, env.left_wheel_body, 1].item()
        cur_wR_y = env.rb_states[0, env.right_wheel_body, 1].item()
        cur_w_y = 0.5 * (cur_wL_y + cur_wR_y)

        if not took_off and step <= 5:
            prep_base_vys.append(abs(b_vel[1]))
            prep_com_vys.append(abs(true_com_vel[0, 1].item()))
            prep_wheel_ys.append(cur_w_y)
            prep_com_ys.append(true_com_pos[0, 1].item())

        if is_air and has_took:
            max_clearance_cm = max(max_clearance_cm, clr_min_cm)

        # Takeoff detection
        if is_air and has_took and not took_off:
            took_off = True
            takeoff_step = step
            takeoff_com_y = float(true_com_pos[0, 1].item())
            takeoff_pitch = cur_pitch_deg

        # Touchdown detection
        if took_off and (not is_air) and (not landed):
            landed = True
            touchdown_step = step
            touchdown_com_y = float(true_com_pos[0, 1].item())
            touchdown_pitch = cur_pitch_deg
            touchdown_roll = cur_roll_deg
            touchdown_yaw = cur_yaw_deg

        # Post-touchdown recovery tracking (2 seconds window post touchdown)
        if landed and step > touchdown_step + 5:
            post_rec_vys.append(b_vel[1])
            tau_L = env.torques[0, env.left_wheel_idx].item()
            tau_R = env.torques[0, env.right_wheel_idx].item()
            torque_wL_rec.append(tau_L)
            torque_wR_rec.append(tau_R)
            torque_fwd_rec.append(tau_L + tau_R)

        # Record observation slot 9
        if not landed:
            obs_slot9_pre_td = float(obs[0, 9].item())
        else:
            if obs_slot9_post_td is None:
                obs_slot9_post_td = float(obs[0, 9].item())

        # ZeroCmd: strictly post-touchdown intervention on input to Actor only
        obs_for_actor = obs.clone()
        if zero_cmd and landed:
            obs_for_actor[:, 9] = 0.0

        with torch.no_grad():
            actions = ac_jump.act_inference(obs_for_actor)

        # Single Minimal Intervention: Hip extension trim during descent
        comp_cond = is_air and (env.jump_phase[0].item() >= phase_threshold)
        if comp_cond and (hip_bias != 0.0):
            actions[:, env.left_hip_idx] += hip_bias
            actions[:, env.right_hip_idx] += hip_bias

        obs, _, _, _, _ = env.step(actions)

        if env.reset_buf[0].item() != 0 and step < total_jump_steps - 5:
            abnormal_resets += 1

    fwd_dist_cm = (touchdown_com_y - takeoff_com_y) * 100.0 if (took_off and landed) else 0.0
    rec_vy_last10 = float(np.mean(post_rec_vys[-10:])) if len(post_rec_vys) >= 10 else 0.0
    rec_vy_last10_vals = [float(v) for v in post_rec_vys[-10:]]

    prep_max_base_vy = float(np.max(prep_base_vys)) if len(prep_base_vys) > 0 else 0.0
    prep_wheel_slip_cm = abs(prep_wheel_ys[-1] - prep_wheel_ys[0]) * 100.0 if len(prep_wheel_ys) > 1 else 0.0

    # Criteria
    passed_pre_runup = (prep_max_base_vy <= 0.20) and (prep_wheel_slip_cm <= 2.0)
    passed_fwd_8cm = fwd_dist_cm >= 8.0
    passed_fwd_12cm = fwd_dist_cm >= 12.0
    passed_clr_20cm = max_clearance_cm >= 20.0
    passed_pitch_10deg = abs(touchdown_pitch) <= 10.0
    passed_roll_10deg = abs(touchdown_roll) <= 10.0
    passed_rec_vy_003 = abs(rec_vy_last10) <= 0.030
    passed_no_reset = abnormal_resets == 0

    joint_passed_baseline = (passed_pre_runup and passed_fwd_8cm and passed_clr_20cm and passed_pitch_10deg and passed_roll_10deg and passed_rec_vy_003 and passed_no_reset)
    joint_passed_phase4 = (passed_pre_runup and passed_fwd_12cm and passed_clr_20cm and passed_pitch_10deg and passed_roll_10deg and passed_rec_vy_003 and passed_no_reset)

    mean_tau_L_last10 = float(np.mean(torque_wL_rec[-10:])) if len(torque_wL_rec) >= 10 else 0.0
    mean_tau_R_last10 = float(np.mean(torque_wR_rec[-10:])) if len(torque_wR_rec) >= 10 else 0.0
    mean_tau_fwd_last10 = float(np.mean(torque_fwd_rec[-10:])) if len(torque_fwd_rec) >= 10 else 0.0

    result = {
        "seed": seed,
        "zero_cmd": zero_cmd,
        "fwd_dist_cm": fwd_dist_cm,
        "max_clearance_cm": max_clearance_cm,
        "takeoff_pitch_deg": takeoff_pitch,
        "touchdown_pitch_deg": touchdown_pitch,
        "touchdown_roll_deg": touchdown_roll,
        "touchdown_yaw_deg": touchdown_yaw,
        "post_rec_vy_last10": rec_vy_last10,
        "post_rec_vy_last10_vals": rec_vy_last10_vals,
        "obs_slot9_pre_td": obs_slot9_pre_td,
        "obs_slot9_post_td": obs_slot9_post_td,
        "mean_tau_L_last10": mean_tau_L_last10,
        "mean_tau_R_last10": mean_tau_R_last10,
        "mean_tau_fwd_last10": mean_tau_fwd_last10,
        "abnormal_resets": abnormal_resets,
        "passed_pre_runup": passed_pre_runup,
        "passed_fwd_8cm": passed_fwd_8cm,
        "passed_fwd_12cm": passed_fwd_12cm,
        "passed_clr_20cm": passed_clr_20cm,
        "passed_pitch_10deg": passed_pitch_10deg,
        "passed_roll_10deg": passed_roll_10deg,
        "passed_rec_vy_003": passed_rec_vy_003,
        "passed_no_reset": passed_no_reset,
        "joint_passed_baseline": joint_passed_baseline,
        "joint_passed_phase4": joint_passed_phase4,
    }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3001)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--zero_cmd", action="store_true")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()
    res = eval_zerocmd_seed(args.seed, args.model, out_path=args.out, zero_cmd=args.zero_cmd)
    print(f"[EVAL DONE] Seed {args.seed} | zero_cmd={args.zero_cmd} | Fwd: {res['fwd_dist_cm']:.2f}cm | Clr: {res['max_clearance_cm']:.2f}cm | C4: {res['post_rec_vy_last10']:.4f} m/s | Pass C4: {res['passed_rec_vy_003']} | Joint Pass: {res['joint_passed_phase4']}")
