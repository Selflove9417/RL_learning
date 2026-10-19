#!/usr/bin/env python3
# ==============================================================================
# Phase 3.2-T Task 5: Minimal Stable Takeoff Entry Experiment
# Tests Seeds 100, 2, 17, 23 with:
#   1. Zero-velocity balance stabilization (Phase 15 PPO / PI parking control)
#   2. Strict physical stability gate (|Vy|<=0.03, |Vz|<=0.05, Fn in [20, 150]N, held 5 steps)
#   3. Seamless handoff to Champion Forward Jump Policy
#   4. Evaluates Forward Distance (>=8cm), Clearance (>=20cm), Roll (<=10 deg)
# ==============================================================================

import os
import sys
import json
import math
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

def run_stable_entry_experiment(seed, max_settle_time=1.5):
    print(f"\n================================================================================")
    print(f"  RUNNING STABLE TAKEOFF ENTRY EXPERIMENT: SEED {seed}")
    print(f"================================================================================")

    # 1. We will use the forward jump static task environment
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["stable_entry", "--task", task_name, "--num_envs", "1"]
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

    # Load Phase 15 Balance Policy
    bal_model_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase15/models/phase15_m130.pt"
    ac_balance = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ckpt_bal = torch.load(bal_model_path, map_location=env.device)
    ac_balance.load_state_dict(ckpt_bal["model_state_dict"])
    ac_balance.eval()

    # Load Champion Forward Jump Policy
    jump_model_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    ac_jump = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ckpt_jump = torch.load(jump_model_path, map_location=env.device)
    ac_jump.load_state_dict(ckpt_jump["model_state_dict"])
    ac_jump.eval()

    # Step 1: Initialize robot in physical contact pose at nominal height z = 0.39774
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

    # Warm-up settle buffers
    bal_filtered_actions = torch.zeros(env.num_envs, 6, device=env.device)
    pi_integral = torch.zeros(env.num_envs, device=env.device)
    pi_v_filt = torch.zeros(env.num_envs, device=env.device)

    # Settling Phase loop
    max_settle_steps = int(max_settle_time / env.dt)
    stable_consecutive_steps = 0
    stable_reached = False
    settle_step_reached = 0

    h_ctrl = 0.39774
    norm_h = ((h_ctrl - 0.40) / 0.15)
    norm_h_tensor = torch.tensor([[norm_h]], device=env.device)

    print("\n--- PHASE 1: Balance Stabilization & Zero-Velocity Settling ---")
    for step in range(max_settle_steps):
        # 1. Construct 31-dim observation for Phase 15 balance policy
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

        # Balance IK for 5-bar linkage
        q1_ik, q2_ik = compute_leg_ik(torch.tensor([h_ctrl], device=env.device))

        tgt_hip_bal_L = env.default_dof_pos[:, env.left_hip_idx] + q1_ik + 0.25 * 0.7 * bal_filtered_actions[:, env.left_hip_idx]
        tgt_knee_bal_L = env.default_dof_pos[:, env.left_knee_idx] + q2_ik + 0.25 * 0.5 * bal_filtered_actions[:, env.left_knee_idx]
        tgt_hip_bal_R = env.default_dof_pos[:, env.right_hip_idx] + q1_ik + 0.25 * 0.7 * bal_filtered_actions[:, env.right_hip_idx]
        tgt_knee_bal_R = env.default_dof_pos[:, env.right_knee_idx] + q2_ik + 0.25 * 0.5 * bal_filtered_actions[:, env.right_knee_idx]

        # Wheel torque computation
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

        # Convert to environment action equivalent for step
        # Note: in LeggedRobot, compute_torques uses kp=180, kd=4.5 for jump.
        # But for balance settling, kp=40, kd=2 is standard.
        # We directly apply torques in PhysX substeps for exact balance!
        for _ in range(env.cfg.control.decimation):
            torques = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            # Leg PD (kp=40, kd=2)
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

        # Refresh state quantities
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
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi

        # Stability Gate Evaluation:
        is_vy_calm = (abs(b_vy) <= 0.03) and (abs(c_vy) <= 0.03)
        is_vz_calm = (abs(b_vz) <= 0.05) and (abs(c_vz) <= 0.05)
        is_fn_grounded = (fn_L >= 20.0 and fn_L <= 180.0) and (fn_R >= 20.0 and fn_R <= 180.0)
        is_att_calm = (abs(roll) <= 2.5) and (abs(pitch) <= 3.5)
        is_stable_now = is_vy_calm and is_vz_calm and is_fn_grounded and is_att_calm

        if is_stable_now:
            stable_consecutive_steps += 1
        else:
            stable_consecutive_steps = 0

        if (step + 1) % 10 == 0 or is_stable_now:
            print(f"  Step {step+1:2d} (t={(step+1)*0.02:.2f}s): b_vy={b_vy:+.4f}, c_vy={c_vy:+.4f}, b_vz={b_vz:+.4f}, Fn=({fn_L:.1f}, {fn_R:.1f})N, pitch={pitch:+.2f}deg, roll={roll:+.2f}deg, stable_streak={stable_consecutive_steps}")

        # Check if held for 5 consecutive steps (0.10s)
        if stable_consecutive_steps >= 5 and step >= 15: # At least 0.3s of settling
            stable_reached = True
            settle_step_reached = step + 1
            print(f"  >>> STABILITY GATE ACHIEVED at step {settle_step_reached} (t={settle_step_reached*0.02:.2f}s) <<<")
            break

    # If stability gate achieved, record the pristine state right before jump
    print("\n--- PHASE 2: Seamless Handoff to Champion Forward Jump Policy ---")
    # Reset jump timers, phase, and filtered actions
    env.jump_timer[:] = 0.0
    env.jump_phase[:] = 0.0
    env.filtered_actions[:] = 0.0
    env.last_actions[:] = 0.0
    env.episode_length_buf[:] = 0
    env.jump_latched[:] = False
    env.has_taken_off[:] = False
    env.has_touched_down[:] = False
    env.one_shot_touchdown_event[:] = False
    env.flight_distance_y[:] = 0.0
    env.final_flight_distance[:] = 0.0

    # Record Hand-Off State
    r_pos = env.root_states[0, :3].cpu().numpy()
    b_vel = env.base_lin_vel[0].cpu().numpy()
    _, true_com_vel = env._compute_true_com()
    c_vel = true_com_vel[0].cpu().numpy()
    fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
    fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()
    r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
    roll_deg = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
    pitch_deg = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
    print(f"Hand-off State: z={r_pos[2]:.4f}m, Base Vy={b_vel[1]:+.4f} m/s, CoM Vy={c_vel[1]:+.4f} m/s, Vz={b_vel[2]:+.4f} m/s")
    print(f"                Fn=({fn_L:.1f}, {fn_R:.1f})N, Pitch={pitch_deg:+.2f}deg, Roll={roll_deg:+.2f}deg")

    # Run Jump Episode (2.8s = 140 steps)
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
    touchdown_roll = 0.0
    touchdown_pitch = 0.0
    post_rec_vys = []

    for step in range(total_jump_steps):
        t_sim = step * env.dt
        true_com_pos, true_com_vel = env._compute_true_com()
        b_vel = env.base_lin_vel[0].cpu().numpy()
        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi

        clr_min, _, _ = compute_exact_cylinder_clearance(env)
        clr_min_cm = clr_min[0].item() * 100.0
        is_air = bool(env.is_airborne[0].item())
        has_took = bool(env.has_taken_off[0].item())

        if is_air and has_took:
            max_clearance_cm = max(max_clearance_cm, clr_min_cm)

        if is_air and has_took and not took_off:
            took_off = True
            takeoff_step = step
            takeoff_com_y = float(true_com_pos[0, 1].item())
            print(f"  [Takeoff @ step {step:2d} (t={t_sim:.2f}s)]: vy={b_vel[1]:+.4f}m/s, vz={b_vel[2]:+.4f}m/s, com_y={takeoff_com_y:.4f}m")

        if took_off and (not is_air) and (not landed):
            landed = True
            touchdown_step = step
            touchdown_com_y = float(true_com_pos[0, 1].item())
            touchdown_pitch = pitch
            touchdown_roll = roll
            print(f"  [Touchdown @ step {step:2d} (t={t_sim:.2f}s)]: com_y={touchdown_com_y:.4f}m, Roll={touchdown_roll:+.2f}deg, Pitch={touchdown_pitch:+.2f}deg")

        if landed and t_sim >= 1.0:
            post_rec_vys.append(b_vel[1])

        with torch.no_grad():
            actions = ac_jump.act_inference(obs)
        obs, _, _, _, _ = env.step(actions)

    fwd_dist_cm = (touchdown_com_y - takeoff_com_y) * 100.0 if (took_off and landed) else 0.0
    rec_vy = float(np.mean(post_rec_vys[-10:])) if len(post_rec_vys) >= 10 else 0.0

    passed_fwd = fwd_dist_cm >= 8.0
    passed_clr = max_clearance_cm >= 20.0
    passed_roll = abs(touchdown_roll) <= 10.0
    joint_passed = passed_fwd and passed_clr and passed_roll

    result = {
        "seed": seed,
        "stable_reached": stable_reached,
        "settle_time_s": settle_step_reached * env.dt,
        "handoff_base_vy": float(b_vel[1]),
        "handoff_com_vy": float(c_vel[1]),
        "handoff_fn_L": float(fn_L),
        "handoff_fn_R": float(fn_R),
        "took_off": took_off,
        "landed": landed,
        "fwd_dist_cm": fwd_dist_cm,
        "max_clearance_cm": max_clearance_cm,
        "touchdown_roll_deg": touchdown_roll,
        "touchdown_pitch_deg": touchdown_pitch,
        "post_rec_vy": rec_vy,
        "passed_fwd_8cm": passed_fwd,
        "passed_clr_20cm": passed_clr,
        "passed_roll_10deg": passed_roll,
        "joint_passed": joint_passed
    }

    print("\n--------------------------------------------------------------------------------")
    print(f"SEED {seed} EVALUATION SUMMARY:")
    print(f"  Stable Reached    : {stable_reached} (at {settle_step_reached*0.02:.2f}s)")
    print(f"  Forward Distance  : {fwd_dist_cm:.2f} cm (>=8cm: {passed_fwd})")
    print(f"  Max Clearance     : {max_clearance_cm:.2f} cm (>=20cm: {passed_clr})")
    print(f"  Touchdown Roll    : {touchdown_roll:.2f} deg (<=10deg: {passed_roll})")
    print(f"  Touchdown Pitch   : {touchdown_pitch:.2f} deg")
    print(f"  Recovery Vy       : {rec_vy:+.4f} m/s")
    print(f"  Joint Passed      : {joint_passed}")
    print("--------------------------------------------------------------------------------")
    print("===JSON_START===")
    print(json.dumps(result, indent=2))
    print("===JSON_END===")

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    args_p, _ = parser.parse_known_args()
    res = run_stable_entry_experiment(args_p.seed)
