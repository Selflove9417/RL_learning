#!/usr/bin/env python3
# ==============================================================================
# Phase 3.2-U Task 2 & 3: Minimal Single-Variable Roll Correction
# Case B: Aerial Descent Landing Preparation Action Symmetrization
#
# Intervenes ONLY when airborne during flight/descent phase (phase >= 0.15):
#   act_hip_sym = 0.5 * (act_hip_L + act_hip_R)
#   act_knee_sym = 0.5 * (act_knee_L + act_knee_R)
#   actions[:, left_hip] = act_hip_sym
#   actions[:, right_hip] = act_hip_sym
#   actions[:, left_knee] = act_knee_sym
#   actions[:, right_knee] = act_knee_sym
#
# Ground push-off (phase < 0.15 or grounded): 100% untouched!
# Wheel torques: 100% untouched!
# Checkpoints & PD gains: 100% untouched!
# ==============================================================================

import os
import sys
import json
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

def run_corrected_entry_experiment(seed, symmetrize_air=True, air_phase_threshold=0.15):
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["eval_corrected", "--task", task_name, "--num_envs", "1"]
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

    # Reset and place at ground contact pose z = 0.397739
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

    max_settle_steps = int(1.5 / env.dt)
    stable_consecutive_steps = 0
    stable_reached = False
    settle_step_reached = 0

    h_ctrl = 0.39774
    norm_h = ((h_ctrl - 0.40) / 0.15)
    norm_h_tensor = torch.tensor([[norm_h]], device=env.device)

    # Balance stabilization
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
        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi

        is_stable_now = (abs(b_vy) <= 0.03) and (abs(c_vy) <= 0.03) and (abs(b_vz) <= 0.05) and (fn_L >= 20.0 and fn_L <= 180.0) and (fn_R >= 20.0 and fn_R <= 180.0) and (abs(roll) <= 2.5) and (abs(pitch) <= 3.5)
        if is_stable_now:
            stable_consecutive_steps += 1
        else:
            stable_consecutive_steps = 0

        if stable_consecutive_steps >= 5 and step >= 15:
            stable_reached = True
            settle_step_reached = step + 1
            break

    # Reset Jump environment trackers
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

        if took_off and (not is_air) and (not landed):
            landed = True
            touchdown_step = step
            touchdown_com_y = float(true_com_pos[0, 1].item())
            touchdown_pitch = pitch
            touchdown_roll = roll

        if landed and t_sim >= 1.0:
            post_rec_vys.append(b_vel[1])

        with torch.no_grad():
            actions = ac_jump.act_inference(obs)

        # CASE B MINIMAL INTERVENTION: Symmetrize leg actions strictly while airborne in flight/descent phase!
        if symmetrize_air and is_air and (env.jump_phase[0].item() >= air_phase_threshold):
            act_hip_sym = 0.5 * (actions[:, env.left_hip_idx] + actions[:, env.right_hip_idx])
            act_knee_sym = 0.5 * (actions[:, env.left_knee_idx] + actions[:, env.right_knee_idx])
            actions[:, env.left_hip_idx] = act_hip_sym
            actions[:, env.right_hip_idx] = act_hip_sym
            actions[:, env.left_knee_idx] = act_knee_sym
            actions[:, env.right_knee_idx] = act_knee_sym

        obs, _, _, _, _ = env.step(actions)

    fwd_dist_cm = (touchdown_com_y - takeoff_com_y) * 100.0 if (took_off and landed) else 0.0
    rec_vy = float(np.mean(post_rec_vys[-10:])) if len(post_rec_vys) >= 10 else 0.0

    passed_fwd = fwd_dist_cm >= 8.0
    passed_clr = max_clearance_cm >= 20.0
    passed_roll = abs(touchdown_roll) <= 10.0
    passed_pitch = abs(touchdown_pitch) <= 10.0
    passed_rec_vy = abs(rec_vy) <= 0.03
    joint_passed = passed_fwd and passed_clr and passed_roll and passed_pitch and passed_rec_vy

    result = {
        "seed": seed,
        "symmetrize_air": symmetrize_air,
        "air_phase_threshold": air_phase_threshold,
        "stable_reached": stable_reached,
        "settle_time_s": settle_step_reached * env.dt,
        "handoff_base_vy": float(b_vy),
        "handoff_com_vy": float(c_vy),
        "fwd_dist_cm": fwd_dist_cm,
        "max_clearance_cm": max_clearance_cm,
        "touchdown_roll_deg": touchdown_roll,
        "touchdown_pitch_deg": touchdown_pitch,
        "post_rec_vy": rec_vy,
        "passed_fwd_8cm": passed_fwd,
        "passed_clr_20cm": passed_clr,
        "passed_roll_10deg": passed_roll,
        "passed_pitch_10deg": passed_pitch,
        "passed_rec_vy_003": passed_rec_vy,
        "joint_passed": joint_passed
    }

    print(f"Seed {seed:3d} (Symmetrize={symmetrize_air}): Fwd={fwd_dist_cm:.2f}cm, Clr={max_clearance_cm:.2f}cm, Roll={touchdown_roll:.2f}deg, Pitch={touchdown_pitch:.2f}deg, RecVy={rec_vy:+.4f}m/s -> JointPass={joint_passed}")
    return result

if __name__ == "__main__":
    seeds = [100, 2, 17, 23]
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        seeds = [int(sys.argv[1])]
    
    print("\n================================================================================")
    print("  RUNNING CASE B MINIMAL INTERVENTION EXPERIMENT (SEEDS: 100, 2, 17, 23)")
    print("================================================================================")
    results = {}
    for s in seeds:
        results[s] = run_corrected_entry_experiment(s, symmetrize_air=True, air_phase_threshold=0.15)

    print("\n================================================================================")
    print("  SUMMARY OF CASE B RESULTS:")
    print("================================================================================")
    for s, res in results.items():
        print(f"Seed {s:3d}: Fwd={res['fwd_dist_cm']:6.2f}cm (>=8), Clr={res['max_clearance_cm']:5.2f}cm (>=20), Roll={res['touchdown_roll_deg']:+6.2f}deg (<=10), Pitch={res['touchdown_pitch_deg']:+6.2f}deg (<=10), RecVy={res['post_rec_vy']:+.4f} (<=0.03) -> {res['joint_passed']}")

    with open("evaluation_assets/phase3_2_t_ground_reset/case_b_results_4seeds.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to evaluation_assets/phase3_2_t_ground_reset/case_b_results_4seeds.json")
