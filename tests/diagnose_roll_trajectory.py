#!/usr/bin/env python3
# ==============================================================================
# Phase 3.2-U Task 1: High-Resolution Roll Trajectory & Dynamics Diagnostic
# Tracks:
#   1. Switching instant (Step 0) torque & target discontinuity
#   2. Push-off phase (Steps 0-12): Fn_L, Fn_R, Iz_L, Iz_R, tau_hip/knee L vs R, omega_x, roll
#   3. Flight phase (Steps 12-25): omega_x, roll, airborne symmetry
#   4. Aerial descent & landing prep (Steps 25-35): q_hip_L vs q_hip_R, q_knee_L vs q_knee_R,
#      wheel height L vs R, touchdown instant roll & pitch
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

def diagnose_roll(seed):
    print(f"\n================================================================================")
    print(f"  DIAGNOSING ROLL DYNAMICS: SEED {seed}")
    print(f"================================================================================")

    task_name = "myrobot_forward_jump_static"
    sys.argv = ["diag_roll", "--task", task_name, "--num_envs", "1"]
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

    # Place robot at ground contact pose z = 0.397739
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

    print(f"Settling complete at step {settle_step_reached}: Roll={roll:+.2f}deg, Pitch={pitch:+.2f}deg, Base Vy={b_vy:+.4f}m/s")

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

    env.compute_observations()
    obs = env.obs_buf.clone()

    # Track detailed frames
    frames = []
    impulse_L = 0.0
    impulse_R = 0.0
    took_off = False
    landed = False
    takeoff_step = None
    touchdown_step = None

    total_jump_steps = int(1.5 / env.dt) # 75 steps is plenty to cover flight and touchdown

    for step in range(total_jump_steps):
        t_sim = step * env.dt
        true_com_pos, true_com_vel = env._compute_true_com()
        b_vel = env.base_lin_vel[0].cpu().numpy()
        b_ang_vel = env.base_ang_vel[0].cpu().numpy()
        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
        yaw = torch.atan2(torch.sin(y_raw[0]), torch.cos(y_raw[0])).item() * 180.0 / np.pi

        clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(env)
        fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
        fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()

        q_hip_L = env.dof_pos[0, env.left_hip_idx].item()
        q_hip_R = env.dof_pos[0, env.right_hip_idx].item()
        q_knee_L = env.dof_pos[0, env.left_knee_idx].item()
        q_knee_R = env.dof_pos[0, env.right_knee_idx].item()

        is_air = bool(env.is_airborne[0].item())
        has_took = bool(env.has_taken_off[0].item())

        if not is_air and step <= 15: # Push-off phase
            impulse_L += fn_L * env.dt
            impulse_R += fn_R * env.dt

        if is_air and has_took and not took_off:
            took_off = True
            takeoff_step = step

        if took_off and (not is_air) and (not landed):
            landed = True
            touchdown_step = step

        # Get policy actions
        with torch.no_grad():
            actions = ac_jump.act_inference(obs)

        # Before stepping, let's see what torques will be computed
        hip_targets_L = env.default_dof_pos[:, env.left_hip_idx] + actions[:, env.left_hip_idx] * env.hip_action_scale
        hip_targets_R = env.default_dof_pos[:, env.right_hip_idx] + actions[:, env.right_hip_idx] * env.hip_action_scale
        knee_targets_L = env.default_dof_pos[:, env.left_knee_idx] + actions[:, env.left_knee_idx] * env.knee_action_scale
        knee_targets_R = env.default_dof_pos[:, env.right_knee_idx] + actions[:, env.right_knee_idx] * env.knee_action_scale

        frame_info = {
            "step": step,
            "t_s": round(t_sim, 3),
            "phase": round(env.jump_phase[0].item(), 3),
            "is_air": is_air,
            "has_took": has_took,
            "roll_deg": round(roll, 2),
            "pitch_deg": round(pitch, 2),
            "omega_x_radps": round(float(b_ang_vel[0]), 3),
            "fn_L_N": round(fn_L, 1),
            "fn_R_N": round(fn_R, 1),
            "clr_L_cm": round(clr_L[0].item() * 100.0, 2),
            "clr_R_cm": round(clr_R[0].item() * 100.0, 2),
            "q_hip_L": round(q_hip_L, 4),
            "q_hip_R": round(q_hip_R, 4),
            "delta_q_hip": round(q_hip_L - q_hip_R, 4),
            "q_knee_L": round(q_knee_L, 4),
            "q_knee_R": round(q_knee_R, 4),
            "delta_q_knee": round(q_knee_L - q_knee_R, 4),
            "act_hip_L": round(actions[0, env.left_hip_idx].item(), 4),
            "act_hip_R": round(actions[0, env.right_hip_idx].item(), 4),
            "act_knee_L": round(actions[0, env.left_knee_idx].item(), 4),
            "act_knee_R": round(actions[0, env.right_knee_idx].item(), 4),
            "torques": [round(t, 2) for t in env.torques[0].cpu().numpy().tolist()] if hasattr(env, "torques") else []
        }
        frames.append(frame_info)

        obs, _, _, _, _ = env.step(actions)

    print(f"\n--- LIFTOFF & PUSH-OFF DYNAMICS (Takeoff at step {takeoff_step}, Touchdown at step {touchdown_step}) ---")
    print(f"Total Push-Off Vertical Impulse: Left={impulse_L:.2f} N*s, Right={impulse_R:.2f} N*s, Delta(R-L)={impulse_R - impulse_L:+.2f} N*s")

    print("\n--- TIMELINE OF CRITICAL STEPS (0 to 35) ---")
    print(f"{'Step':>4} | {'t(s)':>5} | {'Phase':>5} | {'Air':>5} | {'Roll(deg)':>9} | {'wx(rad/s)':>9} | {'Fn_L(N)':>7} | {'Fn_R(N)':>7} | {'clr_L':>6} | {'clr_R':>6} | {'q_hip_L':>7} | {'q_hip_R':>7} | {'dq_hip':>7}")
    print("-" * 115)
    for f in frames[:38]:
        print(f"{f['step']:4d} | {f['t_s']:5.3f} | {f['phase']:5.3f} | {str(f['is_air']):>5} | {f['roll_deg']:+9.2f} | {f['omega_x_radps']:+9.3f} | {f['fn_L_N']:7.1f} | {f['fn_R_N']:7.1f} | {f['clr_L_cm']:6.2f} | {f['clr_R_cm']:6.2f} | {f['q_hip_L']:+7.3f} | {f['q_hip_R']:+7.3f} | {f['delta_q_hip']:+7.3f}")

    return {
        "seed": seed,
        "takeoff_step": takeoff_step,
        "touchdown_step": touchdown_step,
        "impulse_L": impulse_L,
        "impulse_R": impulse_R,
        "delta_impulse": impulse_R - impulse_L,
        "touchdown_roll": frames[touchdown_step]['roll_deg'] if touchdown_step else None,
        "frames": frames
    }

if __name__ == "__main__":
    seeds = [100, 2, 17, 23]
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        seeds = [int(sys.argv[1])]
    results = {}
    for s in seeds:
        results[s] = diagnose_roll(s)
    with open("evaluation_assets/phase3_2_t_ground_reset/roll_diagnostic_seeds.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDiagnostic completed and saved to evaluation_assets/phase3_2_t_ground_reset/roll_diagnostic_seeds.json")
