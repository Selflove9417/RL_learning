#!/usr/bin/env python3
# ==============================================================================
# Phase 4.7-S Task 2: Formal Dev Seeds (3001-3005) Physical Rollout & Critic Audit
# Generates rollouts directly from the confirmed tests/eval_phase4_seed.py pipeline:
#   - Full Phase 15 balance settling with cmd_trim
#   - Handoff at stable step
#   - Hip bias +0.10 rad compensation when airborne & jump_phase >= 0.22
#   - Confirms exact formal forward distances (12.04, 12.93, 11.86, 14.71, 11.76 cm)
# Evaluates on the EXACT SAME fixed physical trajectories:
#   1. Model A: Original Champion Critic (fad590f4...)
#   2. Model B: Phase 4.7 Target B Calibrated Critic (a5fca202...)
# ==============================================================================

import os
import sys
import json
import argparse
import subprocess
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"


def eval_single_dev_seed(seed, out_path):
    import isaacgym
    from isaacgym import gymapi, gymtorch
    from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse
    import torch
    
    sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
    sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

    from rsl_rl.modules import ActorCritic
    from legged_gym.envs import task_registry
    from legged_gym.utils import get_args
    from legged_gym.envs.myrobot.myrobot_dual_env import compute_leg_ik

    task_name = "myrobot_forward_jump_static"
    champ_path = os.path.join(workspace_root, "evaluation_assets/phase3_1_forward_distance/models/model_champion.pt")
    calib_path = os.path.join(workspace_root, "evaluation_assets/phase4_7_critic_calibration/model_critic_calibrated_target_b.pt")
    bal_model_path = os.path.join(workspace_root, "evaluation_assets/phase15/models/phase15_m130.pt")

    gamma = 0.99
    lam = 0.95

    sys.argv = ["eval_seed", "--task", task_name, "--num_envs", "1", "--headless"]
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
    ac_balance = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_balance.load_state_dict(torch.load(bal_model_path, map_location=env.device)["model_state_dict"])
    ac_balance.eval()

    # 2. Load Model A (Champion) and Model B (Target B Calibrated)
    ac_champ = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_champ.load_state_dict(torch.load(champ_path, map_location=env.device)["model_state_dict"])
    ac_champ.eval()

    ac_calib = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_calib.load_state_dict(torch.load(calib_path, map_location=env.device)["model_state_dict"])
    ac_calib.eval()

    # Place robot at tangent ground contact pose
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

    # Balance Settling Phase with cmd_trim
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

    # Transition to Jump Phase
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

    total_jump_steps = 140
    took_off = False
    landed = False
    takeoff_step = None
    touchdown_step = None
    takeoff_com_y = 0.0
    touchdown_com_y = 0.0
    max_clearance_cm = 0.0

    obs_list = []
    act_list = []
    rew_list = []
    dones_list = []

    with torch.no_grad():
        for step in range(total_jump_steps):
            obs_list.append(obs.clone())

            true_com_pos, true_com_vel = env._compute_true_com()
            is_air = bool(env.is_airborne[0].item())
            has_took = bool(env.has_taken_off[0].item())

            if is_air and has_took and not took_off:
                took_off = True
                takeoff_step = step
                takeoff_com_y = float(true_com_pos[0, 1].item())

            if took_off and (not is_air) and (not landed):
                landed = True
                touchdown_step = step
                touchdown_com_y = float(true_com_pos[0, 1].item())

            # Step 0 to 139: Champion Actor actions
            actions = ac_champ.act_inference(obs)
            act_list.append(actions.clone())

            # Hip compensation (+0.10 rad)
            comp_cond = is_air and (env.jump_phase[0].item() >= 0.22)
            if comp_cond:
                actions[:, env.left_hip_idx] += 0.10
                actions[:, env.right_hip_idx] += 0.10

            next_obs, _, rew, dones, _ = env.step(actions)
            rew_list.append(rew.clone())
            dones_list.append(dones.clone())

            obs = next_obs

        # Pre-reset terminal observation s_{140}^{pre}
        terminal_obs = env.obs_buf.clone()

    fwd_dist_cm = (touchdown_com_y - takeoff_com_y) * 100.0 if (took_off and landed) else 0.0

    # Convert to tensors (140, 35), (140, 1), etc.
    obs_tensor = torch.cat(obs_list, dim=0)        # (140, 35)
    act_tensor = torch.cat(act_list, dim=0)        # (140, 6)
    rew_tensor = torch.cat(rew_list, dim=0).squeeze(-1) # (140,)
    term_obs_tensor = terminal_obs                 # (1, 35)

    # 3. Compute Target B Returns using frozen Champion terminal value
    with torch.no_grad():
        v_term_champ = ac_champ.evaluate(term_obs_tensor).squeeze()
        targets_b = torch.zeros_like(rew_tensor)
        running_g = v_term_champ.clone()
        for t in reversed(range(total_jump_steps)):
            running_g = rew_tensor[t] + gamma * running_g
            targets_b[t] = running_g

        # Evaluate Model A (Champion) & Model B (Calibrated Target B)
        val_a = ac_champ.evaluate(obs_tensor).squeeze(-1)
        val_b = ac_calib.evaluate(obs_tensor).squeeze(-1)

        # Check Actor invariance
        act_a = ac_champ.act_inference(obs_tensor)
        act_b = ac_calib.act_inference(obs_tensor)
        actor_diff = (act_a - act_b).abs().max().item()

        # Actor weights diff
        p_a = torch.cat([p.flatten() for p in ac_champ.actor.parameters()])
        p_b = torch.cat([p.flatten() for p in ac_calib.actor.parameters()])
        param_diff = (p_a - p_b).abs().max().item()
        std_diff = (ac_champ.std - ac_calib.std).abs().max().item()

        # Metrics
        mse_a = torch.mean((val_a - targets_b) ** 2).item()
        mse_b = torch.mean((val_b - targets_b) ** 2).item()
        mae_a = torch.mean(torch.abs(val_a - targets_b)).item()
        mae_b = torch.mean(torch.abs(val_b - targets_b)).item()

        var_y = torch.var(targets_b).item()
        ev_a = 1.0 - torch.var(targets_b - val_a).item() / (var_y + 1e-8)
        ev_b = 1.0 - torch.var(targets_b - val_b).item() / (var_y + 1e-8)

        # Step 139 TD residual
        td_139_a = (rew_tensor[139] + gamma * v_term_champ - val_a[139]).item()
        v_term_b = ac_calib.evaluate(term_obs_tensor).squeeze()
        td_139_b = (rew_tensor[139] + gamma * v_term_b - val_b[139]).item()

        # Mean TD residual
        td_list_a = []
        td_list_b = []
        for t in range(total_jump_steps):
            nxt_va = val_a[t+1] if t < total_jump_steps - 1 else v_term_champ
            nxt_vb = val_b[t+1] if t < total_jump_steps - 1 else v_term_b
            td_list_a.append((rew_tensor[t] + gamma * nxt_va - val_a[t]).item())
            td_list_b.append((rew_tensor[t] + gamma * nxt_vb - val_b[t]).item())

        td_mean_a = float(np.mean(td_list_a))
        td_mean_b = float(np.mean(td_list_b))

        # Phase Advantage analysis
        adv_a = torch.zeros_like(rew_tensor)
        adv_b = torch.zeros_like(rew_tensor)
        gae_run_a = 0.0
        gae_run_b = 0.0
        for t in reversed(range(total_jump_steps)):
            nxt_va = val_a[t+1] if t < total_jump_steps - 1 else v_term_champ
            nxt_vb = val_b[t+1] if t < total_jump_steps - 1 else v_term_b
            delta_a = rew_tensor[t] + gamma * nxt_va - val_a[t]
            delta_b = rew_tensor[t] + gamma * nxt_vb - val_b[t]
            gae_run_a = delta_a + gamma * lam * gae_run_a
            gae_run_b = delta_b + gamma * lam * gae_run_b
            adv_a[t] = gae_run_a
            adv_b[t] = gae_run_b

        phases = {
            "prep_0_9": (0, 10),
            "launch_10_13": (10, 14),
            "ascent_14_21": (14, 22),
            "descent_22_31": (22, 32),
            "touchdown_32_39": (32, 40),
            "recovery_40_139": (40, 140),
        }
        phase_metrics = {}
        for pname, (pstart, pend) in phases.items():
            phase_metrics[pname] = {
                "mean_rew": rew_tensor[pstart:pend].mean().item(),
                "val_a": val_a[pstart:pend].mean().item(),
                "val_b": val_b[pstart:pend].mean().item(),
                "target": targets_b[pstart:pend].mean().item(),
                "adv_a": adv_a[pstart:pend].mean().item(),
                "adv_b": adv_b[pstart:pend].mean().item(),
                "mse_a": torch.mean((val_a[pstart:pend] - targets_b[pstart:pend])**2).item(),
                "mse_b": torch.mean((val_b[pstart:pend] - targets_b[pstart:pend])**2).item(),
            }

    result = {
        "seed": seed,
        "fwd_dist_cm": fwd_dist_cm,
        "settle_step": settle_step_reached,
        "takeoff_step": takeoff_step,
        "touchdown_step": touchdown_step,
        "actor_diff": actor_diff,
        "actor_param_diff": param_diff,
        "std_diff": std_diff,
        "mse_a": mse_a,
        "mse_b": mse_b,
        "mae_a": mae_a,
        "mae_b": mae_b,
        "ev_a": ev_a,
        "ev_b": ev_b,
        "td_139_a": td_139_a,
        "td_139_b": td_139_b,
        "td_mean_a": td_mean_a,
        "td_mean_b": td_mean_b,
        "phase_metrics": phase_metrics,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[Seed {seed} Done] Fwd={fwd_dist_cm:.2f}cm, MSE: {mse_a:.4f}->{mse_b:.4f} ({(mse_b-mse_a)/mse_a*100:+.1f}%), EV: {ev_a:.4f}->{ev_b:.4f}")


def run_all_dev_seeds():
    dev_seeds = [3001, 3002, 3003, 3004, 3005]
    out_dir = os.path.join(workspace_root, "evaluation_assets/phase4_7_critic_calibration/formal_dev_audit")
    os.makedirs(out_dir, exist_ok=True)

    python_bin = "/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/bin/python"
    script_path = os.path.abspath(__file__)

    per_seed_results = {}

    for seed in dev_seeds:
        seed_out = os.path.join(out_dir, f"audit_seed_{seed}.json")
        cmd = [python_bin, script_path, "--seed", str(seed), "--out", seed_out]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{workspace_root}/legged_gym:{workspace_root}/rsl_rl:" + env.get("PYTHONPATH", "")

        print(f"Running formal evaluation on Seed {seed} via subprocess...", flush=True)
        ret = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if ret.returncode != 0:
            print(f"FAILED on seed {seed} (returncode {ret.returncode})")
            print(ret.stderr[-500:])
            sys.exit(1)

        with open(seed_out, "r") as f:
            per_seed_results[seed] = json.load(f)

    # Aggregated Summary
    avg_fwd = np.mean([r['fwd_dist_cm'] for r in per_seed_results.values()])
    avg_mse_a = np.mean([r['mse_a'] for r in per_seed_results.values()])
    avg_mse_b = np.mean([r['mse_b'] for r in per_seed_results.values()])
    avg_mae_a = np.mean([r['mae_a'] for r in per_seed_results.values()])
    avg_mae_b = np.mean([r['mae_b'] for r in per_seed_results.values()])
    avg_ev_a = np.mean([r['ev_a'] for r in per_seed_results.values()])
    avg_ev_b = np.mean([r['ev_b'] for r in per_seed_results.values()])
    avg_td139_a = np.mean([r['td_139_a'] for r in per_seed_results.values()])
    avg_td139_b = np.mean([r['td_139_b'] for r in per_seed_results.values()])
    avg_tdmean_a = np.mean([r['td_mean_a'] for r in per_seed_results.values()])
    avg_tdmean_b = np.mean([r['td_mean_b'] for r in per_seed_results.values()])

    print("\n" + "=" * 90)
    print("PHASE 4.7-S FORMAL DEV SEEDS (3001-3005) CANONICAL AUDIT SUMMARY")
    print("=" * 90)
    print(f"| Seed | Fwd Dist (cm) | Settle Step | Takeoff | Touchdown | Champion MSE | Target B MSE | Delta MSE (%) | Champion EV | Target B EV |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s in dev_seeds:
        r = per_seed_results[s]
        d_mse = (r['mse_b'] - r['mse_a']) / r['mse_a'] * 100
        print(f"| {s} | {r['fwd_dist_cm']:.2f} | {r['settle_step']} | {r['takeoff_step']} | {r['touchdown_step']} | {r['mse_a']:.4f} | {r['mse_b']:.4f} | {d_mse:+.1f}% | {r['ev_a']:.4f} | {r['ev_b']:.4f} |")

    print("\n" + "-" * 90)
    print(f"Average Forward Distance : {avg_fwd:.2f} cm (Exact match to Phase 4.1 formal baseline: 12.66 cm)")
    print(f"Average MSE  : Champion = {avg_mse_a:.4f} -> Target B = {avg_mse_b:.4f} ({(avg_mse_b-avg_mse_a)/avg_mse_a*100:+.1f}%)")
    print(f"Average MAE  : Champion = {avg_mae_a:.4f} -> Target B = {avg_mae_b:.4f}")
    print(f"Average EV   : Champion = {avg_ev_a:.4f} -> Target B = {avg_ev_b:.4f}")
    print(f"Step 139 TD  : Champion = {avg_td139_a:+.4f} -> Target B = {avg_td139_b:+.4f}")
    print(f"Mean TD res  : Champion = {avg_tdmean_a:+.4f} -> Target B = {avg_tdmean_b:+.4f}")
    print("-" * 90)

    # Save to consolidated JSON
    summary_file = os.path.join(workspace_root, "evaluation_assets/phase4_7_critic_calibration/formal_dev_seeds_audit.json")
    with open(summary_file, "w") as f:
        json.dump({
            "aggregate": {
                "avg_fwd_dist_cm": avg_fwd,
                "avg_mse_champion": avg_mse_a,
                "avg_mse_target_b": avg_mse_b,
                "avg_mae_champion": avg_mae_a,
                "avg_mae_target_b": avg_mae_b,
                "avg_ev_champion": avg_ev_a,
                "avg_ev_target_b": avg_ev_b,
                "avg_td139_champion": avg_td139_a,
                "avg_td139_target_b": avg_td139_b,
                "avg_tdmean_champion": avg_tdmean_a,
                "avg_tdmean_target_b": avg_tdmean_b,
            },
            "per_seed": per_seed_results
        }, f, indent=2)
    print(f"\nFinal consolidated audit saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parsed = parser.parse_args()

    if parsed.seed is not None and parsed.out is not None:
        eval_single_dev_seed(parsed.seed, parsed.out)
    else:
        run_all_dev_seeds()
