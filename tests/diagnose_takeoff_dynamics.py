import os
import sys

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz
import torch
import numpy as np

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args


def record_trajectory(seed, model_path):
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["eval_dyn", "--task", task_name, "--num_envs", "1"]
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

    # 1. Balance model
    bal_model_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase15/models/phase15_m130.pt"
    ac_balance = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ckpt_bal = torch.load(bal_model_path, map_location=env.device)
    ac_balance.load_state_dict(ckpt_bal["model_state_dict"])
    ac_balance.eval()

    # 2. Jump model
    ac_jump = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ckpt_jump = torch.load(model_path, map_location=env.device)
    ac_jump.load_state_dict(ckpt_jump["model_state_dict"])
    ac_jump.eval()

    # Reset
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

    # Settling
    bal_filtered_actions = torch.zeros(env.num_envs, 6, device=env.device)
    pi_v_filt = torch.zeros(env.num_envs, device=env.device)
    h_ctrl = 0.39774
    norm_h = ((h_ctrl - 0.40) / 0.15)
    norm_h_tensor = torch.tensor([[norm_h]], device=env.device)
    kp_leg, kd_leg = 150.0, 6.0
    tau_limit_w = 12.0

    for step in range(75):
        dof_pos_bal = (env.dof_pos - env.default_dof_pos).clone()
        dof_pos_bal[:, env.wheel_indices] = 0.0
        v_fwd = env.base_lin_vel[:, 1]
        pi_v_filt = 0.85 * pi_v_filt + 0.15 * v_fwd
        q_yaw, _, _ = get_euler_xyz(env.base_quat)
        bal_cmds = torch.zeros(env.num_envs, 3, device=env.device)
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

        from legged_gym.envs.myrobot.myrobot_dual_env import compute_leg_ik
        q1_ik, q2_ik = compute_leg_ik(torch.tensor([h_ctrl], device=env.device))

        tgt_hip_L = env.default_dof_pos[:, env.left_hip_idx] + q1_ik + 0.25 * 0.7 * bal_filtered_actions[:, env.left_hip_idx]
        tgt_knee_L = env.default_dof_pos[:, env.left_knee_idx] + q2_ik + 0.25 * 0.5 * bal_filtered_actions[:, env.left_knee_idx]
        tgt_hip_R = env.default_dof_pos[:, env.right_hip_idx] + q1_ik + 0.25 * 0.7 * bal_filtered_actions[:, env.right_hip_idx]
        tgt_knee_R = env.default_dof_pos[:, env.right_knee_idx] + q2_ik + 0.25 * 0.5 * bal_filtered_actions[:, env.right_knee_idx]

        base_tau_L = 7.0 * torch.clamp(bal_filtered_actions[:, env.left_wheel_idx], -1.0, 1.0) - 0.3 * env.dof_vel[:, env.left_wheel_idx]
        base_tau_R = 7.0 * torch.clamp(bal_filtered_actions[:, env.right_wheel_idx], -1.0, 1.0) - 0.3 * env.dof_vel[:, env.right_wheel_idx]
        wheel_speed_error = env.dof_vel[:, env.left_wheel_idx] - env.dof_vel[:, env.right_wheel_idx]
        delta_tau_omega = torch.clamp(0.10 * wheel_speed_error, -1.5, 1.5)
        tau_c = 0.5 * (base_tau_L + base_tau_R)
        tau_d = 0.5 * (base_tau_R - base_tau_L) + delta_tau_omega
        tau_d_clamped = torch.clamp(tau_d, -tau_limit_w, tau_limit_w)
        tau_c_limit = torch.clamp(tau_limit_w - torch.abs(tau_d_clamped), min=0.0)
        tau_c_allocated = torch.clamp(tau_c, -tau_c_limit, tau_c_limit)
        tau_w_L = tau_c_allocated - tau_d_clamped
        tau_w_R = tau_c_allocated + tau_d_clamped

        for _ in range(env.cfg.control.decimation):
            torques = torch.zeros(env.num_envs, 6, device=env.device)
            torques[:, env.left_hip_idx] = kp_leg * (tgt_hip_L - env.dof_pos[:, env.left_hip_idx]) - kd_leg * env.dof_vel[:, env.left_hip_idx]
            torques[:, env.right_hip_idx] = kp_leg * (tgt_hip_R - env.dof_pos[:, env.right_hip_idx]) - kd_leg * env.dof_vel[:, env.right_hip_idx]
            torques[:, env.left_knee_idx] = kp_leg * (tgt_knee_L - env.dof_pos[:, env.left_knee_idx]) - kd_leg * env.dof_vel[:, env.left_knee_idx]
            torques[:, env.right_knee_idx] = kp_leg * (tgt_knee_R - env.dof_pos[:, env.right_knee_idx]) - kd_leg * env.dof_vel[:, env.right_knee_idx]
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

        # Check stable
        b_vy = env.base_lin_vel[:, 1]
        _, c_vel = env._compute_true_com()
        c_vy = c_vel[:, 1]
        b_vz = env.base_lin_vel[:, 2]
        fn_L = env.contact_forces[:, env.left_wheel_body, 2]
        fn_R = env.contact_forces[:, env.right_wheel_body, 2]
        r_raw, p_raw, _ = get_euler_xyz(env.base_quat)
        pitch_deg = torch.atan2(torch.sin(r_raw), torch.cos(r_raw)) * 180.0 / np.pi
        roll_deg = torch.atan2(torch.sin(p_raw), torch.cos(p_raw)) * 180.0 / np.pi
        is_stable = (
            (b_vy.abs() <= 0.03) & (c_vy.abs() <= 0.03) & (b_vz.abs() <= 0.05) &
            (fn_L >= 20.0) & (fn_L <= 180.0) & (fn_R >= 20.0) & (fn_R <= 180.0) &
            (pitch_deg.abs() <= 2.5) & (roll_deg.abs() <= 3.5)
        )
        if step >= 15 and is_stable.all():
            break

    # Jump Execution Phase
    env.jump_timer[:] = 0.0
    env.jump_phase[:] = 0.0
    env.jump_latched[:] = False
    env.has_taken_off[:] = False
    env.has_touched_down[:] = False
    env.one_shot_touchdown_event[:] = False
    env.filtered_actions[:] = 0.0
    env.last_actions[:] = 0.0
    env.episode_length_buf[:] = 0
    env.reset_buf[:] = 0

    trace = []
    total_steps = 40 # Steps 0 to 40 covers prep, launch, flight, and touchdown

    # For impulse integration (launch phase: phi in [0.05, 0.22])
    impulse_y = 0.0
    impulse_z = 0.0
    mg = 22.0 * 9.81 # 215.82 N

    for j_step in range(total_steps):
        # Refresh obs
        env._refresh_physical_state(update_has_taken_off=False)
        env.compute_observations()
        obs = env.obs_buf.clone()

        with torch.no_grad():
            act_raw = ac_jump.act_inference(obs)

        # Apply aerial compensation if applicable
        executed_actions = act_raw.clone()
        if env.is_airborne.any() and (env.jump_phase[0].item() >= 0.22):
            executed_actions[:, env.left_hip_idx] += 0.10
            executed_actions[:, env.right_hip_idx] += 0.10

        # Action smoothing
        clipped = torch.clip(executed_actions, -100.0, 100.0)
        env.last_actions[:] = env.filtered_actions
        env.filtered_actions[:] = 0.80 * clipped + 0.20 * env.last_actions

        # Step physics
        env.jump_timer += env.dt
        landing_completed = env.has_taken_off & env.has_touched_down & (env.jump_timer >= 0.50 * env.jump_cycle_time)
        env.jump_latched |= landing_completed
        clamped_timer = torch.where(
            env.jump_latched,
            torch.clamp(env.jump_timer, max=0.85 * env.jump_cycle_time),
            env.jump_timer
        )
        env.jump_phase[:] = clamped_timer / env.jump_cycle_time

        # Target angles
        # Hip: default + 0.80 * act[0], Knee: default + 0.70 * act[1], Wheel: vel target = 20 * act[2]
        tgt_hip_L = env.default_dof_pos[0, env.left_hip_idx].item() + 0.80 * env.filtered_actions[0, env.left_hip_idx].item()
        tgt_knee_L = env.default_dof_pos[0, env.left_knee_idx].item() + 0.70 * env.filtered_actions[0, env.left_knee_idx].item()
        tgt_hip_R = env.default_dof_pos[0, env.right_hip_idx].item() + 0.80 * env.filtered_actions[0, env.right_hip_idx].item()
        tgt_knee_R = env.default_dof_pos[0, env.right_knee_idx].item() + 0.70 * env.filtered_actions[0, env.right_knee_idx].item()

        # Step via super(MyRobotJump, env).step(...)
        super(type(env).__bases__[0].__bases__[0], env).step(executed_actions)

        # Record physics state
        c_pos, c_vel = env._compute_true_com()
        b_pos = env.root_states[0, :3].cpu().numpy()
        b_vel = env.base_lin_vel[0].cpu().numpy()
        b_ang = env.base_ang_vel[0].cpu().numpy()
        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat[0:1])
        pitch = torch.atan2(torch.sin(r_raw), torch.cos(r_raw)).item() * 180.0 / np.pi
        roll = torch.atan2(torch.sin(p_raw), torch.cos(p_raw)).item() * 180.0 / np.pi

        fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
        fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()
        fy_L = env.contact_forces[0, env.left_wheel_body, 1].item()
        fy_R = env.contact_forces[0, env.right_wheel_body, 1].item()

        torques = env.torques[0].cpu().numpy()
        dof_pos = env.dof_pos[0].cpu().numpy()
        dof_vel = env.dof_vel[0].cpu().numpy()

        is_air = env.is_airborne[0].item()
        phi = env.jump_phase[0].item()

        # Accumulate impulses during launch (steps 5 to 11, before liftoff)
        if not is_air and j_step >= 5:
            impulse_y += (fy_L + fy_R) * env.dt
            impulse_z += (fn_L + fn_R - mg) * env.dt

        # Compute takeoff forward velocity reward at this step
        vy = b_vel[1]
        target_vy = 0.22
        vz_gate = np.clip((b_vel[2] - 0.8) / 0.8, 0.0, 1.0)
        vel_tracking = np.exp(-((vy - target_vy) ** 2) / (2.0 * (0.06 ** 2)))
        in_launch = (phi >= 0.05) and (phi <= 0.22) and (not is_air)
        rew_takeoff_vy = 3.0 * vel_tracking * vz_gate if in_launch else 0.0

        trace.append({
            "step": j_step,
            "phi": phi,
            "is_air": is_air,
            "com_vy": c_vel[0, 1].item(),
            "com_vz": c_vel[0, 2].item(),
            "base_vy": b_vel[1],
            "base_vz": b_vel[2],
            "pitch_deg": pitch,
            "roll_deg": roll,
            "pitch_rate": b_ang[0],
            "fn_L": fn_L,
            "fn_R": fn_R,
            "fy_L": fy_L,
            "fy_R": fy_R,
            "tgt_hip_L": tgt_hip_L,
            "tgt_knee_L": tgt_knee_L,
            "tgt_hip_R": tgt_hip_R,
            "tgt_knee_R": tgt_knee_R,
            "act_hip_L": dof_pos[env.left_hip_idx],
            "act_knee_L": dof_pos[env.left_knee_idx],
            "act_hip_R": dof_pos[env.right_hip_idx],
            "act_knee_R": dof_pos[env.right_knee_idx],
            "vel_hip_L": dof_vel[env.left_hip_idx],
            "vel_knee_L": dof_vel[env.left_knee_idx],
            "tau_hip_L": torques[env.left_hip_idx],
            "tau_knee_L": torques[env.left_knee_idx],
            "tau_wheel_L": torques[env.left_wheel_idx],
            "tau_wheel_R": torques[env.right_wheel_idx],
            "rew_takeoff_vy": rew_takeoff_vy,
        })

    return trace, impulse_y, impulse_z


def compare_single_seed(seed):
    champ_path = "evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    iter5_path = "legged_gym/logs/myrobot_forward_jump_phase4/Oct17_10-50-44_phase4_4_pilot_sigmoid_5iter/model_5.pt"

    trace_c, imp_y_c, imp_z_c = record_trajectory(seed, champ_path)
    trace_5, imp_y_5, imp_z_5 = record_trajectory(seed, iter5_path)

    lift_c = next(t["step"] for t in trace_c if t["is_air"])
    lift_5 = next(t["step"] for t in trace_5 if t["is_air"])

    t_lift_c = trace_c[lift_c]
    t_lift_5 = trace_5[lift_5]

    peak_tau_knee_c = max(trace_c[i]["tau_knee_L"] for i in range(lift_c))
    peak_tau_knee_5 = max(trace_5[i]["tau_knee_L"] for i in range(lift_5))
    peak_tau_w_c = max(abs(trace_c[i]["tau_wheel_L"]) for i in range(lift_c))
    peak_tau_w_5 = max(abs(trace_5[i]["tau_wheel_L"]) for i in range(lift_5))

    sum_rew_vy_c = sum(t["rew_takeoff_vy"] for t in trace_c)
    sum_rew_vy_5 = sum(t["rew_takeoff_vy"] for t in trace_5)

    res = {
        "seed": seed,
        "lift_step_c": lift_c,
        "lift_step_5": lift_5,
        "liftoff_vy_c": t_lift_c["com_vy"],
        "liftoff_vy_5": t_lift_5["com_vy"],
        "liftoff_vz_c": t_lift_c["com_vz"],
        "liftoff_vz_5": t_lift_5["com_vz"],
        "impulse_y_c": imp_y_c,
        "impulse_y_5": imp_y_5,
        "impulse_z_c": imp_z_c,
        "impulse_z_5": imp_z_5,
        "peak_knee_c": peak_tau_knee_c,
        "peak_knee_5": peak_tau_knee_5,
        "peak_wheel_c": peak_tau_w_c,
        "peak_wheel_5": peak_tau_w_5,
        "pitch_lift_c": t_lift_c["pitch_deg"],
        "pitch_lift_5": t_lift_5["pitch_deg"],
        "pitch_rate_lift_c": t_lift_c["pitch_rate"],
        "pitch_rate_lift_5": t_lift_5["pitch_rate"],
        "sum_rew_vy_c": sum_rew_vy_c,
        "sum_rew_vy_5": sum_rew_vy_5,
        "step_8_c": trace_c[8],
        "step_8_5": trace_5[8],
        "step_9_c": trace_c[9],
        "step_9_5": trace_5[9],
        "step_10_c": trace_c[10],
        "step_10_5": trace_5[10],
    }
    return res

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3001)
    parser.add_argument("--out", type=str, default="")
    cli_args = parser.parse_args()

    data = compare_single_seed(cli_args.seed)
    print(f"\nSeed {data['seed']}:")
    print(f"  Liftoff Step: Baseline = Step {data['lift_step_c']} | Iter 5 = Step {data['lift_step_5']}")
    print(f"  Liftoff CoM Vy: Baseline = {data['liftoff_vy_c']:.4f} m/s | Iter 5 = {data['liftoff_vy_5']:.4f} m/s (Delta = {data['liftoff_vy_5'] - data['liftoff_vy_c']:+.4f})")
    print(f"  Liftoff CoM Vz: Baseline = {data['liftoff_vz_c']:.4f} m/s | Iter 5 = {data['liftoff_vz_5']:.4f} m/s (Delta = {data['liftoff_vz_5'] - data['liftoff_vz_c']:+.4f})")
    print(f"  Contact Impulse Y (N*s): Baseline = {data['impulse_y_c']:.4f} | Iter 5 = {data['impulse_y_5']:.4f} (Delta = {data['impulse_y_5'] - data['impulse_y_c']:+.4f})")
    print(f"  Contact Impulse Z (N*s): Baseline = {data['impulse_z_c']:.4f} | Iter 5 = {data['impulse_z_5']:.4f} (Delta = {data['impulse_z_5'] - data['impulse_z_c']:+.4f})")
    print(f"  Takeoff Vy Rew Sum: Baseline = {data['sum_rew_vy_c']:.4f} | Iter 5 = {data['sum_rew_vy_5']:.4f} (Delta = {data['sum_rew_vy_5'] - data['sum_rew_vy_c']:+.4f})")

    if cli_args.out:
        with open(cli_args.out, "w") as f:
            json.dump(data, f, indent=2)
