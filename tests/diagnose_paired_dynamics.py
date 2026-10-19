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
from legged_gym.envs.myrobot.myrobot_dual_env import compute_exact_cylinder_clearance, compute_leg_ik

def run_paired_dynamics(seed):
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["paired_dyn", "--task", task_name, "--num_envs", "2"]
    args = get_args()
    args.headless = True
    args.seed = seed

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.normalization.clip_actions = 100.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.num_envs = 2
    env_cfg.seed = seed
    env_cfg.noise.add_noise = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # 1. Balance model
    bal_model_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase15/models/phase15_m130.pt"
    ac_balance = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ckpt_bal = torch.load(bal_model_path, map_location=env.device)
    ac_balance.load_state_dict(ckpt_bal["model_state_dict"])
    ac_balance.eval()

    # 2. Champion Model (Env 0)
    champ_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    ac_champ = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ckpt_c = torch.load(champ_path, map_location=env.device)
    ac_champ.load_state_dict(ckpt_c["model_state_dict"])
    ac_champ.eval()

    # 3. Iter 5 Model (Env 1)
    iter5_path = "/home/xy/IsaacTu_jump/legged_gym/logs/myrobot_forward_jump_phase4/Oct17_10-50-44_phase4_4_pilot_sigmoid_5iter/model_5.pt"
    ac_iter5 = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ckpt_5 = torch.load(iter5_path, map_location=env.device)
    ac_iter5.load_state_dict(ckpt_5["model_state_dict"])
    ac_iter5.eval()

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

    # Settling Phase
    bal_filtered_actions = torch.zeros(env.num_envs, 6, device=env.device)
    pi_v_filt = torch.zeros(env.num_envs, device=env.device)
    h_ctrl = 0.39774
    norm_h = ((h_ctrl - 0.40) / 0.15)
    norm_h_tensor = torch.tensor([[norm_h], [norm_h]], device=env.device)
    kp_leg, kd_leg = 150.0, 6.0
    tau_limit_w = 12.0
    q1_ik, q2_ik = compute_leg_ik(torch.tensor([h_ctrl], device=env.device))

    for step in range(75):
        dof_pos_bal = (env.dof_pos - env.default_dof_pos).clone()
        dof_pos_bal[:, env.wheel_indices] = 0.0
        v_fwd = env.base_lin_vel[:, 1]
        pi_v_filt = 0.85 * pi_v_filt + 0.15 * v_fwd
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

    # Jump Phase
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

    trace_c = []
    trace_5 = []

    impulse_y_c, impulse_y_5 = 0.0, 0.0
    impulse_z_c, impulse_z_5 = 0.0, 0.0
    mg = 22.0 * 9.81

    total_jump_steps = 40

    for step in range(total_jump_steps):
        env._refresh_physical_state(update_has_taken_off=False)
        env.compute_observations()
        obs = env.obs_buf.clone()

        with torch.no_grad():
            act_c = ac_champ.act_inference(obs[0:1])
            act_5 = ac_iter5.act_inference(obs[1:2])
            act_raw = torch.cat([act_c, act_5], dim=0)

        executed_actions = act_raw.clone()
        comp_mask = env.is_airborne & (env.jump_phase >= 0.22)
        if comp_mask.any():
            executed_actions[comp_mask, env.left_hip_idx] += 0.10
            executed_actions[comp_mask, env.right_hip_idx] += 0.10

        clipped = torch.clip(executed_actions, -100.0, 100.0)
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

        # Step physics via LeggedRobot base
        super(type(env).__bases__[0].__bases__[0], env).step(executed_actions)

        # Measurements
        c_pos, c_vel = env._compute_true_com()
        b_vel = env.base_lin_vel.cpu().numpy()
        b_ang = env.base_ang_vel.cpu().numpy()
        r_raw, p_raw, _ = get_euler_xyz(env.base_quat)
        pitch = (torch.atan2(torch.sin(r_raw), torch.cos(r_raw)) * 180.0 / np.pi).cpu().numpy()
        roll = (torch.atan2(torch.sin(p_raw), torch.cos(p_raw)) * 180.0 / np.pi).cpu().numpy()

        fn_L = env.contact_forces[:, env.left_wheel_body, 2].cpu().numpy()
        fn_R = env.contact_forces[:, env.right_wheel_body, 2].cpu().numpy()
        fy_L = env.contact_forces[:, env.left_wheel_body, 1].cpu().numpy()
        fy_R = env.contact_forces[:, env.right_wheel_body, 1].cpu().numpy()

        torques = env.torques.cpu().numpy()
        dof_pos = env.dof_pos.cpu().numpy()
        dof_vel = env.dof_vel.cpu().numpy()
        is_air = env.is_airborne.cpu().numpy()

        # Impulse integration (steps 5 to liftoff)
        if not is_air[0] and step >= 5:
            impulse_y_c += (fy_L[0] + fy_R[0]) * env.dt
            impulse_z_c += (fn_L[0] + fn_R[0] - mg) * env.dt
        if not is_air[1] and step >= 5:
            impulse_y_5 += (fy_L[1] + fy_R[1]) * env.dt
            impulse_z_5 += (fn_L[1] + fn_R[1] - mg) * env.dt

        # Takeoff vy reward
        def calc_rew_vy(vy, vz, phi, air):
            target_vy = 0.22
            vz_gate = np.clip((vz - 0.8) / 0.8, 0.0, 1.0)
            vel_tracking = np.exp(-((vy - target_vy) ** 2) / (2.0 * (0.06 ** 2)))
            in_launch = (phi >= 0.05) and (phi <= 0.22) and (not air)
            return 3.0 * vel_tracking * vz_gate if in_launch else 0.0

        rew_vy_c = calc_rew_vy(b_vel[0, 1], b_vel[0, 2], env.jump_phase[0].item(), is_air[0])
        rew_vy_5 = calc_rew_vy(b_vel[1, 1], b_vel[1, 2], env.jump_phase[1].item(), is_air[1])

        record_c = {
            "step": step, "phi": env.jump_phase[0].item(), "is_air": bool(is_air[0]),
            "com_vy": c_vel[0, 1].item(), "com_vz": c_vel[0, 2].item(),
            "base_vy": b_vel[0, 1], "base_vz": b_vel[0, 2],
            "pitch_deg": pitch[0], "roll_deg": roll[0], "pitch_rate": b_ang[0, 0],
            "fn_tot": fn_L[0] + fn_R[0], "fy_tot": fy_L[0] + fy_R[0],
            "knee_tau_L": torques[0, env.left_knee_idx], "wheel_tau_L": torques[0, env.left_wheel_idx],
            "knee_vel_L": dof_vel[0, env.left_knee_idx], "rew_vy": rew_vy_c
        }
        record_5 = {
            "step": step, "phi": env.jump_phase[1].item(), "is_air": bool(is_air[1]),
            "com_vy": c_vel[1, 1].item(), "com_vz": c_vel[1, 2].item(),
            "base_vy": b_vel[1, 1], "base_vz": b_vel[1, 2],
            "pitch_deg": pitch[1], "roll_deg": roll[1], "pitch_rate": b_ang[1, 0],
            "fn_tot": fn_L[1] + fn_R[1], "fy_tot": fy_L[1] + fy_R[1],
            "knee_tau_L": torques[1, env.left_knee_idx], "wheel_tau_L": torques[1, env.left_wheel_idx],
            "knee_vel_L": dof_vel[1, env.left_knee_idx], "rew_vy": rew_vy_5
        }
        trace_c.append(record_c)
        trace_5.append(record_5)

    print("=== TRACE C (Champion) ===")
    for t in trace_c[:16]:
        print(f"step {t['step']:2d} phi={t['phi']:.2f} air={t['is_air']} fn={t['fn_tot']:6.1f} com_vz={t['com_vz']:6.2f} com_vy={t['com_vy']:6.2f} base_vz={t['base_vz']:6.2f} base_vy={t['base_vy']:6.2f}")

    print("=== TRACE 5 (Iter 5) ===")
    for t in trace_5[:16]:
        print(f"step {t['step']:2d} phi={t['phi']:.2f} air={t['is_air']} fn={t['fn_tot']:6.1f} com_vz={t['com_vz']:6.2f} com_vy={t['com_vy']:6.2f} base_vz={t['base_vz']:6.2f} base_vy={t['base_vy']:6.2f}")

    # Liftoff condition: true takeoff occurs during/after push-off (step >= 8) when airborne
    lift_c = next((t["step"] for t in trace_c if t["step"] >= 8 and t["is_air"]), -1)
    lift_5 = next((t["step"] for t in trace_5 if t["step"] >= 8 and t["is_air"]), -1)

    out = {
        "seed": seed,
        "lift_c": lift_c,
        "lift_5": lift_5,
        "com_vy_lift_c": trace_c[lift_c]["com_vy"],
        "com_vy_lift_5": trace_5[lift_5]["com_vy"],
        "com_vz_lift_c": trace_c[lift_c]["com_vz"],
        "com_vz_lift_5": trace_5[lift_5]["com_vz"],
        "impulse_y_c": impulse_y_c,
        "impulse_y_5": impulse_y_5,
        "impulse_z_c": impulse_z_c,
        "impulse_z_5": impulse_z_5,
        "pitch_lift_c": trace_c[lift_c]["pitch_deg"],
        "pitch_lift_5": trace_5[lift_5]["pitch_deg"],
        "pitch_rate_lift_c": trace_c[lift_c]["pitch_rate"],
        "pitch_rate_lift_5": trace_5[lift_5]["pitch_rate"],
        "sum_rew_vy_c": sum(t["rew_vy"] for t in trace_c),
        "sum_rew_vy_5": sum(t["rew_vy"] for t in trace_5),
        "step_9_c": trace_c[9],
        "step_9_5": trace_5[9],
        "step_10_c": trace_c[10],
        "step_10_5": trace_5[10],
    }
    def to_py(x):
        if isinstance(x, (np.floating, np.float32, np.float64)):
            return float(x)
        elif isinstance(x, (np.integer, np.int32, np.int64)):
            return int(x)
        elif isinstance(x, (np.ndarray, list, tuple)):
            return [to_py(i) for i in x]
        elif isinstance(x, dict):
            return {k: to_py(v) for k, v in x.items()}
        elif torch.is_tensor(x):
            return to_py(x.item() if x.numel() == 1 else x.tolist())
        return x

    return to_py(out)

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3001)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()
    data = run_paired_dynamics(args.seed)
    print(json.dumps(data, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
