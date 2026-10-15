"""
Rigorous Multi-Seed Evaluation Script for MyRobot Jump PPO Policy (Phase 2.1 Audit)
-----------------------------------------------------------------------------------
Evaluates trained PPO jump checkpoints across multiple independent random seeds.
Measures:
1. Strict 20 cm reach rate (h_jump >= 0.20 m) vs 18 cm rate (h_jump >= 0.18 m)
2. True bilateral lift-off
3. Single continuous airborne flight duration vs cumulative airborne time
4. Dynamic trajectory metrics: takeoff vz, apex COM height, touchdown vz, peak landing impact
5. Attitude regulation: max flight pitch/roll, recovery phase pitch/roll
6. Actuator saturation: Hip (140 Nm), Knee (120 Nm) torque and angular speed
7. Non-wheel ground collisions and unexpected resets
8. Post-landing displacement (dx, dy) and upright recovery
9. Joint pass rate (height >= 20 cm AND stable upright landing without crash)
"""

import os
import sys
import math
import json
import argparse

# Isaac Gym MUST be imported before torch
import isaacgym
from isaacgym import gymapi, gymtorch
import torch
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args


def quat_to_euler(q):
    """Convert quaternion [x, y, z, w] to euler angles [roll, pitch, yaw] in radians."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def evaluate_checkpoint(model_path, seed=1, num_envs=16, eval_steps=300):
    print("=" * 80)
    print(f"Loading checkpoint : {model_path}")
    print(f"Random seed        : {seed}")
    print(f"Number of envs     : {num_envs}")
    print(f"Evaluation steps   : {eval_steps} ({eval_steps * 0.02:.1f} s, {eval_steps * 0.02 / 2.0:.1f} jump cycles)")
    print("=" * 80)

    # Set up arguments for Isaac Gym
    sys.argv = [sys.argv[0], "--headless", "--task", "myrobot_jump"]
    args = get_args()
    args.headless = True
    args.task = "myrobot_jump"
    args.seed = seed

    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_jump")
    env_cfg.seed = seed
    env_cfg.env.num_envs = num_envs
    env_cfg.env.episode_length_s = eval_steps * 0.02 + 5.0  # prevent timeout resets during eval
    env_cfg.commands.init_jump_height = 0.20
    env_cfg.commands.target_jump_height = 0.20
    env_cfg.domain_rand.push_robots = False

    # Create environment
    env, _ = task_registry.make_env(name="myrobot_jump", args=args, env_cfg=env_cfg)

    # Load Actor model
    checkpoint = torch.load(model_path, map_location=env.device)
    
    from rsl_rl.modules import ActorCritic
    actor_critic = ActorCritic(
        num_actor_obs=env_cfg.env.num_observations,
        num_critic_obs=env_cfg.env.num_observations,
        num_actions=env_cfg.env.num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation='elu',
        init_noise_std=1.0,
    ).to(env.device)
    actor_critic.load_state_dict(checkpoint['model_state_dict'])
    actor_critic.eval()
    policy = actor_critic.act_inference

    # Reset environment
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    initial_pos = env.root_states[:, 0:3].clone()

    # Tracking metrics per env
    max_bilateral_clearance = torch.zeros(num_envs, device=env.device)
    max_clearance_L = torch.zeros(num_envs, device=env.device)
    max_clearance_R = torch.zeros(num_envs, device=env.device)
    max_com_z = torch.zeros(num_envs, device=env.device)

    # Flight duration tracking
    current_continuous_air_time = torch.zeros(num_envs, device=env.device)
    max_continuous_air_time = torch.zeros(num_envs, device=env.device)
    cumulative_air_time = torch.zeros(num_envs, device=env.device)

    # Dynamic velocities
    takeoff_vz_recorded = torch.zeros(num_envs, device=env.device)
    touchdown_vz_recorded = torch.zeros(num_envs, device=env.device)
    prev_airborne = torch.zeros(num_envs, device=env.device, dtype=torch.bool)

    # Forces and torques
    peak_impact_forces = torch.zeros(num_envs, device=env.device)
    max_hip_torque = torch.zeros(num_envs, device=env.device)
    max_knee_torque = torch.zeros(num_envs, device=env.device)
    max_hip_vel = torch.zeros(num_envs, device=env.device)
    max_knee_vel = torch.zeros(num_envs, device=env.device)

    # Attitude angles
    max_pitch_deg = torch.zeros(num_envs, device=env.device)
    max_roll_deg = torch.zeros(num_envs, device=env.device)
    recovery_pitch_sum = torch.zeros(num_envs, device=env.device)
    recovery_steps_count = torch.zeros(num_envs, device=env.device)

    # Contacts & Resets
    non_wheel_contacts = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
    unexpected_reset_count = torch.zeros(num_envs, device=env.device, dtype=torch.long)

    # Step simulation
    for step in range(eval_steps):
        with torch.no_grad():
            actions = policy(obs)
        step_res = env.step(actions)
        obs, _, _, dones, _ = step_res

        # Count unexpected resets
        unexpected_reset_count += dones.long()

        # Update clearances
        clr_L = env.wheel_clearance[:, 0]
        clr_R = env.wheel_clearance[:, 1]
        bilateral_clr = env.bilateral_clearance

        max_clearance_L = torch.maximum(max_clearance_L, clr_L)
        max_clearance_R = torch.maximum(max_clearance_R, clr_R)
        max_bilateral_clearance = torch.maximum(max_bilateral_clearance, bilateral_clr)
        max_com_z = torch.maximum(max_com_z, env.root_states[:, 2])

        airborne = env.is_airborne
        cumulative_air_time += airborne.float() * env.dt

        # Continuous flight tracking
        current_continuous_air_time = torch.where(
            airborne,
            current_continuous_air_time + env.dt,
            torch.zeros_like(current_continuous_air_time)
        )
        max_continuous_air_time = torch.maximum(max_continuous_air_time, current_continuous_air_time)

        # Takeoff detection: transitioned from grounded to airborne during pushoff (phase > 0.08)
        just_took_off = airborne & (~prev_airborne) & (env.jump_phase > 0.08)
        for i in range(num_envs):
            if just_took_off[i].item() and takeoff_vz_recorded[i].item() == 0.0:
                takeoff_vz_recorded[i] = env.base_lin_vel[i, 2]

        # Touchdown detection: transitioned from airborne to grounded after flight (phase > 0.22)
        just_touched_down = (~airborne) & prev_airborne & (env.jump_phase > 0.22)
        for i in range(num_envs):
            if just_touched_down[i].item() and touchdown_vz_recorded[i].item() == 0.0:
                touchdown_vz_recorded[i] = env.base_lin_vel[i, 2]

        prev_airborne[:] = airborne

        # Contact impact
        fn_max = torch.maximum(
            env.contact_forces[:, env.left_wheel_body, 2],
            env.contact_forces[:, env.right_wheel_body, 2]
        )
        peak_impact_forces = torch.maximum(peak_impact_forces, fn_max)

        # Attitude angles
        for i in range(num_envs):
            quat = env.root_states[i, 3:7].cpu().numpy()
            r, p, y = quat_to_euler(quat)
            p_deg = abs(math.degrees(p))
            r_deg = abs(math.degrees(r))
            max_pitch_deg[i] = max(max_pitch_deg[i].item(), p_deg)
            max_roll_deg[i] = max(max_roll_deg[i].item(), r_deg)

            # Recovery phase stats (phi in [0.55, 1.0])
            if env.jump_phase[i].item() >= 0.55:
                recovery_pitch_sum[i] += p_deg
                recovery_steps_count[i] += 1

        # Joint torques & velocities
        hip_torques = torch.maximum(
            torch.abs(env.torques[:, env.left_hip_idx]),
            torch.abs(env.torques[:, env.right_hip_idx])
        )
        knee_torques = torch.maximum(
            torch.abs(env.torques[:, env.left_knee_idx]),
            torch.abs(env.torques[:, env.right_knee_idx])
        )
        max_hip_torque = torch.maximum(max_hip_torque, hip_torques)
        max_knee_torque = torch.maximum(max_knee_torque, knee_torques)

        hip_vels = torch.maximum(
            torch.abs(env.dof_vel[:, env.left_hip_idx]),
            torch.abs(env.dof_vel[:, env.right_hip_idx])
        )
        knee_vels = torch.maximum(
            torch.abs(env.dof_vel[:, env.left_knee_idx]),
            torch.abs(env.dof_vel[:, env.right_knee_idx])
        )
        max_hip_vel = torch.maximum(max_hip_vel, hip_vels)
        max_knee_vel = torch.maximum(max_knee_vel, knee_vels)

        # Check non-wheel ground collisions (> 15 N)
        if len(env.termination_contact_indices) > 0:
            link_f = torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1)
            bad_contact = torch.any(link_f > 15.0, dim=1)
            non_wheel_contacts |= bad_contact

    # Compute final displacements
    final_pos = env.root_states[:, 0:3].clone()
    disp_xy = final_pos[:, 0:2] - initial_pos[:, 0:2]

    # URDF effort limits
    HIP_EFFORT_LIMIT = 140.0
    KNEE_EFFORT_LIMIT = 120.0
    HIP_VEL_LIMIT = 12.0
    KNEE_VEL_LIMIT = 14.0

    # Compile results
    results = []
    print("\n" + "=" * 115)
    print(f"{'Env':>3} | {'Max Clr (m)':>11} | {'20cm?':>6} | {'18cm?':>6} | {'Single Air':>10} | {'Cumul Air':>10} | {'Takeoff vz':>10} | {'Land F':>8} | {'Pitch':>6} | {'Crash?':>6} | {'Status':>8}")
    print("-" * 115)

    reach_20cm_count = 0
    reach_18cm_count = 0
    stable_landing_count = 0
    joint_pass_count = 0

    clr_list = []
    single_air_list = []
    cumul_air_list = []
    impact_list = []
    takeoff_vz_list = []
    touchdown_vz_list = []
    disp_x_list = []
    disp_y_list = []

    for i in range(num_envs):
        b_clr = max_bilateral_clearance[i].item()
        l_clr = max_clearance_L[i].item()
        r_clr = max_clearance_R[i].item()
        s_air = max_continuous_air_time[i].item()
        c_air = cumulative_air_time[i].item()
        v_to = takeoff_vz_recorded[i].item()
        v_td = touchdown_vz_recorded[i].item()
        f_land = peak_impact_forces[i].item()
        p_max = max_pitch_deg[i].item()
        r_max = max_roll_deg[i].item()
        rec_count = max(recovery_steps_count[i].item(), 1.0)
        p_rec = recovery_pitch_sum[i].item() / rec_count
        crashed = non_wheel_contacts[i].item()
        resets = unexpected_reset_count[i].item()
        dx = disp_xy[i, 0].item()
        dy = disp_xy[i, 1].item()

        reach_20 = (b_clr >= 0.200)
        reach_18 = (b_clr >= 0.180)
        liftoff = (b_clr >= 0.010)
        stable = (p_max < 30.0) and (not crashed) and (resets == 0)
        joint_pass = reach_20 and stable

        if reach_20: reach_20cm_count += 1
        if reach_18: reach_18cm_count += 1
        if stable: stable_landing_count += 1
        if joint_pass: joint_pass_count += 1

        clr_list.append(b_clr)
        single_air_list.append(s_air)
        cumul_air_list.append(c_air)
        impact_list.append(f_land)
        takeoff_vz_list.append(v_to)
        touchdown_vz_list.append(v_td)
        disp_x_list.append(dx)
        disp_y_list.append(dy)

        status_str = "PASS-20" if joint_pass else ("PASS-18" if (reach_18 and stable) else ("CRASH" if crashed else "LOW"))
        print(f"{i:>3} | {b_clr:11.4f} | {str(reach_20):>6} | {str(reach_18):>6} | {s_air:9.2f}s | {c_air:9.2f}s | {v_to:8.2f}m/s | {f_land:7.1f}N | {p_max:5.1f}° | {str(crashed):>6} | {status_str:>8}")

        results.append({
            "env_id": i,
            "seed": seed,
            "bilateral_clearance_m": b_clr,
            "left_clearance_m": l_clr,
            "right_clearance_m": r_clr,
            "reach_20cm": reach_20,
            "reach_18cm": reach_18,
            "bilateral_liftoff": liftoff,
            "single_flight_time_s": s_air,
            "cumulative_air_time_s": c_air,
            "takeoff_vz_mps": v_to,
            "touchdown_vz_mps": v_td,
            "max_com_z_m": max_com_z[i].item(),
            "peak_impact_force_N": f_land,
            "max_pitch_deg": p_max,
            "max_roll_deg": r_max,
            "recovery_mean_pitch_deg": p_rec,
            "displacement_x_m": dx,
            "displacement_y_m": dy,
            "non_wheel_collision": crashed,
            "unexpected_resets": resets,
            "stable_landing": stable,
            "joint_pass_20cm": joint_pass,
            "max_hip_torque_Nm": max_hip_torque[i].item(),
            "hip_torque_saturation_pct": (max_hip_torque[i].item() / HIP_EFFORT_LIMIT) * 100.0,
            "max_knee_torque_Nm": max_knee_torque[i].item(),
            "knee_torque_saturation_pct": (max_knee_torque[i].item() / KNEE_EFFORT_LIMIT) * 100.0,
            "max_hip_vel_rads": max_hip_vel[i].item(),
            "hip_vel_saturation_pct": (max_hip_vel[i].item() / HIP_VEL_LIMIT) * 100.0,
            "max_knee_vel_rads": max_knee_vel[i].item(),
            "knee_vel_saturation_pct": (max_knee_vel[i].item() / KNEE_VEL_LIMIT) * 100.0,
        })

    print("-" * 115)
    reach_20_rate = (reach_20cm_count / num_envs) * 100.0
    reach_18_rate = (reach_18cm_count / num_envs) * 100.0
    stable_rate = (stable_landing_count / num_envs) * 100.0
    joint_rate = (joint_pass_count / num_envs) * 100.0

    print(f"Summary for Seed {seed} ({num_envs} environments):")
    print(f"  Strict 20 cm Reach Rate : {reach_20_rate:.2f}% ({reach_20cm_count}/{num_envs})")
    print(f"  18 cm Reach Rate        : {reach_18_rate:.2f}% ({reach_18cm_count}/{num_envs})")
    print(f"  Stable Landing Rate     : {stable_rate:.2f}% ({stable_landing_count}/{num_envs})")
    print(f"  Joint Pass Rate (20cm)  : {joint_rate:.2f}% ({joint_pass_count}/{num_envs})")
    print(f"  Bilateral Clearance     : Mean = {np.mean(clr_list)*100:.2f} cm, Min = {np.min(clr_list)*100:.2f} cm, Max = {np.max(clr_list)*100:.2f} cm")
    print(f"  Single Flight Duration  : Mean = {np.mean(single_air_list):.3f} s (Min = {np.min(single_air_list):.3f} s, Max = {np.max(single_air_list):.3f} s)")
    print(f"  Cumulative Air Time     : Mean = {np.mean(cumul_air_list):.2f} s")
    print(f"  Takeoff vz              : Mean = {np.mean(takeoff_vz_list):.2f} m/s")
    print(f"  Touchdown vz            : Mean = {np.mean(touchdown_vz_list):.2f} m/s")
    print(f"  Peak Landing Force      : Mean = {np.mean(impact_list):.1f} N")
    print(f"  Post-Landing Disp       : dx = {np.mean(disp_x_list):+.3f} m, dy = {np.mean(disp_y_list):+.3f} m")
    print(f"  Actuator Peak Torques   : Hip = {torch.max(max_hip_torque).item():.1f} Nm ({torch.max(max_hip_torque).item()/HIP_EFFORT_LIMIT*100:.1f}%), Knee = {torch.max(max_knee_torque).item():.1f} Nm ({torch.max(max_knee_torque).item()/KNEE_EFFORT_LIMIT*100:.1f}%)")
    print(f"  Actuator Peak Speeds    : Hip = {torch.max(max_hip_vel).item():.1f} rad/s ({torch.max(max_hip_vel).item()/HIP_VEL_LIMIT*100:.1f}%), Knee = {torch.max(max_knee_vel).item():.1f} rad/s ({torch.max(max_knee_vel).item()/KNEE_VEL_LIMIT*100:.1f}%)")
    print("=" * 115)

    return {
        "seed": seed,
        "num_envs": num_envs,
        "eval_steps": eval_steps,
        "reach_20cm_rate_pct": reach_20_rate,
        "reach_18cm_rate_pct": reach_18_rate,
        "stable_landing_rate_pct": stable_rate,
        "joint_pass_rate_pct": joint_rate,
        "mean_clearance_m": float(np.mean(clr_list)),
        "min_clearance_m": float(np.min(clr_list)),
        "max_clearance_m": float(np.max(clr_list)),
        "mean_single_flight_time_s": float(np.mean(single_air_list)),
        "mean_cumulative_air_time_s": float(np.mean(cumul_air_list)),
        "mean_takeoff_vz_mps": float(np.mean(takeoff_vz_list)),
        "mean_touchdown_vz_mps": float(np.mean(touchdown_vz_list)),
        "mean_peak_impact_N": float(np.mean(impact_list)),
        "mean_disp_x_m": float(np.mean(disp_x_list)),
        "mean_disp_y_m": float(np.mean(disp_y_list)),
        "max_hip_torque_Nm": float(torch.max(max_hip_torque).item()),
        "max_knee_torque_Nm": float(torch.max(max_knee_torque).item()),
        "max_hip_vel_rads": float(torch.max(max_hip_vel).item()),
        "max_knee_vel_rads": float(torch.max(max_knee_vel).item()),
        "envs": results
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate MyRobot Jump PPO Policy")
    parser.add_argument("model_path", type=str, nargs="?", default="/home/xy/IsaacTu_jump/legged_gym/logs/myrobot_jump/Oct14_15-29-57_jump_20cm_stage5_landing/model_100.pt")
    parser.add_argument("out_json", type=str, nargs="?", default="/home/xy/IsaacTu_jump/evaluation_assets/jump_feasibility/eval_stage5_audit.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--eval_steps", type=int, default=300)
    
    parsed, unknown = parser.parse_known_args()

    all_seed_summaries = []
    os.makedirs(os.path.dirname(parsed.out_json), exist_ok=True)

    for s in parsed.seeds:
        res = evaluate_checkpoint(parsed.model_path, seed=s, num_envs=parsed.num_envs, eval_steps=parsed.eval_steps)
        all_seed_summaries.append(res)

    # Save complete audit json
    audit_data = {
        "model_path": parsed.model_path,
        "seeds_evaluated": parsed.seeds,
        "num_envs_per_seed": parsed.num_envs,
        "eval_steps_per_env": parsed.eval_steps,
        "overall_reach_20cm_rate_pct": float(np.mean([s["reach_20cm_rate_pct"] for s in all_seed_summaries])),
        "overall_reach_18cm_rate_pct": float(np.mean([s["reach_18cm_rate_pct"] for s in all_seed_summaries])),
        "overall_stable_landing_rate_pct": float(np.mean([s["stable_landing_rate_pct"] for s in all_seed_summaries])),
        "overall_joint_pass_rate_pct": float(np.mean([s["joint_pass_rate_pct"] for s in all_seed_summaries])),
        "overall_mean_clearance_cm": float(np.mean([s["mean_clearance_m"] for s in all_seed_summaries]) * 100.0),
        "overall_min_clearance_cm": float(np.min([s["min_clearance_m"] for s in all_seed_summaries]) * 100.0),
        "overall_max_clearance_cm": float(np.max([s["max_clearance_m"] for s in all_seed_summaries]) * 100.0),
        "overall_single_flight_duration_s": float(np.mean([s["mean_single_flight_time_s"] for s in all_seed_summaries])),
        "overall_cumulative_air_time_s": float(np.mean([s["mean_cumulative_air_time_s"] for s in all_seed_summaries])),
        "seed_summaries": all_seed_summaries
    }

    with open(parsed.out_json, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    print("\n" + "#" * 80)
    print("                      OVERALL 5-SEED AUDIT SUMMARY")
    print("#" * 80)
    print(f"Model Path                   : {parsed.model_path}")
    print(f"Seeds Tested                 : {parsed.seeds} (Total {len(parsed.seeds) * parsed.num_envs} environment runs)")
    print(f"Strict 20 cm Reach Rate      : {audit_data['overall_reach_20cm_rate_pct']:.2f}%")
    print(f"18 cm Reach Rate             : {audit_data['overall_reach_18cm_rate_pct']:.2f}%")
    print(f"Stable Upright Landing Rate  : {audit_data['overall_stable_landing_rate_pct']:.2f}%")
    print(f"Joint Pass Rate (20cm+Stable): {audit_data['overall_joint_pass_rate_pct']:.2f}%")
    print(f"Bilateral Clearance          : Mean = {audit_data['overall_mean_clearance_cm']:.2f} cm, Min = {audit_data['overall_min_clearance_cm']:.2f} cm, Max = {audit_data['overall_max_clearance_cm']:.2f} cm")
    print(f"Single Jump Flight Duration  : Mean = {audit_data['overall_single_flight_duration_s']:.3f} s (Ballistic ideal ~0.404 s)")
    print(f"Cumulative Air Time (6.0s)   : Mean = {audit_data['overall_cumulative_air_time_s']:.2f} s")
    print(f"Audit Results File           : {parsed.out_json}")
    print("#" * 80)
