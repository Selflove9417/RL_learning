#!/usr/bin/env python3
# ==============================================================================
# MyRobot Forward Jump: Multi-Seed Benchmark & Validation Tool
# Evaluates Stage 8 Baseline vs. Phase 1 Trial 2 across 5 independent random seeds
# Accurately records single-jump physical metrics without inter-cycle pollution
# ==============================================================================

import os
import sys
import json
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz
import torch

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.envs.myrobot.myrobot_dual_env import compute_exact_cylinder_clearance


def evaluate_policy_single_jump(env, policy, is_forward=True, duration_s=2.0):
    """
    Evaluates exactly one complete jump cycle:
    Static preparation (0.0 - 0.12) -> Push-off (0.12 - 0.20) -> Flight (0.20 - 0.65) ->
    Touchdown cushion (0.65 - 0.75) -> Post-landing stationary balance recovery (0.75 - 2.0).
    """
    total_steps = int(duration_s / env.dt)
    env.reset()
    if not is_forward:
        env.commands[:, 1] = 0.0

    # Get body masses for CoM
    num_bodies = env.gym.get_actor_rigid_body_count(env.envs[0], 0)
    rb_masses = []
    for b in range(num_bodies):
        prop = env.gym.get_actor_rigid_body_properties(env.envs[0], 0)[b]
        rb_masses.append(prop.mass)
    rb_masses_t = torch.tensor(rb_masses, device=env.device, dtype=torch.float).unsqueeze(0).unsqueeze(-1)
    total_mass = rb_masses_t.sum().item()

    # Trackers
    takeoff_step = None
    touchdown_step = None
    max_clearance_cm = 0.0
    takeoff_root_y = 0.0
    takeoff_com_y = 0.0
    takeoff_root_vz = 0.0
    takeoff_root_vy = 0.0
    takeoff_com_vz = 0.0
    takeoff_com_vy = 0.0
    touchdown_pitch = 0.0
    touchdown_roll = 0.0
    touchdown_yaw = 0.0
    initial_yaw = 0.0
    final_yaw = 0.0

    peak_hip_tau = 0.0
    peak_knee_tau = 0.0
    peak_wheel_tau = 0.0

    hip_sat_count = 0
    knee_sat_count = 0
    wheel_sat_count = 0

    recovery_vys = []
    recovery_vxs = []
    abnormal_resets = 0

    took_off = False
    landed = False

    for step in range(total_steps):
        t_sim = step * env.dt
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        # Whole-body CoM
        rb_pos = env.rb_states[:, :, 0:3]
        rb_vel = env.rb_states[:, :, 7:10]
        com_pos = (rb_pos * rb_masses_t).sum(dim=1) / total_mass
        com_vel = (rb_vel * rb_masses_t).sum(dim=1) / total_mass

        root_pos = env.root_states[0, 0:3].cpu().numpy()
        root_vel_w = env.root_states[0, 7:10].cpu().numpy()
        base_lin_vel = env.base_lin_vel[0].cpu().numpy()

        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
        yaw = torch.atan2(torch.sin(y_raw[0]), torch.cos(y_raw[0])).item() * 180.0 / np.pi

        if step == 0:
            initial_yaw = yaw
        final_yaw = yaw

        clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(env)
        clr_min_cm = clr_min[0].item() * 100.0

        fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
        fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()

        is_air = bool(env.is_airborne[0].item())
        has_took = bool(env.has_taken_off[0].item())

        tau_hip = max(abs(env.torques[0, env.left_hip_idx].item()), abs(env.torques[0, env.right_hip_idx].item()))
        tau_knee = max(abs(env.torques[0, env.left_knee_idx].item()), abs(env.torques[0, env.right_knee_idx].item()))
        tau_wheel = max(abs(env.torques[0, env.left_wheel_idx].item()), abs(env.torques[0, env.right_wheel_idx].item()))

        peak_hip_tau = max(peak_hip_tau, tau_hip)
        peak_knee_tau = max(peak_knee_tau, tau_knee)
        peak_wheel_tau = max(peak_wheel_tau, tau_wheel)

        if tau_hip >= 119.5: hip_sat_count += 1
        if tau_knee >= 119.5: knee_sat_count += 1
        if tau_wheel >= 7.95: wheel_sat_count += 1

        if is_air and has_took:
            max_clearance_cm = max(max_clearance_cm, clr_min_cm)

        # Detect takeoff transition
        if is_air and has_took and not took_off:
            took_off = True
            takeoff_step = step
            takeoff_root_y = float(root_pos[1])
            takeoff_com_y = float(com_pos[0, 1].item())
            takeoff_root_vz = float(root_vel_w[2])
            takeoff_root_vy = float(root_vel_w[1])
            takeoff_com_vz = float(com_vel[0, 2].item())
            takeoff_com_vy = float(com_vel[0, 1].item())

        # Detect touchdown transition
        if took_off and (not is_air) and (not landed):
            landed = True
            touchdown_step = step
            touchdown_root_y = float(root_pos[1])
            touchdown_com_y = float(com_pos[0, 1].item())
            touchdown_pitch = pitch
            touchdown_roll = roll
            touchdown_yaw = yaw

        # Recovery monitoring (phase 0.45 - 0.95, after landing)
        phase = env.jump_phase[0].item()
        if landed and phase >= 0.45 and phase <= 0.95:
            recovery_vys.append(base_lin_vel[1])
            recovery_vxs.append(base_lin_vel[0])

        obs = env.get_observations()
        with torch.no_grad():
            actions = policy(obs)
        env.step(actions)

        if env.reset_buf[0].item() != 0 and step < total_steps - 5:
            abnormal_resets += 1

    valid_jump = took_off and landed and (max_clearance_cm >= 10.0)

    if took_off and landed:
        flight_duration_s = (touchdown_step - takeoff_step) * env.dt
        flight_steps = touchdown_step - takeoff_step
        root_forward_dist_cm = (touchdown_root_y - takeoff_root_y) * 100.0
        com_forward_dist_cm = (touchdown_com_y - takeoff_com_y) * 100.0
    else:
        flight_duration_s = 0.0
        flight_steps = 0
        root_forward_dist_cm = 0.0
        com_forward_dist_cm = 0.0

    mean_rec_vy = float(np.mean(recovery_vys)) if len(recovery_vys) > 0 else 0.0
    mean_rec_vx = float(np.mean(recovery_vxs)) if len(recovery_vxs) > 0 else 0.0
    yaw_drift = abs(final_yaw - initial_yaw)

    return {
        "valid_jump": valid_jump,
        "abnormal_resets": abnormal_resets,
        "max_clearance_cm": max_clearance_cm,
        "flight_duration_s": flight_duration_s,
        "flight_steps": flight_steps,
        "root_forward_dist_cm": root_forward_dist_cm,
        "com_forward_dist_cm": com_forward_dist_cm,
        "takeoff_root_vy_mps": takeoff_root_vy,
        "takeoff_root_vz_mps": takeoff_root_vz,
        "takeoff_com_vy_mps": takeoff_com_vy,
        "takeoff_com_vz_mps": takeoff_com_vz,
        "touchdown_pitch_deg": touchdown_pitch,
        "touchdown_roll_deg": touchdown_roll,
        "touchdown_yaw_deg": touchdown_yaw,
        "yaw_drift_deg": yaw_drift,
        "recovery_vy_mps": mean_rec_vy,
        "recovery_vx_mps": mean_rec_vx,
        "peak_hip_tau_nm": peak_hip_tau,
        "peak_knee_tau_nm": peak_knee_tau,
        "peak_wheel_tau_nm": peak_wheel_tau,
        "hip_sat_count": hip_sat_count,
        "knee_sat_count": knee_sat_count,
        "wheel_sat_count": wheel_sat_count
    }


def run_benchmark():
    output_dir = os.path.join(workspace_root, "evaluation_assets", "phase1_forward_jump")
    os.makedirs(output_dir, exist_ok=True)
    benchmark_json_path = os.path.join(output_dir, "multi_seed_benchmark_summary.json")

    sys.argv = ["benchmark_fwd", "--task", "myrobot_forward_jump", "--num_envs", "1"]
    args = get_args()
    args.headless = True

    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_forward_jump")
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 10.0
    env_cfg.noise.add_noise = False
    env_cfg.env.enable_camera_sensors = False

    seeds = [42, 100, 200, 300, 400]

    baseline_model_path = os.path.join(workspace_root, "evaluation_assets", "frozen_stage8", "model_40.pt")
    trial2_model_path = os.path.join(output_dir, "models", "model_30_yaw_aligned.pt")

    models = [
        {"name": "Stage 8 Baseline (In-Place)", "path": baseline_model_path, "is_forward": False},
        {"name": "Trial 2 (Forward Jump)", "path": trial2_model_path, "is_forward": True}
    ]

    all_results = {}

    for m_info in models:
        m_name = m_info["name"]
        m_path = m_info["path"]
        is_fwd = m_info["is_forward"]

        print(f"\n" + "=" * 75)
        print(f"🔬 Multi-Seed Benchmark: {m_name}")
        print(f"📦 Model: {m_path}")
        print("=" * 75)

        seed_records = []

        for seed in seeds:
            args.seed = seed
            env_cfg.seed = seed
            env, _ = task_registry.make_env(name="myrobot_forward_jump", args=args, env_cfg=env_cfg)

            ac = ActorCritic(
                num_actor_obs=env_cfg.env.num_observations,
                num_critic_obs=env_cfg.env.num_observations,
                num_actions=env_cfg.env.num_actions,
                actor_hidden_dims=[512, 256, 128],
                critic_hidden_dims=[512, 256, 128],
                activation="elu",
                init_noise_std=1.0
            ).to(env.device)
            ckpt = torch.load(m_path, map_location=env.device)
            ac.load_state_dict(ckpt["model_state_dict"])
            ac.eval()

            rec = evaluate_policy_single_jump(env, ac.act_inference, is_forward=is_fwd, duration_s=2.0)
            rec["seed"] = seed
            seed_records.append(rec)
            print(f"  Seed {seed:3d}: Valid={rec['valid_jump']}, Clr={rec['max_clearance_cm']:5.2f}cm, "
                  f"FwdDist(Root)={rec['root_forward_dist_cm']:+5.2f}cm, FwdDist(CoM)={rec['com_forward_dist_cm']:+5.2f}cm, "
                  f"T_flight={rec['flight_duration_s']:.3f}s, YawDrift={rec['yaw_drift_deg']:4.2f}deg, "
                  f"RecVy={rec['recovery_vy_mps']:+.3f}m/s, PeakKnee={rec['peak_knee_tau_nm']:.1f}Nm")

        # Compute summary statistics
        def agg(key):
            vals = [r[key] for r in seed_records]
            return {
                "mean": round(float(np.mean(vals)), 3),
                "std": round(float(np.std(vals)), 3),
                "min": round(float(np.min(vals)), 3),
                "max": round(float(np.max(vals)), 3)
            }

        summary = {
            "model_path": m_path,
            "seeds": seeds,
            "total_trials": len(seeds),
            "valid_jumps": sum(1 for r in seed_records if r["valid_jump"]),
            "abnormal_resets": sum(r["abnormal_resets"] for r in seed_records),
            "max_clearance_cm": agg("max_clearance_cm"),
            "flight_duration_s": agg("flight_duration_s"),
            "root_forward_dist_cm": agg("root_forward_dist_cm"),
            "com_forward_dist_cm": agg("com_forward_dist_cm"),
            "takeoff_root_vy_mps": agg("takeoff_root_vy_mps"),
            "takeoff_root_vz_mps": agg("takeoff_root_vz_mps"),
            "takeoff_com_vy_mps": agg("takeoff_com_vy_mps"),
            "takeoff_com_vz_mps": agg("takeoff_com_vz_mps"),
            "touchdown_pitch_deg": agg("touchdown_pitch_deg"),
            "touchdown_roll_deg": agg("touchdown_roll_deg"),
            "yaw_drift_deg": agg("yaw_drift_deg"),
            "recovery_vy_mps": agg("recovery_vy_mps"),
            "recovery_vx_mps": agg("recovery_vx_mps"),
            "peak_hip_tau_nm": agg("peak_hip_tau_nm"),
            "peak_knee_tau_nm": agg("peak_knee_tau_nm"),
            "peak_wheel_tau_nm": agg("peak_wheel_tau_nm"),
            "knee_saturation_steps": agg("knee_sat_count"),
            "seed_records": seed_records
        }
        all_results[m_name] = summary

    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Multi-seed benchmark summary successfully saved to: {benchmark_json_path}")


if __name__ == "__main__":
    run_benchmark()
