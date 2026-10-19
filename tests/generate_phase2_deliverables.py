#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Phase 2 Static Forward Jump Deliverables Generator
# 1. Evaluates Baseline (Trial 2 hot-start) vs Champion (model_30) across 5 seeds
# 2. Generates 50 Hz high-frequency telemetry CSV
# 3. Renders 720p HD demonstration MP4 video
# 4. Saves champion model and multi-seed comparative summary JSON
# ==============================================================================

import os
import sys
import csv
import json
import shutil
import numpy as np
import imageio

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


def eval_model_seeds(env_name, model_path, seeds, duration_s=2.8):
    """Evaluates a model across seeds in a separate clean sub-process per seed using eval_single_seed_fwd_static.py"""
    worker_script = os.path.join(workspace_root, "tests", "eval_single_seed_fwd_static.py")
    python_bin = sys.executable

    seed_results = []
    for seed in seeds:
        out_json = f"/tmp/eval_p2_{os.path.basename(model_path)}_seed_{seed}.json"
        cmd = [python_bin, worker_script, "--seed", str(seed), "--model_path", model_path, "--out_json", out_json]
        env = os.environ.copy()
        import subprocess
        res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if res.returncode == 0:
            with open(out_json) as f:
                d = json.load(f)
            seed_results.append(d)
        else:
            print(f"Error evaluating seed {seed}: {res.stderr}")
    return seed_results


def aggregate_metrics(seed_data):
    if not seed_data:
        return {}
    valid_count = sum(1 for d in seed_data if d["valid_jump"])
    resets_count = sum(d["abnormal_resets"] for d in seed_data)
    com_dists = [d["com_forward_dist_cm"] for d in seed_data]
    clearances = [d["max_clearance_cm"] for d in seed_data]
    vy_takeoffs = [d["takeoff_com_vy_mps"] for d in seed_data]
    vz_takeoffs = [d["takeoff_com_vz_mps"] for d in seed_data]
    pre_roll_vys = [d["pre_push_vy_mps"] for d in seed_data]
    pitch_touchdowns = [d["touchdown_pitch_deg"] for d in seed_data]
    roll_touchdowns = [d["touchdown_roll_deg"] for d in seed_data]
    yaw_touchdowns = [d["touchdown_yaw_deg"] for d in seed_data]
    recovery_vys = [d["recovery_vy_mps"] for d in seed_data]
    recovery_vxs = [d["recovery_vx_mps"] for d in seed_data]
    yaw_drifts = [d["yaw_drift_deg"] for d in seed_data]
    peak_hip_taus = [d["peak_hip_tau_nm"] for d in seed_data]
    peak_knee_taus = [d["peak_knee_tau_nm"] for d in seed_data]
    peak_wheel_taus = [d["peak_wheel_tau_nm"] for d in seed_data]

    # Count clearance >= 20.0 cm threshold
    clr_above_20_count = sum(1 for c in clearances if c >= 20.0)

    return {
        "valid_jumps": f"{valid_count}/{len(seed_data)}",
        "valid_jump_rate_pct": round(valid_count / len(seed_data) * 100.0, 1),
        "abnormal_resets": resets_count,
        "clearance_ge_20cm_rate": f"{clr_above_20_count}/{len(seed_data)} ({clr_above_20_count/len(seed_data)*100.0:.1f}%)",
        "com_dist_mean_cm": round(float(np.mean(com_dists)), 2),
        "com_dist_std_cm": round(float(np.std(com_dists)), 2),
        "com_dist_min_cm": round(float(np.min(com_dists)), 2),
        "com_dist_max_cm": round(float(np.max(com_dists)), 2),
        "clearance_mean_cm": round(float(np.mean(clearances)), 2),
        "clearance_std_cm": round(float(np.std(clearances)), 2),
        "clearance_min_cm": round(float(np.min(clearances)), 2),
        "clearance_max_cm": round(float(np.max(clearances)), 2),
        "takeoff_vy_mean_mps": round(float(np.mean(vy_takeoffs)), 3),
        "takeoff_vy_std_mps": round(float(np.std(vy_takeoffs)), 3),
        "takeoff_vz_mean_mps": round(float(np.mean(vz_takeoffs)), 3),
        "pre_roll_vy_mean_mps": round(float(np.mean(pre_roll_vys)), 3),
        "pitch_touchdown_mean_deg": round(float(np.mean(pitch_touchdowns)), 2),
        "roll_touchdown_mean_deg": round(float(np.mean(roll_touchdowns)), 2),
        "yaw_touchdown_mean_deg": round(float(np.mean(yaw_touchdowns)), 2),
        "yaw_drift_mean_deg": round(float(np.mean(yaw_drifts)), 2),
        "recovery_vy_mean_mps": round(float(np.mean(recovery_vys)), 3),
        "recovery_vx_mean_mps": round(float(np.mean(recovery_vxs)), 3),
        "peak_hip_tau_nm": round(float(np.max(peak_hip_taus)), 2),
        "peak_knee_tau_nm": round(float(np.max(peak_knee_taus)), 2),
        "peak_wheel_tau_nm": round(float(np.max(peak_wheel_taus)), 2),
        "seed_details": seed_data
    }


def generate_telemetry_and_video(champion_model_path, out_csv_path, out_mp4_path, seed=42):
    print(f"\n🎬 Recording Telemetry CSV and Demonstration MP4 (Seed {seed})...")
    sys.argv = ["eval_p2_demo", "--task", "myrobot_forward_jump_static", "--num_envs", "1"]
    args = get_args()
    args.headless = False
    args.seed = seed

    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_forward_jump_static")
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.env.episode_length_s = 2.8
    env_cfg.noise.add_noise = False
    env_cfg.env.enable_camera_sensors = True

    env, _ = task_registry.make_env(name="myrobot_forward_jump_static", args=args, env_cfg=env_cfg)

    # Set camera
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1280
    camera_props.height = 720
    camera_props.enable_tensors = False
    camera_props.horizontal_fov = 65.0
    camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    cam_offset = np.array([-1.4, -2.1, 0.85])

    # Load policy
    ac = ActorCritic(
        num_actor_obs=env_cfg.env.num_observations,
        num_critic_obs=env_cfg.env.num_observations,
        num_actions=env_cfg.env.num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0
    ).to(env.device)
    ckpt = torch.load(champion_model_path, map_location=env.device)
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    policy = ac.act_inference

    obs, _ = env.reset()

    frames = []
    total_steps = int(2.8 / env.dt)

    csv_file = open(out_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "time_s", "step", "jump_phase", "jump_latched",
        "root_x", "root_y", "root_z",
        "com_x", "com_y", "com_z",
        "root_vx", "root_vy", "root_vz",
        "com_vx", "com_vy", "com_vz",
        "base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z",
        "roll_deg", "pitch_deg", "yaw_deg",
        "is_airborne", "has_taken_off", "has_touched_down",
        "clr_min_cm", "clr_L_cm", "clr_R_cm",
        "fn_L_N", "fn_R_N",
        "flight_distance_y_cm",
        "tau_hip_L_Nm", "tau_knee_L_Nm", "tau_wheel_L_Nm",
        "tau_hip_R_Nm", "tau_knee_R_Nm", "tau_wheel_R_Nm"
    ])

    for step in range(total_steps):
        t_sim = step * env.dt

        # Compute states
        true_com_pos, true_com_vel = env._compute_true_com()
        root_pos = env.root_states[0, 0:3].cpu().numpy()
        root_vel = env.root_states[0, 7:10].cpu().numpy()
        base_lin_vel = env.base_lin_vel[0].cpu().numpy()

        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
        yaw = torch.atan2(torch.sin(y_raw[0]), torch.cos(y_raw[0])).item() * 180.0 / np.pi

        clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(env)
        clr_min_cm = clr_min[0].item() * 100.0
        clr_L_cm = clr_L[0].item() * 100.0
        clr_R_cm = clr_R[0].item() * 100.0

        fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
        fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()

        is_air = int(env.is_airborne[0].item())
        has_took = int(env.has_taken_off[0].item())
        has_land = int(env.has_touched_down[0].item())
        latched = int(env.jump_latched[0].item())
        phase = env.jump_phase[0].item()

        fwd_dist_cm = env.flight_distance_y[0].item() * 100.0

        t_hip_L = env.torques[0, env.left_hip_idx].item()
        t_knee_L = env.torques[0, env.left_knee_idx].item()
        t_wheel_L = env.torques[0, env.left_wheel_idx].item()
        t_hip_R = env.torques[0, env.right_hip_idx].item()
        t_knee_R = env.torques[0, env.right_knee_idx].item()
        t_wheel_R = env.torques[0, env.right_wheel_idx].item()

        # Telemetry write
        csv_writer.writerow([
            f"{t_sim:.3f}", step, f"{phase:.4f}", latched,
            f"{root_pos[0]:.4f}", f"{root_pos[1]:.4f}", f"{root_pos[2]:.4f}",
            f"{true_com_pos[0, 0].item():.4f}", f"{true_com_pos[0, 1].item():.4f}", f"{true_com_pos[0, 2].item():.4f}",
            f"{root_vel[0]:.4f}", f"{root_vel[1]:.4f}", f"{root_vel[2]:.4f}",
            f"{true_com_vel[0, 0].item():.4f}", f"{true_com_vel[0, 1].item():.4f}", f"{true_com_vel[0, 2].item():.4f}",
            f"{base_lin_vel[0]:.4f}", f"{base_lin_vel[1]:.4f}", f"{base_lin_vel[2]:.4f}",
            f"{roll:.2f}", f"{pitch:.2f}", f"{yaw:.2f}",
            is_air, has_took, has_land,
            f"{clr_min_cm:.2f}", f"{clr_L_cm:.2f}", f"{clr_R_cm:.2f}",
            f"{fn_L:.2f}", f"{fn_R:.2f}",
            f"{fwd_dist_cm:.2f}",
            f"{t_hip_L:.2f}", f"{t_knee_L:.2f}", f"{t_wheel_L:.2f}",
            f"{t_hip_R:.2f}", f"{t_knee_R:.2f}", f"{t_wheel_R:.2f}"
        ])

        # Camera frame
        cam_target = gymapi.Vec3(root_pos[0], root_pos[1], root_pos[2] + 0.12)
        cam_eye = gymapi.Vec3(root_pos[0] + cam_offset[0], root_pos[1] + cam_offset[1], root_pos[2] + cam_offset[2])
        env.gym.set_camera_location(camera_handle, env.envs[0], cam_eye, cam_target)
        env.gym.render_all_camera_sensors(env.sim)
        img = env.gym.get_camera_image(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
        img = img.reshape((camera_props.height, camera_props.width, 4))[:, :, :3]
        frames.append(img)

        with torch.no_grad():
            actions = policy(obs)
        obs, _, rewards, dones, infos = env.step(actions)

    csv_file.close()
    print(f"✅ Telemetry CSV written to: {out_csv_path}")

    print(f"📹 Rendering MP4 video with {len(frames)} frames to: {out_mp4_path}...")
    imageio.mimwrite(out_mp4_path, frames, fps=50, quality=8)
    print(f"✅ Video successfully created ({os.path.getsize(out_mp4_path)/1024/1024:.2f} MB)")


def main():
    print("=" * 80)
    print("🚀 MyRobot Phase 2 Static Forward Jump: Full Deliverables Generator")
    print("=" * 80)

    deliverable_dir = os.path.join(workspace_root, "evaluation_assets", "phase2_static_forward_jump")
    model_dir = os.path.join(deliverable_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    baseline_model_path = os.path.join(workspace_root, "evaluation_assets", "phase1_forward_jump", "models", "model_30_yaw_aligned.pt")
    repaired_run_dir = os.path.join(workspace_root, "legged_gym", "logs", "myrobot_forward_jump_static", "Oct16_13-08-31_phase2_static_fwd_jump_repaired")
    champion_source_path = os.path.join(repaired_run_dir, "model_30.pt")
    runnerup_source_path = os.path.join(repaired_run_dir, "model_20.pt")

    champion_target_path = os.path.join(model_dir, "model_champion.pt")
    runnerup_target_path = os.path.join(model_dir, "model_candidate_iter20.pt")

    shutil.copyfile(champion_source_path, champion_target_path)
    shutil.copyfile(runnerup_source_path, runnerup_target_path)
    print(f"✅ Champion model archived to: {champion_target_path}")

    seeds = [42, 100, 200, 300, 400]

    # 1. Evaluate baseline across 5 seeds
    print("\n--> Step 1: Evaluating Baseline Trial 2 in Static Environment across 5 seeds...")
    baseline_seed_data = eval_model_seeds("myrobot_forward_jump_static", baseline_model_path, seeds)
    baseline_summary = aggregate_metrics(baseline_seed_data)

    # 2. Evaluate champion across 5 seeds
    print("\n--> Step 2: Evaluating Phase 2 Champion (Iter 30) across 5 seeds...")
    champion_seed_data = eval_model_seeds("myrobot_forward_jump_static", champion_target_path, seeds)
    champion_summary = aggregate_metrics(champion_seed_data)

    # 3. Evaluate candidate model_20 across 5 seeds
    print("\n--> Step 3: Evaluating Phase 2 Candidate (Iter 20) across 5 seeds...")
    runnerup_seed_data = eval_model_seeds("myrobot_forward_jump_static", runnerup_target_path, seeds)
    runnerup_summary = aggregate_metrics(runnerup_seed_data)

    # 4. Generate Telemetry CSV & Demonstration MP4
    out_csv = os.path.join(deliverable_dir, "static_forward_jump_telemetry.csv")
    out_mp4 = os.path.join(deliverable_dir, "static_forward_jump_demonstration.mp4")
    generate_telemetry_and_video(champion_target_path, out_csv, out_mp4, seed=42)

    # 5. Save consolidated evaluation summary JSON
    summary_json_path = os.path.join(deliverable_dir, "phase2_static_jump_evaluation_summary.json")
    final_output = {
        "metadata": {
            "task": "myrobot_forward_jump_static",
            "phase": "Phase 2 Static Forward Jump Short-Round Training",
            "date": "2026-10-16",
            "seeds_evaluated": seeds,
            "baseline_model": baseline_model_path,
            "champion_model": champion_target_path,
            "runnerup_model": runnerup_target_path
        },
        "baseline_trial2_in_static_env": baseline_summary,
        "phase2_champion_iter30": champion_summary,
        "phase2_candidate_iter20": runnerup_summary
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Consolidated multi-seed benchmark saved to: {summary_json_path}")
    print("=" * 80)
    print("🎉 All deliverables successfully generated!")
    print("=" * 80)


if __name__ == "__main__":
    main()
