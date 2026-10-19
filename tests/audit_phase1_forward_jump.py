#!/usr/bin/env python3
# ==============================================================================
# MyRobot Forward Jump: Comprehensive Dynamics Consistency & Telemetry Audit
# ==============================================================================

import os
import sys
import json
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

# Isaac Gym must be imported before torch
import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse
import torch

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.envs.myrobot.myrobot_dual_env import compute_exact_cylinder_clearance


def run_audit():
    output_dir = os.path.join(workspace_root, "evaluation_assets", "phase1_forward_jump")
    os.makedirs(output_dir, exist_ok=True)
    audit_json_path = os.path.join(output_dir, "detailed_com_audit.json")

    sys.argv = ["audit_fwd", "--task", "myrobot_forward_jump", "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = 42

    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_forward_jump")
    env_cfg.env.num_envs = 1
    env_cfg.seed = 42
    env_cfg.env.episode_length_s = 10.0
    env_cfg.noise.add_noise = False
    env_cfg.env.enable_camera_sensors = False

    env, _ = task_registry.make_env(name="myrobot_forward_jump", args=args, env_cfg=env_cfg)

    trial_model_path = os.path.join(output_dir, "models", "model_30_yaw_aligned.pt")
    ac = ActorCritic(
        num_actor_obs=env_cfg.env.num_observations,
        num_critic_obs=env_cfg.env.num_observations,
        num_actions=env_cfg.env.num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0
    ).to(env.device)
    ckpt = torch.load(trial_model_path, map_location=env.device)
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    policy = ac.act_inference

    # Collect rigid body masses
    num_bodies = env.gym.get_actor_rigid_body_count(env.envs[0], 0)
    rb_masses = []
    body_names = []
    for b in range(num_bodies):
        prop = env.gym.get_actor_rigid_body_properties(env.envs[0], 0)[b]
        name = env.gym.get_actor_rigid_body_names(env.envs[0], 0)[b]
        rb_masses.append(prop.mass)
        body_names.append(name)
    rb_masses_t = torch.tensor(rb_masses, device=env.device, dtype=torch.float).unsqueeze(0).unsqueeze(-1)  # (1, num_bodies, 1)
    total_mass = rb_masses_t.sum().item()
    print(f"Rigid bodies ({num_bodies}): {list(zip(body_names, rb_masses))}")
    print(f"Total robot mass: {total_mass:.4f} kg")

    env.reset()

    logs = []
    for step in range(150):
        t_sim = step * env.dt
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        # Rigid body states in World Frame
        rb_pos = env.rb_states[:, :, 0:3]  # (1, num_bodies, 3)
        rb_vel = env.rb_states[:, :, 7:10]  # (1, num_bodies, 3)

        # Whole-body CoM
        com_pos = (rb_pos * rb_masses_t).sum(dim=1) / total_mass  # (1, 3)
        com_vel = (rb_vel * rb_masses_t).sum(dim=1) / total_mass  # (1, 3)

        root_pos = env.root_states[0, 0:3].cpu().numpy()
        root_vel_w = env.root_states[0, 7:10].cpu().numpy()
        base_lin_vel = env.base_lin_vel[0].cpu().numpy()  # Local Base frame
        base_ang_vel = env.base_ang_vel[0].cpu().numpy()

        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
        yaw = torch.atan2(torch.sin(y_raw[0]), torch.cos(y_raw[0])).item() * 180.0 / np.pi

        is_air = bool(env.is_airborne[0].item())
        has_took = bool(env.has_taken_off[0].item())
        phase = env.jump_phase[0].item()

        fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
        fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()

        clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(env)
        clr_min_cm = clr_min[0].item() * 100.0
        clr_L_cm = clr_L[0].item() * 100.0
        clr_R_cm = clr_R[0].item() * 100.0

        tau_hip_L = env.torques[0, env.left_hip_idx].item()
        tau_knee_L = env.torques[0, env.left_knee_idx].item()
        tau_wheel_L = env.torques[0, env.left_wheel_idx].item()
        tau_hip_R = env.torques[0, env.right_hip_idx].item()
        tau_knee_R = env.torques[0, env.right_knee_idx].item()
        tau_wheel_R = env.torques[0, env.right_wheel_idx].item()

        flight_d_y = env.flight_distance_y[0].item()
        final_flight_d = env.final_flight_distance[0].item()

        logs.append({
            "step": step, "time": round(t_sim, 3), "phase": round(phase, 4),
            "is_air": is_air, "has_took": has_took,
            "root_x": float(root_pos[0]), "root_y": float(root_pos[1]), "root_z": float(root_pos[2]),
            "com_x": float(com_pos[0, 0].item()), "com_y": float(com_pos[0, 1].item()), "com_z": float(com_pos[0, 2].item()),
            "root_vx_w": float(root_vel_w[0]), "root_vy_w": float(root_vel_w[1]), "root_vz_w": float(root_vel_w[2]),
            "com_vx_w": float(com_vel[0, 0].item()), "com_vy_w": float(com_vel[0, 1].item()), "com_vz_w": float(com_vel[0, 2].item()),
            "base_vx_loc": float(base_lin_vel[0]), "base_vy_loc": float(base_lin_vel[1]), "base_vz_loc": float(base_lin_vel[2]),
            "roll": round(roll, 3), "pitch": round(pitch, 3), "yaw": round(yaw, 3),
            "ang_vel_z": round(float(base_ang_vel[2]), 4),
            "clr_min_cm": round(clr_min_cm, 3), "clr_L_cm": round(clr_L_cm, 3), "clr_R_cm": round(clr_R_cm, 3),
            "fn_L": round(fn_L, 2), "fn_R": round(fn_R, 2),
            "tau_hip_L": round(tau_hip_L, 2), "tau_hip_R": round(tau_hip_R, 2),
            "tau_knee_L": round(tau_knee_L, 2), "tau_knee_R": round(tau_knee_R, 2),
            "tau_wheel_L": round(tau_wheel_L, 2), "tau_wheel_R": round(tau_wheel_R, 2),
            "flight_d_y": round(flight_d_y, 4), "final_flight_d": round(final_flight_d, 4)
        })

        obs = env.get_observations()
        with torch.no_grad():
            actions = policy(obs)
        env.step(actions)

    with open(audit_json_path, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"✅ Detailed dynamics audit saved to {audit_json_path}")


if __name__ == "__main__":
    run_audit()
