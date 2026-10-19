#!/usr/bin/env python3
# ==============================================================================
# Script: diagnose_zero_pos_anomaly.py
# Deep inspection of the 2940N force and whole-body kinematics at z=10m
# ==============================================================================

import os
import sys

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse
import torch
import numpy as np

from legged_gym.envs import task_registry
from legged_gym.utils import get_args

def diagnose():
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["diag_anomaly", "--task", task_name, "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = 17

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.normalization.clip_actions = 100.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.num_envs = 1
    env_cfg.seed = 17
    env_cfg.noise.add_noise = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # 1. Inspect actor, bodies, masses
    actor = env.actor_handles[0]
    num_bodies = env.num_bodies
    props = env.gym.get_actor_rigid_body_properties(env.envs[0], actor)
    masses = [p.mass for p in props]
    total_mass = sum(masses)
    # body_names from env
    body_names = env.gym.get_actor_rigid_body_names(env.envs[0], actor)

    print(f"Total mass: {total_mass:.4f} kg")
    for b in range(num_bodies):
        print(f"  Body {b} ({body_names[b]}): mass = {masses[b]:.4f} kg, com = ({props[b].com.x:.4f}, {props[b].com.y:.4f}, {props[b].com.z:.4f})")

    # 2. Place robot in sky (z = 10.0m) at q=0
    print("\n--- TEST: High altitude (z = 10.0m), q = 0.00 ---")
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.root_states[:, 2] = 10.0
    env.root_states[:, 7:13] = 0.0
    env.dof_pos[:, :] = 0.0
    env.dof_vel[:, :] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    # Check state before substep 1
    com_pos_0, com_vel_0 = env._compute_true_com()
    print("Before substep 1 (t = 0):")
    print(f"  root pos: {env.root_states[0, 0:3].cpu().numpy()}")
    print(f"  root vel: {env.root_states[0, 7:10].cpu().numpy()}")
    print(f"  true com pos: {com_pos_0[0].cpu().numpy()}")
    print(f"  true com vel: {com_vel_0[0].cpu().numpy()}")

    # Step 1 PhysX substep (dt = 0.005s) with zero torques
    env.torques = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    com_pos_1, com_vel_1 = env._compute_true_com()
    print("\nAfter substep 1 (t = 0.005s, zero torques):")
    print(f"  root pos: {env.root_states[0, 0:3].cpu().numpy()}")
    print(f"  root lin vel: {env.root_states[0, 7:10].cpu().numpy()}")
    print(f"  root ang vel: {env.root_states[0, 10:13].cpu().numpy()}")
    print(f"  true com pos: {com_pos_1[0].cpu().numpy()}")
    print(f"  true com vel: {com_vel_1[0].cpu().numpy()}")

    # Momentum computation
    total_lin_momentum = np.zeros(3)
    for b in range(num_bodies):
        pos_b = env.rb_states[0, b, 0:3].cpu().numpy()
        vel_b = env.rb_states[0, b, 7:10].cpu().numpy()
        cf_b = env.contact_forces[0, b].cpu().numpy()
        p_b = masses[b] * vel_b
        total_lin_momentum += p_b
        print(f"  Body {b} ({body_names[b]}): vel = {vel_b}, contact_force = {cf_b}, p_z = {p_b[2]:+.4f} kg*m/s")

    print(f"\nTotal linear momentum: {total_lin_momentum} kg*m/s")
    print(f"Total momentum / total mass: {total_lin_momentum / total_mass} m/s")
    print(f"Expected gravity drop vel (-g * dt): { -9.81 * 0.005 } m/s")
    print(f"Discrepancy (com_vz - expected): { (total_lin_momentum[2] / total_mass) - (-9.81 * 0.005) :.6f} m/s")

if __name__ == "__main__":
    diagnose()
