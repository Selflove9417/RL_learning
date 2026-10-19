#!/usr/bin/env python3
# ==============================================================================
# Comprehensive Physical Settling Exploration Script
# Tests different settling configurations, controllers, and durations
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

def explore_settling():
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["settle_explore", "--task", task_name, "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = 42

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.normalization.clip_actions = 100.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.num_envs = 1
    env_cfg.seed = 42
    env_cfg.noise.add_noise = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # Explore Option 1: What happens with zero action settling if placed on ground?
    print("--- Option 1: Contact placement z=0.397739, zero action settling for 30 steps (0.6s) ---")
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.root_states[:, 2] = 0.397739
    env.root_states[:, 7:13] = 0.0
    env.dof_vel[:, :] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    for step in range(30):
        env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        if (step + 1) % 5 == 0 or step == 0:
            fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
            fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()
            b_vy = env.base_lin_vel[0, 1].item()
            b_vz = env.base_lin_vel[0, 2].item()
            _, true_com_vel = env._compute_true_com()
            c_vy = true_com_vel[0, 1].item()
            w_L = env.dof_vel[0, env.left_wheel_idx].item()
            w_R = env.dof_vel[0, env.right_wheel_idx].item()
            r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
            pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
            roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
            print(f"step {step+1:2d} (t={(step+1)*0.02:.2f}s): z={env.root_states[0, 2].item():.4f}, fn=({fn_L:.1f}, {fn_R:.1f}), b_vy={b_vy:.4f}, c_vy={c_vy:.4f}, w=({w_L:.2f}, {w_R:.2f}), pitch={pitch:.2f}, roll={roll:.2f}")

    # Explore Option 2: Pitch angle adjustment for CoM balance
    # CoM is behind wheel axle by 12.7mm. If robot pitches forward by 2.05 deg, CoM is over wheels.
    print("\n--- Option 2: Pitch +2.05 deg, Contact placement, zero actions ---")
    from isaacgym.torch_utils import quat_from_euler_xyz
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.root_states[:, 2] = 0.397739
    new_quat = quat_from_euler_xyz(torch.tensor([0.0], device=env.device), torch.tensor([2.05 * np.pi / 180.0], device=env.device), torch.tensor([0.0], device=env.device))
    env.root_states[:, 3:7] = new_quat
    env.root_states[:, 7:13] = 0.0
    env.dof_vel[:, :] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    for step in range(30):
        env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        if (step + 1) % 5 == 0 or step == 0:
            fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
            fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()
            b_vy = env.base_lin_vel[0, 1].item()
            b_vz = env.base_lin_vel[0, 2].item()
            _, true_com_vel = env._compute_true_com()
            c_vy = true_com_vel[0, 1].item()
            w_L = env.dof_vel[0, env.left_wheel_idx].item()
            w_R = env.dof_vel[0, env.right_wheel_idx].item()
            r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
            pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
            roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
            print(f"step {step+1:2d} (t={(step+1)*0.02:.2f}s): z={env.root_states[0, 2].item():.4f}, fn=({fn_L:.1f}, {fn_R:.1f}), b_vy={b_vy:.4f}, c_vy={c_vy:.4f}, w=({w_L:.2f}, {w_R:.2f}), pitch={pitch:.2f}, roll={roll:.2f}")

if __name__ == "__main__":
    explore_settling()
