#!/usr/bin/env python3
# ==============================================================================
# Script to test physical settling and contact physics in IsaacGym
# ==============================================================================

import os
import sys

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse, quat_from_euler_xyz
import torch
import numpy as np

from legged_gym.envs import task_registry
from legged_gym.utils import get_args

def test_settling():
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["settle_test", "--task", task_name, "--num_envs", "1"]
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

    print("=== TEST 1: Nominal drop (root_z = 0.4000) with zero actions ===")
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    # Check 1 substep / 1 step evolution
    for i in range(5):
        obs, _, _, _, _ = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        pos_L = env.rb_states[0, env.left_wheel_body, 2].item()
        pos_R = env.rb_states[0, env.right_wheel_body, 2].item()
        fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
        fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()
        base_vy = env.base_lin_vel[0, 1].item()
        base_vz = env.base_lin_vel[0, 2].item()
        true_com_pos, true_com_vel = env._compute_true_com()
        com_vy = true_com_vel[0, 1].item()
        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        print(f"Step {i+1} (t={(i+1)*0.02:.2f}s): z_root={env.root_states[0, 2].item():.5f}, clr_L={(pos_L-env.wheel_radius)*1000:.2f}mm, clr_R={(pos_R-env.wheel_radius)*1000:.2f}mm, fn_L={fn_L:.1f}N, fn_R={fn_R:.1f}N, base_vy={base_vy:.4f}, com_vy={com_vy:.4f}, pitch={pitch:.2f}deg, roll={roll:.2f}deg")

    print("\n=== TEST 2: Ground contact placement (root_z = 0.397739) ===")
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

    for i in range(5):
        obs, _, _, _, _ = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        pos_L = env.rb_states[0, env.left_wheel_body, 2].item()
        pos_R = env.rb_states[0, env.right_wheel_body, 2].item()
        fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
        fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()
        base_vy = env.base_lin_vel[0, 1].item()
        base_vz = env.base_lin_vel[0, 2].item()
        true_com_pos, true_com_vel = env._compute_true_com()
        com_vy = true_com_vel[0, 1].item()
        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        print(f"Step {i+1} (t={(i+1)*0.02:.2f}s): z_root={env.root_states[0, 2].item():.5f}, clr_L={(pos_L-env.wheel_radius)*1000:.2f}mm, clr_R={(pos_R-env.wheel_radius)*1000:.2f}mm, fn_L={fn_L:.1f}N, fn_R={fn_R:.1f}N, base_vy={base_vy:.4f}, com_vy={com_vy:.4f}, pitch={pitch:.2f}deg, roll={roll:.2f}deg")

if __name__ == "__main__":
    test_settling()
