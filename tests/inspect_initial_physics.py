#!/usr/bin/env python3
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

def inspect_physics():
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["inspect", "--task", task_name, "--num_envs", "1"]
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

    # Let's inspect:
    # 1. dt and substeps
    sim_params = env.sim_params
    print(f"sim dt: {env.dt}, physx dt: {sim_params.dt}, substeps: {sim_params.substeps}")
    print(f"num bodies: {env.num_bodies}, feet_indices: {env.feet_indices}")
    print(f"left_wheel_body: {env.left_wheel_body}, right_wheel_body: {env.right_wheel_body}")

    # Inspect initial root_states
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    root_pos = env.root_states[0, 0:3].cpu().numpy()
    root_quat = env.root_states[0, 3:7].cpu().numpy()
    left_wheel_pos = env.rb_states[0, env.left_wheel_body, 0:3].cpu().numpy()
    right_wheel_pos = env.rb_states[0, env.right_wheel_body, 0:3].cpu().numpy()

    print(f"Initial root pos: {root_pos}")
    print(f"Left wheel pos: {left_wheel_pos}, wheel_radius: {env.wheel_radius}")
    print(f"Left wheel bottom z: {left_wheel_pos[2] - env.wheel_radius:.6f} m")
    print(f"Right wheel pos: {right_wheel_pos}")
    print(f"Right wheel bottom z: {right_wheel_pos[2] - env.wheel_radius:.6f} m")

    true_com_pos, true_com_vel = env._compute_true_com()
    print(f"Initial true CoM pos: {true_com_pos[0].cpu().numpy()}")
    print(f"CoM y - wheel y: {true_com_pos[0, 1].item() - (left_wheel_pos[1] + right_wheel_pos[1])/2.0:.6f} m")
    print(f"CoM x - wheel x: {true_com_pos[0, 0].item() - (left_wheel_pos[0] + right_wheel_pos[0])/2.0:.6f} m")

if __name__ == "__main__":
    inspect_physics()
