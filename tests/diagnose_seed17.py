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

def diagnose_seed17():
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["diag17", "--task", task_name, "--num_envs", "1"]
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

    print("=== SEED 17: Reset inspection ===")
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    print("State A:")
    print("  root pos:", env.root_states[0, 0:3].cpu().numpy())
    print("  root lin vel:", env.root_states[0, 7:10].cpu().numpy())
    print("  root ang vel:", env.root_states[0, 10:13].cpu().numpy())
    print("  dof pos:", env.dof_pos[0].cpu().numpy())
    print("  dof vel:", env.dof_vel[0].cpu().numpy())

    # Now step 1 with zero actions
    obs, _, _, _, _ = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    print("\nState B:")
    print("  root pos:", env.root_states[0, 0:3].cpu().numpy())
    print("  root lin vel:", env.root_states[0, 7:10].cpu().numpy())
    print("  root ang vel:", env.root_states[0, 10:13].cpu().numpy())
    print("  base lin vel (local):", env.base_lin_vel[0].cpu().numpy())
    print("  dof pos:", env.dof_pos[0].cpu().numpy())
    print("  dof vel:", env.dof_vel[0].cpu().numpy())
    print("  torques:", env.torques[0].cpu().numpy())
    print("  contact forces L:", env.contact_forces[0, env.left_wheel_body].cpu().numpy())
    print("  contact forces R:", env.contact_forces[0, env.right_wheel_body].cpu().numpy())

    # All rigid bodies
    for b in range(env.num_bodies):
        cf = env.contact_forces[0, b].cpu().numpy()
        pos = env.rb_states[0, b, 0:3].cpu().numpy()
        vel = env.rb_states[0, b, 7:10].cpu().numpy()
        print(f"  Body {b} pos={pos}, vel={vel}, cf={cf}")

if __name__ == "__main__":
    diagnose_seed17()
