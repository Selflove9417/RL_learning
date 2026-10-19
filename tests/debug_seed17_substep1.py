#!/usr/bin/env python3
import os
import sys

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
import torch
import numpy as np

from legged_gym.envs import task_registry
from legged_gym.utils import get_args

def debug_seed(seed):
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["diag", "--task", task_name, "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = seed

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.normalization.clip_actions = 100.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.noise.add_noise = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # Let's inspect root states and rigid body states right after fresh reset_idx
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    print("--- BEFORE SIMULATE ---")
    print(f"root_states: pos={env.root_states[0, :3].cpu().numpy()}, lin_vel={env.root_states[0, 7:10].cpu().numpy()}, ang_vel={env.root_states[0, 10:13].cpu().numpy()}")
    print(f"rb_states 0 (base): lin_vel={env.rb_states[0, 0, 7:10].cpu().numpy()}")
    print(f"rb_states 3 (wheel_L): lin_vel={env.rb_states[0, 3, 7:10].cpu().numpy()}")
    print(f"rb_states 6 (wheel_R): lin_vel={env.rb_states[0, 6, 7:10].cpu().numpy()}")
    print(f"dof_pos: {env.dof_pos[0].cpu().numpy()}")
    print(f"dof_vel: {env.dof_vel[0].cpu().numpy()}")

    # Now simulate 1 single substep (5ms)
    env.torques = env._compute_torques(torch.zeros(env.num_envs, env.num_actions, device=env.device))
    print(f"Torques computed: {env.torques[0].cpu().numpy()}")
    env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    print("\n--- AFTER 1 SUBSTEP (5ms) ---")
    print(f"root_states: pos={env.root_states[0, :3].cpu().numpy()}, lin_vel={env.root_states[0, 7:10].cpu().numpy()}, ang_vel={env.root_states[0, 10:13].cpu().numpy()}")
    print(f"rb_states 0 (base): lin_vel={env.rb_states[0, 0, 7:10].cpu().numpy()}")
    print(f"rb_states 3 (wheel_L): lin_vel={env.rb_states[0, 3, 7:10].cpu().numpy()}")
    print(f"rb_states 6 (wheel_R): lin_vel={env.rb_states[0, 6, 7:10].cpu().numpy()}")
    print(f"contact forces body 0: {env.contact_forces[0, 0].cpu().numpy()}")
    print(f"contact forces body 3 (wheel_L): {env.contact_forces[0, 3].cpu().numpy()}")
    print(f"contact forces body 6 (wheel_R): {env.contact_forces[0, 6].cpu().numpy()}")

if __name__ == "__main__":
    debug_seed(17)
