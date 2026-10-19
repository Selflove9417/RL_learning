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

def audit_episode():
    task_name = "myrobot_forward_jump_phase4"
    sys.argv = ["audit_ep", "--task", task_name, "--num_envs", "64"]
    args = get_args()
    args.headless = True
    args.seed = 1

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = 64
    env_cfg.seed = 1
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    print("=== Task 3: Episode Length & Types Audit ===")
    print("type(env.dt)               :", type(env.dt))
    print("repr(env.dt)               :", repr(env.dt))
    print("type(env.max_episode_length_s):", type(env.max_episode_length_s))
    print("repr(env.max_episode_length_s):", repr(env.max_episode_length_s))
    print("env.max_episode_length_s / env.dt:", env.max_episode_length_s / env.dt)
    print("repr(env.max_episode_length_s / env.dt):", repr(env.max_episode_length_s / env.dt))
    print("np.ceil(env.max_episode_length_s / env.dt):", np.ceil(env.max_episode_length_s / env.dt))
    print("env.max_episode_length     :", env.max_episode_length)
    print("train_cfg.runner.num_steps_per_env:", train_cfg.runner.num_steps_per_env)

    # Now step the environment for 145 steps to see exact behavior at 139, 140, 141
    print("\n=== Stepping 145 steps to track done / timeout / episode_length_buf ===")
    obs, _ = env.reset()

    for s in range(145):
        # zero action or random action
        act = torch.zeros(64, 6, device=env.device)
        obs, _, rew, dones, infos = env.step(act)
        ep_len = env.episode_length_buf[0].item()
        d_val = dones[0].item()
        to_val = infos["time_outs"][0].item() if "time_outs" in infos else None
        reset_val = env.reset_buf[0].item()

        if s in [137, 138, 139, 140, 141, 142]:
            print(f"Step {s:3d} (after step): ep_len_buf={ep_len:3d}, done={d_val}, timeout={to_val}, reset_buf={reset_val}")

if __name__ == "__main__":
    audit_episode()
