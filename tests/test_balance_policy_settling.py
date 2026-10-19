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

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.envs.myrobot.myrobot_dual_env import compute_exact_cylinder_clearance

def test_balance_settling():
    bal_path = "evaluation_assets/phase15/models/phase15_m130.pt"
    if not os.path.exists(bal_path):
        print("Balance model not found!")
        return

    task_name = "myrobot_forward_jump_static"
    sys.argv = ["test_bal", "--task", task_name, "--num_envs", "1"]
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

    # Note: balance policy in phase 15 has 31-dim obs:
    # Let's check observation dimension of balance policy
    ckpt_bal = torch.load(bal_path, map_location=env.device)
    print("Balance policy in_features:", ckpt_bal['model_state_dict']['actor.0.weight'].shape)

if __name__ == "__main__":
    test_balance_settling()
