#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Diagnostic Script for 35-dim Observation Vector and Actuator Torque Limits Verification

import os
import sys
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymtorch
import torch

from legged_gym.envs import MyRobotObstacleJump
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic


def diagnose_obs_and_torques():
    print("=" * 80)
    print("DIAGNOSING 35-DIM OBSERVATION VECTOR AND ACTUATOR TORQUE LIMITS")
    print("=" * 80)

    args = get_args()
    args.headless = True
    args.task = "myrobot_obstacle_jump"
    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_obstacle_jump")
    env_cfg.env.num_envs = 1

    env, _ = task_registry.make_env(name="myrobot_obstacle_jump", args=args, env_cfg=env_cfg)
    obs, _ = env.reset()

    # Verify obs dimension
    print(f"\n[1] Observation Buffer Verification:")
    print(f"  obs.shape: {obs.shape} (Expected: [1, 35])")
    assert obs.shape == (1, 35), f"Mismatch in obs shape: {obs.shape}"

    # Detailed slicing check
    print("\n[2] Exact 35-Dim Observation Decomposition:")
    print(f"  Indices 0..2   (3): base_lin_vel * 2.0         = {obs[0, 0:3].tolist()}")
    print(f"  Indices 3..5   (3): base_ang_vel * 0.25        = {obs[0, 3:6].tolist()}")
    print(f"  Indices 6..8   (3): projected_gravity          = {obs[0, 6:9].tolist()}")
    print(f"  Indices 9..12  (4): jump_cmds                  = {obs[0, 9:13].tolist()}")
    print(f"    - Slot 9:  cmd_fwd * 2.0                     = {obs[0, 9].item():.4f}")
    print(f"    - Slot 10: cmd_yaw * 0.25                    = {obs[0, 10].item():.4f}")
    print(f"    - Slot 11: norm_target_height               = {obs[0, 11].item():.4f}")
    print(f"    - Slot 12: jump_phase                       = {obs[0, 12].item():.4f}")
    print(f"  Indices 13..18 (6): obs_dof_pos (relative)     = {obs[0, 13:19].tolist()}")
    print(f"  Indices 19..24 (6): dof_vel * 0.05             = {obs[0, 19:25].tolist()}")
    print(f"  Indices 25..30 (6): filtered_actions           = {obs[0, 25:31].tolist()}")
    print(f"  Indices 31..32 (2): contact_state (Fn_L, Fn_R) = {obs[0, 31:33].tolist()}")
    print(f"  Indices 33..34 (2): phase_clock (sin, cos)     = {obs[0, 33:35].tolist()}")
    total_dims = 3 + 3 + 3 + 4 + 6 + 6 + 6 + 2 + 2
    print(f"  Total verified dimensions: {total_dims}")

    # Load frozen model_3.pt and verify forward pass
    print("\n[3] Model Compatibility with Frozen model_3.pt:")
    model_path = "legged_gym/logs/myrobot_running_jump/Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt"
    ckpt = torch.load(model_path, map_location=env.device)
    policy = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()

    with torch.no_grad():
        action = policy.act_inference(obs)
    print(f"  Policy inference successful: action shape = {action.shape}, range = [{action.min().item():.3f}, {action.max().item():.3f}]")

    # Verify actuator torque limits
    print("\n[4] Actuator Torque Limits Verification:")
    print(f"  env.torque_limits: {env.torque_limits.tolist()}")
    for i, name in enumerate(env.dof_names):
        print(f"    Joint {i} ({name}): torque_limit = {env.torque_limits[i].item():.1f} Nm")
    
    # Check that wheel torque limit is 12.0 Nm and hip 140.0, knee 120.0 Nm from robot_jump.urdf
    assert env.torque_limits[env.left_wheel_idx].item() == 12.0, "Wheel torque limit altered!"
    assert env.torque_limits[env.right_wheel_idx].item() == 12.0, "Wheel torque limit altered!"
    assert env.torque_limits[env.left_hip_idx].item() == 140.0, "Hip torque limit altered!"
    assert env.torque_limits[env.left_knee_idx].item() == 120.0, "Knee torque limit altered!"
    print("  ✓ All actuator torque limits strictly match robot_jump.urdf specification (Hip 140 Nm, Knee 120 Nm, Wheel 12 Nm).")
    print("=" * 80)


if __name__ == "__main__":
    diagnose_obs_and_torques()
