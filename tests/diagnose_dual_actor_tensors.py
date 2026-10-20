#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Diagnostic Script for Dual Actor Tensor Mapping and Reset Indexing Verification

import os
import sys
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
import torch

from legged_gym.envs import MyRobotObstacleJump
from legged_gym.utils import task_registry, get_args


def diagnose_dual_actor_tensors():
    print("=" * 80)
    print("DIAGNOSING DUAL ACTOR TENSORS IN ISAAC GYM")
    print("=" * 80)

    args = get_args()
    args.headless = True
    args.task = "myrobot_obstacle_jump"
    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_obstacle_jump")
    num_envs = 4
    env_cfg.env.num_envs = num_envs

    env, _ = task_registry.make_env(name="myrobot_obstacle_jump", args=args, env_cfg=env_cfg)

    print(f"\n[1] Environment and Actor Counts:")
    print(f"  num_envs: {env.num_envs}")
    print(f"  actors_per_env: 2 (Actor 0: Robot, Actor 1: Obstacle)")
    print(f"  robot_num_bodies: {env.num_bodies} (7 bodies)")
    print(f"  robot_num_dofs: {env.num_dofs} (6 DOFs)")
    print(f"  obstacle_num_bodies: 1 (1 body)")
    print(f"  obstacle_num_dofs: 0 (0 DOFs, fixed base)")

    print(f"\n[2] Tensor Shapes and Memory Layout:")
    print(f"  all_root_states shape: {env.all_root_states.shape} (num_envs, 2, 13)")
    print(f"  root_states (Robot) shape: {env.root_states.shape} (num_envs, 13)")
    print(f"  obstacle_root_states shape: {env.obstacle_root_states.shape} (num_envs, 13)")
    print(f"  all_rb_states shape: {env.all_rb_states.shape} (num_envs, 8, 13)")
    print(f"  all_contact_forces shape: {env.all_contact_forces.shape} (num_envs, 8, 3)")
    print(f"  contact_forces (Robot) shape: {env.contact_forces.shape} (num_envs, 7, 3)")
    print(f"  obstacle_contact_forces shape: {env.obstacle_contact_forces.shape} (num_envs, 3)")
    print(f"  dof_state shape: {env.dof_state.shape} (num_envs * 6, 2)")

    # Verify tensor sharing / memory slicing
    print(f"\n[3] Pointer / Memory Slicing Verification:")
    # Robot root states should share memory with all_root_states[:, 0, :]
    env.all_root_states[1, 0, 0] = 999.0
    assert env.root_states[1, 0].item() == 999.0, "root_states does not view all_root_states!"
    env.all_root_states[1, 0, 0] = 0.0  # restore
    print("  ✓ root_states is a direct view into all_root_states[:, 0, :]")

    # Obstacle root states should share memory with all_root_states[:, 1, :]
    env.all_root_states[2, 1, 0] = 888.0
    assert env.obstacle_root_states[2, 0].item() == 888.0, "obstacle_root_states does not view all_root_states!"
    env.all_root_states[2, 1, 0] = 0.0  # restore
    print("  ✓ obstacle_root_states is a direct view into all_root_states[:, 1, :]")

    # Robot contact forces view
    env.all_contact_forces[0, 7, 2] = 777.0
    assert env.obstacle_contact_forces[0, 2].item() == 777.0, "obstacle_contact_forces does not view all_contact_forces!"
    env.all_contact_forces[0, 7, 2] = 0.0  # restore
    print("  ✓ obstacle_contact_forces is a direct view into all_contact_forces[:, 7, :] (body 7)")

    print(f"\n[4] Reset Indexing Test on Subsets:")
    # Test reset_idx on subset [1, 3]
    subset_ids = torch.tensor([1, 3], device=env.device, dtype=torch.long)
    print(f"  Calling reset_idx on envs {subset_ids.tolist()}...")
    env.reset_idx(subset_ids)
    print("  ✓ reset_idx succeeded with no PhysX crashes or CUDA errors.")

    # Check actor IDs passed to set_dof_state_tensor_indexed:
    robot_actor_ids = (subset_ids * 2).to(dtype=torch.int32)
    print(f"  Robot actor IDs in sim: {robot_actor_ids.tolist()} (Actor 2 and Actor 6)")
    obstacle_actor_ids = (subset_ids * 2 + 1).to(dtype=torch.int32)
    print(f"  Obstacle actor IDs in sim: {obstacle_actor_ids.tolist()} (Actor 3 and Actor 7)")

    # Execute a step to confirm everything steps cleanly
    actions = torch.zeros(num_envs, 6, device=env.device)
    obs, _, rews, dones, _ = env.step(actions)
    print("  ✓ env.step succeeded cleanly.")
    print("=" * 80)


if __name__ == "__main__":
    diagnose_dual_actor_tensors()
