#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Multi-Cycle Continuous Evaluation on Obstacle Environment
# Evaluates whether the baseline policy exhibits kinematic degradation or instability
# across multiple jump cycles when attempting to navigate through an obstacle corridor.

import os
import sys
import math
import json
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymtorch
from isaacgym.torch_utils import get_euler_xyz
import torch
from legged_gym.envs import MyRobotObstacleJump
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic


def verify_continuous_degradation(seed=42, num_steps=600):
    model_path = "legged_gym/logs/myrobot_running_jump/Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt"

    args = get_args()
    args.headless = True
    args.seed = seed
    args.task = "myrobot_obstacle_jump"

    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_obstacle_jump")
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.env.continuous_mode = True
    env_cfg.trigger.mode = "spatial"

    env, _ = task_registry.make_env(name="myrobot_obstacle_jump", args=args, env_cfg=env_cfg)
    obs, _ = env.reset()

    ckpt = torch.load(model_path, map_location=env.device)
    policy = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()

    cycles_data = []
    current_cycle = None
    step_in_cycle = 0

    print(f"\n[INFO] Running continuous obstacle multi-cycle test for {num_steps} steps...")

    for step_i in range(num_steps):
        with torch.no_grad():
            actions = policy.act_inference(obs)
        obs, _, rews, dones, infos = env.step(actions)

        fsm = env.fsm_state[0].item()
        pos_y = env.root_states[0, 1].item()
        pos_z = env.root_states[0, 2].item()
        vw_y = env.root_states[0, 8].item()
        clr_cm = env.bilateral_clearance[0].item() * 100.0
        is_col = env.has_obstacle_collision[0].item()
        obs_fn = env.peak_obstacle_contact_force[0].item()

        if fsm == env.STATE_JUMP and current_cycle is None:
            current_cycle = {
                "cycle_idx": len(cycles_data) + 1,
                "start_step": step_i,
                "takeoff_y": pos_y,
                "takeoff_vw": vw_y,
                "max_clr_cm": 0.0,
                "peak_col_N": 0.0,
                "touchdown_y": None,
                "touchdown_vw": None,
                "recovered": False
            }

        if current_cycle is not None:
            if clr_cm > current_cycle["max_clr_cm"]:
                current_cycle["max_clr_cm"] = clr_cm
            if obs_fn > current_cycle["peak_col_N"]:
                current_cycle["peak_col_N"] = obs_fn

            if fsm == env.STATE_RECOVER and current_cycle["touchdown_y"] is None:
                current_cycle["touchdown_y"] = pos_y
                current_cycle["touchdown_vw"] = vw_y

            if fsm == env.STATE_BALANCE and current_cycle["touchdown_y"] is not None:
                current_cycle["recovered"] = True
                current_cycle["end_step"] = step_i
                current_cycle["flight_dist_m"] = current_cycle["touchdown_y"] - current_cycle["takeoff_y"]
                cycles_data.append(current_cycle)
                print(f"  Cycle #{current_cycle['cycle_idx']} | Liftoff: {current_cycle['takeoff_y']:.3f}m | "
                      f"Touchdown: {current_cycle['touchdown_y']:.3f}m | FlightDist: {current_cycle['flight_dist_m']*100:.1f}cm | "
                      f"MaxClr: {current_cycle['max_clr_cm']:.1f}cm | PeakObstacleForce: {current_cycle['peak_col_N']:.1f}N")
                current_cycle = None

    print(f"\n[SUMMARY] Total completed jump attempts: {len(cycles_data)}")
    for c in cycles_data:
        print(f"  Cycle {c['cycle_idx']}: Flight Dist = {c.get('flight_dist_m', 0.0)*100:.1f}cm, "
              f"Peak Collision Force = {c['peak_col_N']:.1f}N")


if __name__ == "__main__":
    verify_continuous_degradation()
