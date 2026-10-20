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
    env_cfg.trigger.spatial_trigger_dist = 0.09
    env_cfg.obstacle.lead_dist = 0.60  # Reposition obstacle 60cm ahead for each subsequent cycle

    env, _ = task_registry.make_env(name="myrobot_obstacle_jump", args=args, env_cfg=env_cfg)
    obs, _ = env.reset()

    ckpt = torch.load(model_path, map_location=env.device)
    policy = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()

    cycles_data = []
    current_cycle = None

    print(f"\n[INFO] Running continuous obstacle multi-cycle test for {num_steps} steps...")
    print(f"       Obstacle dynamically repositioned 0.60m ahead of robot after each cycle recovery.")
    print(f"       Per-cycle contact forces, clearance, and collision states are zeroed at each takeoff.")

    for step_i in range(num_steps):
        with torch.no_grad():
            actions = policy.act_inference(obs)
        obs, _, rews, dones, infos = env.step(actions)

        fsm = env.fsm_state[0].item()
        pos_y = env.root_states[0, 1].item()
        pos_z = env.root_states[0, 2].item()
        vw_y = env.root_states[0, 8].item()
        fn_L = env.contact_forces[0, env.feet_indices[0], 2].item()
        fn_R = env.contact_forces[0, env.feet_indices[1], 2].item()
        clr_cm = env.bilateral_clearance[0].item() * 100.0
        obs_fn = torch.norm(env.obstacle_contact_forces[0]).item()

        if fsm == env.STATE_JUMP and current_cycle is None:
            current_cycle = {
                "cycle_idx": len(cycles_data) + 1,
                "trigger_step": step_i,
                "trigger_y": pos_y,
                "obstacle_pos_y": env.obstacle_py,
                "liftoff_step": None,
                "liftoff_y": None,
                "liftoff_vw": None,
                "max_clr_cm": 0.0,
                "peak_col_N": 0.0,
                "touchdown_step": None,
                "touchdown_y": None,
                "touchdown_vw": None,
                "recovered": False
            }

        if current_cycle is not None:
            if clr_cm > current_cycle["max_clr_cm"]:
                current_cycle["max_clr_cm"] = clr_cm
            if obs_fn > current_cycle["peak_col_N"]:
                current_cycle["peak_col_N"] = obs_fn

            is_air = (fn_L < 1.0 and fn_R < 1.0) and (clr_cm > 1.5)
            if is_air and current_cycle["liftoff_y"] is None:
                current_cycle["liftoff_step"] = step_i
                current_cycle["liftoff_y"] = pos_y
                current_cycle["liftoff_vw"] = vw_y

            if current_cycle["liftoff_y"] is not None and current_cycle["touchdown_y"] is None:
                if (fn_L > 20.0 or fn_R > 20.0) and step_i > current_cycle["liftoff_step"] + 3:
                    current_cycle["touchdown_step"] = step_i
                    current_cycle["touchdown_y"] = pos_y
                    current_cycle["touchdown_vw"] = vw_y

            # In case FSM changes to RECOVER without discrete contact threshold:
            if fsm == env.STATE_RECOVER and current_cycle["touchdown_y"] is None:
                current_cycle["touchdown_step"] = step_i
                current_cycle["touchdown_y"] = pos_y
                current_cycle["touchdown_vw"] = vw_y

            if fsm == env.STATE_BALANCE and current_cycle["touchdown_y"] is not None:
                current_cycle["recovered"] = True
                current_cycle["end_step"] = step_i
                
                # Airborne distance (true flight) vs FSM total displacement
                liftoff_pos = current_cycle["liftoff_y"] if current_cycle["liftoff_y"] is not None else current_cycle["trigger_y"]
                current_cycle["airborne_dist_cm"] = (current_cycle["touchdown_y"] - liftoff_pos) * 100.0
                current_cycle["fsm_total_dist_cm"] = (current_cycle["touchdown_y"] - current_cycle["trigger_y"]) * 100.0
                cycles_data.append(current_cycle)

                print(f"  Cycle #{current_cycle['cycle_idx']} | "
                      f"ObsPos: {current_cycle['obstacle_pos_y']:.3f}m | "
                      f"Liftoff: {liftoff_pos:.3f}m | "
                      f"TD: {current_cycle['touchdown_y']:.3f}m | "
                      f"AirFlightDist: {current_cycle['airborne_dist_cm']:.1f}cm (FSM_Total: {current_cycle['fsm_total_dist_cm']:.1f}cm) | "
                      f"MaxClr: {current_cycle['max_clr_cm']:.1f}cm | "
                      f"CyclePeakObsForce: {current_cycle['peak_col_N']:.1f}N")
                current_cycle = None

    print(f"\n[SUMMARY] Total completed jump attempts: {len(cycles_data)}")
    for c in cycles_data:
        print(f"  Cycle {c['cycle_idx']}: ObsPos={c['obstacle_pos_y']:.3f}m, "
              f"Airborne Flight={c['airborne_dist_cm']:.1f}cm, "
              f"Peak Collision Force={c['peak_col_N']:.1f}N")

    out_file = "evaluation_assets/continuous_obstacle_jump_results.json"
    with open(out_file, "w") as f:
        json.dump(cycles_data, f, indent=2)
    print(f"\n[INFO] Continuous jump results saved to {out_file}")


if __name__ == "__main__":
    verify_continuous_degradation()
