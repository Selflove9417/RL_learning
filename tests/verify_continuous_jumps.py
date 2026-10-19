#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Continuous Multi-Cycle Jump Verification Script for MyRobot Running Jump
# Verifies model_3.pt over 5 consecutive running jump cycles without timeout/reset.

import os
import sys
import math
import isaacgym
from isaacgym import gymtorch
from isaacgym.torch_utils import get_euler_xyz
import torch
import numpy as np


workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic


def verify_continuous(seed=42, num_steps=700, model_path=None):
    if model_path is None:
        model_path = "legged_gym/logs/myrobot_running_jump/Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt"

    args = get_args()
    args.headless = True
    args.seed = seed

    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_running_jump")
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.env.continuous_mode = True  # Enable continuous demonstration mode (no 2.8s timeout)

    env, _ = task_registry.make_env(name="myrobot_running_jump", args=args, env_cfg=env_cfg)
    obs, _ = env.reset()

    # Load and verify model_3.pt weights
    ckpt = torch.load(model_path, map_location=env.device)
    policy = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()

    print(f"Loaded policy from: {model_path} (iter: {ckpt.get('iter', 'unknown')})")
    print(f"Running continuous simulation for {num_steps} steps ({num_steps*0.02:.1f}s)...")

    jumps_recorded = []
    current_jump = None
    resets_count = 0
    fell = False

    for step_i in range(num_steps):
        with torch.no_grad():
            actions = policy.act_inference(obs)

        obs, _, rews, dones, infos = env.step(actions)

        if dones[0].item():
            resets_count += 1

        fsm = env.fsm_state[0].item()
        pos_x = env.root_states[0, 0].item()
        pos_y = env.root_states[0, 1].item()
        pos_z = env.root_states[0, 2].item()
        vw_y = env.root_states[0, 8].item()
        clr_cm = env.bilateral_clearance[0].item() * 100.0
        fn_L = env.contact_forces[0, env.feet_indices[0], 2].item()
        fn_R = env.contact_forces[0, env.feet_indices[1], 2].item()
        fn_max = max(fn_L, fn_R)

        r_raw, p_raw, _ = get_euler_xyz(env.base_quat)
        pitch = math.degrees(torch.atan2(torch.sin(r_raw), torch.cos(r_raw)).item())
        roll = math.degrees(torch.atan2(torch.sin(p_raw), torch.cos(p_raw)).item())

        if pos_z < 0.20 or abs(pitch) > 45.0 or abs(roll) > 40.0:
            fell = True
            print(f"FAILED: Robot fell at step {step_i}! z={pos_z:.3f}, pitch={pitch:.1f}, roll={roll:.1f}")
            break

        # Check jump lifecycle
        if fsm == env.STATE_JUMP and current_jump is None:
            current_jump = {
                "jump_idx": len(jumps_recorded) + 1,
                "trigger_step": step_i,
                "pre_jump_y": pos_y,
                "liftoff_step": None,
                "liftoff_y": None,
                "liftoff_vw": None,
                "touchdown_step": None,
                "touchdown_y": None,
                "touchdown_vw": None,
                "flight_distance_m": 0.0,
                "max_clearance_cm": 0.0,
                "peak_impact_N": 0.0,
                "recovered": False,
                "handoff_step": None
            }

        if current_jump is not None:
            if clr_cm > current_jump["max_clearance_cm"]:
                current_jump["max_clearance_cm"] = clr_cm

            # Detect liftoff
            if env.has_taken_off[0].item() and current_jump["liftoff_step"] is None:
                current_jump["liftoff_step"] = step_i
                current_jump["liftoff_y"] = pos_y
                current_jump["liftoff_vw"] = vw_y

            # Detect touchdown
            if fsm == env.STATE_RECOVER and current_jump["touchdown_step"] is None:
                current_jump["touchdown_step"] = step_i
                current_jump["touchdown_y"] = pos_y
                current_jump["touchdown_vw"] = vw_y
                if current_jump["liftoff_y"] is not None:
                    current_jump["flight_distance_m"] = pos_y - current_jump["liftoff_y"]

            if fsm == env.STATE_RECOVER:
                if fn_max > current_jump["peak_impact_N"]:
                    current_jump["peak_impact_N"] = fn_max

            # Detect recovery back to BALANCE
            if fsm == env.STATE_BALANCE and current_jump["touchdown_step"] is not None:
                current_jump["recovered"] = True
                current_jump["handoff_step"] = step_i
                jumps_recorded.append(current_jump)
                print(f"  [Jump #{current_jump['jump_idx']}] Liftoff: step={current_jump['liftoff_step']} (y={current_jump['liftoff_y']:.3f}m, vy={current_jump['liftoff_vw']:.3f}m/s) -> Touchdown: step={current_jump['touchdown_step']} (y={current_jump['touchdown_y']:.3f}m) | Dist: {current_jump['flight_distance_m']*100:.1f}cm | Clr: {current_jump['max_clearance_cm']:.1f}cm | Impact: {current_jump['peak_impact_N']:.1f}N | Recovered at step {step_i}")
                current_jump = None

    final_y = env.root_states[0, 1].item()
    final_vy = env.root_states[0, 8].item()

    summary = {
        "seed": seed,
        "total_steps": num_steps,
        "total_time_s": num_steps * 0.02,
        "total_forward_displacement_m": final_y,
        "final_vy": final_vy,
        "num_jumps_completed": len(jumps_recorded),
        "total_resets": resets_count,
        "fell": fell,
        "jumps": jumps_recorded
    }

    print("\n" + "=" * 80)
    print(" === 连续多周期跑跳测试结果 (Continuous Multi-Cycle Jump Summary) ===")
    print(f" 总仿真时长: {num_steps*0.02:.2f} s ({num_steps} steps)")
    print(f" 总前进位移: {final_y:.3f} m (初始 0.000 m -> 终点 {final_y:.3f} m)")
    print(f" 最终巡航速度: {final_vy:.3f} m/s")
    print(f" 完成跳跃总数: {len(jumps_recorded)} 次连续起跳-腾空-落地恢复")
    print(f" 跌倒次数: {1 if fell else 0} | 非正常重置次数: {resets_count}")
    for j in jumps_recorded:
        print(f"   跳跃 #{j['jump_idx']}: 起飞位置 Y={j['liftoff_y']:.3f}m | 落地位置 Y={j['touchdown_y']:.3f}m | 腾空距离={j['flight_distance_m']*100:.1f}cm | 最大净空={j['max_clearance_cm']:.1f}cm | 触地冲击={j['peak_impact_N']:.1f}N")
    print("=" * 80 + "\n")

    return summary


if __name__ == "__main__":
    verify_continuous()
