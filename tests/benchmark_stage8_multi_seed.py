# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Benchmark Script: Stage 8 Dynamic Running Jump Baseline across 10 Random Seeds
#
# Metrics Collected:
# 1. 触发前速度 (Pre-jump forward velocity at Step 39)
# 2. 实际离地瞬间前向速度 (Liftoff forward velocity)
# 3. 触地瞬间前向速度 (Touchdown forward velocity)
# 4. 恢复后的前向速度 (Resumed steady-state forward velocity before timeout)
# 5. 腾空位移 (Airborne forward displacement delta y)
# 6. 最大净空 (Max ground clearance)
# 7. 跌倒率 (Fall rate across 10 seeds)

import os
import sys
import math
import json
import argparse
import subprocess

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))


def run_single_seed_benchmark(seed):
    import isaacgym
    from isaacgym import gymtorch
    import torch
    import numpy as np
    from legged_gym.envs import MyRobotRunningJump
    from legged_gym.utils import task_registry, get_args

    args = get_args()
    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_running_jump")
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed

    env, _ = task_registry.make_env(name="myrobot_running_jump", args=args, env_cfg=env_cfg)
    env.reset()

    obs = env.get_observations()

    pre_jump_vy = 0.0
    liftoff_step = -1
    touchdown_step = -1
    timeout_step = -1
    liftoff_vy = 0.0
    touchdown_vy = 0.0
    liftoff_pos_y = 0.0
    touchdown_pos_y = 0.0
    max_clearance = 0.0
    fell = False
    fall_step = -1
    fall_reason = ""
    resumed_vy = 0.0

    for step_i in range(140):
        # Capture pre-jump velocity at step 39
        if step_i == 39:
            pre_jump_vy = env.base_lin_vel[0, 1].item()

        with torch.no_grad():
            actions = env.ac_jump.act_inference(obs)

        obs, _, rews, dones, infos = env.step(actions)

        vy = env.base_lin_vel[0, 1].item()
        height = env.root_states[0, 2].item()
        pos_y = env.root_states[0, 1].item()
        clr = env.bilateral_clearance[0].item()
        took_off = env.has_taken_off[0].item()
        touched_down = env.has_touched_down[0].item()
        is_timeout = infos.get("time_outs", [False])[0].item()

        if is_timeout and timeout_step == -1:
            timeout_step = step_i

        if clr > max_clearance:
            max_clearance = clr

        # Detect liftoff event
        if took_off and liftoff_step == -1:
            liftoff_step = step_i
            liftoff_vy = vy
            liftoff_pos_y = pos_y

        # Detect touchdown event
        if touched_down and touchdown_step == -1:
            touchdown_step = step_i
            touchdown_vy = vy
            touchdown_pos_y = pos_y

        # Steady-state resumed cruising velocity before timeout
        if step_i >= 120 and timeout_step == -1:
            resumed_vy = vy

        # Check fall
        if dones[0].item() and not is_timeout and timeout_step == -1:
            fell = True
            fall_step = step_i
            r_raw, p_raw, _ = isaacgym.torch_utils.get_euler_xyz(env.base_quat)
            pitch = torch.atan2(torch.sin(r_raw), torch.cos(r_raw)).item()
            roll = torch.atan2(torch.sin(p_raw), torch.cos(p_raw)).item()
            if height < 0.20:
                fall_reason = f"Height drop ({height:.2f}m < 0.20m)"
            elif abs(pitch) > math.radians(45.0):
                fall_reason = f"Pitch tilt ({math.degrees(pitch):.1f} deg > 45 deg)"
            elif abs(roll) > math.radians(40.0):
                fall_reason = f"Roll tilt ({math.degrees(roll):.1f} deg > 40 deg)"
            else:
                fall_reason = "Unspecified fall"
            break

    # Calculate flight displacement
    if touchdown_step != -1 and liftoff_step != -1:
        flight_disp = touchdown_pos_y - liftoff_pos_y
    else:
        flight_disp = 0.0

    result = {
        "seed": seed,
        "pre_jump_vy": float(pre_jump_vy),
        "liftoff_step": int(liftoff_step),
        "liftoff_vy": float(liftoff_vy),
        "touchdown_step": int(touchdown_step),
        "touchdown_vy": float(touchdown_vy),
        "resumed_vy": float(resumed_vy if not fell else 0.0),
        "flight_disp": float(flight_disp),
        "max_clearance": float(max_clearance),
        "fell": bool(fell),
        "fall_step": int(fall_step),
        "fall_reason": str(fall_reason)
    }

    print("\n===RESULT_START===")
    print(json.dumps(result))
    print("===RESULT_END===")


def run_multi_seed_orchestrator(seeds):
    import numpy as np

    results = []

    print("=" * 80)
    print(f" STAGE 8 RUNNING JUMP BASELINE BENCHMARK ACROSS {len(seeds)} SEEDS")
    print("=" * 80)

    python_executable = sys.executable
    script_path = os.path.abspath(__file__)

    for s in seeds:
        print(f"Evaluating Seed {s:2d}...", end="", flush=True)
        cmd = [python_executable, script_path, "--seed", str(s)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if res.returncode != 0:
            print(f" [CRASH ERROR (code {res.returncode})]")
            print("STDERR:", res.stderr[-500:])
            continue

        # Parse JSON output from subprocess
        output = res.stdout
        start_tag = "===RESULT_START==="
        end_tag = "===RESULT_END==="
        if start_tag in output and end_tag in output:
            json_str = output.split(start_tag)[1].split(end_tag)[0].strip()
            data = json.loads(json_str)
            results.append(data)
            status = "FELL at Step " + str(data["fall_step"]) if data["fell"] else "PASSED"
            print(f" [{status}] Pre-vy: {data['pre_jump_vy']:.3f} | Takeoff-vy: {data['liftoff_vy']:.3f} | "
                  f"Touchdown-vy: {data['touchdown_vy']:.3f} | Resumed-vy: {data['resumed_vy']:.3f} | "
                  f"MaxClr: {data['max_clearance']*100:.1f}cm | Disp: {data['flight_disp']*100:.1f}cm")
        else:
            print(" [PARSE ERROR: No result markers in output]")
            print(output[-500:])

    # Print Full Markdown Summary Table
    print("\n" + "=" * 115)
    print(" 10-SEED BENCHMARK SUMMARY TABLE")
    print("=" * 115)
    header = (f"| {'Seed':4s} | {'起跳前速度':10s} | {'离地瞬时速度':12s} | {'触地瞬时速度':12s} | "
              f"{'恢复后速度':10s} | {'腾空位移':8s} | {'最大净空':8s} | {'状态 / 跌倒原因':24s} |")
    print(header)
    print("|" + "|".join(["-" * (len(col) + 2) for col in header.split("|")[1:-1]]) + "|")

    for r in results:
        status_str = f"跌倒 (Step {r['fall_step']}, {r['fall_reason']})" if r["fell"] else "成功存活"
        print(f"| {r['seed']:4d} | {r['pre_jump_vy']:8.3f} m/s | {r['liftoff_vy']:10.3f} m/s | "
              f"{r['touchdown_vy']:10.3f} m/s | {r['resumed_vy']:8.3f} m/s | "
              f"{r['flight_disp']*100:6.1f} cm | {r['max_clearance']*100:6.1f} cm | {status_str:24s} |")

    print("=" * 115)

    # Compute Statistics for surviving and all runs
    num_falls = sum(1 for r in results if r["fell"])
    fall_rate = (num_falls / len(seeds)) * 100.0

    survived = [r for r in results if not r["fell"]]
    print(f"\n[总体评估统计指标] (总测试种子数: {len(seeds)})")
    print(f"  - 跌倒率 (Fall Rate)           : {fall_rate:.1f}% ({num_falls}/{len(seeds)})")
    print(f"  - 平均触发前速度 (Pre-jump vy)  : {np.mean([r['pre_jump_vy'] for r in results]):.4f} +/- {np.std([r['pre_jump_vy'] for r in results]):.4f} m/s")
    print(f"  - 平均离地前向速度 (Liftoff vy) : {np.mean([r['liftoff_vy'] for r in results]):.4f} +/- {np.std([r['liftoff_vy'] for r in results]):.4f} m/s")
    print(f"  - 平均触地前向速度 (Touchdown vy): {np.mean([r['touchdown_vy'] for r in results]):.4f} +/- {np.std([r['touchdown_vy'] for r in results]):.4f} m/s")
    print(f"  - 平均腾空水平位移 (Flight Disp): {np.mean([r['flight_disp'] for r in results])*100:.2f} +/- {np.std([r['flight_disp'] for r in results])*100:.2f} cm")
    print(f"  - 平均最大净空 (Max Clearance) : {np.mean([r['max_clearance'] for r in results])*100:.2f} +/- {np.std([r['max_clearance'] for r in results])*100:.2f} cm")

    if len(survived) > 0:
        print(f"  - 存活回合恢复巡航速度 (Resumed vy): {np.mean([r['resumed_vy'] for r in survived]):.4f} +/- {np.std([r['resumed_vy'] for r in survived]):.4f} m/s")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 8 Running Jump Baseline Multi-Seed Benchmark")
    parser.add_argument("--seed", type=int, default=None, help="Evaluate a single seed")
    custom_args, _ = parser.parse_known_args()

    if custom_args.seed is not None:
        run_single_seed_benchmark(custom_args.seed)
    else:
        seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        run_multi_seed_orchestrator(seeds)
