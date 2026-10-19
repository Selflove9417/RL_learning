# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Paired Physical Benchmark: Controlled Iter 3 vs Iter 1 vs Stage 8 Baseline
# Evaluates Fixed Paired Seeds (11, 13, 15, 20) and Unseen Generalization Seeds (25, 27, 31, 35)

import os
import sys
import math
import json
import csv
import subprocess

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))


def run_single_eval():
    import isaacgym
    from isaacgym import gymtorch
    from isaacgym.torch_utils import get_euler_xyz
    import torch
    import numpy as np
    from legged_gym.envs import MyRobotRunningJump
    from legged_gym.utils import task_registry, get_args
    from rsl_rl.modules import ActorCritic

    args = get_args()
    args.headless = True
    seed = args.seed
    model_path = args.model_path

    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_running_jump")
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed

    env, _ = task_registry.make_env(name="myrobot_running_jump", args=args, env_cfg=env_cfg)
    obs, _ = env.reset()

    ckpt = torch.load(model_path, map_location=env.device)
    policy = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()

    pre_jump_vw = None
    liftoff_vw = None
    touchdown_vw = None
    takeoff_y = None
    touchdown_y = None
    max_clearance = 0.0
    peak_impact = 0.0
    fell = False
    liftoff_step = -1
    touchdown_step = -1

    for step_i in range(140):
        with torch.no_grad():
            actions = policy.act_inference(obs)

        obs, _, rews, dones, infos = env.step(actions)

        vw_y = env.root_states[0, 8].item()
        pos_y = env.root_states[0, 1].item()
        pos_z = env.root_states[0, 2].item()
        clr = env.bilateral_clearance[0].item() * 100.0
        fn_L = env.contact_forces[0, env.feet_indices[0], 2].item()
        fn_R = env.contact_forces[0, env.feet_indices[1], 2].item()

        if clr > max_clearance:
            max_clearance = clr

        if step_i == 39:
            pre_jump_vw = vw_y

        is_contact_lost = (fn_L < 1.0 and fn_R < 1.0)
        if is_contact_lost and liftoff_step == -1 and step_i >= 40:
            liftoff_step = step_i
            liftoff_vw = vw_y
            takeoff_y = pos_y

        if liftoff_step != -1 and touchdown_step == -1 and step_i > liftoff_step + 3:
            if fn_L > 20.0 or fn_R > 20.0:
                touchdown_step = step_i
                touchdown_vw = vw_y
                touchdown_y = pos_y

        if touchdown_step != -1 and step_i >= touchdown_step:
            impact = max(fn_L, fn_R)
            if impact > peak_impact:
                peak_impact = impact

        r_raw, p_raw, _ = get_euler_xyz(env.base_quat)
        pitch = math.degrees(torch.atan2(torch.sin(r_raw), torch.cos(r_raw)).item())
        roll = math.degrees(torch.atan2(torch.sin(p_raw), torch.cos(p_raw)).item())
        if pos_z < 0.20 or abs(pitch) > 45.0 or abs(roll) > 40.0:
            fell = True
            break

    final_vw = env.root_states[0, 8].item()
    flight_dist = (touchdown_y - takeoff_y) if (takeoff_y is not None and touchdown_y is not None) else 0.0

    retention = (liftoff_vw / pre_jump_vw) if (pre_jump_vw and abs(pre_jump_vw) > 0.01 and liftoff_vw is not None) else 0.0

    result = {
        "seed": seed,
        "pre_jump_vw": pre_jump_vw,
        "liftoff_vw": liftoff_vw,
        "retention_pct": retention * 100.0,
        "max_clearance_cm": max_clearance,
        "flight_dist_m": flight_dist,
        "peak_impact_N": peak_impact,
        "final_vw": final_vw,
        "fell": fell,
        "liftoff_step": liftoff_step,
        "touchdown_step": touchdown_step
    }

    print("\n===BENCHMARK_RESULT===")
    print(json.dumps(result))
    print("===BENCHMARK_END===")


def run_worker(seed, model_path):
    cmd = [
        sys.executable,
        "-u",
        __file__,
        "--seed", str(seed),
        "--model_path", model_path
    ]
    sub_env = os.environ.copy()
    sub_env["RUN_BENCH_WORKER"] = "1"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=sub_env)
    stdout, stderr = proc.communicate()

    if "===BENCHMARK_RESULT===" in stdout:
        block = stdout.split("===BENCHMARK_RESULT===")[1].split("===BENCHMARK_END===")[0].strip()
        return json.loads(block)
    else:
        raise RuntimeError(f"Benchmark failed for seed {seed} ({model_path}):\n{stderr[-600:]}")


def main():
    if os.environ.get("RUN_BENCH_WORKER") == "1":
        run_single_eval()
        return

    paired_seeds = [11, 13, 15, 20]
    generalization_seeds = [25, 27, 31, 35]
    all_seeds = paired_seeds + generalization_seeds

    models = {
        "Stage 8 Baseline": "evaluation_assets/frozen_stage8/model_40.pt",
        "Iter 1 Checkpoint": "legged_gym/logs/myrobot_running_jump/Oct19_20-39-35_running_jump_020ms_iter1/model_1.pt",
        "Controlled Iter 3": "legged_gym/logs/myrobot_running_jump/Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt"
    }

    print("=" * 140)
    print(" MYROBOT RUNNING JUMP: PAIRED BENCHMARK EVALUATION ACROSS MODELS AND SEEDS")
    print(" * Models Evaluated:")
    print("   1. Stage 8 Baseline (model_40.pt)")
    print("   2. Iter 1 Checkpoint (model_1.pt)")
    print("   3. Controlled Iter 3 (model_3.pt)")
    print(" * Seeds: Paired [11, 13, 15, 20], Unseen Generalization [25, 27, 31, 35]")
    print("=" * 140)

    results = {m_name: {} for m_name in models}

    for m_name, m_path in models.items():
        print(f"\nEvaluating Model: {m_name} ({m_path})...")
        for s in all_seeds:
            print(f"  - Running Seed {s:2d}...", end="", flush=True)
            res = run_worker(s, m_path)
            results[m_name][s] = res
            print(f" Done. (Pre: {res['pre_jump_vw']:.3f}m/s -> Liftoff: {res['liftoff_vw']:.3f}m/s | Clr: {res['max_clearance_cm']:.1f}cm | Fell: {res['fell']})")

    # Detailed Summary Table
    print("\n" + "=" * 145)
    print(" PAIRED SEEDS SUMMARY (SEEDS 11, 13, 15, 20)")
    print("=" * 145)
    print("| Seed | Model               | Pre v_w(Y) | Liftoff v_w(Y) | Retention | Max Clr (cm) | Flight Dist (m) | Peak Impact (N) | Final v_w(Y) | Fall? |")
    print("|------|---------------------|------------|----------------|-----------|--------------|-----------------|-----------------|--------------|-------|")

    for s in paired_seeds:
        for m_name in models:
            r = results[m_name][s]
            fell_str = "FALL" if r["fell"] else "OK  "
            print(f"|  {s:2d}  | {m_name:19s} | {r['pre_jump_vw']:9.3f}m/s | {r['liftoff_vw']:12.3f}m/s | {r['retention_pct']:8.1f}% | {r['max_clearance_cm']:11.1f}cm | {r['flight_dist_m']:14.3f}m | {r['peak_impact_N']:14.1f}N | {r['final_vw']:11.3f}m/s | {fell_str} |")
        print("|------|---------------------|------------|----------------|-----------|--------------|-----------------|-----------------|--------------|-------|")

    print("\n" + "=" * 145)
    print(" UNSEEN GENERALIZATION SEEDS SUMMARY (SEEDS 25, 27, 31, 35)")
    print("=" * 145)
    print("| Seed | Model               | Pre v_w(Y) | Liftoff v_w(Y) | Retention | Max Clr (cm) | Flight Dist (m) | Peak Impact (N) | Final v_w(Y) | Fall? |")
    print("|------|---------------------|------------|----------------|-----------|--------------|-----------------|-----------------|--------------|-------|")

    for s in generalization_seeds:
        for m_name in models:
            r = results[m_name][s]
            fell_str = "FALL" if r["fell"] else "OK  "
            print(f"|  {s:2d}  | {m_name:19s} | {r['pre_jump_vw']:9.3f}m/s | {r['liftoff_vw']:12.3f}m/s | {r['retention_pct']:8.1f}% | {r['max_clearance_cm']:11.1f}cm | {r['flight_dist_m']:14.3f}m | {r['peak_impact_N']:14.1f}N | {r['final_vw']:11.3f}m/s | {fell_str} |")
        print("|------|---------------------|------------|----------------|-----------|--------------|-----------------|-----------------|--------------|-------|")

    # Aggregate Statistics
    print("\n" + "=" * 115)
    print(" AGGREGATE PERFORMANCE COMPARISON ACROSS ALL 8 SEEDS")
    print("=" * 115)
    print("| Model               | Mean Liftoff v_w(Y) | Mean Retention | Mean Max Clr | Mean Flight Dist | Fall Rate |")
    print("|---------------------|---------------------|----------------|--------------|------------------|-----------|")

    for m_name in models:
        all_r = [results[m_name][s] for s in all_seeds]
        mean_lo_vw = sum(r["liftoff_vw"] for r in all_r) / len(all_r)
        mean_ret = sum(r["retention_pct"] for r in all_r) / len(all_r)
        mean_clr = sum(r["max_clearance_cm"] for r in all_r) / len(all_r)
        mean_dist = sum(r["flight_dist_m"] for r in all_r) / len(all_r)
        fall_rate = sum(1 for r in all_r if r["fell"]) / len(all_r) * 100.0

        print(f"| {m_name:19s} |      {mean_lo_vw:7.3f} m/s     |     {mean_ret:6.1f}%    |   {mean_clr:7.1f} cm   |     {mean_dist:7.3f} m    |   {fall_rate:5.1f}%   |")
    print("=" * 115)


if __name__ == '__main__':
    main()
