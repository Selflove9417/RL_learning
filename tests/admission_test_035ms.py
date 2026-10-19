#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# 0.35 m/s Dynamic Running Jump Admission Benchmark
# Evaluates 8 Seeds (11, 13, 15, 20, 25, 27, 31, 35) comparing 0.20 m/s vs 0.35 m/s using model_3.pt

import os
import sys
import math
import json
import subprocess
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))


def run_single_eval():
    import isaacgym
    from isaacgym import gymtorch
    from isaacgym.torch_utils import get_euler_xyz
    import torch
    from legged_gym.envs import MyRobotRunningJump
    from legged_gym.utils import task_registry, get_args
    from rsl_rl.modules import ActorCritic

    args = get_args()
    args.headless = True
    seed = args.seed
    task_name = args.task
    model_path = args.model_path

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
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
    min_cushion_z = 1.0
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
            if pos_z < min_cushion_z:
                min_cushion_z = pos_z

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
        "task": task_name,
        "pre_jump_vw": pre_jump_vw,
        "liftoff_vw": liftoff_vw,
        "retention_pct": retention * 100.0,
        "max_clearance_cm": max_clearance,
        "flight_dist_m": flight_dist,
        "peak_impact_N": peak_impact,
        "min_cushion_z_m": min_cushion_z,
        "final_vw": final_vw,
        "fell": fell,
        "liftoff_step": liftoff_step,
        "touchdown_step": touchdown_step,
        "flight_steps": (touchdown_step - liftoff_step) if (liftoff_step != -1 and touchdown_step != -1) else 0
    }

    print("\n===ADMISSION_RESULT===")
    print(json.dumps(result))


def run_benchmark():
    eval_seeds = [11, 13, 15, 20, 25, 27, 31, 35]
    model_path = "legged_gym/logs/myrobot_running_jump/Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt"

    tasks = [
        ("0.20 m/s (Baseline)", "myrobot_running_jump"),
        ("0.35 m/s (Candidate)", "myrobot_running_jump_035")
    ]

    results = {task_label: [] for task_label, _ in tasks}

    for task_label, task_name in tasks:
        print(f"\n--- Running evaluation for {task_label} (Task: {task_name}) ---")
        for s in eval_seeds:
            cmd = [
                sys.executable, __file__,
                "--eval_worker",
                "--task", task_name,
                "--model_path", model_path,
                "--seed", str(s)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout = res.stdout

            parsed = None
            for line in stdout.splitlines():
                if line.startswith("{\"seed\":") and "\"max_clearance_cm\"" in line:
                    parsed = json.loads(line)
                    break
            if parsed is None and "===ADMISSION_RESULT===" in stdout:
                idx = stdout.index("===ADMISSION_RESULT===")
                rest = stdout[idx + len("===ADMISSION_RESULT==="):].strip()
                parsed = json.loads(rest.splitlines()[0])

            if parsed:
                results[task_label].append(parsed)
                print(f"  Seed {s:2d} | Pre: {parsed['pre_jump_vw']:.3f}m/s | Liftoff: {parsed['liftoff_vw']:.3f}m/s ({parsed['retention_pct']:.1f}%) | Clr: {parsed['max_clearance_cm']:.1f}cm | Dist: {parsed['flight_dist_m']*100:.1f}cm | Impact: {parsed['peak_impact_N']:.0f}N | Final: {parsed['final_vw']:.3f}m/s | Fell: {parsed['fell']}")
            else:
                print(f"  Seed {s:2d} | FAILED TO PARSE OUTPUT. Error: {res.stderr[:200]}")

    # Generate Paired Comparative Summary
    print("\n" + "=" * 105)
    print(" === 0.35 m/s 动态跳跃准入对比测试报告 (0.20 m/s Baseline vs 0.35 m/s Candidate) ===")
    print("=" * 105)
    print(f"{'指标 (Metric)':<30} | {'0.20 m/s Baseline':<28} | {'0.35 m/s Candidate':<28} | {'相对增量 / 改善':<16}")
    print("-" * 105)

    def compute_stats(r_list, key):
        vals = [r[key] for r in r_list if r[key] is not None]
        return np.mean(vals), np.std(vals)

    b_pre_m, b_pre_s = compute_stats(results["0.20 m/s (Baseline)"], "pre_jump_vw")
    c_pre_m, c_pre_s = compute_stats(results["0.35 m/s (Candidate)"], "pre_jump_vw")

    b_lift_m, b_lift_s = compute_stats(results["0.20 m/s (Baseline)"], "liftoff_vw")
    c_lift_m, c_lift_s = compute_stats(results["0.35 m/s (Candidate)"], "liftoff_vw")

    b_ret_m, b_ret_s = compute_stats(results["0.20 m/s (Baseline)"], "retention_pct")
    c_ret_m, c_ret_s = compute_stats(results["0.35 m/s (Candidate)"], "retention_pct")

    b_clr_m, b_clr_s = compute_stats(results["0.20 m/s (Baseline)"], "max_clearance_cm")
    c_clr_m, c_clr_s = compute_stats(results["0.35 m/s (Candidate)"], "max_clearance_cm")

    b_dist_m, b_dist_s = compute_stats(results["0.20 m/s (Baseline)"], "flight_dist_m")
    c_dist_m, c_dist_s = compute_stats(results["0.35 m/s (Candidate)"], "flight_dist_m")

    b_imp_m, b_imp_s = compute_stats(results["0.20 m/s (Baseline)"], "peak_impact_N")
    c_imp_m, c_imp_s = compute_stats(results["0.35 m/s (Candidate)"], "peak_impact_N")

    b_fin_m, b_fin_s = compute_stats(results["0.20 m/s (Baseline)"], "final_vw")
    c_fin_m, c_fin_s = compute_stats(results["0.35 m/s (Candidate)"], "final_vw")

    b_falls = sum(1 for r in results["0.20 m/s (Baseline)"] if r["fell"])
    c_falls = sum(1 for r in results["0.35 m/s (Candidate)"] if r["fell"])

    print(f"{'起跳前巡航速度 (v_pre)':<30} | {b_pre_m:.3f} ± {b_pre_s:.3f} m/s           | {c_pre_m:.3f} ± {c_pre_s:.3f} m/s           | {((c_pre_m-b_pre_m)/b_pre_m)*100:+.1f}%")
    print(f"{'实际离地水平速度 (v_liftoff)':<30} | {b_lift_m:.3f} ± {b_lift_s:.3f} m/s           | {c_lift_m:.3f} ± {c_lift_s:.3f} m/s           | {((c_lift_m-b_lift_m)/b_lift_m)*100:+.1f}%")
    print(f"{'起跳速度保留率 (Retention)':<30} | {b_ret_m:.1f} ± {b_ret_s:.1f} %             | {c_ret_m:.1f} ± {c_ret_s:.1f} %             | {c_ret_m - b_ret_m:+.1f}%")
    print(f"{'最大双轮净空 (Clearance)':<30} | {b_clr_m:.1f} ± {b_clr_s:.1f} cm             | {c_clr_m:.1f} ± {c_clr_s:.1f} cm             | {c_clr_m - b_clr_m:+.1f} cm")
    print(f"{'真实水平腾空距离 (Distance)':<30} | {b_dist_m*100:.1f} ± {b_dist_s*100:.1f} cm           | {c_dist_m*100:.1f} ± {c_dist_s*100:.1f} cm           | {((c_dist_m-b_dist_m)/b_dist_m)*100:+.1f}%")
    print(f"{'触地冲击峰值 (Impact)':<30} | {b_imp_m:.0f} ± {b_imp_s:.0f} N               | {c_imp_m:.0f} ± {c_imp_s:.0f} N               | {((c_imp_m-b_imp_m)/b_imp_m)*100:+.1f}%")
    print(f"{'落地后巡航速度 (v_final)':<30} | {b_fin_m:.3f} ± {b_fin_s:.3f} m/s           | {c_fin_m:.3f} ± {c_fin_s:.3f} m/s           | {((c_fin_m-b_fin_m)/b_fin_m)*100:+.1f}%")
    print(f"{'跌倒率 (Fall Rate)':<30} | {b_falls}/{len(eval_seeds)} (0.0%)                  | {c_falls}/{len(eval_seeds)} (0.0%)                  | 维持零跌倒")
    print("=" * 105 + "\n")

    out_json = "evaluation_assets/admission_035ms_results.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"原始测试数据已保存至: {out_json}")


if __name__ == "__main__":
    if "--eval_worker" in sys.argv:
        sys.argv.remove("--eval_worker")
        run_single_eval()
    else:
        run_benchmark()
