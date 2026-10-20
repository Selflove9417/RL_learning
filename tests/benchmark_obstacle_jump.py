#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Physical Obstacle Clearance Benchmark for MyRobot (model_3.pt)
# Rigorous physics evaluation across 8 benchmark seeds with Phase 2 strict criteria.
#
# ============================================================================
# SUPERSEDED -- HISTORIC SCRIPT, KEPT FOR REPRODUCING THE OLD BASELINE ONLY.
# Its geometry is wrong and its numbers cannot be compared with current runs:
#   * it measures the wheel front as `pos_y + wheel_radius`, omitting the ~0.114 m
#     longitudinal offset of the wheel centres and the forward extent of the legs
#     (the true front is up to 0.22 m further ahead);
#   * it reads per-episode flags AFTER env.step() returned, i.e. after reset_idx
#     already cleared them, which is where `touchdown_pos_y = 0.0` and the
#     impossible `flight_dist = -0.106 m` came from;
#   * it counts any Isaac Gym contact record as a collision, including
#     speculative proximity records with `lambda == 0` that carry no impulse;
#   * its recovery criterion accepts 65 % of the commanded speed.
# For current results use:
#   python tests/evaluate_obstacle_jump.py --pipeline cpu
#   python tests/evaluate_obstacle_jump_continuous.py --pipeline cpu
# ============================================================================

import os
import sys
import math
import json
import argparse
import subprocess
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))


def run_single_seed_eval():
    import isaacgym
    from isaacgym import gymtorch
    from isaacgym.torch_utils import get_euler_xyz
    import torch
    from legged_gym.envs import MyRobotObstacleJump
    from legged_gym.utils import task_registry, get_args
    from rsl_rl.modules import ActorCritic

    trigger_mode = "spatial"
    if "--trigger_mode" in sys.argv:
        idx = sys.argv.index("--trigger_mode")
        trigger_mode = sys.argv[idx + 1]
        del sys.argv[idx:idx+2]
    if "--eval_worker" in sys.argv:
        sys.argv.remove("--eval_worker")

    args = get_args()
    args.headless = True
    seed = args.seed
    task_name = args.task if args.task != 'anymal_c_flat' else 'myrobot_obstacle_jump'
    model_path = args.model_path if args.model_path is not None else "legged_gym/logs/myrobot_running_jump/Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt"



    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.env.episode_length_s = 4.0   # 200 steps (no early timeout reset)
    env_cfg.trigger.mode = trigger_mode

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    obs, _ = env.reset()

    ckpt = torch.load(model_path, map_location=env.device)
    policy = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()

    R = env.wheel_radius  # 0.070m
    obs_h = env.obstacle_h
    obs_w = env.obstacle_w
    obs_py = env.obstacle_py
    obs_front_y = obs_py - 0.5 * obs_w
    obs_rear_y = obs_py + 0.5 * obs_w

    pre_jump_vw = None
    liftoff_vw = None
    touchdown_vw = None
    liftoff_pos_y = None
    touchdown_pos_y = None
    liftoff_step = -1
    touchdown_step = -1
    peak_impact = 0.0
    fell = False

    # Observation window for post-landing velocity recovery (fixed timeout reset pollution)
    recovery_vels = []

    for step_i in range(180):
        with torch.no_grad():
            actions = policy.act_inference(obs)

        obs, _, rews, dones, infos = env.step(actions)

        vw_y = env.root_states[0, 8].item()
        pos_y = env.root_states[0, 1].item()
        pos_z = env.root_states[0, 2].item()
        fn_L = env.contact_forces[0, env.feet_indices[0], 2].item()
        fn_R = env.contact_forces[0, env.feet_indices[1], 2].item()

        # Track approach velocity
        if step_i == 35:
            pre_jump_vw = vw_y

        # Detect liftoff
        is_airborne = (fn_L < 1.0 and fn_R < 1.0) and (env.bilateral_clearance[0].item() > 0.015)
        if is_airborne and liftoff_step == -1 and env.fsm_state[0].item() == env.STATE_JUMP:
            liftoff_step = step_i
            liftoff_vw = vw_y
            liftoff_pos_y = pos_y

        # Detect touchdown
        if liftoff_step != -1 and touchdown_step == -1 and step_i > liftoff_step + 4:
            if fn_L > 20.0 or fn_R > 20.0:
                touchdown_step = step_i
                touchdown_vw = vw_y
                touchdown_pos_y = pos_y

        if touchdown_step != -1 and step_i >= touchdown_step:
            impact = max(fn_L, fn_R)
            if impact > peak_impact:
                peak_impact = impact

            # Collect recovery window samples: 15 to 40 steps after touchdown (0.3s ~ 0.8s)
            if touchdown_step + 15 <= step_i <= touchdown_step + 40:
                recovery_vels.append(vw_y)

        # Fall check
        r_raw, p_raw, _ = get_euler_xyz(env.base_quat)
        pitch = math.degrees(torch.atan2(torch.sin(r_raw), torch.cos(r_raw)).item())
        roll = math.degrees(torch.atan2(torch.sin(p_raw), torch.cos(p_raw)).item())
        if pos_z < 0.20 or abs(pitch) > 45.0 or abs(roll) > 40.0:
            fell = True
            break

    # Geometry calculations
    flight_dist = (touchdown_pos_y - liftoff_pos_y) if (liftoff_pos_y is not None and touchdown_pos_y is not None) else 0.0
    liftoff_wheel_front = (liftoff_pos_y + R) if liftoff_pos_y is not None else 0.0
    touchdown_wheel_rear = (touchdown_pos_y - R) if touchdown_pos_y is not None else 0.0

    takeoff_margin = (obs_front_y - liftoff_wheel_front) if liftoff_pos_y is not None else -1.0
    landing_margin = (touchdown_wheel_rear - obs_rear_y) if touchdown_pos_y is not None else -1.0

    has_collision = env.has_obstacle_collision[0].item()
    peak_obs_force = env.peak_obstacle_contact_force[0].item()
    min_obs_clr_cm = env.min_obstacle_clearance[0].item() * 100.0

    mean_recovery_vw = float(np.mean(recovery_vels)) if len(recovery_vels) > 0 else 0.0
    target_v = env.target_forward_vel
    vel_retention_pct = (mean_recovery_vw / target_v * 100.0) if target_v > 0 else 0.0

    # Phase 2 Strict Success Criteria:
    # 1. Forward approach without stopping (pre_jump_vw >= 0.15)
    # 2. Liftoff occurs before obstacle front edge (takeoff_margin > 0.005)
    # 3. Touchdown occurs past obstacle rear edge (landing_margin > 0.005)
    # 4. Zero collision / penetration (not has_collision)
    # 5. Clearance above obstacle top > 0 (min_obs_clr_cm > 0.0)
    # 6. Post-landing velocity recovery >= 65% of target
    # 7. Robot did not fall
    phase2_success = (
        (pre_jump_vw is not None and pre_jump_vw >= 0.15) and
        (takeoff_margin > 0.005) and
        (landing_margin > 0.005) and
        (not has_collision) and
        (min_obs_clr_cm > 0.0) and
        (vel_retention_pct >= 65.0) and
        (not fell)
    )

    result = {
        "seed": seed,
        "task": task_name,
        "trigger_mode": trigger_mode,
        "obstacle_h_cm": obs_h * 100.0,
        "obstacle_w_cm": obs_w * 100.0,
        "obstacle_py_m": obs_py,
        "obs_front_y_m": obs_front_y,
        "obs_rear_y_m": obs_rear_y,
        "pre_jump_vw": pre_jump_vw,
        "liftoff_step": liftoff_step,
        "liftoff_vw": liftoff_vw,
        "liftoff_pos_y": liftoff_pos_y,
        "liftoff_wheel_front": liftoff_wheel_front,
        "takeoff_margin_cm": takeoff_margin * 100.0,
        "touchdown_step": touchdown_step,
        "touchdown_vw": touchdown_vw,
        "touchdown_pos_y": touchdown_pos_y,
        "touchdown_wheel_rear": touchdown_wheel_rear,
        "landing_margin_cm": landing_margin * 100.0,
        "flight_dist_m": flight_dist,
        "flight_steps": (touchdown_step - liftoff_step) if (liftoff_step != -1 and touchdown_step != -1) else 0,
        "has_obstacle_collision": has_collision,
        "peak_obstacle_force_N": peak_obs_force,
        "min_obstacle_clearance_cm": min_obs_clr_cm,
        "peak_impact_N": peak_impact,
        "recovery_vw": mean_recovery_vw,
        "vel_retention_pct": vel_retention_pct,
        "fell": fell,
        "phase2_success": phase2_success
    }

    print("\n===OBSTACLE_BENCHMARK_RESULT===")
    print(json.dumps(result))


def run_benchmark():
    eval_seeds = [11, 13, 15, 20, 25, 27, 31, 35]
    model_path = "legged_gym/logs/myrobot_running_jump/Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt"

    eval_modes = [
        ("0.35 m/s Spatial Trigger", "myrobot_obstacle_jump", "spatial"),
        ("0.35 m/s Step-40 Trigger (Legacy)", "myrobot_obstacle_jump", "step")
    ]

    all_results = {}

    for mode_label, task_name, trigger_mode in eval_modes:
        print(f"\n{'='*80}")
        print(f" >>> Evaluating {mode_label} (Task: {task_name}, Trigger: {trigger_mode}) <<<")
        print(f"{'='*80}")
        mode_data = []

        for s in eval_seeds:
            cmd = [
                sys.executable, __file__,
                "--eval_worker",
                "--task", task_name,
                "--model_path", model_path,
                "--seed", str(s),
                "--trigger_mode", trigger_mode
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{os.path.join(workspace_root, 'legged_gym')}:{os.path.join(workspace_root, 'rsl_rl')}:{env.get('PYTHONPATH', '')}"

            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            found = False
            for line in res.stdout.split("\n"):
                if line.startswith("===OBSTACLE_BENCHMARK_RESULT==="):
                    found = True
                    continue
                if found and line.strip().startswith("{"):
                    try:
                        record = json.loads(line.strip())
                        mode_data.append(record)
                        status_str = "SUCCESS" if record["phase2_success"] else "FAIL"
                        print(f"  Seed {s:2d} | Liftoff: {record['liftoff_pos_y']:.3f}m | Landing: {record['touchdown_pos_y']:.3f}m | "
                              f"Dist: {record['flight_dist_m']*100:.1f}cm | TakeoffMarg: {record['takeoff_margin_cm']:+.1f}cm | "
                              f"LandMarg: {record['landing_margin_cm']:+.1f}cm | Collision: {record['has_obstacle_collision']} "
                              f"(F={record['peak_obstacle_force_N']:.1f}N) | Clr: {record['min_obstacle_clearance_cm']:+.1f}cm | "
                              f"Recov_vw: {record['recovery_vw']:.3f}m/s ({record['vel_retention_pct']:.1f}%) | Result: {status_str}")
                    except Exception as e:
                        print(f"  Seed {s} JSON parse error: {e}")
                    break
            if not found:
                print(f"  Seed {s} failed to produce benchmark output!")
                print("  Stderr:", res.stderr[-500:] if len(res.stderr) > 0 else "None")

        all_results[mode_label] = mode_data

    # Save to evaluation_assets
    output_path = os.path.join(workspace_root, "evaluation_assets", "obstacle_jump_baseline_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved benchmark results to: {output_path}")

    # Print summary table
    print("\n" + "=" * 95)
    print(" === 真实物理障碍跨越测试总结 (True Physical Obstacle Jump Benchmark Summary) ===")
    print("=" * 95)
    for mode_label, data in all_results.items():
        if len(data) == 0:
            continue
        n_total = len(data)
        n_success = sum(1 for d in data if d["phase2_success"])
        n_collision = sum(1 for d in data if d["has_obstacle_collision"])
        n_fell = sum(1 for d in data if d["fell"])
        mean_flight = np.mean([d["flight_dist_m"] * 100.0 for d in data])
        mean_takeoff_margin = np.mean([d["takeoff_margin_cm"] for d in data])
        mean_landing_margin = np.mean([d["landing_margin_cm"] for d in data])
        mean_obs_clr = np.mean([d["min_obstacle_clearance_cm"] for d in data])
        mean_recov_v = np.mean([d["recovery_vw"] for d in data])
        mean_retention = np.mean([d["vel_retention_pct"] for d in data])

        print(f"\n模式: {mode_label}")
        print(f"  测试种子总数: {n_total}")
        print(f"  Phase 2 严格跨越成功率: {n_success}/{n_total} ({n_success/n_total*100:.1f}%)")
        print(f"  障碍碰撞率: {n_collision}/{n_total} ({n_collision/n_total*100:.1f}%)")
        print(f"  跌倒失败率: {n_fell}/{n_total} ({n_fell/n_total*100:.1f}%)")
        print(f"  平均飞行距离: {mean_flight:.1f} cm")
        print(f"  平均起跳前缘裕度: {mean_takeoff_margin:+.1f} cm (正值表示在障碍前离地)")
        print(f"  平均着陆后缘裕度: {mean_landing_margin:+.1f} cm (正值表示在障碍后着陆)")
        print(f"  跨障最小净空: {mean_obs_clr:+.1f} cm (负值表示穿模/碰撞)")
        print(f"  着陆后巡航恢复速度: {mean_recov_v:.3f} m/s (速度保留率: {mean_retention:.1f}%)")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_worker", action="store_true")
    parser.add_argument("--task", type=str, default="myrobot_obstacle_jump")
    parser.add_argument("--model_path", type=str, default="legged_gym/logs/myrobot_running_jump/Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--trigger_mode", type=str, default="spatial")
    args, unknown = parser.parse_known_args()

    if args.eval_worker:
        run_single_seed_eval()
    else:
        run_benchmark()
