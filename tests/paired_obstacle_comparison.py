#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Paired Physical Control Experiment: With Obstacle vs Without Obstacle
# Evaluates model_3.pt under identical random seeds and identical takeoff spatial triggering.
#
# ============================================================================
# SUPERSEDED -- HISTORIC SCRIPT, KEPT FOR REPRODUCING THE OLD BASELINE ONLY.
# Do not quote its output as evidence about the current task:
#   * the "without obstacle" arm only moves the box to y = 10 m; the box stays in
#     the scene, so it is not a free run;
#   * the wheel front is `pos_y + wheel_radius`, which is up to 0.22 m behind the
#     true URDF envelope, so both the trigger distance and the margins are wrong;
#   * it counts any contact record as a hit, including zero-impulse speculative
#     proximity records (a real 9.885 mm near-miss was reported as a 577-2333 N
#     strike);
#   * it reads flags after env.step() returned, i.e. after reset_idx cleared them;
#   * hence its claim that all 8 seeds make first contact BEFORE liftoff does not
#     reproduce under the corrected geometry: 2 of 8 do (seeds 25 and 31).
# For current results use:
#   python tests/evaluate_obstacle_jump.py --pipeline cpu
#   python tests/analyze_first_contact.py --pipeline cpu
# ============================================================================

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

    from legged_gym.envs import MyRobotObstacleJump
    from legged_gym.utils import task_registry, get_args
    from rsl_rl.modules import ActorCritic

    with_obstacle = (os.environ.get("WITH_OBSTACLE", "1") == "1")

    args = get_args()
    args.headless = True
    seed = args.seed
    args.task = "myrobot_obstacle_jump"
    model_path = "legged_gym/logs/myrobot_running_jump/Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt"

    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_obstacle_jump")
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.env.episode_length_s = 4.0
    env_cfg.trigger.mode = "spatial"
    env_cfg.trigger.spatial_trigger_dist = 0.09

    if not with_obstacle:
        # Move obstacle far away (10m) so robot has completely flat, unobstructed ground
        env_cfg.obstacle.pos_y = 10.0

    env, _ = task_registry.make_env(name="myrobot_obstacle_jump", args=args, env_cfg=env_cfg)
    obs, _ = env.reset()

    ckpt = torch.load(model_path, map_location=env.device)
    policy = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()

    R = env.wheel_radius
    obs_front_y = 0.30 - 0.5 * 0.04  # 0.28m

    step_log = []
    liftoff_step = -1
    liftoff_pos_y = None
    liftoff_vw = None
    obs_contact_step = -1
    obs_contact_force_peak = 0.0
    touchdown_step = -1
    touchdown_pos_y = None
    touchdown_vw = None
    recovery_vels = []

    for step_i in range(160):
        # In Condition B (without obstacle), trigger takeoff at the exact same spatial coordinate
        # (when robot base_y reaches 0.12m, matching Condition A's trigger condition)
        if not with_obstacle:
            robot_front_y = env.root_states[0, 1].item() + R
            dist_to_virtual_front = obs_front_y - robot_front_y
            in_balance = (env.fsm_state[0].item() == env.STATE_BALANCE)
            is_cruising = env.root_states[0, 8].item() > 0.10
            reached_window = (dist_to_virtual_front <= 0.09) and (dist_to_virtual_front >= -0.02)
            has_warmed_up = env.episode_length_buf[0].item() >= 20
            if in_balance and is_cruising and reached_window and has_warmed_up:
                env.request_teleop_jump(torch.tensor([True], device=env.device))

        with torch.no_grad():
            actions = policy.act_inference(obs)
        obs, _, rews, dones, infos = env.step(actions)

        fsm = env.fsm_state[0].item()
        vw_y = env.root_states[0, 8].item()
        pos_y = env.root_states[0, 1].item()
        pos_z = env.root_states[0, 2].item()
        fn_L = env.contact_forces[0, env.feet_indices[0], 2].item()
        fn_R = env.contact_forces[0, env.feet_indices[1], 2].item()
        clr_cm = env.bilateral_clearance[0].item() * 100.0

        if with_obstacle:
            obs_fn = torch.norm(env.obstacle_contact_forces[0]).item()
        else:
            obs_fn = 0.0

        is_airborne = (fn_L < 1.0 and fn_R < 1.0) and (clr_cm > 1.5)
        if is_airborne and liftoff_step == -1 and fsm == env.STATE_JUMP:
            liftoff_step = step_i
            liftoff_vw = vw_y
            liftoff_pos_y = pos_y

        if with_obstacle and obs_fn > 1.0 and obs_contact_step == -1:
            obs_contact_step = step_i
        if obs_fn > obs_contact_force_peak:
            obs_contact_force_peak = obs_fn

        if liftoff_step != -1 and touchdown_step == -1 and step_i > liftoff_step + 4:
            if fn_L > 20.0 or fn_R > 20.0:
                touchdown_step = step_i
                touchdown_vw = vw_y
                touchdown_pos_y = pos_y

        if touchdown_step != -1:
            if touchdown_step + 15 <= step_i <= touchdown_step + 40:
                recovery_vels.append(vw_y)

        step_log.append({
            "step": step_i,
            "fsm": fsm,
            "pos_y": pos_y,
            "pos_z": pos_z,
            "vw_y": vw_y,
            "fn_ground": max(fn_L, fn_R),
            "fn_obs": obs_fn,
            "clr_cm": clr_cm
        })

    flight_dist = (touchdown_pos_y - liftoff_pos_y) if (liftoff_pos_y and touchdown_pos_y) else 0.0
    mean_rec_vel = float(np.mean(recovery_vels)) if len(recovery_vels) > 0 else 0.0

    res = {
        "seed": seed,
        "with_obstacle": with_obstacle,
        "liftoff_step": liftoff_step,
        "liftoff_vw": liftoff_vw,
        "liftoff_pos_y": liftoff_pos_y,
        "obs_contact_step": obs_contact_step,
        "obs_contact_force_peak": obs_contact_force_peak,
        "touchdown_step": touchdown_step,
        "touchdown_vw": touchdown_vw,
        "touchdown_pos_y": touchdown_pos_y,
        "flight_dist_m": flight_dist,
        "flight_steps": (touchdown_step - liftoff_step) if (liftoff_step != -1 and touchdown_step != -1) else 0,
        "mean_recovery_vel": mean_rec_vel,
        "step_log": step_log
    }

    out_file = f"/tmp/paired_trial_seed{seed}_{'with' if with_obstacle else 'without'}.json"
    with open(out_file, "w") as f:
        json.dump(res, f)
    print(f"RESULT_FILE:{out_file}")


def execute_subproc(seed, with_obstacle):
    cmd = [
        sys.executable, __file__,
        "--paired_worker",
        "--seed", str(seed),
        "--headless"
    ]
    env = os.environ.copy()
    env["WITH_OBSTACLE"] = "1" if with_obstacle else "0"
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"Error executing subprocess for seed {seed}: {res.stderr}")
        return None
    for line in res.stdout.splitlines():
        if line.startswith("RESULT_FILE:"):
            fn = line.split(":", 1)[1].strip()
            with open(fn, "r") as f:
                data = json.load(f)
            os.remove(fn)
            return data
    print(f"Could not find RESULT_FILE in stdout: {res.stdout}")
    return None


def run_benchmark():
    print("=" * 80)
    print("PAIRED PHYSICAL CONTROL EXPERIMENT: WITH VS WITHOUT OBSTACLE (model_3.pt)")
    print("=" * 80)

    # First run detailed step-level comparison on Seed 11
    print("\nRunning paired trial for Seed 11...")
    res_with_11 = execute_subproc(seed=11, with_obstacle=True)
    res_without_11 = execute_subproc(seed=11, with_obstacle=False)

    print("\n[1] Seed 11 Detailed Trajectory Comparison:")
    print(f"  Liftoff Step:           With Obstacle = {res_with_11['liftoff_step']}  |  Without Obstacle = {res_without_11['liftoff_step']}")
    print(f"  Liftoff Velocity (vy):  With Obstacle = {res_with_11['liftoff_vw']:.4f} m/s  |  Without Obstacle = {res_without_11['liftoff_vw']:.4f} m/s")
    print(f"  Liftoff Pos Y:          With Obstacle = {res_with_11['liftoff_pos_y']:.4f} m  |  Without Obstacle = {res_without_11['liftoff_pos_y']:.4f} m")
    print(f"  First Obs Contact Step: With Obstacle = {res_with_11['obs_contact_step']} (Peak Force: {res_with_11['obs_contact_force_peak']:.1f} N)  |  Without = None")
    print(f"  Touchdown Step:         With Obstacle = {res_with_11['touchdown_step']}  |  Without Obstacle = {res_without_11['touchdown_step']}")
    print(f"  Touchdown Pos Y:        With Obstacle = {res_with_11['touchdown_pos_y']:.4f} m  |  Without Obstacle = {res_without_11['touchdown_pos_y']:.4f} m")
    print(f"  Flight Distance:        With Obstacle = {res_with_11['flight_dist_m']*100:.2f} cm  |  Without Obstacle = {res_without_11['flight_dist_m']*100:.2f} cm")
    print(f"  Recovery Velocity:      With Obstacle = {res_with_11['mean_recovery_vel']:.4f} m/s  |  Without Obstacle = {res_without_11['mean_recovery_vel']:.4f} m/s")

    print("\n[2] Step-by-Step Trajectory during Jump Phase (Steps 58 to 86):")
    print(f"{'Step':>4} | {'Condition':>10} | {'Pos Y (m)':>10} | {'Pos Z (m)':>10} | {'Vel Y (m/s)':>11} | {'Clearance (cm)':>14} | {'Fn_Obs (N)':>10}")
    print("-" * 80)
    for s in range(58, 87):
        pt_w = res_with_11['step_log'][s]
        pt_wo = res_without_11['step_log'][s]
        print(f"{s:>4} | {'WITH_OBS':>10} | {pt_w['pos_y']:>10.4f} | {pt_w['pos_z']:>10.4f} | {pt_w['vw_y']:>11.4f} | {pt_w['clr_cm']:>14.1f} | {pt_w['fn_obs']:>10.1f}")
        print(f"{s:>4} | {'NO_OBS':>10} | {pt_wo['pos_y']:>10.4f} | {pt_wo['pos_z']:>10.4f} | {pt_wo['vw_y']:>11.4f} | {pt_wo['clr_cm']:>14.1f} | {pt_wo['fn_obs']:>10.1f}")
        print("-" * 80)

    # Now run across 8 seeds
    seeds = [11, 13, 15, 20, 25, 27, 31, 35]
    batch_results = []
    print("\n[3] 8-Seed Summary Table:")
    print(f"{'Seed':>4} | {'Mode':>8} | {'Liftoff Y':>10} | {'Liftoff vy':>10} | {'TD Y':>10} | {'FlightDist':>10} | {'PeakObsN':>10} | {'Recov_vy':>10}")
    print("-" * 85)

    for sd in seeds:
        rw = execute_subproc(sd, with_obstacle=True)
        rwo = execute_subproc(sd, with_obstacle=False)
        batch_results.append({"with": rw, "without": rwo})
        print(f"{sd:>4} | {'WITH':>8} | {rw['liftoff_pos_y']:>10.3f} | {rw['liftoff_vw']:>10.3f} | {rw['touchdown_pos_y']:>10.3f} | {rw['flight_dist_m']*100:>9.1f}cm | {rw['obs_contact_force_peak']:>10.1f} | {rw['mean_recovery_vel']:>10.3f}")
        print(f"{sd:>4} | {'WITHOUT':>8} | {rwo['liftoff_pos_y']:>10.3f} | {rwo['liftoff_vw']:>10.3f} | {rwo['touchdown_pos_y']:>10.3f} | {rwo['flight_dist_m']*100:>9.1f}cm | {rwo['obs_contact_force_peak']:>10.1f} | {rwo['mean_recovery_vel']:>10.3f}")
        print("-" * 85)

    out_file = "evaluation_assets/paired_obstacle_comparison_results.json"
    with open(out_file, "w") as f:
        clean_res = []
        for pair in batch_results:
            clean_res.append({
                "seed": pair["with"]["seed"],
                "with_obstacle": {k: v for k, v in pair["with"].items() if k != "step_log"},
                "without_obstacle": {k: v for k, v in pair["without"].items() if k != "step_log"}
            })
        json.dump(clean_res, f, indent=2)
    print(f"\n[INFO] Paired comparison saved to {out_file}")


if __name__ == "__main__":
    if "--paired_worker" in sys.argv:
        sys.argv.remove("--paired_worker")
        run_single_eval()
    else:
        run_benchmark()
