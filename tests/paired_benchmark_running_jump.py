# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Paired Physical Benchmark: Frozen Stage 8 vs Running Jump Iter 1
# Evaluated on Unparticipated Random Seeds (Seeds 11 ~ 20)

import os
import sys
import math
import json
import subprocess

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))


def run_single_eval():
    import isaacgym
    from isaacgym import gymtorch
    import torch
    import numpy as np
    from legged_gym.envs import MyRobotRunningJump
    from legged_gym.utils import task_registry, get_args
    from rsl_rl.modules import ActorCritic

    args = get_args()
    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_running_jump")
    env_cfg.env.num_envs = 1
    env_cfg.seed = args.seed

    env, _ = task_registry.make_env(name="myrobot_running_jump", args=args, env_cfg=env_cfg)
    env.reset()

    # Load evaluated model into evaluation policy
    model_path = getattr(args, 'model_path', 'evaluation_assets/frozen_stage8/model_40.pt')
    ckpt = torch.load(model_path, map_location=env.device)
    eval_policy = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    eval_policy.load_state_dict(ckpt['model_state_dict'])
    eval_policy.eval()

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
        if step_i == 39:
            pre_jump_vy = env.base_lin_vel[0, 1].item()

        with torch.no_grad():
            actions = eval_policy.act_inference(obs)

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

        if took_off and liftoff_step == -1:
            liftoff_step = step_i
            liftoff_vy = vy
            liftoff_pos_y = pos_y

        if touched_down and touchdown_step == -1:
            touchdown_step = step_i
            touchdown_vy = vy
            touchdown_pos_y = pos_y

        if step_i >= 120 and timeout_step == -1:
            resumed_vy = vy

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

    if touchdown_step != -1 and liftoff_step != -1:
        flight_disp = touchdown_pos_y - liftoff_pos_y
    else:
        flight_disp = 0.0

    result = {
        "seed": args.seed,
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


def run_worker(model_path, seed):
    cmd = [
        sys.executable,
        "-u",
        __file__,
        "--model_path", model_path,
        "--seed", str(seed)
    ]
    sub_env = os.environ.copy()
    sub_env["RUN_WORKER"] = "1"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=sub_env)
    stdout, stderr = proc.communicate()

    if "===RESULT_START===" in stdout:
        block = stdout.split("===RESULT_START===")[1].split("===RESULT_END===")[0].strip()
        try:
            return json.loads(block)
        except Exception as e:
            return {"seed": seed, "fell": True, "pre_jump_vy": 0.0, "liftoff_vy": 0.0, "touchdown_vy": 0.0, "resumed_vy": 0.0, "flight_disp": 0.0, "max_clearance": 0.0, "fall_reason": f"JSON parse error: {e}"}
    else:
        return {"seed": seed, "fell": True, "pre_jump_vy": 0.0, "liftoff_vy": 0.0, "touchdown_vy": 0.0, "resumed_vy": 0.0, "flight_disp": 0.0, "max_clearance": 0.0, "fall_reason": f"Execution failed: {stderr[-200:] if stderr else 'Unknown'}"}


def main():
    if os.environ.get("RUN_WORKER") == "1":
        run_single_eval()
        return

    import numpy as np

    stage8_model = "evaluation_assets/frozen_stage8/model_40.pt"
    iter1_model = "legged_gym/logs/myrobot_running_jump/Oct19_20-39-35_running_jump_020ms_iter1/model_1.pt"
    seeds = list(range(11, 21)) # 10 unparticipated random seeds: 11, 12, ..., 20

    print("=" * 115)
    print("  PAIRED BENCHMARK: FROZEN STAGE 8 VS RUNNING JUMP ITER 1")
    print(f"  * Unparticipated Evaluation Seeds : {seeds}")
    print(f"  * Baseline Model (Stage 8)       : {stage8_model}")
    print(f"  * Evaluated Model (Iter 1)       : {iter1_model}")
    print("=" * 115)

    results_stage8 = []
    results_iter1 = []

    for seed in seeds:
        print(f"Evaluating Seed {seed:2d}...", flush=True)
        res_s8 = run_worker(stage8_model, seed)
        res_it1 = run_worker(iter1_model, seed)
        results_stage8.append(res_s8)
        results_iter1.append(res_it1)

        s8_status = "FELL" if res_s8["fell"] else "PASS"
        it1_status = "FELL" if res_it1["fell"] else "PASS"
        print(f"  Seed {seed:2d} | Stage 8: {s8_status} (v_lo={res_s8['liftoff_vy']:.3f}, clr={res_s8['max_clearance']*100:.1f}cm) | Iter 1: {it1_status} (v_lo={res_it1['liftoff_vy']:.3f}, clr={res_it1['max_clearance']*100:.1f}cm)")

    print("\n" + "=" * 115)
    print(" PAIRED COMPARISON TABLE ACROSS 10 UNPARTICIPATED SEEDS (SEEDS 11 ~ 20)")
    print("=" * 115)
    print(f"| Seed | Model        | 起跳前速度  | 实际离地速度 | 触地前向速度 | 恢复后巡航速度 | 腾空位移  | 最大净空  | 存活状态 |")
    print(f"|------|--------------|------------|--------------|--------------|----------------|-----------|-----------|----------|")

    for s8, it1 in zip(results_stage8, results_iter1):
        sd = s8["seed"]
        s8_st = "PASS" if not s8["fell"] else f"FAIL ({s8.get('fall_reason', '')[:10]})"
        it1_st = "PASS" if not it1["fell"] else f"FAIL ({it1.get('fall_reason', '')[:10]})"
        print(f"| {sd:4d} | Stage 8 (BL) | {s8['pre_jump_vy']:8.3f}m/s | {s8['liftoff_vy']:10.3f}m/s | {s8['touchdown_vy']:10.3f}m/s | {s8['resumed_vy']:12.3f}m/s | {s8['flight_disp']*100:7.1f}cm | {s8['max_clearance']*100:7.1f}cm | {s8_st:8s} |")
        print(f"|      | Iter 1 (New) | {it1['pre_jump_vy']:8.3f}m/s | {it1['liftoff_vy']:10.3f}m/s | {it1['touchdown_vy']:10.3f}m/s | {it1['resumed_vy']:12.3f}m/s | {it1['flight_disp']*100:7.1f}cm | {it1['max_clearance']*100:7.1f}cm | {it1_st:8s} |")
        print(f"|------|--------------|------------|--------------|--------------|----------------|-----------|-----------|----------|")

    # Aggregate metrics
    def stats(res_list):
        survived = [r for r in res_list if not r["fell"]]
        fall_rate = (len(res_list) - len(survived)) / len(res_list) * 100.0
        if not survived:
            return {"fall_rate": fall_rate, "pre_vy": 0, "lo_vy": 0, "td_vy": 0, "res_vy": 0, "disp": 0, "clr": 0}
        return {
            "fall_rate": fall_rate,
            "pre_vy": float(np.mean([r["pre_jump_vy"] for r in survived])),
            "lo_vy": float(np.mean([r["liftoff_vy"] for r in survived])),
            "td_vy": float(np.mean([r["touchdown_vy"] for r in survived])),
            "res_vy": float(np.mean([r["resumed_vy"] for r in survived])),
            "disp": float(np.mean([r["flight_disp"] * 100.0 for r in survived])),
            "clr": float(np.mean([r["max_clearance"] * 100.0 for r in survived])),
        }

    s8_stat = stats(results_stage8)
    it1_stat = stats(results_iter1)

    print("\n" + "=" * 80)
    print(" AGGREGATE SUMMARY COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Stage 8 Baseline':<18} | {'Running Jump Iter 1':<18}")
    print("-" * 80)
    print(f"{'Fall Rate (%)':<35} | {s8_stat['fall_rate']:<18.1f} | {it1_stat['fall_rate']:<18.1f}")
    print(f"{'Pre-jump Velocity vy (m/s)':<35} | {s8_stat['pre_vy']:<18.3f} | {it1_stat['pre_vy']:<18.3f}")
    print(f"{'Liftoff Velocity vy (m/s)':<35} | {s8_stat['lo_vy']:<18.3f} | {it1_stat['lo_vy']:<18.3f}")
    print(f"{'Touchdown Velocity vy (m/s)':<35} | {s8_stat['td_vy']:<18.3f} | {it1_stat['td_vy']:<18.3f}")
    print(f"{'Resumed Velocity vy (m/s)':<35} | {s8_stat['res_vy']:<18.3f} | {it1_stat['res_vy']:<18.3f}")
    print(f"{'Flight Displacement (cm)':<35} | {s8_stat['disp']:<18.1f} | {it1_stat['disp']:<18.1f}")
    print(f"{'Max Ground Clearance (cm)':<35} | {s8_stat['clr']:<18.1f} | {it1_stat['clr']:<18.1f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
