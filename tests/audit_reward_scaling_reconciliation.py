#!/usr/bin/env python3
# ==============================================================================
# Phase 4.7 Task B: Reward Scaling Reconciliation Audit
# Evaluates every reward term on frozen Champion baseline trajectory (Seed 3001)
# Reconciles:
# 1. Config weight (cfg_scale)
# 2. Effective weight (after _prepare_reward_function multiplies by dt = 0.02)
# 3. Raw function return value (at peak / typical step)
# 4. Step reward_buf increment
# 5. Episode cumulative reward
# 6. Extras per-second conversion (/ 2.8s)
# Explains why touchdown_forward_distance = 0.0275 vs theoretical 2.0 * sigmoid(d) ~ 1.26
# ==============================================================================

import os
import sys
import json
import isaacgym
import torch
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args


def audit_reward_scaling():
    task_name = "myrobot_forward_jump_phase4"
    sys.argv = ["audit_rew", "--task", task_name, "--num_envs", "1", "--headless"]
    args = get_args()
    args.seed = 3001

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = 1
    env_cfg.seed = 3001
    env_cfg.domain_rand.push_robots = False
    env_cfg.noise.add_noise = False

    # Extract raw config scales before make_env modifies them
    cfg_scales = dict(env_cfg.rewards.scales.__dict__)
    # Filter special attributes
    cfg_scales = {k: v for k, v in cfg_scales.items() if not k.startswith("__") and isinstance(v, (int, float)) and v != 0}

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # Load Champion
    champ_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    ac = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(torch.load(champ_path, map_location=env.device)["model_state_dict"])
    ac.eval()

    obs, _ = env.reset()

    reward_names = list(env.reward_names)
    effective_scales = {name: env.reward_scales[name] for name in reward_names}

    raw_fn_vals = {name: [] for name in reward_names}
    step_increments = {name: [] for name in reward_names}

    T = 140
    td_step = -1

    with torch.no_grad():
        for step in range(T):
            act = ac.act_inference(obs)

            # Record raw function returns before step updates physics
            for name in reward_names:
                idx = env.reward_names.index(name)
                fn = env.reward_functions[idx]
                raw_val = fn()[0].item()
                raw_fn_vals[name].append(raw_val)
                step_increments[name].append(raw_val * effective_scales[name])

            next_obs, _, rew, dones, infos = env.step(act)

            if env.has_touched_down[0].item() and td_step < 0:
                td_step = step

            obs = next_obs

    # Compile table
    reconciliation_table = []
    for name in reward_names:
        c_scale = cfg_scales.get(name, 0.0)
        e_scale = effective_scales[name]
        raw_vals = np.array(raw_fn_vals[name])
        step_incs = np.array(step_increments[name])
        ep_sum = float(np.sum(step_incs))
        extras_rate = ep_sum / 2.8

        peak_step = int(np.argmax(np.abs(raw_vals)))
        peak_raw = float(raw_vals[peak_step])
        peak_inc = float(step_incs[peak_step])
        mean_raw = float(np.mean(raw_vals))

        reconciliation_table.append({
            "name": name,
            "cfg_scale": c_scale,
            "effective_scale": e_scale,
            "dt_multiplied": abs(e_scale - c_scale * 0.02) < 1e-6,
            "peak_raw_val": peak_raw,
            "peak_step": peak_step,
            "mean_raw_val": mean_raw,
            "peak_step_increment": peak_inc,
            "episode_cumulative": ep_sum,
            "extras_per_sec": extras_rate,
        })

    # Sort by absolute episode cumulative
    reconciliation_table.sort(key=lambda x: abs(x["episode_cumulative"]), reverse=True)

    print("=" * 115)
    print("PHASE 4.7 TASK B: REWARD SCALING RECONCILIATION AUDIT TABLE (Seed 3001)")
    print("=" * 115)
    print(f"{'Reward Name':<32} | {'Cfg Scale':<10} {'Eff Scale':<10} | {'Peak Raw':<10} {'Peak Inc':<10} | {'Episode Cum':<12} {'Extras (/s)':<12}")
    print("-" * 115)
    for r in reconciliation_table:
        print(f"{r['name']:<32} | {r['cfg_scale']:<10.4f} {r['effective_scale']:<10.4f} | {r['peak_raw_val']:<10.4f} {r['peak_step_increment']:<10.4f} | {r['episode_cumulative']:<+12.4f} {r['extras_per_sec']:<+12.4f}")

    # Detailed focus on touchdown_forward_distance
    td_entry = next(e for e in reconciliation_table if e["name"] == "touchdown_forward_distance")
    print("\n" + "=" * 80)
    print("FOCUS INVESTIGATION: touchdown_forward_distance")
    print("=" * 80)
    print(f"1. Config Scale (rewards.scales.touchdown_forward_distance): {td_entry['cfg_scale']}")
    print(f"2. Effective Scale in env.reward_scales: {td_entry['effective_scale']} (pre-multiplied by dt = 0.02 in _prepare_reward_function)")
    print(f"3. Raw function return on touchdown step ({td_entry['peak_step']}): {td_entry['peak_raw_val']:.4f}")
    print(f"   Theoretical unscaled value: cfg_scale * raw_val = 2.0 * {td_entry['peak_raw_val']:.4f} = {2.0 * td_entry['peak_raw_val']:.4f}")
    print(f"4. Actual single-step reward_buf increment: eff_scale * raw_val = 0.04 * {td_entry['peak_raw_val']:.4f} = {td_entry['peak_step_increment']:.4f}")
    print(f"5. Episode Cumulative Reward (active for exactly 1 step): {td_entry['episode_cumulative']:.4f}")
    print(f"6. TensorBoard Extras per-second rate: {td_entry['extras_per_sec']:.4f}")
    print(f"\nROOT CAUSE EXPLANATION:")
    print("  In LeggedRobot._prepare_reward_function(), ALL non-zero reward scales are multiplied by self.dt (0.02).")
    print("  For rate-based continuous rewards (e.g. tracking_lin_vel = 4.0), multiplying by dt turns the rate into a per-step Riemann sum:")
    print("    sum_t (raw_t * scale * dt) ~ integral (raw * scale) dt = scale * duration * raw_avg.")
    print("    e.g. 4.0 * 2.8s * 1.0 ~ 8.0 ~ 11.2 points across 140 steps.")
    print("  HOWEVER, touchdown_forward_distance is a DISCRETE ONE-SHOT EVENT (fires on exactly 1 transition step).")
    print("  Multiplying its scale 2.0 by dt=0.02 diluted the single-step impulse by 50x (down to 0.04).")
    print(f"  Consequently, the robot receives only 0.04 * {td_entry['peak_raw_val']:.4f} = {td_entry['episode_cumulative']:.4f} points instead of the intuitive 2.0 * {td_entry['peak_raw_val']:.4f} = {2.0 * td_entry['peak_raw_val']:.4f} points!")

    out_file = os.path.join(workspace_root, "evaluation_assets", "phase4_6_critic_calibration", "reward_scaling_reconciliation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "reconciliation_table": reconciliation_table,
            "touchdown_focus": td_entry,
        }, f, indent=2)
    print(f"\nSaved reward reconciliation to: {out_file}")


if __name__ == "__main__":
    audit_reward_scaling()
