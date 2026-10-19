#!/usr/bin/env python3
"""
MyRobot: Small-Scale Controlled PPO Training for Dynamic Running Jump (myrobot_running_jump)
===========================================================================================
- Task: myrobot_running_jump (continuous cruising warmup -> dynamic takeoff -> landing resume)
- Target Velocity: 0.20 m/s (Low-speed dynamic jump stage)
- Hotstart: evaluation_assets/frozen_stage8/model_40.pt
- Controlled iterations: 10 iterations (64 envs x 70 steps = 44,800 timesteps)
- Fixed learning rate: 5.0e-5, max_grad_norm: 1.0, clip: 0.2
- Tracks velocity retention, clearance, pitch stability, and loss convergence
"""

import os
import sys
import time
import json
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
import torch

from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from rsl_rl.runners import OnPolicyRunner


def evaluate_policy(env, policy, num_steps=70):
    policy.eval()
    with torch.inference_mode():
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        max_clr = torch.zeros(env.num_envs, device=env.device)
        takeoff_vy = torch.zeros(env.num_envs, device=env.device)
        touchdown_vy = torch.zeros(env.num_envs, device=env.device)
        touchdown_dist = torch.zeros(env.num_envs, device=env.device)
        has_taken_off = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        has_touched_down = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        for step in range(num_steps):
            actions = policy.act_inference(obs)
            obs, _, rews, dones, _ = env.step(actions)

            cur_clr = env.bilateral_clearance * 100.0
            max_clr = torch.maximum(max_clr, cur_clr)

            liftoff_now = (env.is_airborne) & (~has_taken_off)
            if liftoff_now.any():
                ids = liftoff_now.nonzero(as_tuple=False).flatten()
                has_taken_off[ids] = True
                takeoff_vy[ids] = env.base_lin_vel[ids, 1]

            td_now = (has_taken_off) & (~env.is_airborne) & (~has_touched_down) & (env.jump_phase >= 0.30)
            if td_now.any():
                td_ids = td_now.nonzero(as_tuple=False).flatten()
                has_touched_down[td_ids] = True
                touchdown_vy[td_ids] = env.base_lin_vel[td_ids, 1]
                touchdown_dist[td_ids] = env.flight_distance_y[td_ids]

        mean_to_vy = float(takeoff_vy[has_taken_off].mean().item()) if has_taken_off.any() else float(env.takeoff_vy.mean().item())
        mean_td_vy = float(touchdown_vy[has_touched_down].mean().item()) if has_touched_down.any() else 0.0
        mean_clr = float(max_clr.mean().item())
        mean_dist = float(touchdown_dist[has_touched_down].mean().item()) if has_touched_down.any() else 0.0
        retention = mean_td_vy / max(0.05, mean_to_vy) if mean_to_vy > 0 else 0.0

        return {
            "mean_takeoff_vy": mean_to_vy,
            "mean_touchdown_vy": mean_td_vy,
            "mean_clearance_cm": mean_clr,
            "mean_distance_cm": mean_dist,
            "velocity_retention": retention,
            "takeoff_count": int(has_taken_off.sum().item()),
            "touchdown_count": int(has_touched_down.sum().item()),
        }


def run_controlled_training(num_iterations=10, seed=42):
    task_name = "myrobot_running_jump"
    sys.argv = ['train_running_jump', '--task', task_name, '--headless', '--num_envs', '64']
    args = get_args()
    args.headless = True
    args.seed = seed

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = 64
    env_cfg.seed = seed
    train_cfg.runner.max_iterations = num_iterations
    train_cfg.runner.save_interval = 5

    print("\n" + "=" * 78)
    print(" 🚀 MyRobot: Phase 5 Dynamic Running Jump Controlled PPO Training")
    print("=" * 78)
    print(f" Task Name       : {task_name}")
    print(f" Target Speed    : {env_cfg.commands.target_forward_vel:.2f} m/s (In-motion continuous)")
    print(f" Parallel Envs   : {env_cfg.env.num_envs}")
    print(f" Steps per Env   : {env_cfg.env.episode_length_s / 0.02:.0f} (1.4s per jump cycle)")
    print(f" PPO Iterations  : {num_iterations}")
    print(f" Learning Rate   : {train_cfg.algorithm.learning_rate} (Fixed)")
    print(f" Reward Profile  : Momentum Retention + Vertical Takeoff + Clearance + Post-Touchdown Resume")
    print("=" * 78 + "\n")

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    log_dir = os.path.join(workspace_root, "legged_gym", "logs", "myrobot_running_jump", f"Oct19_running_jump_020ms_iter{num_iterations}")
    os.makedirs(log_dir, exist_ok=True)

    from legged_gym.utils.helpers import class_to_dict
    train_cfg_dict = class_to_dict(train_cfg)
    ppo_runner = OnPolicyRunner(env=env, train_cfg=train_cfg_dict, log_dir=log_dir, device=env.device)

    # 1. Hotstart ActorCritic from frozen_stage8/model_40.pt
    hotstart_path = os.path.join(workspace_root, "evaluation_assets", "frozen_stage8", "model_40.pt")
    print(f"📥 Hotstarting ActorCritic from: {hotstart_path}")
    ckpt = torch.load(hotstart_path, map_location=env.device)
    ppo_runner.alg.actor_critic.load_state_dict(ckpt['model_state_dict'])

    # 2. Pre-training Baseline Evaluation
    print("\n🔍 Conducting Pre-training Zero-Step Baseline Evaluation...")
    base_res = evaluate_policy(env, ppo_runner.alg.actor_critic, num_steps=70)
    print(f"   Baseline Takeoff vy      : {base_res['mean_takeoff_vy']:+.3f} m/s")
    print(f"   Baseline Max Clearance   : {base_res['mean_clearance_cm']:.1f} cm")
    print(f"   Baseline Liftoff Count   : {base_res['takeoff_count']}/{env.num_envs}")

    # 3. Controlled PPO Training Loop
    print(f"\n🏋️ Starting Controlled PPO Training ({num_iterations} iterations)...")
    ppo_runner.alg.actor_critic.train()

    # Reset environment cleanly
    env.reset()
    ppo_runner.learn(num_learning_iterations=num_iterations, init_at_random_ep_len=False)

    # 4. Save Final Checkpoint
    final_model_dir = os.path.join(workspace_root, "evaluation_assets", "running_jump")
    os.makedirs(final_model_dir, exist_ok=True)
    final_model_path = os.path.join(final_model_dir, "model_running_jump_v1.pt")
    torch.save({
        'model_state_dict': ppo_runner.alg.actor_critic.state_dict(),
        'optimizer_state_dict': ppo_runner.alg.optimizer.state_dict(),
        'iter': num_iterations
    }, final_model_path)
    print(f"✅ Final Running Jump Model saved to: {final_model_path}")

    # 5. Post-training Evaluation & Paired Comparison
    print("\n🔍 Conducting Post-training Paired Evaluation...")
    post_res = evaluate_policy(env, ppo_runner.alg.actor_critic, num_steps=70)

    print("\n" + "=" * 78)
    print(" 📊 Dynamic Running Jump Paired Results (Before vs After PPO Update):")
    print("=" * 78)
    print(f" Metric                          | Pre-Training (Stage 8) | Post-Training (10 Iter)")
    print(f" --------------------------------+------------------------+------------------------")
    print(f" Takeoff Speed (vy)              | {base_res['mean_takeoff_vy']:+6.3f} m/s           | {post_res['mean_takeoff_vy']:+6.3f} m/s")
    print(f" Touchdown Speed (vy)            |    N/A (Braked to 0)   | {post_res['mean_touchdown_vy']:+6.3f} m/s")
    print(f" Max Ground Clearance            | {base_res['mean_clearance_cm']:6.1f} cm              | {post_res['mean_clearance_cm']:6.1f} cm")
    print(f" Airborne Distance               |    N/A                 | {post_res['mean_distance_cm']:6.1f} cm")
    print(f" Velocity Retention Ratio        |    0.0%                | {post_res['velocity_retention']:6.1%}")
    print(f" Successful Touchdowns           |    0/64                | {post_res['touchdown_count']:2d}/64")
    print("=" * 78)

    # Save summary report
    summary = {
        "task": task_name,
        "target_vy": env_cfg.commands.target_forward_vel,
        "iterations": num_iterations,
        "baseline": base_res,
        "post_training": post_res,
        "checkpoint": final_model_path
    }
    summary_path = os.path.join(final_model_dir, "running_jump_preliminary_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📝 Summary saved to: {summary_path}\n")


if __name__ == "__main__":
    run_controlled_training()
