#!/usr/bin/env python3
# ==============================================================================
# Phase 4.1 Training Entry Point: Single-Variable Forward Jump Incremental PPO
# Synchronous episodic rollouts, warm-started from Champion, Phase 15 settling
# ==============================================================================

import os
import sys
import json
import shutil
import subprocess
from datetime import datetime

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import class_to_dict, get_args, task_registry
import torch


def save_run_snapshot(log_dir, env_cfg, train_cfg, args, ppo_runner=None):
    if log_dir is None:
        return
    os.makedirs(log_dir, exist_ok=True)
    snapshot = {
        "created_at": datetime.now().isoformat(),
        "phase": "Phase 4.1 Incremental Curriculum",
        "env_cfg": class_to_dict(env_cfg),
        "train_cfg": class_to_dict(train_cfg),
        "arguments": vars(args),
    }
    with open(os.path.join(log_dir, "training_snapshot.json"), "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, ensure_ascii=False, default=str)

    source_dir = os.path.join(log_dir, "source_snapshot")
    os.makedirs(source_dir, exist_ok=True)
    for rel_p in [
        "legged_gym/envs/myrobot/myrobot_forward_jump_phase4_env.py",
        "legged_gym/envs/myrobot/myrobot_forward_jump_phase4_config.py",
        "scripts/train_phase4.py",
    ]:
        full_p = os.path.join(workspace_root, rel_p)
        if os.path.isfile(full_p):
            shutil.copy2(full_p, os.path.join(source_dir, os.path.basename(rel_p)))


def main():
    task_name = "myrobot_forward_jump_phase4"
    if "--task" not in sys.argv:
        sys.argv.extend(["--task", task_name])
    args = get_args()

    print(f"==================================================")
    print(f" [Phase 4.1] Starting PPO Incremental Training")
    print(f" Task: {task_name}")
    print(f" Device: {args.rl_device}")
    print(f"==================================================")

    env, env_cfg = task_registry.make_env(name=task_name, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=task_name, args=args)

    save_run_snapshot(ppo_runner.log_dir, env_cfg, train_cfg, args, ppo_runner=ppo_runner)

    # CRITICAL: init_at_random_ep_len=False for strict synchronous episodic jumping!
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)

    print(f"\nTraining completed! Artifacts saved to: {ppo_runner.log_dir}")


if __name__ == "__main__":
    main()
