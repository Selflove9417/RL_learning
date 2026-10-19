"""
Phase 4.8-Pilot-A: Single-Iteration Actor PPO Parameter Update & Instrumentation
Strict constraints:
1. Exactly 1 PPO iteration (num_learning_iterations=1, init_at_random_ep_len=False).
2. Warm-started from model_critic_calibrated_target_b.pt (Actor = Champion, Critic = Target B).
3. Learning rate = 1.0e-4, 5 epochs x 4 mini-batches = 20 optimizer steps.
4. Independent experiment logging directory, no overwriting of existing baselines.
5. Captures all 20 mini-batch updates: losses, KL, clip fraction, grad norms, parameter shifts.
6. Saves model_pilot_a_iter1.pt and training_telemetry.json to evaluation_assets/phase4_8_pilot_a/.
"""

from isaacgym import gymapi
import os
import sys
import copy
import json
import time
import shutil
import hashlib
import numpy as np
import torch
import torch.nn as nn

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import class_to_dict, get_args, task_registry
from rsl_rl.modules import ActorCritic


def compute_sha256(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    print("=" * 80)
    print("MYROBOT PHASE 4.8-PILOT-A: SINGLE-ITERATION PPO EXPERIMENT")
    print("=" * 80)

    # 1. Pre-training verification & configuration setup
    task_name = "myrobot_forward_jump_phase4"
    num_envs = 64
    sys.argv = ["pilot_a", "--task", task_name, "--num_envs", str(num_envs), "--headless"]
    args = get_args()
    args.seed = 6001

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs
    env_cfg.seed = 6001
    env_cfg.domain_rand.push_robots = False
    env_cfg.noise.add_noise = False

    # Independent experiment configuration
    timestamp = time.strftime("%b%d_%H-%M-%S")
    train_cfg.seed = 6001
    train_cfg.runner.experiment_name = "myrobot_forward_jump_phase4"
    train_cfg.runner.run_name = f"{timestamp}_phase4_8_pilot_a_1iter"
    train_cfg.runner.max_iterations = 1
    train_cfg.runner.save_interval = 1
    train_cfg.runner.resume = False

    # Warm-start from Target B calibrated Critic model
    calib_path = os.path.join(workspace_root, "evaluation_assets/phase4_7_critic_calibration/model_critic_calibrated_target_b.pt")
    champ_path = os.path.join(workspace_root, "evaluation_assets/phase3_1_forward_distance/models/model_champion.pt")
    
    assert os.path.isfile(calib_path), f"Calibrated model not found: {calib_path}"
    assert os.path.isfile(champ_path), f"Champion model not found: {champ_path}"

    train_cfg.runner.warm_start = True
    train_cfg.runner.warm_start_path = calib_path
    train_cfg.runner.warm_start_optimizer = False

    # Algorithm hyperparameters
    train_cfg.algorithm.learning_rate = 1.0e-4
    train_cfg.algorithm.desired_kl = 0.01
    train_cfg.algorithm.enable_early_stopping = True
    train_cfg.algorithm.early_stop_kl = 0.04
    train_cfg.algorithm.max_grad_norm = 1.0
    train_cfg.algorithm.num_learning_epochs = 5
    train_cfg.algorithm.num_mini_batches = 4
    train_cfg.algorithm.clip_param = 0.2
    train_cfg.algorithm.value_loss_coef = 1.0
    train_cfg.algorithm.entropy_coef = 0.005
    train_cfg.algorithm.schedule = 'adaptive'
    train_cfg.algorithm.gamma = 0.99
    train_cfg.algorithm.lam = 0.95

    # Calculate planned updates
    planned_updates = train_cfg.algorithm.num_learning_epochs * train_cfg.algorithm.num_mini_batches
    print(f"\n[Pre-Training Audit]")
    print(f"  Task Name                 : {task_name}")
    print(f"  Seed                      : {train_cfg.seed}")
    print(f"  Target Learning Rate      : {train_cfg.algorithm.learning_rate}")
    print(f"  Learning Epochs           : {train_cfg.algorithm.num_learning_epochs}")
    print(f"  Mini-Batches per Epoch    : {train_cfg.algorithm.num_mini_batches}")
    print(f"  Total Planned Updates     : {planned_updates} calls to optimizer.step()")
    print(f"  Clip Parameter            : {train_cfg.algorithm.clip_param}")
    print(f"  Desired KL (Adaptive)     : {train_cfg.algorithm.desired_kl}")
    print(f"  Early Stop KL Threshold   : {train_cfg.algorithm.early_stop_kl}")
    print(f"  Max Grad Norm (Clipping)  : {train_cfg.algorithm.max_grad_norm}")
    print(f"  Warm Start Path           : {calib_path}")
    print(f"  Initial Model SHA256      : {compute_sha256(calib_path)}")

    # 2. Build Environment and OnPolicyRunner
    env, env_cfg = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=task_name, args=args, train_cfg=train_cfg)

    # 3. Model parameter fidelity verification
    ac_champ = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_champ.load_state_dict(torch.load(champ_path, map_location=env.device)["model_state_dict"])
    ac_champ.eval()

    ac_calib = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_calib.load_state_dict(torch.load(calib_path, map_location=env.device)["model_state_dict"])
    ac_calib.eval()

    ac_runner = runner.alg.actor_critic

    with torch.no_grad():
        actor_runner_p = torch.cat([p.flatten() for p in ac_runner.actor.parameters()])
        actor_champ_p = torch.cat([p.flatten() for p in ac_champ.actor.parameters()])
        actor_diff = (actor_runner_p - actor_champ_p).abs().max().item()
        std_diff = (ac_runner.std - ac_champ.std).abs().max().item()

        critic_runner_p = torch.cat([p.flatten() for p in ac_runner.critic.parameters()])
        critic_calib_p = torch.cat([p.flatten() for p in ac_calib.critic.parameters()])
        critic_diff = (critic_runner_p - critic_calib_p).abs().max().item()

    print(f"\n[Model Initialization Invariance Check]")
    print(f"  Runner Actor diff vs Champion : {actor_diff:.2e} (Expected: 0.00e+00)")
    print(f"  Runner Std diff vs Champion   : {std_diff:.2e} (Expected: 0.00e+00)")
    print(f"  Runner Critic diff vs Target B: {critic_diff:.2e} (Expected: 0.00e+00)")
    assert actor_diff == 0.0, "Runner Actor does not match Champion baseline!"
    assert std_diff == 0.0, "Runner action std does not match Champion baseline!"
    assert critic_diff == 0.0, "Runner Critic does not match Target B calibrated model!"

    # Snapshot initial parameters for exact post-training delta measurement
    with torch.no_grad():
        init_actor_params = torch.cat([p.detach().flatten() for p in ac_runner.actor.parameters()]).clone()
        init_critic_params = torch.cat([p.detach().flatten() for p in ac_runner.critic.parameters()]).clone()
        init_std = ac_runner.std.detach().clone()

    # 4. Instrument PPO optimizer step to record every single update
    step_telemetry = []
    orig_optimizer_step = runner.alg.optimizer.step

    def instrumented_optimizer_step(*step_args, **step_kwargs):
        with torch.no_grad():
            cur_lr = runner.alg.optimizer.param_groups[0]['lr']
            actor_params = [p for p in ac_runner.actor.parameters() if p.grad is not None]
            critic_params = [p for p in ac_runner.critic.parameters() if p.grad is not None]
            actor_gn = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in actor_params]), 2).item() if actor_params else 0.0
            critic_gn = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in critic_params]), 2).item() if critic_params else 0.0
            cur_std = ac_runner.std.detach().clone()
            
            rec = {
                "step_index": len(step_telemetry) + 1,
                "learning_rate": cur_lr,
                "actor_grad_norm_unclipped": actor_gn,
                "critic_grad_norm_unclipped": critic_gn,
                "action_std_mean": cur_std.mean().item(),
                "action_std_min": cur_std.min().item(),
                "action_std_max": cur_std.max().item(),
            }
            step_telemetry.append(rec)
        return orig_optimizer_step(*step_args, **step_kwargs)

    runner.alg.optimizer.step = instrumented_optimizer_step

    # 5. Execute exactly 1 Iteration of PPO Training
    print(f"\n[Executing Phase 4.8 Pilot A PPO Iteration 1]...")
    start_time = time.time()
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
    elapsed_time = time.time() - start_time
    print(f"Iteration 1 completed in {elapsed_time:.2f}s!")

    # 6. Post-training parameter shift analysis
    with torch.no_grad():
        final_actor_params = torch.cat([p.detach().flatten() for p in ac_runner.actor.parameters()]).clone()
        final_critic_params = torch.cat([p.detach().flatten() for p in ac_runner.critic.parameters()]).clone()
        final_std = ac_runner.std.detach().clone()

        actor_delta_l2 = torch.norm(final_actor_params - init_actor_params, 2).item()
        actor_rel_delta = actor_delta_l2 / (torch.norm(init_actor_params, 2).item() + 1e-8)

        critic_delta_l2 = torch.norm(final_critic_params - init_critic_params, 2).item()
        critic_rel_delta = critic_delta_l2 / (torch.norm(init_critic_params, 2).item() + 1e-8)

        std_delta_l2 = torch.norm(final_std - init_std, 2).item()

    print("\n--- Parameter Update Magnitude ---")
    print(f"  Actor Parameter L2 Delta    : {actor_delta_l2:.6f} (Relative: {actor_rel_delta * 100:.4f}%)")
    print(f"  Critic Parameter L2 Delta   : {critic_delta_l2:.6f} (Relative: {critic_rel_delta * 100:.4f}%)")
    print(f"  Action Std L2 Delta         : {std_delta_l2:.6f}")
    print(f"  Final Action Std Mean       : {final_std.mean().item():.6f}")

    # Check for NaN / Inf
    assert not torch.isnan(final_actor_params).any(), "NaN detected in Actor parameters!"
    assert not torch.isinf(final_actor_params).any(), "Inf detected in Actor parameters!"
    assert not torch.isnan(final_critic_params).any(), "NaN detected in Critic parameters!"
    assert not torch.isinf(final_critic_params).any(), "Inf detected in Critic parameters!"

    # 7. Extract full PPO diagnostics
    diag = runner.alg.last_update_diagnostics
    print("\n--- PPO Aggregated Diagnostics ---")
    print(f"  Updates Executed            : {diag['updates_executed']} / {diag['updates_planned']}")
    print(f"  Early Stopping Triggered    : {bool(diag['early_stop_triggered'])}")
    if diag['early_stop_triggered']:
        print(f"  Early Stopping Reason       : {diag['early_stop_reason']}")
    print(f"  KL Divergence (Mean)        : {diag['kl_mean']:.6e}")
    print(f"  KL Divergence (P95)         : {diag['kl_p95']:.6e}")
    print(f"  KL Divergence (Max)         : {diag['kl_max']:.6e}")
    print(f"  Surrogate Loss (Mean)       : {diag['surrogate_loss']:.6f}")
    print(f"  Value Loss (Mean)           : {diag['value_loss']:.6f}")
    print(f"  Clip Fraction (Mean)        : {diag['clip_fraction'] * 100:.2f}%")
    print(f"  Actor Grad Norm (Mean)      : {diag['actor_grad_norm_mean']:.4f}")
    print(f"  Critic Grad Norm (Mean)     : {diag['critic_grad_norm_mean']:.4f}")
    print(f"  Total Grad Norm (Mean)      : {diag['grad_norm_mean']:.4f}")
    print(f"  Entropy (Mean)              : {diag['entropy']:.4f}")

    # Combine step telemetry with algorithm outputs
    for i in range(len(step_telemetry)):
        if "actual_lr_step_list" in diag and i < len(diag["actual_lr_step_list"]):
            step_telemetry[i]["lr_verified"] = diag["actual_lr_step_list"][i]

    # 8. Archive trained model and telemetry
    out_dir = os.path.join(workspace_root, "evaluation_assets/phase4_8_pilot_a")
    os.makedirs(out_dir, exist_ok=True)

    saved_model_src = os.path.join(runner.log_dir, "model_1.pt")
    assert os.path.isfile(saved_model_src), f"Expected saved checkpoint not found at: {saved_model_src}"

    saved_model_dst = os.path.join(out_dir, "model_pilot_a_iter1.pt")
    shutil.copy2(saved_model_src, saved_model_dst)
    pilot_sha256 = compute_sha256(saved_model_dst)

    print(f"\n[Model Archiving]")
    print(f"  Saved Model Path            : {saved_model_dst}")
    print(f"  Saved Model SHA256          : {pilot_sha256}")

    telemetry_payload = {
        "metadata": {
            "experiment": "Phase 4.8-Pilot-A",
            "task_name": task_name,
            "seed": train_cfg.seed,
            "num_envs": num_envs,
            "timestamp": timestamp,
            "log_dir": runner.log_dir,
            "calib_init_path": calib_path,
            "calib_init_sha256": compute_sha256(calib_path),
            "pilot_model_path": saved_model_dst,
            "pilot_model_sha256": pilot_sha256,
            "elapsed_seconds": elapsed_time,
        },
        "hyperparameters": {
            "learning_rate": train_cfg.algorithm.learning_rate,
            "num_learning_epochs": train_cfg.algorithm.num_learning_epochs,
            "num_mini_batches": train_cfg.algorithm.num_mini_batches,
            "planned_updates": planned_updates,
            "clip_param": train_cfg.algorithm.clip_param,
            "desired_kl": train_cfg.algorithm.desired_kl,
            "early_stop_kl": train_cfg.algorithm.early_stop_kl,
            "max_grad_norm": train_cfg.algorithm.max_grad_norm,
            "gamma": train_cfg.algorithm.gamma,
            "lam": train_cfg.algorithm.lam,
        },
        "parameter_shifts": {
            "actor_delta_l2": actor_delta_l2,
            "actor_rel_delta": actor_rel_delta,
            "critic_delta_l2": critic_delta_l2,
            "critic_rel_delta": critic_rel_delta,
            "std_delta_l2": std_delta_l2,
            "initial_std_mean": init_std.mean().item(),
            "final_std_mean": final_std.mean().item(),
        },
        "aggregated_diagnostics": diag,
        "step_by_step_updates": step_telemetry,
    }

    telemetry_file = os.path.join(out_dir, "training_telemetry.json")
    with open(telemetry_file, "w") as f:
        json.dump(telemetry_payload, f, indent=2, default=str)
    print(f"  Training Telemetry Saved To : {telemetry_file}")

    print("\n" + "=" * 80)
    print("PHASE 4.8-PILOT-A PPO TRAINING SUCCESSFUL")
    print("=" * 80)


if __name__ == "__main__":
    main()
