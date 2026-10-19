#!/usr/bin/env python3
"""
Phase 4.9-Pilot-C: Fixed Low-Learning-Rate Single-Update Controlled PPO Experiment
Strict constraints:
1. Re-initialize from Phase 4.8 Pilot B checkpoint (Actor & Critic both from model_pilot_b_step1.pt).
2. Freshly initialized Adam optimizer (DO NOT inherit history optimizer state).
3. Fixed low learning rate: lr = 5.0e-5, schedule = 'fixed' (Adaptive schedule DISABLED).
4. EXACTLY ONE optimizer.step() executed (strictly 1 mini-batch update of batch size 2240).
5. Early stop KL threshold: 0.020 (conservative safety threshold).
6. Save model_pilot_c_step1.pt and telemetry to evaluation_assets/phase4_9_pilot_c/.
7. Detailed telemetry: per-dimension KL (hips, knees, wheels), Clip fraction, parameter shifts, grad norms.
"""

import os
import sys
import copy
import json
import time
import shutil
import hashlib
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
import torch
import torch.nn as nn

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import class_to_dict, get_args, task_registry
from rsl_rl.modules import ActorCritic


def compute_sha256(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    print("=" * 80)
    print("MYROBOT PHASE 4.9-PILOT-C: CONTROLLED SINGLE PPO UPDATE EXPERIMENT")
    print("=" * 80)

    # 1. Environment & configuration setup
    task_name = "myrobot_forward_jump_phase4"
    num_envs = 64
    seed = 6001
    sys.argv = ["pilot_c", "--task", task_name, "--num_envs", str(num_envs), "--headless"]
    args = get_args()
    args.seed = seed

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.domain_rand.push_robots = False
    env_cfg.noise.add_noise = False
    assert getattr(env_cfg.env, "zero_cmd_post_touchdown", False) is True, "zero_cmd_post_touchdown must be True!"

    # Independent experiment configuration
    timestamp = time.strftime("%b%d_%H-%M-%S")
    train_cfg.seed = seed
    train_cfg.runner.experiment_name = "myrobot_forward_jump_phase4"
    train_cfg.runner.run_name = f"{timestamp}_phase4_9_pilot_c_1step_fixed_lr"
    train_cfg.runner.max_iterations = 1
    train_cfg.runner.save_interval = 1
    train_cfg.runner.resume = False

    # Checkpoint paths
    pilot_b_path = os.path.join(workspace_root, "evaluation_assets/phase4_8_pilot_b/model_pilot_b_step1.pt")
    assert os.path.isfile(pilot_b_path), f"Pilot B model not found: {pilot_b_path}"
    pilot_b_sha256 = compute_sha256(pilot_b_path)
    expected_pilot_b_sha = "64aedcfd0156a5ed512c4c8c1c97f06caefae8c7e662322a143dc3ec9decc29b"

    print(f"\n[Asset Identity Pre-Check]")
    print(f"  Pilot B Path    : {pilot_b_path}")
    print(f"  Pilot B SHA256  : {pilot_b_sha256}")
    assert pilot_b_sha256 == expected_pilot_b_sha, "Pilot B SHA256 mismatch!"

    # Warm-start configuration: load weights from Pilot B, fresh Adam optimizer
    train_cfg.runner.warm_start = True
    train_cfg.runner.warm_start_path = pilot_b_path
    train_cfg.runner.warm_start_optimizer = False  # DO NOT inherit history optimizer state!

    # Strict Fixed Low-LR Hyperparameters
    target_lr = 5.0e-5
    train_cfg.algorithm.learning_rate = target_lr
    train_cfg.algorithm.schedule = 'fixed'          # DISABLED adaptive schedule
    train_cfg.algorithm.desired_kl = None           # No adaptive scaling
    train_cfg.algorithm.enable_early_stopping = True
    train_cfg.algorithm.early_stop_kl = 0.020       # Conservative safety check threshold
    train_cfg.algorithm.max_grad_norm = 1.0
    train_cfg.algorithm.num_learning_epochs = 5
    train_cfg.algorithm.num_mini_batches = 4        # Batch size = 64 * 140 // 4 = 2240
    train_cfg.algorithm.clip_param = 0.2
    train_cfg.algorithm.value_loss_coef = 1.0
    train_cfg.algorithm.entropy_coef = 0.005
    train_cfg.algorithm.gamma = 0.99
    train_cfg.algorithm.lam = 0.95

    print(f"\n[Pilot C Hyperparameters]")
    print(f"  Target Learning Rate      : {train_cfg.algorithm.learning_rate} (Strictly Fixed)")
    print(f"  Schedule                  : {train_cfg.algorithm.schedule} (Adaptive Disabled)")
    print(f"  Clip Parameter            : {train_cfg.algorithm.clip_param}")
    print(f"  Early Stop KL Threshold   : {train_cfg.algorithm.early_stop_kl}")
    print(f"  Max Grad Norm             : {train_cfg.algorithm.max_grad_norm}")
    print(f"  Num Mini-Batches          : {train_cfg.algorithm.num_mini_batches} (Batch Size = {64*140 // 4} = 2240)")

    # 2. Build Environment and OnPolicyRunner
    env, env_cfg = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=task_name, args=args, train_cfg=train_cfg)

    ac_runner = runner.alg.actor_critic

    # Verify model initialization invariance vs Pilot B
    ac_pilot_b = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_pilot_b.load_state_dict(torch.load(pilot_b_path, map_location=env.device)["model_state_dict"])
    ac_pilot_b.eval()

    with torch.no_grad():
        actor_runner_p = torch.cat([p.flatten() for p in ac_runner.actor.parameters()])
        actor_pilot_b_p = torch.cat([p.flatten() for p in ac_pilot_b.actor.parameters()])
        actor_diff = (actor_runner_p - actor_pilot_b_p).abs().max().item()
        std_diff = (ac_runner.std - ac_pilot_b.std).abs().max().item()

        critic_runner_p = torch.cat([p.flatten() for p in ac_runner.critic.parameters()])
        critic_pilot_b_p = torch.cat([p.flatten() for p in ac_pilot_b.critic.parameters()])
        critic_diff = (critic_runner_p - critic_pilot_b_p).abs().max().item()

    print(f"\n[Model Initialization Invariance Check]")
    print(f"  Runner Actor diff vs Pilot B  : {actor_diff:.2e} (Expected: 0.00e+00)")
    print(f"  Runner Std diff vs Pilot B    : {std_diff:.2e} (Expected: 0.00e+00)")
    print(f"  Runner Critic diff vs Pilot B : {critic_diff:.2e} (Expected: 0.00e+00)")
    assert actor_diff == 0.0, "Runner Actor does not match Pilot B baseline!"
    assert std_diff == 0.0, "Runner action std does not match Pilot B baseline!"
    assert critic_diff == 0.0, "Runner Critic does not match Pilot B baseline!"

    # Snapshot initial parameters for exact delta analysis
    with torch.no_grad():
        init_actor_params = torch.cat([p.detach().flatten() for p in ac_runner.actor.parameters()]).clone()
        init_critic_params = torch.cat([p.detach().flatten() for p in ac_runner.critic.parameters()]).clone()
        init_std = ac_runner.std.detach().clone()

    # 3. Instrument PPO to enforce EXACTLY ONE optimizer.step()
    orig_generator_func = runner.alg.storage.mini_batch_generator

    def single_step_mini_batch_generator(num_mini_batches, num_epochs=8):
        gen = orig_generator_func(num_mini_batches, num_epochs)
        batch = next(gen)
        yield batch
        return

    runner.alg.storage.mini_batch_generator = single_step_mini_batch_generator

    # Instrument optimizer.step() to record telemetry and ensure step count == 1
    optimizer_step_records = []
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
                "step_index": len(optimizer_step_records) + 1,
                "learning_rate": cur_lr,
                "actor_grad_norm_unclipped": actor_gn,
                "critic_grad_norm_unclipped": critic_gn,
                "total_grad_norm_unclipped": float(np.sqrt(actor_gn**2 + critic_gn**2)),
                "action_std": cur_std.cpu().numpy().tolist(),
                "action_std_mean": cur_std.mean().item(),
                "action_std_min": cur_std.min().item(),
                "action_std_max": cur_std.max().item(),
            }
            optimizer_step_records.append(rec)

        if len(optimizer_step_records) > 1:
            raise RuntimeError(f"Optimizer step called more than once! Count = {len(optimizer_step_records)}")

        return orig_optimizer_step(*step_args, **step_kwargs)

    runner.alg.optimizer.step = instrumented_optimizer_step

    # 4. Intercept storage before update to capture the exact rollout data
    rollout_snapshot = {}
    orig_update = runner.alg.update

    def captured_update():
        with torch.no_grad():
            rollout_snapshot["observations"] = runner.alg.storage.observations.clone() # (140, 64, 35)
            rollout_snapshot["actions"] = runner.alg.storage.actions.clone()           # (140, 64, 6)
            rollout_snapshot["rewards"] = runner.alg.storage.rewards.clone()           # (140, 64, 1)
            rollout_snapshot["dones"] = runner.alg.storage.dones.clone()               # (140, 64, 1)
            rollout_snapshot["values"] = runner.alg.storage.values.clone()             # (140, 64, 1)
            rollout_snapshot["returns"] = runner.alg.storage.returns.clone()           # (140, 64, 1)
            rollout_snapshot["advantages"] = runner.alg.storage.advantages.clone()     # (140, 64, 1)
            rollout_snapshot["actions_log_prob"] = runner.alg.storage.actions_log_prob.clone() # (140, 64, 1)
            rollout_snapshot["mu"] = runner.alg.storage.mu.clone()                     # (140, 64, 6)
            rollout_snapshot["sigma"] = runner.alg.storage.sigma.clone()               # (140, 64, 6)
        return orig_update()

    runner.alg.update = captured_update

    # 5. Execute 1 Iteration with strictly 1 optimizer step
    print(f"\n[Executing Phase 4.9 Pilot C Training]...")
    start_time = time.time()
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f}s!")

    # Verify optimizer step count
    print(f"\n[Optimizer Step Verification]")
    print(f"  Actual optimizer.step() calls: {len(optimizer_step_records)} (Expected: 1)")
    assert len(optimizer_step_records) == 1, f"Expected exactly 1 optimizer update, got {len(optimizer_step_records)}!"
    step_info = optimizer_step_records[0]
    print(f"  Actual Learning Rate         : {step_info['learning_rate']} (Expected: {target_lr})")
    assert step_info['learning_rate'] == target_lr, f"Learning rate was modified! Got {step_info['learning_rate']}"
    print(f"  Unclipped Actor Grad Norm    : {step_info['actor_grad_norm_unclipped']:.4f}")
    print(f"  Unclipped Critic Grad Norm   : {step_info['critic_grad_norm_unclipped']:.4f}")
    print(f"  Unclipped Total Grad Norm    : {step_info['total_grad_norm_unclipped']:.4f}")

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

    print(f"\n--- Parameter Shifts (Pilot C vs Pilot B Baseline) ---")
    print(f"  Actor L2 Delta    : {actor_delta_l2:.6f} (Relative: {actor_rel_delta * 100:.4f}%)")
    print(f"  Critic L2 Delta   : {critic_delta_l2:.6f} (Relative: {critic_rel_delta * 100:.4f}%)")
    print(f"  Action Std Delta  : {std_delta_l2:.6f}")
    print(f"  Final Action Std  : {final_std.cpu().numpy().tolist()}")

    # Assert no NaN / Inf
    assert not torch.isnan(final_actor_params).any(), "NaN in Actor parameters!"
    assert not torch.isinf(final_actor_params).any(), "Inf in Actor parameters!"
    assert not torch.isnan(final_critic_params).any(), "NaN in Critic parameters!"
    assert not torch.isinf(final_critic_params).any(), "Inf in Critic parameters!"

    # 7. Comprehensive Rollout Evaluation & KL Audit
    print(f"\n[Comprehensive Rollout Evaluation & KL Audit]")
    obs_all = rollout_snapshot["observations"].flatten(0, 1).to(env.device)   # (8960, 35)
    actions_all = rollout_snapshot["actions"].flatten(0, 1).to(env.device)     # (8960, 6)
    old_mu_all = rollout_snapshot["mu"].flatten(0, 1).to(env.device)           # (8960, 6)
    old_sigma_all = rollout_snapshot["sigma"].flatten(0, 1).to(env.device)     # (8960, 6)
    old_log_prob_all = rollout_snapshot["actions_log_prob"].flatten(0, 1).to(env.device) # (8960, 1)
    returns_all = rollout_snapshot["returns"].flatten(0, 1).to(env.device)     # (8960, 1)

    ac_runner.eval()
    with torch.no_grad():
        ac_runner.act(obs_all)
        new_mu_all = ac_runner.action_mean
        new_sigma_all = ac_runner.action_std
        new_log_prob_all = ac_runner.get_actions_log_prob(actions_all)
        new_values = ac_runner.evaluate(obs_all)

        # Analytical Gaussian KL formula per dimension
        kl_per_dim = torch.log(new_sigma_all / old_sigma_all + 1e-8) + \
                     (torch.square(old_sigma_all) + torch.square(old_mu_all - new_mu_all)) / (2.0 * torch.square(new_sigma_all)) - 0.5
        kl_total = torch.sum(kl_per_dim, dim=-1) # (8960,)

        # PPO Ratio & Clip fraction on the exact actual rollout
        log_ratio = new_log_prob_all - old_log_prob_all.squeeze(-1)
        ratio = torch.exp(log_ratio) # (8960,)
        clipped_mask = (torch.abs(ratio - 1.0) > 0.20)
        clip_fraction = torch.mean(clipped_mask.float()).item()

        # Value loss on rollout
        value_loss = torch.mean(torch.square(new_values - returns_all)).item()

        # Policy entropy
        entropy = torch.sum(0.5 + 0.5 * np.log(2 * np.pi) + torch.log(new_sigma_all[0])).item()

    kl_np = kl_total.cpu().numpy()
    mean_kl = float(np.mean(kl_np))
    median_kl = float(np.median(kl_np))
    p95_kl = float(np.percentile(kl_np, 95))
    max_kl = float(np.max(kl_np))

    action_names = ["left_hip", "left_knee", "left_wheel", "right_hip", "right_knee", "right_wheel"]
    kl_dim_stats = {}
    for i, name in enumerate(action_names):
        dim_kl = kl_per_dim[:, i].cpu().numpy()
        kl_dim_stats[name] = {
            "mean": float(np.mean(dim_kl)),
            "std": float(np.std(dim_kl)),
            "p50": float(np.percentile(dim_kl, 50)),
            "p95": float(np.percentile(dim_kl, 95)),
            "max": float(np.max(dim_kl)),
        }

    print(f"\n--- KL Divergence Metrics ---")
    print(f"  Mean Total KL       : {mean_kl:.6f} (Threshold: <= 0.0200)")
    print(f"  Median Total KL     : {median_kl:.6f}")
    print(f"  P95 Total KL        : {p95_kl:.6f}")
    print(f"  Max Total KL        : {max_kl:.6f}")
    print(f"  Rollout Clip Frac   : {clip_fraction * 100:.2f}%")
    print(f"  Rollout Value Loss  : {value_loss:.4f}")
    print(f"  Policy Entropy      : {entropy:.4f}")

    print(f"\n--- Per-Dimension KL Breakdown ---")
    for name, s in kl_dim_stats.items():
        print(f"  {name:15s}: Mean={s['mean']:.6f}, P95={s['p95']:.6f}, Max={s['max']:.6f}")

    # Check KL threshold requirement
    early_stop_kl = 0.020
    is_kl_passed = mean_kl <= early_stop_kl
    print(f"\n[KL Safety Gate Check]")
    if is_kl_passed:
        print(f"  Mean KL ({mean_kl:.6f}) <= {early_stop_kl:.4f} -> CONSERVATIVE KL GATE PASSED!")
    else:
        print(f"  Mean KL ({mean_kl:.6f}) > {early_stop_kl:.4f} -> TRAINING ADMISSION FAILED!")

    # 8. Save independent experiment model & telemetry
    out_dir = os.path.join(workspace_root, "evaluation_assets/phase4_9_pilot_c")
    os.makedirs(out_dir, exist_ok=True)

    out_model_path = os.path.join(out_dir, "model_pilot_c_step1.pt")
    out_telemetry_path = os.path.join(out_dir, "training_telemetry.json")
    out_rollout_path = os.path.join(out_dir, "rollout_pilot_c.pt")

    # Save checkpoint
    torch.save({
        "model_state_dict": ac_runner.state_dict(),
        "optimizer_state_dict": runner.alg.optimizer.state_dict(),
        "iter": 1,
        "step_count": 1,
    }, out_model_path)
    model_sha256 = compute_sha256(out_model_path)
    print(f"\nModel saved to: {out_model_path}")
    print(f"Model SHA256  : {model_sha256}")

    # Save rollout
    torch.save(rollout_snapshot, out_rollout_path)
    print(f"Rollout saved to: {out_rollout_path}")

    # Prepare telemetry
    telemetry = {
        "status": "PASSED" if is_kl_passed else "ADMISSION_FAILED",
        "timestamp": timestamp,
        "base_model": pilot_b_path,
        "base_model_sha256": pilot_b_sha256,
        "output_model": out_model_path,
        "output_model_sha256": model_sha256,
        "num_envs": num_envs,
        "num_steps": 140,
        "seed": seed,
        "optimizer_step_count": len(optimizer_step_records),
        "learning_rate": target_lr,
        "step_info": step_info,
        "parameter_shifts": {
            "actor_delta_l2": actor_delta_l2,
            "actor_relative_delta_pct": actor_rel_delta * 100,
            "critic_delta_l2": critic_delta_l2,
            "critic_relative_delta_pct": critic_rel_delta * 100,
            "std_delta_l2": std_delta_l2,
            "initial_action_std": init_std.cpu().numpy().tolist(),
            "final_action_std": final_std.cpu().numpy().tolist(),
        },
        "kl_metrics": {
            "mean_kl": mean_kl,
            "median_kl": median_kl,
            "p95_kl": p95_kl,
            "max_kl": max_kl,
            "early_stop_kl_threshold": early_stop_kl,
            "kl_gate_passed": is_kl_passed,
            "per_dimension": kl_dim_stats,
        },
        "training_diagnostics": {
            "rollout_clip_fraction": clip_fraction,
            "rollout_value_loss": value_loss,
            "entropy": entropy,
            "rsl_rl_last_diagnostics": runner.alg.last_update_diagnostics,
        }
    }

    with open(out_telemetry_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"Telemetry saved to: {out_telemetry_path}")

    print("\n" + "=" * 80)
    print(f"PHASE 4.9 PILOT C TRAINING FINISHED! STATUS: {telemetry['status']}")
    print("=" * 80)

    if not is_kl_passed:
        print("\n[WARNING] KL threshold exceeded. Model retained for read-only diagnosis.")
        sys.exit(1)


if __name__ == "__main__":
    main()
