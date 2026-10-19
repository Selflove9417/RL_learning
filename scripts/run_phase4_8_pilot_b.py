"""
Phase 4.8-Pilot-B: Fixed Low Learning Rate Single-Update Control Experiment
Strict constraints:
1. Re-initialize from Phase 4.7 Target B checkpoint (Actor = Champion, Critic = Target B).
   DO NOT use Pilot A weights!
2. Fixed low learning rate: lr = 5.0e-5, schedule = 'fixed' (Adaptive schedule DISABLED).
3. EXACTLY ONE optimizer.step() executed (strictly 1 mini-batch update, not a 20-update iteration).
4. Same seed (6001), 64 envs, 140-step rollout, mini-batch size = 2240 (num_mini_batches = 4).
5. Comprehensive pre/post-update telemetry & KL audit across dimensions and phases.
6. Saves model_pilot_b_step1.pt and telemetry to evaluation_assets/phase4_8_pilot_b/.
"""

import os
import sys
import copy
import json
import time
import shutil
import hashlib
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
    print("MYROBOT PHASE 4.8-PILOT-B: FIXED LOW-LR SINGLE OPTIMIZER STEP EXPERIMENT")
    print("=" * 80)

    # 1. Environment & configuration setup
    task_name = "myrobot_forward_jump_phase4"
    num_envs = 64
    seed = 6001
    sys.argv = ["pilot_b", "--task", task_name, "--num_envs", str(num_envs), "--headless"]
    args = get_args()
    args.seed = seed

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.domain_rand.push_robots = False
    env_cfg.noise.add_noise = False

    # Independent experiment configuration
    timestamp = time.strftime("%b%d_%H-%M-%S")
    train_cfg.seed = seed
    train_cfg.runner.experiment_name = "myrobot_forward_jump_phase4"
    train_cfg.runner.run_name = f"{timestamp}_phase4_8_pilot_b_1step_fixed_lr"
    train_cfg.runner.max_iterations = 1
    train_cfg.runner.save_interval = 1
    train_cfg.runner.resume = False

    # Checkpoint paths
    champ_path = os.path.join(workspace_root, "evaluation_assets/phase3_1_forward_distance/models/model_champion.pt")
    calib_path = os.path.join(workspace_root, "evaluation_assets/phase4_7_critic_calibration/model_critic_calibrated_target_b.pt")

    assert os.path.isfile(champ_path), f"Champion model not found: {champ_path}"
    assert os.path.isfile(calib_path), f"Calibrated model not found: {calib_path}"

    champ_sha256 = compute_sha256(champ_path)
    calib_sha256 = compute_sha256(calib_path)

    expected_champ_sha = "fad590f4699abac8ae2b3e635d9cb88ef2658da85bbb37008e13362bb3c5b950"
    expected_calib_sha = "a5fca202c667da64476bb36044f9fdd814cae430753b57e349699b2e679f5ed0"

    print(f"\n[Asset Identity Pre-Check]")
    print(f"  Champion Path             : {champ_path}")
    print(f"  Champion SHA256           : {champ_sha256}")
    print(f"  Target B Path             : {calib_path}")
    print(f"  Target B SHA256           : {calib_sha256}")
    assert champ_sha256 == expected_champ_sha, "Champion SHA256 mismatch!"
    assert calib_sha256 == expected_calib_sha, "Target B SHA256 mismatch!"

    # Warm-start configuration
    train_cfg.runner.warm_start = True
    train_cfg.runner.warm_start_path = calib_path
    train_cfg.runner.warm_start_optimizer = False

    # Strict Fixed Low-LR Hyperparameters
    target_lr = 5.0e-5
    train_cfg.algorithm.learning_rate = target_lr
    train_cfg.algorithm.schedule = 'fixed'          # DISABLED adaptive schedule
    train_cfg.algorithm.desired_kl = None           # No adaptive scaling
    train_cfg.algorithm.enable_early_stopping = True
    train_cfg.algorithm.early_stop_kl = 0.04
    train_cfg.algorithm.max_grad_norm = 1.0
    train_cfg.algorithm.num_learning_epochs = 5
    train_cfg.algorithm.num_mini_batches = 4
    train_cfg.algorithm.clip_param = 0.2
    train_cfg.algorithm.value_loss_coef = 1.0
    train_cfg.algorithm.entropy_coef = 0.005
    train_cfg.algorithm.gamma = 0.99
    train_cfg.algorithm.lam = 0.95

    print(f"\n[Pilot B Hyperparameters]")
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

    # Verify model initialization invariance vs Champion and Target B
    ac_champ = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_champ.load_state_dict(torch.load(champ_path, map_location=env.device)["model_state_dict"])
    ac_champ.eval()

    ac_calib = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_calib.load_state_dict(torch.load(calib_path, map_location=env.device)["model_state_dict"])
    ac_calib.eval()

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

    # Snapshot initial parameters for exact delta analysis
    with torch.no_grad():
        init_actor_params = torch.cat([p.detach().flatten() for p in ac_runner.actor.parameters()]).clone()
        init_critic_params = torch.cat([p.detach().flatten() for p in ac_runner.critic.parameters()]).clone()
        init_std = ac_runner.std.detach().clone()

    # 3. Instrument PPO to enforce EXACTLY ONE optimizer.step()
    # Intercept mini_batch_generator so it yields ONLY the first mini-batch and terminates!
    orig_generator_func = runner.alg.storage.mini_batch_generator
    batch_indices_captured = []
    
    def single_step_mini_batch_generator(num_mini_batches, num_epochs=8):
        # We record the exact permutation indices
        batch_size = runner.alg.storage.num_envs * runner.alg.storage.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        # Call original generator
        gen = orig_generator_func(num_mini_batches, num_epochs)
        # Yield strictly the FIRST mini-batch
        batch = next(gen)
        yield batch
        # Terminate generator here!
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
    print(f"\n[Executing Phase 4.8 Pilot B Training]...")
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

    print(f"\n--- Parameter Shifts (Pilot B vs Target B Baseline) ---")
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

    ac_runner.eval()
    with torch.no_grad():
        ac_runner.act(obs_all)
        new_mu_all = ac_runner.action_mean
        new_sigma_all = ac_runner.action_std
        new_log_prob_all = ac_runner.get_actions_log_prob(actions_all)

        # Analytical Gaussian KL formula per dimension
        kl_per_dim = torch.log(new_sigma_all / old_sigma_all + 1e-5) + \
                     (torch.square(old_sigma_all) + torch.square(old_mu_all - new_mu_all)) / (2.0 * torch.square(new_sigma_all)) - 0.5
        kl_total = torch.sum(kl_per_dim, dim=-1) # (8960,)

        # PPO Ratio & Clip fraction on the exact actual rollout
        log_ratio = new_log_prob_all - old_log_prob_all.squeeze(-1)
        ratio = torch.exp(log_ratio) # (8960,)
        clipped_mask = (torch.abs(ratio - 1.0) > 0.20)
        clip_fraction = torch.mean(clipped_mask.float()).item()

    kl_np = kl_total.cpu().numpy()
    kl_mean = float(np.mean(kl_np))
    kl_median = float(np.median(kl_np))
    kl_p90 = float(np.percentile(kl_np, 90))
    kl_p95 = float(np.percentile(kl_np, 95))
    kl_p99 = float(np.percentile(kl_np, 99))
    kl_max = float(np.max(kl_np))

    print(f"\n--- Post-Update Full Rollout KL (8,960 transitions) ---")
    print(f"  KL Mean           : {kl_mean:.6f}")
    print(f"  KL Median         : {kl_median:.6f}")
    print(f"  KL P90            : {kl_p90:.6f}")
    print(f"  KL P95            : {kl_p95:.6f}")
    print(f"  KL P99            : {kl_p99:.6f}")
    print(f"  KL Max            : {kl_max:.6f}")
    print(f"  Ratio Mean        : {ratio.mean().item():.6f} (Min: {ratio.min().item():.6f}, Max: {ratio.max().item():.6f})")
    print(f"  Ratio P05 / P95   : {np.percentile(ratio.cpu().numpy(), 5):.6f} / {np.percentile(ratio.cpu().numpy(), 95):.6f}")
    print(f"  Actual Clip Frac  : {clip_fraction * 100:.2f}%")

    dim_names = ['Left Hip', 'Left Knee', 'Left Wheel', 'Right Hip', 'Right Knee', 'Right Wheel']
    dim_audit = {}
    print(f"\n--- Per-Action Dimension KL Contribution ---")
    for d, name in enumerate(dim_names):
        kl_d = kl_per_dim[:, d].cpu().numpy()
        d_mu = (new_mu_all[:, d] - old_mu_all[:, d]).abs().cpu().numpy()
        contrib_pct = float(np.mean(kl_d) / kl_mean * 100) if kl_mean > 0 else 0.0
        dim_audit[name] = {
            "kl_mean": float(np.mean(kl_d)),
            "kl_pct": contrib_pct,
            "d_mu_mean": float(np.mean(d_mu)),
            "d_mu_max": float(np.max(d_mu)),
            "std_old": float(old_sigma_all[0, d].item()),
            "std_new": float(new_sigma_all[0, d].item()),
        }
        print(f"  {name:12s}: KL={np.mean(kl_d):.6f} ({contrib_pct:5.1f}%) | |d_mu| mean={np.mean(d_mu):.4f}, max={np.max(d_mu):.4f} | std={old_sigma_all[0, d].item():.4f}->{new_sigma_all[0, d].item():.4f}")

    # Phase-based KL analysis (140 steps x 64 envs)
    kl_by_step = kl_total.reshape(140, 64)
    phases = {
        "prep_0_9": (0, 10),
        "launch_10_13": (10, 14),
        "ascent_14_21": (14, 22),
        "descent_22_31": (22, 32),
        "touchdown_32_39": (32, 40),
        "recovery_40_139": (40, 140)
    }
    phase_audit = {}
    print(f"\n--- Task Phase KL Distribution ---")
    for pname, (s_start, s_end) in phases.items():
        kl_ph = kl_by_step[s_start:s_end, :].cpu().numpy().flatten()
        phase_audit[pname] = {
            "steps": f"{s_start}-{s_end-1}",
            "n_samples": int(len(kl_ph)),
            "kl_mean": float(np.mean(kl_ph)),
            "kl_median": float(np.median(kl_ph)),
            "kl_p95": float(np.percentile(kl_ph, 95)),
            "kl_max": float(np.max(kl_ph)),
        }
        print(f"  {pname:15s} (steps {s_start:2d}-{s_end-1:3d}, n={len(kl_ph):4d}): Mean={np.mean(kl_ph):.6f}, Med={np.median(kl_ph):.6f}, P95={np.percentile(kl_ph, 95):.6f}, Max={np.max(kl_ph):.6f}")

    # 8. Save model checkpoint and telemetry to independent directory
    out_dir = os.path.join(workspace_root, "evaluation_assets/phase4_8_pilot_b")
    os.makedirs(out_dir, exist_ok=True)

    # Save rollout snapshot
    rollout_path = os.path.join(out_dir, "rollout_pilot_b.pt")
    torch.save(rollout_snapshot, rollout_path)

    pilot_b_model_path = os.path.join(out_dir, "model_pilot_b_step1.pt")
    # Save the updated model state dict
    torch.save({
        "model_state_dict": ac_runner.state_dict(),
        "optimizer_state_dict": runner.alg.optimizer.state_dict(),
        "iter": 1,
        "step_count": 1,
    }, pilot_b_model_path)

    pilot_b_sha256 = compute_sha256(pilot_b_model_path)
    print(f"\n[Model Checkpoint Saved]")
    print(f"  Path   : {pilot_b_model_path}")
    print(f"  SHA256 : {pilot_b_sha256}")

    # Compile training telemetry
    telemetry = {
        "metadata": {
            "experiment": "Phase 4.8-Pilot-B",
            "task_name": task_name,
            "seed": seed,
            "num_envs": num_envs,
            "timestamp": timestamp,
            "target_b_init_path": calib_path,
            "target_b_init_sha256": calib_sha256,
            "champion_path": champ_path,
            "champion_sha256": champ_sha256,
            "pilot_b_model_path": pilot_b_model_path,
            "pilot_b_model_sha256": pilot_b_sha256,
            "elapsed_seconds": elapsed_time
        },
        "hyperparameters": {
            "learning_rate": target_lr,
            "schedule": "fixed",
            "desired_kl": None,
            "early_stop_kl": 0.04,
            "max_grad_norm": 1.0,
            "clip_param": 0.2,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "planned_updates": 1,
            "actual_updates": len(optimizer_step_records),
            "gamma": 0.99,
            "lam": 0.95
        },
        "optimizer_step_telemetry": step_info,
        "parameter_shifts": {
            "actor_delta_l2": actor_delta_l2,
            "actor_rel_delta": actor_rel_delta,
            "critic_delta_l2": critic_delta_l2,
            "critic_rel_delta": critic_rel_delta,
            "std_delta_l2": std_delta_l2,
            "initial_std": init_std.cpu().numpy().tolist(),
            "final_std": final_std.cpu().numpy().tolist(),
        },
        "rollout_kl_summary": {
            "kl_mean": kl_mean,
            "kl_median": kl_median,
            "kl_p90": kl_p90,
            "kl_p95": kl_p95,
            "kl_p99": kl_p99,
            "kl_max": kl_max,
            "ratio_mean": float(ratio.mean().item()),
            "ratio_min": float(ratio.min().item()),
            "ratio_max": float(ratio.max().item()),
            "ratio_p05": float(np.percentile(ratio.cpu().numpy(), 5)),
            "ratio_p95": float(np.percentile(ratio.cpu().numpy(), 95)),
            "clip_fraction": clip_fraction,
        },
        "dim_contributions": dim_audit,
        "phase_contributions": phase_audit
    }

    telemetry_file = os.path.join(out_dir, "training_telemetry.json")
    with open(telemetry_file, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"  Telemetry saved to: {telemetry_file}")

    print("\n" + "=" * 80)
    print("PHASE 4.8-PILOT-B TRAINING & AUDIT SUCCESSFULLY COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
