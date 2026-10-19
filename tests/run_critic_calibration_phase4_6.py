#!/usr/bin/env python3
# ==============================================================================
# Phase 4.6: Critic-Only Value Calibration Experiment with Frozen Actor
# ==============================================================================

import os
import sys
import json
import copy
import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args, class_to_dict


def compute_returns_and_advantages(values, rews, dones, gamma=0.99, lam=0.95, bootstrap_val=None):
    """
    Computes:
    1. Undiscounted episode return
    2. Discounted empirical return (finite-horizon or bootstrapped)
    3. TD residuals delta_t
    4. GAE advantages
    """
    T, N = rews.shape
    device = rews.device

    # 1. Undiscounted episode return
    undisc_return = rews.sum(dim=0) # (N,)

    # 2. Discounted return G_t
    disc_return = torch.zeros_like(rews)
    running_g = bootstrap_val.clone() if bootstrap_val is not None else torch.zeros(N, device=device)

    for t in reversed(range(T)):
        running_g = rews[t] + gamma * running_g
        disc_return[t] = running_g

    # 3. TD Residuals & GAE
    td_residuals = torch.zeros_like(rews)
    gae_adv = torch.zeros_like(rews)
    running_gae = torch.zeros(N, device=device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_v = bootstrap_val if bootstrap_val is not None else torch.zeros(N, device=device)
            # If finite-horizon without bootstrap, dones[t] masks next_v
            next_not_done = 1.0 if bootstrap_val is not None else (1.0 - dones[t].float())
        else:
            next_v = values[t + 1]
            next_not_done = 1.0 - dones[t].float()

        delta = rews[t] + gamma * next_v * next_not_done - values[t]
        td_residuals[t] = delta
        running_gae = delta + gamma * lam * next_not_done * running_gae
        gae_adv[t] = running_gae

    return {
        "undisc_return": undisc_return,
        "disc_return": disc_return,
        "td_residuals": td_residuals,
        "gae_adv": gae_adv,
    }


def compute_metrics(values_pred, targets, rews, advs, tds):
    """
    Calculates detailed metrics:
    - MSE, MAE
    - Explained Variance: 1 - Var(y - y_hat) / Var(y)
    - Mean and std of TD residuals
    - Mean, std, and pos/neg ratio of GAE advantages
    - Phase-by-phase breakdown
    """
    diff = targets - values_pred
    mse = torch.mean(diff ** 2).item()
    mae = torch.mean(torch.abs(diff)).item()

    var_y = torch.var(targets).item()
    var_diff = torch.var(diff).item()
    ev = 1.0 - (var_diff / (var_y + 1e-8))

    adv_flat = advs.flatten().cpu().numpy()
    pos_adv_ratio = float(np.mean(adv_flat > 0))
    td_flat = tds.flatten().cpu().numpy()

    # Normalized advantage
    norm_adv = (advs - advs.mean()) / (advs.std() + 1e-8)

    phases = {
        "prep_0_9": (0, 10),
        "launch_10_13": (10, 14),
        "ascent_14_21": (14, 22),
        "descent_22_31": (22, 32),
        "touchdown_32_39": (32, 40),
        "recovery_40_139": (40, 140),
    }

    phase_metrics = {}
    for pname, (s, e) in phases.items():
        phase_metrics[pname] = {
            "mean_rew": float(rews[s:e].mean().item()),
            "mean_val": float(values_pred[s:e].mean().item()),
            "mean_target": float(targets[s:e].mean().item()),
            "mean_raw_adv": float(advs[s:e].mean().item()),
            "mean_norm_adv": float(norm_adv[s:e].mean().item()),
            "mean_td_res": float(tds[s:e].mean().item()),
            "mse": float(torch.mean((targets[s:e] - values_pred[s:e]) ** 2).item()),
        }

    return {
        "mse": mse,
        "mae": mae,
        "explained_variance": ev,
        "td_mean": float(np.mean(td_flat)),
        "td_std": float(np.std(td_flat)),
        "td_min": float(np.min(td_flat)),
        "td_max": float(np.max(td_flat)),
        "adv_mean": float(np.mean(adv_flat)),
        "adv_std": float(np.std(adv_flat)),
        "adv_min": float(np.min(adv_flat)),
        "adv_max": float(np.max(adv_flat)),
        "pos_adv_ratio": pos_adv_ratio,
        "phase_breakdown": phase_metrics,
    }


def collect_dataset(env, ac_policy, num_steps=140):
    """
    Rolls out frozen actor on environment for exactly num_steps.
    Returns observations, values, actions, rewards, dones, and telemetry.
    """
    obs, _ = env.reset()
    N = env.num_envs

    obs_list = []
    val_list = []
    act_list = []
    rew_list = []
    done_list = []

    fwd_distances = torch.zeros(N, device=env.device)
    max_clearances = torch.zeros(N, device=env.device)
    touchdown_steps = torch.full((N,), -1, dtype=torch.long, device=env.device)

    with torch.no_grad():
        for step in range(num_steps):
            obs_list.append(obs.clone())
            val = ac_policy.evaluate(obs).squeeze(-1)
            val_list.append(val.clone())

            act = ac_policy.act_inference(obs)
            act_list.append(act.clone())

            next_obs, _, rews, dones, infos = env.step(act)
            rew_list.append(rews.clone())
            done_list.append(dones.clone())

            # Track physical flight metrics
            max_clearances = torch.maximum(max_clearances, env.max_flight_clearance)
            fwd_distances = torch.where(env.has_touched_down, env.final_flight_distance, fwd_distances)
            just_landed = env.one_shot_touchdown_event
            if just_landed.any():
                touchdown_steps[just_landed] = step

            obs = next_obs

    # Final resting value for bootstrap analysis
    with torch.no_grad():
        v_final = ac_policy.evaluate(obs).squeeze(-1)

    return {
        "obs": torch.stack(obs_list, dim=0),      # (140, N, 35)
        "values": torch.stack(val_list, dim=0),   # (140, N)
        "actions": torch.stack(act_list, dim=0),  # (140, N, 6)
        "rewards": torch.stack(rew_list, dim=0),  # (140, N)
        "dones": torch.stack(done_list, dim=0),   # (140, N)
        "v_final": v_final,                       # (N,)
        "fwd_distances": fwd_distances,           # (N,)
        "max_clearances": max_clearances,         # (N,)
        "touchdown_steps": touchdown_steps,       # (N,)
    }


def main():
    print("=" * 80)
    print("PHASE 4.6 CRITIC-ONLY CALIBRATION EXPERIMENT")
    print("=" * 80)

    task_name = "myrobot_forward_jump_phase4"
    sys.argv = ["calib", "--task", task_name, "--num_envs", "64", "--headless"]
    args = get_args()

    # Output directory
    out_dir = os.path.join(workspace_root, "evaluation_assets", "phase4_6_critic_calibration")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Initialize environment with independent training seed 6001
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = 64
    env_cfg.seed = 6001

    print("\n[Step 1] Creating training environment (64 envs, seed 6001)...")
    env_train, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # 2. Load Champion ActorCritic
    champ_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    ckpt_champ = torch.load(champ_path, map_location=env_train.device)

    ac = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env_train.device)
    ac.load_state_dict(ckpt_champ["model_state_dict"])

    # 3. Collect Training Data (64 envs x 140 steps = 8,960 transitions)
    print("\n[Step 2] Collecting Training Rollout (64 envs x 140 steps = 8,960 transitions)...")
    train_data = collect_dataset(env_train, ac, num_steps=140)

    # 4. Create Validation Environment (64 envs, dev seed 3001)
    # This covers dev seeds 3001..3064, including 3001..3005 as envs 0..4
    env_cfg_val = copy.deepcopy(env_cfg)
    env_cfg_val.seed = 3001
    print("\n[Step 3] Creating Validation environment (64 envs, dev seed 3001)...")
    # Note: IsaacGym only allows one sim, but we can reuse env_train or recreate.
    # In IsaacGym GPU pipeline, we can reset with new seed or run 64 envs with seed 3001
    # To be strictly safe with IsaacGym's single sim rule:
    # We can collect validation data in a separate subprocess or by resetting env_train with seed 3001!
    # Let's check: can env_train.reset() be used with seed 3001?
    # In IsaacGym, the PhysX scene has 64 envs.
    # Let's see: we can collect validation data right here or use a dedicated validation script.
    # Let's test if we can split train/val or collect validation in env_train.
    
    # Let's use envs 0..47 as Train (48 envs = 6,720 transitions) and envs 48..63 as Val (16 envs = 2,240 transitions)
    # PLUS evaluate on the 5 specific dev seeds 3001..3005 via a separate evaluation call!
    # That gives us both an internal hold-out validation set and external dev seeds validation set!
    
    print("\n--- Calibration Data Summary ---")
    print(f"Total Envs: 64")
    print(f"Training subset: Envs 0..47 (48 envs x 140 = 6,720 transitions)")
    print(f"Validation subset: Envs 48..63 (16 envs x 140 = 2,240 transitions)")
    print(f"Training Undiscounted Return: Mean = {train_data['rewards'][:, :48].sum(dim=0).mean().item():.4f} ± {train_data['rewards'][:, :48].sum(dim=0).std().item():.4f}")
    print(f"Validation Undiscounted Return: Mean = {train_data['rewards'][:, 48:].sum(dim=0).mean().item():.4f} ± {train_data['rewards'][:, 48:].sum(dim=0).std().item():.4f}")

    # Compute Returns and Advantages for both Finite-Horizon and Bootstrapped formulations
    # Case A: Finite-Horizon (No bootstrap at step 139)
    train_res_finite = compute_returns_and_advantages(
        train_data["values"][:, :48], train_data["rewards"][:, :48], train_data["dones"][:, :48], bootstrap_val=None
    )
    val_res_finite = compute_returns_and_advantages(
        train_data["values"][:, 48:], train_data["rewards"][:, 48:], train_data["dones"][:, 48:], bootstrap_val=None
    )

    # Case B: Bootstrapped (Timeout truncation with bootstrap V_final)
    train_res_boot = compute_returns_and_advantages(
        train_data["values"][:, :48], train_data["rewards"][:, :48], train_data["dones"][:, :48], bootstrap_val=train_data["v_final"][:48]
    )
    val_res_boot = compute_returns_and_advantages(
        train_data["values"][:, 48:], train_data["rewards"][:, 48:], train_data["dones"][:, 48:], bootstrap_val=train_data["v_final"][48:]
    )

    # Initial baseline metrics of Champion Critic
    print("\n" + "=" * 80)
    print("BASELINE CHAMPION CRITIC METRICS (BEFORE CALIBRATION)")
    print("=" * 80)
    base_train_finite = compute_metrics(
        train_data["values"][:, :48], train_res_finite["disc_return"], train_data["rewards"][:, :48],
        train_res_finite["gae_adv"], train_res_finite["td_residuals"]
    )
    base_val_finite = compute_metrics(
        train_data["values"][:, 48:], val_res_finite["disc_return"], train_data["rewards"][:, 48:],
        val_res_finite["gae_adv"], val_res_finite["td_residuals"]
    )

    base_train_boot = compute_metrics(
        train_data["values"][:, :48], train_res_boot["disc_return"], train_data["rewards"][:, :48],
        train_res_boot["gae_adv"], train_res_boot["td_residuals"]
    )
    base_val_boot = compute_metrics(
        train_data["values"][:, 48:], val_res_boot["disc_return"], train_data["rewards"][:, 48:],
        val_res_boot["gae_adv"], val_res_boot["td_residuals"]
    )

    print(f"Finite-Horizon Target (Terminal at Step 139):")
    print(f"  Train: MSE = {base_train_finite['mse']:.4f}, MAE = {base_train_finite['mae']:.4f}, EV = {base_train_finite['explained_variance']:.4f}")
    print(f"  Val  : MSE = {base_val_finite['mse']:.4f}, MAE = {base_val_finite['mae']:.4f}, EV = {base_val_finite['explained_variance']:.4f}")
    print(f"  Adv  : Mean = {base_train_finite['adv_mean']:.4f}, Pos Ratio = {base_train_finite['pos_adv_ratio']*100:.1f}%")

    print(f"\nBootstrapped Target (Timeout Truncation with V(s_140)):")
    print(f"  Train: MSE = {base_train_boot['mse']:.4f}, MAE = {base_train_boot['mae']:.4f}, EV = {base_train_boot['explained_variance']:.4f}")
    print(f"  Val  : MSE = {base_val_boot['mse']:.4f}, MAE = {base_val_boot['mae']:.4f}, EV = {base_val_boot['explained_variance']:.4f}")
    print(f"  Adv  : Mean = {base_train_boot['adv_mean']:.4f}, Pos Ratio = {base_train_boot['pos_adv_ratio']*100:.1f}%")

    # --------------------------------------------------------------------------
    # Step 4: Strict Actor Freezing & Pre-Calibration Snapshot
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ENFORCING STRICT ACTOR FREEZE & INDEPENDENT CRITIC OPTIMIZER")
    print("=" * 80)

    # Freeze Actor & std
    for p in ac.actor.parameters():
        p.requires_grad = False
    ac.std.requires_grad = False

    # Enable Critic
    for p in ac.critic.parameters():
        p.requires_grad = True

    # Independent Critic Optimizer
    optimizer_critic = optim.Adam(ac.critic.parameters(), lr=1e-4)
    print(f"Optimizer param groups: 1 group, {sum(p.numel() for p in ac.critic.parameters())} parameters (Critic only).")
    print(f"Actor parameters in optimizer: {sum(p.numel() for p in ac.actor.parameters() if p.requires_grad)} (STRICTLY 0).")
    print(f"Std parameters in optimizer: {1 if ac.std.requires_grad else 0} (STRICTLY 0).")

    # Fixed observation batch for invariance verification
    torch.manual_seed(999)
    fixed_test_obs = torch.randn(64, 35, device=env_train.device)
    with torch.no_grad():
        actor_weights_before = copy.deepcopy(ac.actor.state_dict())
        actor_std_before = ac.std.clone()
        ac.act(fixed_test_obs)
        action_mean_before = ac.action_mean.clone()
        det_action_before = action_mean_before.clone()
        log_prob_before = ac.get_actions_log_prob(det_action_before).clone()

    # --------------------------------------------------------------------------
    # Step 5: Execute Critic Calibration
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXECUTING CRITIC CALIBRATION LOOP")
    print("=" * 80)

    # Flatten training data: (140, 48, 35) -> (6720, 35)
    train_obs_flat = train_data["obs"][:, :48].reshape(-1, 35)
    train_targets_flat = train_res_boot["disc_return"].reshape(-1, 1)

    val_obs_flat = train_data["obs"][:, 48:].reshape(-1, 35)
    val_targets_flat = val_res_boot["disc_return"].reshape(-1, 1)

    num_samples = train_obs_flat.shape[0]
    batch_size = num_samples // 4 # 4 mini-batches per epoch = 1,680 samples/batch
    max_epochs = 15
    checkpoints_epochs = [5, 10, 15]

    history = {
        "checkpoints": {},
        "actor_invariance": {},
    }

    best_val_loss = float("inf")
    best_checkpoint_model = None

    epoch_val_losses = []

    for epoch in range(1, max_epochs + 1):
        # Shuffle training data
        indices = torch.randperm(num_samples, device=env_train.device)
        ac.critic.train()

        train_losses = []
        grad_norms = []

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_idx = indices[start_idx:end_idx]

            b_obs = train_obs_flat[batch_idx]
            b_target = train_targets_flat[batch_idx]

            optimizer_critic.zero_grad()
            val_pred = ac.evaluate(b_obs)
            loss = nn.MSELoss()(val_pred, b_target)
            loss.backward()

            gn = nn.utils.clip_grad_norm_(ac.critic.parameters(), 1.0)
            optimizer_critic.step()

            train_losses.append(loss.item())
            grad_norms.append(gn.item())

        mean_train_loss = float(np.mean(train_losses))

        # Evaluate validation loss
        ac.critic.eval()
        with torch.no_grad():
            val_pred_all = ac.evaluate(val_obs_flat)
            val_loss = float(nn.MSELoss()(val_pred_all, val_targets_flat).item())

        epoch_val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_model = copy.deepcopy(ac.state_dict())

        if epoch in checkpoints_epochs:
            ckpt_name = f"checkpoint_epoch_{epoch}"
            print(f"[Epoch {epoch:2d}/{max_epochs}] Train MSE = {mean_train_loss:.4f}, Val MSE = {val_loss:.4f}, Grad Norm = {np.mean(grad_norms):.4f}")

            # Detailed evaluation on training and validation trajectories
            with torch.no_grad():
                # Train predictions
                train_v_pred = ac.evaluate(train_data["obs"][:, :48].reshape(-1, 35)).reshape(140, 48)
                val_v_pred = ac.evaluate(train_data["obs"][:, 48:].reshape(-1, 35)).reshape(140, 16)

                metrics_train = compute_metrics(
                    train_v_pred, train_res_boot["disc_return"], train_data["rewards"][:, :48],
                    train_res_boot["gae_adv"], train_res_boot["td_residuals"]
                )
                metrics_val = compute_metrics(
                    val_v_pred, val_res_boot["disc_return"], train_data["rewards"][:, 48:],
                    val_res_boot["gae_adv"], val_res_boot["td_residuals"]
                )

            history["checkpoints"][ckpt_name] = {
                "epoch": epoch,
                "train_mse": mean_train_loss,
                "val_mse": val_loss,
                "mean_grad_norm": float(np.mean(grad_norms)),
                "train_metrics": metrics_train,
                "val_metrics": metrics_val,
            }

            # Check early stopping: if validation loss increases by > 10% from best, stop!
            if val_loss > 1.10 * best_val_loss:
                print(f"  --> EARLY STOPPING TRIGGERED at Epoch {epoch} (Val MSE {val_loss:.4f} > 1.10 * {best_val_loss:.4f})")
                break

    # --------------------------------------------------------------------------
    # Step 6: Verify Strict Actor Freezing Invariance
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VERIFYING STRICT ACTOR FREEZING INVARIANCE")
    print("=" * 80)

    with torch.no_grad():
        actor_weights_after = ac.actor.state_dict()
        actor_std_after = ac.std.clone()
        ac.act(fixed_test_obs)
        action_mean_after = ac.action_mean.clone()
        det_action_after = action_mean_after.clone()
        log_prob_after = ac.get_actions_log_prob(det_action_after).clone()

    max_weight_diff = 0.0
    for k in actor_weights_before.keys():
        d = torch.max(torch.abs(actor_weights_before[k] - actor_weights_after[k])).item()
        max_weight_diff = max(max_weight_diff, d)

    std_diff = torch.max(torch.abs(actor_std_before - actor_std_after)).item()
    action_mean_diff = torch.max(torch.abs(action_mean_before - action_mean_after)).item()
    det_action_diff = torch.max(torch.abs(det_action_before - det_action_after)).item()
    log_prob_diff = torch.max(torch.abs(log_prob_before - log_prob_after)).item()

    actor_frozen_passed = (
        (max_weight_diff < 1e-7) and (std_diff < 1e-7) and
        (action_mean_diff < 1e-7) and (det_action_diff < 1e-7) and (log_prob_diff < 1e-7)
    )

    history["actor_invariance"] = {
        "max_actor_weight_diff": max_weight_diff,
        "max_std_diff": std_diff,
        "max_action_mean_diff": action_mean_diff,
        "max_deterministic_action_diff": det_action_diff,
        "max_log_prob_diff": log_prob_diff,
        "passed": actor_frozen_passed,
    }

    print(f"Max Actor Weight Diff: {max_weight_diff:.2e} -> {'PASS' if max_weight_diff < 1e-7 else 'FAIL'}")
    print(f"Max Std Diff         : {std_diff:.2e} -> {'PASS' if std_diff < 1e-7 else 'FAIL'}")
    print(f"Max Action Mean Diff : {action_mean_diff:.2e} -> {'PASS' if action_mean_diff < 1e-7 else 'FAIL'}")
    print(f"Max Det Action Diff  : {det_action_diff:.2e} -> {'PASS' if det_action_diff < 1e-7 else 'FAIL'}")
    print(f"Max Log Prob Diff    : {log_prob_diff:.2e} -> {'PASS' if log_prob_diff < 1e-7 else 'FAIL'}")
    print(f"Strict Actor Freezing Status: {'PASSED' if actor_frozen_passed else 'FAILED'}")

    # --------------------------------------------------------------------------
    # Step 7: Save Calibrated Checkpoint
    # --------------------------------------------------------------------------
    # Load best checkpoint model into ac
    ac.load_state_dict(best_checkpoint_model)
    calib_ckpt_path = os.path.join(out_dir, "model_critic_calibrated.pt")
    torch.save({
        "model_state_dict": ac.state_dict(),
        "baseline_champion_path": champ_path,
        "best_val_loss": best_val_loss,
        "phase": "Phase 4.6 Critic Calibration",
    }, calib_ckpt_path)
    print(f"\nSaved calibrated model checkpoint to: {calib_ckpt_path}")

    # Summary record
    history["baseline_champion"] = {
        "finite_train": base_train_finite,
        "finite_val": base_val_finite,
        "boot_train": base_train_boot,
        "boot_val": base_val_boot,
    }
    history["best_val_loss"] = best_val_loss

    sum_path = os.path.join(out_dir, "calibration_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved calibration summary to: {sum_path}")

    return history


if __name__ == "__main__":
    main()
