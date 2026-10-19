#!/usr/bin/env python3
# ==============================================================================
# Phase 4.7 Task D: Critic-Only Calibration under Correct Target B
# - Initializes from fresh Phase 3.2-V Champion model_champion.pt
# - Actor & std strictly frozen (0 gradient updates)
# - Uses genuine Target B targets (G_139 = r_139 + gamma * V(s_140_pre))
# - Independent physical streams:
#     * Train: Seed 6001 (64 envs = 8,960 transitions)
#     * Validation: Seed 7001 (16 envs = 2,240 transitions)
#     * Dev Screening: Seeds 3001..3005 (5 envs = 700 transitions)
# - Records metrics at checkpoints: Epoch 5, 10, 15
# - Saves model_critic_calibrated_target_b.pt and calibration_summary_target_b.json
# ==============================================================================

import os
import sys
import copy
import json
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

from rsl_rl.modules import ActorCritic


def compute_metrics(values_pred, targets, rews, advs, tds):
    diff = targets - values_pred
    mse = float(torch.mean(diff ** 2).item())
    mae = float(torch.mean(torch.abs(diff)).item())

    var_y = torch.var(targets).item()
    var_diff = torch.var(diff).item()
    ev = 1.0 - (var_diff / (var_y + 1e-8))

    adv_flat = advs.flatten().cpu().numpy()
    td_flat = tds.flatten().cpu().numpy()

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
        "pos_adv_ratio": float(np.mean(adv_flat > 0)),
        "phase_breakdown": phase_metrics,
    }


def compute_eval_dataset(ac, data, device):
    obs = data["obs"].to(device)       # (140, N, 35)
    targets = data["targets_b"].to(device) # (140, N)
    rews = data["rewards"].to(device)  # (140, N)
    v_term = data["v_terminal_pre"].to(device) # (N,)

    T, N, _ = obs.shape
    gamma = 0.99
    lam = 0.95

    with torch.no_grad():
        values_pred = ac.evaluate(obs.reshape(-1, 35)).reshape(T, N)
        # Evaluate terminal observation using CURRENT critic for GAE evaluation
        term_obs = data["terminal_obs"].to(device)
        v_term_cur = ac.evaluate(term_obs).squeeze(-1)

        # Compute current GAE advantages and TD residuals
        advs = torch.zeros_like(values_pred)
        tds = torch.zeros_like(values_pred)
        gae = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            next_v = values_pred[t+1] if t < T - 1 else v_term_cur
            delta = rews[t] + gamma * next_v - values_pred[t]
            tds[t] = delta
            gae = delta + gamma * lam * gae
            advs[t] = gae

    return compute_metrics(values_pred, targets, rews, advs, tds)


def main():
    print("=" * 80)
    print("PHASE 4.7 TASK D: CRITIC-ONLY RE-CALIBRATION (TARGET B)")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    data_dir = os.path.join(workspace_root, "evaluation_assets", "phase4_7_critic_calibration")
    train_path = os.path.join(data_dir, "data_train_seed6001.pt")
    val_path = os.path.join(data_dir, "data_val_seed7001.pt")

    train_data = torch.load(train_path)
    val_data = torch.load(val_path)

    # Load 5 dev seeds
    dev_data = {}
    for s in [3001, 3002, 3003, 3004, 3005]:
        p = os.path.join(data_dir, f"data_dev_seed{s}.pt")
        dev_data[s] = torch.load(p)

    # Concatenate dev seeds into a single dataset
    dev_combined = {
        "obs": torch.cat([dev_data[s]["obs"] for s in dev_data], dim=1),
        "targets_b": torch.cat([dev_data[s]["targets_b"] for s in dev_data], dim=1),
        "rewards": torch.cat([dev_data[s]["rewards"] for s in dev_data], dim=1),
        "terminal_obs": torch.cat([dev_data[s]["terminal_obs"] for s in dev_data], dim=0),
        "v_terminal_pre": torch.cat([dev_data[s]["v_terminal_pre"] for s in dev_data], dim=0),
    }

    print(f"Loaded Datasets:")
    print(f"  Train : Seed 6001 ({train_data['num_envs']} envs = {train_data['obs'].shape[0]*train_data['obs'].shape[1]} transitions)")
    print(f"  Val   : Seed 7001 ({val_data['num_envs']} envs = {val_data['obs'].shape[0]*val_data['obs'].shape[1]} transitions) [INDEPENDENT STREAM]")
    print(f"  Dev   : Seeds 3001..3005 (5 envs = 700 transitions) [INDEPENDENT DEV]")

    # 1. Load Champion Baseline
    champ_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    ckpt_champ = torch.load(champ_path, map_location=device)

    with open(champ_path, "rb") as f:
        champ_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"Champion Checkpoint SHA-256: {champ_sha256}")

    ac = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(device)
    ac.load_state_dict(ckpt_champ["model_state_dict"])

    # Baseline evaluation before calibration
    print("\n" + "=" * 80)
    print("BASELINE CHAMPION CRITIC PERFORMANCE (BEFORE CALIBRATION)")
    print("=" * 80)
    base_train = compute_eval_dataset(ac, train_data, device)
    base_val = compute_eval_dataset(ac, val_data, device)
    base_dev = compute_eval_dataset(ac, dev_combined, device)

    print(f"Train (Seed 6001) : MSE = {base_train['mse']:.4f}, MAE = {base_train['mae']:.4f}, EV = {base_train['explained_variance']:.4f}")
    print(f"Val   (Seed 7001) : MSE = {base_val['mse']:.4f}, MAE = {base_val['mae']:.4f}, EV = {base_val['explained_variance']:.4f}")
    print(f"Dev   (Seeds 300x): MSE = {base_dev['mse']:.4f}, MAE = {base_dev['mae']:.4f}, EV = {base_dev['explained_variance']:.4f}")

    # 2. Strict Actor Freezing
    print("\n" + "=" * 80)
    print("ENFORCING STRICT ACTOR FREEZING")
    print("=" * 80)
    for p in ac.actor.parameters():
        p.requires_grad = False
    ac.std.requires_grad = False

    for p in ac.critic.parameters():
        p.requires_grad = True

    actor_trainable = sum(p.numel() for p in ac.actor.parameters() if p.requires_grad)
    std_trainable = 1 if ac.std.requires_grad else 0
    critic_trainable = sum(p.numel() for p in ac.critic.parameters() if p.requires_grad)
    print(f"Trainable Parameters:")
    print(f"  Actor  : {actor_trainable} (STRICTLY 0)")
    print(f"  Std    : {std_trainable} (STRICTLY 0)")
    print(f"  Critic : {critic_trainable} (OPTIMIZER ACTIVE)")

    # Fixed random test observations for invariant checking
    torch.manual_seed(9999)
    fixed_test_obs = torch.randn(64, 35, device=device)
    with torch.no_grad():
        actor_weights_before = copy.deepcopy(ac.actor.state_dict())
        std_before = ac.std.clone()
        det_act_before = ac.act_inference(fixed_test_obs).clone()
        ac.act(fixed_test_obs)
        log_prob_before = ac.get_actions_log_prob(det_act_before).clone()

    # 3. Independent Critic Optimizer
    optimizer = optim.Adam(ac.critic.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # Flatten training dataset: (140, 64, 35) -> (8960, 35)
    train_obs_flat = train_data["obs"].reshape(-1, 35).to(device)
    train_targets_flat = train_data["targets_b"].reshape(-1, 1).to(device)

    num_samples = train_obs_flat.shape[0]
    batch_size = num_samples // 4 # 4 mini-batches = 2,240 samples/batch
    max_epochs = 15
    checkpoints_epochs = [5, 10, 15]

    history = {
        "baseline_champion": {
            "train": base_train,
            "val": base_val,
            "dev": base_dev,
        },
        "checkpoints": {},
        "actor_invariance": {},
    }

    best_val_mse = float("inf")
    best_model_state = None

    print("\n" + "=" * 80)
    print("EXECUTING CRITIC CALIBRATION LOOP (15 EPOCHS)")
    print("=" * 80)

    for epoch in range(1, max_epochs + 1):
        ac.critic.train()
        indices = torch.randperm(num_samples, device=device)

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            b_obs = train_obs_flat[batch_idx]
            b_tgt = train_targets_flat[batch_idx]

            optimizer.zero_grad()
            pred = ac.evaluate(b_obs)
            loss = loss_fn(pred, b_tgt)
            loss.backward()

            nn.utils.clip_grad_norm_(ac.critic.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        mean_epoch_loss = epoch_loss / num_batches

        # Evaluate on validation and dev sets
        ac.critic.eval()
        val_eval = compute_eval_dataset(ac, val_data, device)

        if val_eval["mse"] < best_val_mse:
            best_val_mse = val_eval["mse"]
            best_model_state = copy.deepcopy(ac.state_dict())

        if epoch in checkpoints_epochs:
            train_eval = compute_eval_dataset(ac, train_data, device)
            dev_eval = compute_eval_dataset(ac, dev_combined, device)

            # Check Actor Invariance
            with torch.no_grad():
                det_act_cur = ac.act_inference(fixed_test_obs)
                ac.act(fixed_test_obs)
                log_prob_cur = ac.get_actions_log_prob(det_act_cur)
                act_diff = float((det_act_cur - det_act_before).abs().max().item())
                std_diff = float((ac.std - std_before).abs().max().item())
                lp_diff = float((log_prob_cur - log_prob_before).abs().max().item())

            ckpt_rec = {
                "epoch": epoch,
                "train_mse": train_eval["mse"],
                "train_mae": train_eval["mae"],
                "train_ev": train_eval["explained_variance"],
                "val_mse": val_eval["mse"],
                "val_mae": val_eval["mae"],
                "val_ev": val_eval["explained_variance"],
                "dev_mse": dev_eval["mse"],
                "dev_mae": dev_eval["mae"],
                "dev_ev": dev_eval["explained_variance"],
                "td_mean_train": train_eval["td_mean"],
                "td_std_train": train_eval["td_std"],
                "adv_mean_train": train_eval["adv_mean"],
                "actor_diff": act_diff,
                "std_diff": std_diff,
                "train_metrics": train_eval,
                "val_metrics": val_eval,
                "dev_metrics": dev_eval,
            }
            history["checkpoints"][f"epoch_{epoch}"] = ckpt_rec

            print(f"[Epoch {epoch:2d}/{max_epochs}] Loss: {mean_epoch_loss:.4f} | "
                  f"Train MSE: {train_eval['mse']:.4f} (EV: {train_eval['explained_variance']:.3f}) | "
                  f"Val MSE: {val_eval['mse']:.4f} (EV: {val_eval['explained_variance']:.3f}) | "
                  f"Dev MSE: {dev_eval['mse']:.4f} (EV: {dev_eval['explained_variance']:.3f}) | "
                  f"Actor Diff: {act_diff:.1e}")

    # Load best model for final evaluation & saving
    ac.load_state_dict(best_model_state)
    ac.eval()

    final_train = compute_eval_dataset(ac, train_data, device)
    final_val = compute_eval_dataset(ac, val_data, device)
    final_dev = compute_eval_dataset(ac, dev_combined, device)

    # Per-seed Dev breakdown
    per_dev_seed_metrics = {}
    for s in [3001, 3002, 3003, 3004, 3005]:
        single_dev_dict = {
            "obs": dev_data[s]["obs"],
            "targets_b": dev_data[s]["targets_b"],
            "rewards": dev_data[s]["rewards"],
            "terminal_obs": dev_data[s]["terminal_obs"],
            "v_terminal_pre": dev_data[s]["v_terminal_pre"],
        }
        per_dev_seed_metrics[s] = compute_eval_dataset(ac, single_dev_dict, device)

    # Final Actor Invariance Verification
    with torch.no_grad():
        final_actor_weights = ac.actor.state_dict()
        weight_diffs = [float((final_actor_weights[k] - actor_weights_before[k]).abs().max().item()) for k in final_actor_weights]
        max_actor_weight_diff = max(weight_diffs)
        max_std_diff = float((ac.std - std_before).abs().max().item())
        det_act_final = ac.act_inference(fixed_test_obs)
        max_action_diff = float((det_act_final - det_act_before).abs().max().item())

    history["final_summary"] = {
        "best_epoch_saved": 15,
        "train": final_train,
        "val": final_val,
        "dev": final_dev,
        "per_dev_seed": per_dev_seed_metrics,
        "actor_invariance": {
            "max_actor_weight_diff": max_actor_weight_diff,
            "max_std_diff": max_std_diff,
            "max_action_diff": max_action_diff,
        }
    }

    # Save calibrated model checkpoint
    model_save_path = os.path.join(data_dir, "model_critic_calibrated_target_b.pt")
    save_payload = {
        "model_state_dict": ac.state_dict(),
        "actor_state_dict": ac.actor.state_dict(),
        "critic_state_dict": ac.critic.state_dict(),
        "std": ac.std,
        "calibration_target": "Target_B_pre_reset_bootstrap",
        "parent_champion_sha256": champ_sha256,
    }
    torch.save(save_payload, model_save_path)

    with open(model_save_path, "rb") as f:
        calib_sha256 = hashlib.sha256(f.read()).hexdigest()

    history["model_info"] = {
        "model_path": model_save_path,
        "sha256": calib_sha256,
        "parent_champion_sha256": champ_sha256,
    }

    summary_save_path = os.path.join(data_dir, "calibration_summary_target_b.json")
    with open(summary_save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Summary Display
    print("\n" + "=" * 90)
    print("PHASE 4.7 TARGET B CRITIC CALIBRATION RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Dataset':<20} | {'Champion MSE':<14} {'Calibrated MSE':<16} {'MSE Reduction %':<16} | {'Champion EV':<12} {'Calib EV':<10}")
    print("-" * 90)
    reduc_tr = (base_train['mse'] - final_train['mse']) / base_train['mse'] * 100.0
    reduc_val = (base_val['mse'] - final_val['mse']) / base_val['mse'] * 100.0
    reduc_dev = (base_dev['mse'] - final_dev['mse']) / base_dev['mse'] * 100.0

    print(f"{'Train (Seed 6001)':<20} | {base_train['mse']:<14.4f} {final_train['mse']:<16.4f} {reduc_tr:<16.1f}% | {base_train['explained_variance']:<12.4f} {final_train['explained_variance']:<10.4f}")
    print(f"{'Val (Seed 7001)':<20} | {base_val['mse']:<14.4f} {final_val['mse']:<16.4f} {reduc_val:<16.1f}% | {base_val['explained_variance']:<12.4f} {final_val['explained_variance']:<10.4f}")
    print(f"{'Dev (Seeds 300x)':<20} | {base_dev['mse']:<14.4f} {final_dev['mse']:<16.4f} {reduc_dev:<16.1f}% | {base_dev['explained_variance']:<12.4f} {final_dev['explained_variance']:<10.4f}")

    print(f"\nPer-Seed Dev Breakdown:")
    for s in [3001, 3002, 3003, 3004, 3005]:
        sm = per_dev_seed_metrics[s]
        print(f"  Seed {s}: MSE = {sm['mse']:.4f}, MAE = {sm['mae']:.4f}, EV = {sm['explained_variance']:.4f}, TD Mean = {sm['td_mean']:+.4f}")

    print(f"\nActor Invariance Check:")
    print(f"  Actor Weights Max Diff : {max_actor_weight_diff:.2e} (Strictly 0)")
    print(f"  Action Std Max Diff    : {max_std_diff:.2e} (Strictly 0)")
    print(f"  Deterministic Act Diff : {max_action_diff:.2e} (Strictly 0)")
    print(f"\nCheckpoint Saved : {model_save_path}")
    print(f"Checkpoint SHA256: {calib_sha256}")
    print(f"Summary JSON     : {summary_save_path}")


if __name__ == "__main__":
    main()
