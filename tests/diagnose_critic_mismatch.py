#!/usr/bin/env python3
# ==============================================================================
# Phase 4.5: Offline Critic Value Estimation & Reward Mismatch Diagnostic
# Compares old Phase 4.1 Gaussian reward vs Phase 4.4 Sigmoid reward returns,
# Critic predictions, empirical returns, GAE advantages, and loss magnitudes.
# ==============================================================================

import os
import sys

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
import torch
import numpy as np

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args


def run_critic_diagnostic():
    task_name = "myrobot_forward_jump_phase4"
    sys.argv = ["eval_critic", "--task", task_name, "--num_envs", "64"]
    args = get_args()
    args.headless = True
    args.seed = 1

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = 64
    env_cfg.seed = 1

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # Load Champion
    champ_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    ckpt_champ = torch.load(champ_path, map_location=env.device)
    actor_critic = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    actor_critic.load_state_dict(ckpt_champ["model_state_dict"])
    actor_critic.eval()

    obs, _ = env.reset()

    num_steps = 140
    gamma = 0.99
    lam = 0.95

    # Storage for rollout
    obs_list = []
    val_pred_list = []
    act_list = []
    rew_sigmoid_list = []
    rew_gaussian_list = []
    done_list = []

    # Also track settle steps across all 64 envs
    settle_steps = env.last_settle_step_reached.clone().cpu().numpy()
    stable_flags = env.last_stable_reached.clone().cpu().numpy()

    print(f"Settle summary on reset():")
    print(f"  All stable reached: {stable_flags.all()}")
    print(f"  Mean settle steps: {np.mean(settle_steps):.2f} (min={np.min(settle_steps)}, max={np.max(settle_steps)})")
    print(f"  Timed out envs (>=75): {np.sum(settle_steps >= 75)}")

    with torch.no_grad():
        for step in range(num_steps):
            obs_list.append(obs.clone())
            values = actor_critic.evaluate(obs)
            val_pred_list.append(values.squeeze(-1).clone())

            actions = actor_critic.act_inference(obs)
            act_list.append(actions.clone())

            # Step environment with Phase 4.4 Sigmoid reward
            next_obs, _, rew_sigmoid, dones, infos = env.step(actions)
            rew_sigmoid_list.append(rew_sigmoid.clone())
            done_list.append(dones.clone())

            # Simultaneously compute counterfactual old Gaussian reward on the exact same state:
            # 1. Airborne forward displacement with Gaussian tracking centered at 0.100 m (Phase 4.1) or 0.070 m (Phase 3.1)
            # In Phase 4.1: expected_airborne_distance = 0.100 m
            target_d_41 = 0.100
            valid_flight = env.is_airborne & env.has_taken_off & (~env.has_touched_down) & (env.jump_phase <= 0.40)
            clr_gate = torch.clamp((env.max_flight_clearance - 0.185) / 0.025, min=0.0, max=1.0)
            d_air = torch.clamp(env.flight_distance_y, min=0.0)
            dist_tracking_41 = torch.exp(-torch.square(d_air - target_d_41) / (2.0 * (0.04 ** 2)))
            progress_41 = torch.clamp(d_air / target_d_41, min=0.0, max=1.0)
            score_air_41 = (0.5 * progress_41 + 0.5 * dist_tracking_41) * clr_gate * valid_flight.float()

            clr_ok = (env.max_flight_clearance >= 0.200).float()
            dist_score_41 = torch.exp(-torch.square(env.final_flight_distance - target_d_41) / (2.0 * (0.04 ** 2)))
            score_td_41 = dist_score_41 * clr_ok * env.one_shot_touchdown_event.float()

            # Diff in reward = scale_air * (score_air_41 - score_air_sig) + scale_td * (score_td_41 - score_td_sig)
            score_air_sig = torch.sigmoid((d_air - 0.110) / 0.025) * clr_gate * valid_flight.float()
            score_td_sig = torch.sigmoid((env.final_flight_distance - 0.110) / 0.025) * clr_ok * env.one_shot_touchdown_event.float()

            delta_rew = 3.5 * (score_air_41 - score_air_sig) + 2.0 * (score_td_41 - score_td_sig)
            rew_gaussian = rew_sigmoid + delta_rew
            rew_gaussian_list.append(rew_gaussian.clone())

            obs = next_obs

    # Convert to tensors (num_steps, num_envs)
    vals = torch.stack(val_pred_list, dim=0) # (140, 64)
    rews_sig = torch.stack(rew_sigmoid_list, dim=0) # (140, 64)
    rews_gauss = torch.stack(rew_gaussian_list, dim=0) # (140, 64)
    dones_t = torch.stack(done_list, dim=0) # (140, 64)

    # 1. Total Episode Return
    ep_ret_sig = rews_sig.sum(dim=0).cpu().numpy()
    ep_ret_gauss = rews_gauss.sum(dim=0).cpu().numpy()

    # 2. Discounted Empirical Returns G_t
    ret_discounted_sig = torch.zeros_like(rews_sig)
    ret_discounted_gauss = torch.zeros_like(rews_gauss)

    running_g_sig = torch.zeros(64, device=env.device)
    running_g_gauss = torch.zeros(64, device=env.device)

    for t in reversed(range(num_steps)):
        running_g_sig = rews_sig[t] + gamma * running_g_sig
        running_g_gauss = rews_gauss[t] + gamma * running_g_gauss
        ret_discounted_sig[t] = running_g_sig
        ret_discounted_gauss[t] = running_g_gauss

    # 3. GAE Advantage Calculation
    adv_sig = torch.zeros_like(rews_sig)
    adv_gauss = torch.zeros_like(rews_gauss)

    gae_sig = torch.zeros(64, device=env.device)
    gae_gauss = torch.zeros(64, device=env.device)

    for t in reversed(range(num_steps)):
        next_val = vals[t+1] if t < num_steps - 1 else torch.zeros(64, device=env.device)
        next_not_done = 1.0 - dones_t[t].float()

        delta_s = rews_sig[t] + gamma * next_val * next_not_done - vals[t]
        gae_sig = delta_s + gamma * lam * next_not_done * gae_sig
        adv_sig[t] = gae_sig

        delta_g = rews_gauss[t] + gamma * next_val * next_not_done - vals[t]
        gae_gauss = delta_g + gamma * lam * next_not_done * gae_gauss
        adv_gauss[t] = gae_gauss

    # Print Full Comparative Analysis
    print("\n" + "="*80)
    print("REWARD & VALUE ESTIMATION COMPARISON (64 Envs x 140 Steps = 8,960 Transitions)")
    print("="*80)
    print(f"Episode Undiscounted Return:")
    print(f"  Old Gaussian (Phase 4.1): Mean = {np.mean(ep_ret_gauss):.4f} ± {np.std(ep_ret_gauss):.4f} (range [{np.min(ep_ret_gauss):.2f}, {np.max(ep_ret_gauss):.2f}])")
    print(f"  New Sigmoid  (Phase 4.4): Mean = {np.mean(ep_ret_sig):.4f} ± {np.std(ep_ret_sig):.4f} (range [{np.min(ep_ret_sig):.2f}, {np.max(ep_ret_sig):.2f}])")
    print(f"  Mean Return Delta (Sigmoid - Gaussian): {np.mean(ep_ret_sig - ep_ret_gauss):.4f}")

    print(f"\nStep 0 Value Prediction vs Discounted Empirical Return G_0:")
    v0 = vals[0].cpu().numpy()
    g0_sig = ret_discounted_sig[0].cpu().numpy()
    g0_gauss = ret_discounted_gauss[0].cpu().numpy()

    print(f"  Critic Predicted V(s_0): Mean = {np.mean(v0):.4f} ± {np.std(v0):.4f}")
    print(f"  Empirical G_0 (Old Gaussian): Mean = {np.mean(g0_gauss):.4f} ± {np.std(g0_gauss):.4f} | Error (V0 - G0) = {np.mean(v0 - g0_gauss):.4f}")
    print(f"  Empirical G_0 (New Sigmoid) : Mean = {np.mean(g0_sig):.4f} ± {np.std(g0_sig):.4f} | Error (V0 - G0) = {np.mean(v0 - g0_sig):.4f}")

    # Step-level Value MSE across all 8,960 transitions
    mse_val_gauss = torch.mean((vals - ret_discounted_gauss)**2).item()
    mse_val_sig = torch.mean((vals - ret_discounted_sig)**2).item()
    mae_val_gauss = torch.mean(torch.abs(vals - ret_discounted_gauss)).item()
    mae_val_sig = torch.mean(torch.abs(vals - ret_discounted_sig)).item()

    print(f"\nCritic Prediction Accuracy Across All 8,960 Transitions:")
    print(f"  Under Old Gaussian Reward: MSE = {mse_val_gauss:.4f}, MAE = {mae_val_gauss:.4f}")
    print(f"  Under New Sigmoid Reward : MSE = {mse_val_sig:.4f}, MAE = {mae_val_sig:.4f}")

    # GAE Advantage Comparison
    adv_sig_flat = adv_sig.flatten().cpu().numpy()
    adv_gauss_flat = adv_gauss.flatten().cpu().numpy()

    print(f"\nGAE Advantages (Raw unnormalized):")
    print(f"  Old Gaussian: Mean = {np.mean(adv_gauss_flat):.4f}, Std = {np.std(adv_gauss_flat):.4f}, Min = {np.min(adv_gauss_flat):.4f}, Max = {np.max(adv_gauss_flat):.4f}")
    print(f"  New Sigmoid : Mean = {np.mean(adv_sig_flat):.4f}, Std = {np.std(adv_sig_flat):.4f}, Min = {np.min(adv_sig_flat):.4f}, Max = {np.max(adv_sig_flat):.4f}")

    # Normalized Advantages
    norm_adv_sig = (adv_sig_flat - np.mean(adv_sig_flat)) / (np.std(adv_sig_flat) + 1e-8)
    norm_adv_gauss = (adv_gauss_flat - np.mean(adv_gauss_flat)) / (np.std(adv_gauss_flat) + 1e-8)

    print(f"\nNormalized Advantages:")
    print(f"  Old Gaussian: Min = {np.min(norm_adv_gauss):.4f}, Max = {np.max(norm_adv_gauss):.4f}")
    print(f"  New Sigmoid : Min = {np.min(norm_adv_sig):.4f}, Max = {np.max(norm_adv_sig):.4f}")

    # Flight step specific rewards and advantages
    print(f"\nFlight Phase Steps (Steps 12 to 32) Average Rewards:")
    print(f"  Old Gaussian: {rews_gauss[12:33].mean().item():.4f}")
    print(f"  New Sigmoid : {rews_sig[12:33].mean().item():.4f}")
    print(f"  Touchdown step (Step 32) Reward:")
    print(f"  Old Gaussian: {rews_gauss[32].mean().item():.4f}")
    print(f"  New Sigmoid : {rews_sig[32].mean().item():.4f}")

if __name__ == "__main__":
    run_critic_diagnostic()
