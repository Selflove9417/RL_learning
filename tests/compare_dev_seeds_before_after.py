#!/usr/bin/env python3
# ==============================================================================
# Phase 4.6: Comprehensive Comparative Evaluation on Fixed Dev Seeds 3001..3005
# Compares Model A (Champion Critic) vs Model B (Calibrated Critic)
# Evaluates:
# 1. Value prediction error on new Sigmoid reward
# 2. Generalization (Train vs Validation error)
# 3. Explained Variance
# 4. GAE advantage quality & relation to actual trajectory forward distance
# 5. Relative advantage of high-distance (Seed 3004) vs low-distance (Seed 3003)
# 6. Potential distortions in support/landing/recovery phases
# ==============================================================================

import os
import sys
import json
import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse

import torch
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args


def evaluate_dev_seed(seed, ac_champ, ac_calib):
    task_name = "myrobot_forward_jump_phase4"
    sys.argv = ["eval_seed", "--task", task_name, "--num_envs", "1", "--headless"]
    args = get_args()
    args.seed = seed

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    obs, _ = env.reset()

    obs_list = []
    rew_list = []
    done_list = []
    val_a_list = []
    val_b_list = []

    com_y_list = []
    clr_list = []

    fwd_dist_final = 0.0
    clr_max = 0.0
    touchdown_step = -1

    with torch.no_grad():
        for step in range(140):
            obs_list.append(obs.clone())

            # Model A (Champion) value prediction
            va = ac_champ.evaluate(obs).item()
            val_a_list.append(va)

            # Model B (Calibrated) value prediction
            vb = ac_calib.evaluate(obs).item()
            val_b_list.append(vb)

            # Step with frozen Champion Actor
            act = ac_champ.act_inference(obs)
            next_obs, _, rew, dones, infos = env.step(act)

            rew_list.append(rew.item())
            done_list.append(dones.item())

            clr_cur = env.bilateral_clearance[0].item()
            clr_list.append(clr_cur)
            clr_max = max(clr_max, env.max_flight_clearance[0].item())

            if env.has_touched_down[0].item() and touchdown_step < 0:
                touchdown_step = step
                fwd_dist_final = env.final_flight_distance[0].item()

            obs = next_obs

        # Final resting state value
        va_final = ac_champ.evaluate(obs).item()
        vb_final = ac_calib.evaluate(obs).item()

    if fwd_dist_final == 0.0 and env.has_touched_down[0].item():
        fwd_dist_final = env.final_flight_distance[0].item()

    T = 140
    gamma = 0.99
    lam = 0.95

    rews_t = torch.tensor(rew_list)
    vals_a_t = torch.tensor(val_a_list)
    vals_b_t = torch.tensor(val_b_list)

    # 1. Empirical Discounted Return (Bootstrapped with resting value)
    disc_ret_a = torch.zeros(T)
    running_g_a = va_final
    for t in reversed(range(T)):
        running_g_a = rews_t[t] + gamma * running_g_a
        disc_ret_a[t] = running_g_a

    disc_ret_b = torch.zeros(T)
    running_g_b = vb_final
    for t in reversed(range(T)):
        running_g_b = rews_t[t] + gamma * running_g_b
        disc_ret_b[t] = running_g_b

    # Finite horizon return (no bootstrap)
    disc_ret_finite = torch.zeros(T)
    running_g_fin = 0.0
    for t in reversed(range(T)):
        running_g_fin = rews_t[t] + gamma * running_g_fin
        disc_ret_finite[t] = running_g_fin

    # 2. GAE Advantages
    # For Model A
    adv_a = torch.zeros(T)
    gae_a = 0.0
    td_a = torch.zeros(T)
    for t in reversed(range(T)):
        next_v = vals_a_t[t+1] if t < T - 1 else va_final
        delta = rews_t[t] + gamma * next_v - vals_a_t[t]
        td_a[t] = delta
        gae_a = delta + gamma * lam * gae_a
        adv_a[t] = gae_a

    # For Model B
    adv_b = torch.zeros(T)
    gae_b = 0.0
    td_b = torch.zeros(T)
    for t in reversed(range(T)):
        next_v = vals_b_t[t+1] if t < T - 1 else vb_final
        delta = rews_t[t] + gamma * next_v - vals_b_t[t]
        td_b[t] = delta
        gae_b = delta + gamma * lam * gae_b
        adv_b[t] = gae_b

    phases = {
        "prep_0_9": (0, 10),
        "launch_10_13": (10, 14),
        "ascent_14_21": (14, 22),
        "descent_22_31": (22, 32),
        "touchdown_32_39": (32, 40),
        "recovery_40_139": (40, 140),
    }

    phase_adv_a = {p: float(adv_a[s:e].mean().item()) for p, (s, e) in phases.items()}
    phase_adv_b = {p: float(adv_b[s:e].mean().item()) for p, (s, e) in phases.items()}

    mse_a_boot = float(torch.mean((disc_ret_a - vals_a_t)**2).item())
    mse_b_boot = float(torch.mean((disc_ret_b - vals_b_t)**2).item())

    mse_a_fin = float(torch.mean((disc_ret_finite - vals_a_t)**2).item())
    mse_b_fin = float(torch.mean((disc_ret_finite - vals_b_t)**2).item())

    return {
        "seed": seed,
        "fwd_dist_cm": fwd_dist_final * 100.0,
        "max_clearance_cm": clr_max * 100.0,
        "touchdown_step": touchdown_step,
        "undisc_return": float(rews_t.sum().item()),
        "disc_ret_0_boot": float(disc_ret_b[0].item()),
        "v0_model_a": float(vals_a_t[0].item()),
        "v0_model_b": float(vals_b_t[0].item()),
        "mse_model_a_boot": mse_a_boot,
        "mse_model_b_boot": mse_b_boot,
        "mse_model_a_finite": mse_a_fin,
        "mse_model_b_finite": mse_b_fin,
        "adv_mean_model_a": float(adv_a.mean().item()),
        "adv_mean_model_b": float(adv_b.mean().item()),
        "launch_adv_a": phase_adv_a["launch_10_13"],
        "launch_adv_b": phase_adv_b["launch_10_13"],
        "flight_adv_a": float(adv_a[14:32].mean().item()),
        "flight_adv_b": float(adv_b[14:32].mean().item()),
        "touchdown_adv_a": phase_adv_a["touchdown_32_39"],
        "touchdown_adv_b": phase_adv_b["touchdown_32_39"],
        "recovery_adv_a": phase_adv_a["recovery_40_139"],
        "recovery_adv_b": phase_adv_b["recovery_40_139"],
        "phase_adv_a": phase_adv_a,
        "phase_adv_b": phase_adv_b,
    }


def main():
    dev_seeds = [3001, 3002, 3003, 3004, 3005]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Model A (Champion Critic)
    champ_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    ckpt_a = torch.load(champ_path, map_location=device)
    ac_champ = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(device)
    ac_champ.load_state_dict(ckpt_a["model_state_dict"])
    ac_champ.eval()

    # Load Model B (Calibrated Critic)
    calib_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase4_6_critic_calibration/model_critic_calibrated.pt"
    ckpt_b = torch.load(calib_path, map_location=device)
    ac_calib = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(device)
    ac_calib.load_state_dict(ckpt_b["model_state_dict"])
    ac_calib.eval()

    # Note: evaluate each dev seed
    results = []
    # Since IsaacGym can only create one instance per process, we can run seeds using subprocess or in 1 batch.
    # To run all 5 seeds cleanly without recreating gym in same process, let's pass seed via sys.argv or run 5 envs!
    # Let's run a batch of 5 envs with seeds 3001..3005!
    # Let's test if we can do 5 envs:
    task_name = "myrobot_forward_jump_phase4"
    sys.argv = ["eval_batch_5", "--task", task_name, "--num_envs", "5", "--headless"]
    args = get_args()
    args.seed = 3001

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = 5
    env_cfg.seed = 3001

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    obs, _ = env.reset()
    N = 5
    T = 140

    obs_list = []
    rew_list = []
    val_a_list = []
    val_b_list = []
    fwd_distances = torch.zeros(N, device=device)
    max_clearances = torch.zeros(N, device=device)
    touchdown_steps = torch.full((N,), -1, dtype=torch.long, device=device)

    with torch.no_grad():
        for step in range(T):
            obs_list.append(obs.clone())
            va = ac_champ.evaluate(obs).squeeze(-1)
            vb = ac_calib.evaluate(obs).squeeze(-1)
            val_a_list.append(va.clone())
            val_b_list.append(vb.clone())

            act = ac_champ.act_inference(obs)
            next_obs, _, rew, dones, infos = env.step(act)
            rew_list.append(rew.clone())

            max_clearances = torch.maximum(max_clearances, env.max_flight_clearance)
            fwd_distances = torch.where(env.has_touched_down, env.final_flight_distance, fwd_distances)
            just_landed = env.one_shot_touchdown_event
            if just_landed.any():
                touchdown_steps[just_landed] = step

            obs = next_obs

        va_final = ac_champ.evaluate(obs).squeeze(-1)
        vb_final = ac_calib.evaluate(obs).squeeze(-1)

    rews = torch.stack(rew_list, dim=0) # (140, 5)
    vals_a = torch.stack(val_a_list, dim=0) # (140, 5)
    vals_b = torch.stack(val_b_list, dim=0) # (140, 5)

    gamma = 0.99
    lam = 0.95

    # Bootstrapped return
    disc_ret_a = torch.zeros_like(rews)
    disc_ret_b = torch.zeros_like(rews)
    running_a = va_final.clone()
    running_b = vb_final.clone()

    for t in reversed(range(T)):
        running_a = rews[t] + gamma * running_a
        running_b = rews[t] + gamma * running_b
        disc_ret_a[t] = running_a
        disc_ret_b[t] = running_b

    # Finite-horizon return
    disc_ret_fin = torch.zeros_like(rews)
    running_fin = torch.zeros(N, device=device)
    for t in reversed(range(T)):
        running_fin = rews[t] + gamma * running_fin
        disc_ret_fin[t] = running_fin

    # GAE advantages
    adv_a = torch.zeros_like(rews)
    adv_b = torch.zeros_like(rews)
    gae_a = torch.zeros(N, device=device)
    gae_b = torch.zeros(N, device=device)

    for t in reversed(range(T)):
        next_va = vals_a[t+1] if t < T - 1 else va_final
        delta_a = rews[t] + gamma * next_va - vals_a[t]
        gae_a = delta_a + gamma * lam * gae_a
        adv_a[t] = gae_a

        next_vb = vals_b[t+1] if t < T - 1 else vb_final
        delta_b = rews[t] + gamma * next_vb - vals_b[t]
        gae_b = delta_b + gamma * lam * gae_b
        adv_b[t] = gae_b

    phases = {
        "prep_0_9": (0, 10),
        "launch_10_13": (10, 14),
        "ascent_14_21": (14, 22),
        "descent_22_31": (22, 32),
        "touchdown_32_39": (32, 40),
        "recovery_40_139": (40, 140),
    }

    per_seed_results = []
    for i, seed in enumerate(dev_seeds):
        mse_a_boot = float(torch.mean((disc_ret_a[:, i] - vals_a[:, i])**2).item())
        mse_b_boot = float(torch.mean((disc_ret_b[:, i] - vals_b[:, i])**2).item())
        mse_a_fin = float(torch.mean((disc_ret_fin[:, i] - vals_a[:, i])**2).item())
        mse_b_fin = float(torch.mean((disc_ret_fin[:, i] - vals_b[:, i])**2).item())

        rec = {
            "seed": seed,
            "fwd_dist_cm": float(fwd_distances[i].item() * 100.0),
            "max_clearance_cm": float(max_clearances[i].item() * 100.0),
            "touchdown_step": int(touchdown_steps[i].item()),
            "undisc_return": float(rews[:, i].sum().item()),
            "v0_model_a": float(vals_a[0, i].item()),
            "v0_model_b": float(vals_b[0, i].item()),
            "ret0_boot_b": float(disc_ret_b[0, i].item()),
            "mse_model_a_boot": mse_a_boot,
            "mse_model_b_boot": mse_b_boot,
            "mse_model_a_finite": mse_a_fin,
            "mse_model_b_finite": mse_b_fin,
            "launch_adv_a": float(adv_a[10:14, i].mean().item()),
            "launch_adv_b": float(adv_b[10:14, i].mean().item()),
            "flight_adv_a": float(adv_a[14:32, i].mean().item()),
            "flight_adv_b": float(adv_b[14:32, i].mean().item()),
            "touchdown_adv_a": float(adv_a[32:40, i].mean().item()),
            "touchdown_adv_b": float(adv_b[32:40, i].mean().item()),
            "recovery_adv_a": float(adv_a[40:140, i].mean().item()),
            "recovery_adv_b": float(adv_b[40:140, i].mean().item()),
        }
        per_seed_results.append(rec)

    # Print Full Comparative Table
    print("\n" + "=" * 90)
    print("PHASE 4.6 BEFORE VS AFTER COMPARISON ON DEV SEEDS [3001..3005]")
    print("=" * 90)

    print(f"\n1. Value Prediction Error (Bootstrapped Target):")
    print(f"{'Seed':<6} | {'Fwd Dist (cm)':<14} | {'Model A MSE':<13} {'Model B MSE':<13} {'MSE Reduc %':<12} | {'V0 (A)':<8} {'V0 (B)':<8} {'Target G0':<10}")
    print("-" * 90)
    for r in per_seed_results:
        reduc = (r['mse_model_a_boot'] - r['mse_model_b_boot']) / r['mse_model_a_boot'] * 100.0
        print(f"{r['seed']:<6} | {r['fwd_dist_cm']:<14.2f} | {r['mse_model_a_boot']:<13.4f} {r['mse_model_b_boot']:<13.4f} {reduc:<12.1f}% | {r['v0_model_a']:<8.2f} {r['v0_model_b']:<8.2f} {r['ret0_boot_b']:<10.2f}")

    print(f"\n2. GAE Advantage Comparison Across Trajectory Qualities:")
    print(f"{'Seed':<6} | {'Fwd Dist (cm)':<14} | {'Launch Adv A':<14} {'Launch Adv B':<14} | {'Flight Adv A':<14} {'Flight Adv B':<14} | {'TD Adv A':<10} {'TD Adv B':<10}")
    print("-" * 90)
    for r in per_seed_results:
        print(f"{r['seed']:<6} | {r['fwd_dist_cm']:<14.2f} | {r['launch_adv_a']:<14.4f} {r['launch_adv_b']:<14.4f} | {r['flight_adv_a']:<14.4f} {r['flight_adv_b']:<14.4f} | {r['touchdown_adv_a']:<10.4f} {r['touchdown_adv_b']:<10.4f}")

    print(f"\n3. Phase-by-Phase Advantage Comparison (A vs B):")
    print(f"{'Phase':<20} | {'Model A (Champion)':<22} | {'Model B (Calibrated)':<22} | {'Adv Delta (B - A)':<20}")
    print("-" * 80)
    for pname, (s, e) in phases.items():
        mean_adv_a = float(adv_a[s:e].mean().item())
        mean_adv_b = float(adv_b[s:e].mean().item())
        delta = mean_adv_b - mean_adv_a
        print(f"{pname:<20} | {mean_adv_a:<22.4f} | {mean_adv_b:<22.4f} | {delta:<+20.4f}")

    # Distance vs Advantage Correlation Analysis
    fwds = [r['fwd_dist_cm'] for r in per_seed_results]
    launch_as = [r['launch_adv_a'] for r in per_seed_results]
    launch_bs = [r['launch_adv_b'] for r in per_seed_results]
    flight_as = [r['flight_adv_a'] for r in per_seed_results]
    flight_bs = [r['flight_adv_b'] for r in per_seed_results]

    corr_launch_a = float(np.corrcoef(fwds, launch_as)[0, 1])
    corr_launch_b = float(np.corrcoef(fwds, launch_bs)[0, 1])
    corr_flight_a = float(np.corrcoef(fwds, flight_as)[0, 1])
    corr_flight_b = float(np.corrcoef(fwds, flight_bs)[0, 1])

    print("\n" + "=" * 80)
    print("TRAJECTORY QUALITY CORRELATION AUDIT")
    print("=" * 80)
    print(f"Correlation between Forward Distance and Launch Advantage:")
    print(f"  Model A (Champion)  : r = {corr_launch_a:+.4f}")
    print(f"  Model B (Calibrated): r = {corr_launch_b:+.4f}")
    print(f"Correlation between Forward Distance and Flight Advantage:")
    print(f"  Model A (Champion)  : r = {corr_flight_a:+.4f}")
    print(f"  Model B (Calibrated): r = {corr_flight_b:+.4f}")

    # Check Seed 3004 (high distance) vs Seed 3003 (low distance)
    s3004 = next(r for r in per_seed_results if r['seed'] == 3004)
    s3003 = next(r for r in per_seed_results if r['seed'] == 3003)
    print(f"\nExtreme Trajectory Contrast (Seed 3004 [Best] vs Seed 3003 [Lowest]):")
    print(f"  Seed 3004 (Fwd = {s3004['fwd_dist_cm']:.2f} cm):")
    print(f"    Launch Adv: Model A = {s3004['launch_adv_a']:.4f} | Model B = {s3004['launch_adv_b']:.4f}")
    print(f"    Flight Adv: Model A = {s3004['flight_adv_a']:.4f} | Model B = {s3004['flight_adv_b']:.4f}")
    print(f"  Seed 3003 (Fwd = {s3003['fwd_dist_cm']:.2f} cm):")
    print(f"    Launch Adv: Model A = {s3003['launch_adv_a']:.4f} | Model B = {s3003['launch_adv_b']:.4f}")
    print(f"    Flight Adv: Model A = {s3003['flight_adv_a']:.4f} | Model B = {s3003['flight_adv_b']:.4f}")
    print(f"  Relative Advantage (Seed 3004 - Seed 3003):")
    print(f"    Launch Delta: Model A = {s3004['launch_adv_a'] - s3003['launch_adv_a']:+.4f} | Model B = {s3004['launch_adv_b'] - s3003['launch_adv_b']:+.4f}")
    print(f"    Flight Delta: Model A = {s3004['flight_adv_a'] - s3003['flight_adv_a']:+.4f} | Model B = {s3004['flight_adv_b'] - s3003['flight_adv_b']:+.4f}")

    out_file = os.path.join(workspace_root, "evaluation_assets", "phase4_6_critic_calibration", "dev_seeds_comparison.json")
    save_data = {
        "per_seed_results": per_seed_results,
        "correlations": {
            "launch_adv_model_a": corr_launch_a,
            "launch_adv_model_b": corr_launch_b,
            "flight_adv_model_a": corr_flight_a,
            "flight_adv_model_b": corr_flight_b,
        },
        "phase_breakdown": {
            pname: {
                "adv_a": float(adv_a[s:e].mean().item()),
                "adv_b": float(adv_b[s:e].mean().item()),
            } for pname, (s, e) in phases.items()
        }
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved comparison results to: {out_file}")


if __name__ == "__main__":
    main()
