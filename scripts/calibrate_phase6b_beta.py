"""
Phase 6B-Pilot: Beta Calibration Script for Reference-Policy KL Regularization.
Measures:
1. PPO actor loss & gradient norm (||g_actor_ppo||).
2. Reference-Policy KL value & gradient norm (||g_actor_ref_kl||).
3. Gradient norm ratio (||beta * g_ref_kl|| / ||g_ppo||) across candidate beta values.
4. Selects beta_weak and beta_strong based on empirical gradient scaling.
"""

import isaacgym
import os
import sys
import torch
import numpy as np
import copy
from torch.distributions import Normal, kl_divergence

WORKSPACE_ROOT = "/home/xy/IsaacTu"
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "rsl_rl"))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "legged_gym"))

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def calibrate_beta():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.seed = 1
    args.num_envs = 2048
    args.rl_device = 'cuda:0'
    args.sim_device = 'cuda:0'
    args.warm_start = True
    args.warm_start_optimizer = False
    args.learning_rate = 7.59375e-5

    print("=" * 80)
    print("PHASE 6B-PILOT: REFERENCE-POLICY KL REGULARIZATION BETA CALIBRATION")
    print("=" * 80)

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    ppo = ppo_runner.alg
    ac = ppo.actor_critic
    device = ppo.device

    # Load frozen Reference Actor from model_25.pt
    ref_path = "legged_gym/logs/myrobot/Oct10_15-48-32_turning_height_stage5_3_formal/model_25.pt"
    ref_ckpt = torch.load(ref_path, map_location=device)['model_state_dict']
    ref_actor = copy.deepcopy(ac.actor)
    ref_actor_sd = {k.replace('actor.', ''): v for k, v in ref_ckpt.items() if k.startswith('actor.')}
    ref_actor.load_state_dict(ref_actor_sd)
    for p in ref_actor.parameters():
        p.requires_grad = False
    ref_actor.eval()
    ref_std = ref_ckpt['std'].clone().detach().to(device)
    ref_std.requires_grad = False

    # 1. Collect 1 Rollout (24 steps, 2048 envs = 49,152 transitions)
    print("\n[1. Collecting Rollout 1 (49,152 transitions)...]")
    obs = env.get_observations()
    priv_obs = env.get_privileged_observations()
    critic_obs = priv_obs if priv_obs is not None else obs
    obs, critic_obs = obs.to(device), critic_obs.to(device)

    with torch.inference_mode():
        for i in range(ppo_runner.num_steps_per_env):
            actions = ppo.act(obs, critic_obs)
            obs, priv_obs, rewards, dones, infos = env.step(actions)
            critic_obs = priv_obs if priv_obs is not None else obs
            obs, critic_obs, rewards, dones = obs.to(device), critic_obs.to(device), rewards.to(device), dones.to(device)
            ppo.process_env_step(rewards, dones, infos)

        ppo.compute_returns(critic_obs)

    storage = ppo.storage
    generator = storage.mini_batch_generator(ppo.num_mini_batches, ppo.num_learning_epochs)

    # Mini-batch 1
    batch_1 = next(generator)
    obs_b1, c_obs_b1, act_b1, tgt_val_b1, adv_b1, ret_b1, old_logp_b1, old_mu_b1, old_sigma_b1, _, _ = batch_1

    print("\n[2. Step 1: Performing 1 standard PPO update to establish realistic policy drift...]")
    # Loss for mini-batch 1
    ac.act(obs_b1)
    act_logp_b1 = ac.get_actions_log_prob(act_b1)
    val_b1 = ac.evaluate(c_obs_b1)
    ent_b1 = ac.entropy

    ratio_b1 = torch.exp(act_logp_b1 - torch.squeeze(old_logp_b1))
    surr_b1 = -torch.squeeze(adv_b1) * ratio_b1
    surr_clip_b1 = -torch.squeeze(adv_b1) * torch.clamp(ratio_b1, 1.0 - ppo.clip_param, 1.0 + ppo.clip_param)
    surr_loss_b1 = torch.max(surr_b1, surr_clip_b1).mean()

    val_clip_b1 = tgt_val_b1 + (val_b1 - tgt_val_b1).clamp(-ppo.clip_param, ppo.clip_param)
    val_losses_b1 = torch.max((val_b1 - ret_b1).pow(2), (val_clip_b1 - ret_b1).pow(2)).mean()
    total_loss_b1 = surr_loss_b1 + ppo.value_loss_coef * val_losses_b1 - ppo.entropy_coef * ent_b1.mean()

    ppo.optimizer.zero_grad()
    total_loss_b1.backward()
    torch.nn.utils.clip_grad_norm_(ac.parameters(), ppo.max_grad_norm)
    ppo.optimizer.step()

    # Mini-batch 2 (now policy has drifted by 1 gradient step)
    batch_2 = next(generator)
    obs_b2, c_obs_b2, act_b2, tgt_val_b2, adv_b2, ret_b2, old_logp_b2, old_mu_b2, old_sigma_b2, _, _ = batch_2

    print("\n[3. Mini-batch 2: Measuring PPO Actor Gradient vs Reference KL Gradient...]")
    ac.act(obs_b2)
    mu_b2 = ac.action_mean
    sigma_b2 = ac.action_std
    act_logp_b2 = ac.get_actions_log_prob(act_b2)
    ent_b2 = ac.entropy

    ratio_b2 = torch.exp(act_logp_b2 - torch.squeeze(old_logp_b2))
    surr_b2 = -torch.squeeze(adv_b2) * ratio_b2
    surr_clip_b2 = -torch.squeeze(adv_b2) * torch.clamp(ratio_b2, 1.0 - ppo.clip_param, 1.0 + ppo.clip_param)
    surr_loss_b2 = torch.max(surr_b2, surr_clip_b2).mean()
    actor_ppo_loss = surr_loss_b2 - ppo.entropy_coef * ent_b2.mean()

    # Reference policy forward on same obs
    with torch.no_grad():
        mu_ref_b2 = ref_actor(obs_b2)
        ref_sigma_expanded = ref_std.repeat(obs_b2.shape[0], 1)

    # Reference KL: KL(ref || curr)
    p_ref = Normal(mu_ref_b2, ref_sigma_expanded)
    p_curr = Normal(mu_b2, sigma_b2)
    ref_kl_elementwise = kl_divergence(p_ref, p_curr).sum(dim=-1)
    ref_kl_loss = ref_kl_elementwise.mean()

    # Separate Actor gradients
    actor_params = list(ac.actor.parameters())

    # Gradient of PPO loss w.r.t actor params
    g_actor_ppo = torch.autograd.grad(actor_ppo_loss, actor_params, retain_graph=True)
    norm_actor_ppo = torch.norm(torch.stack([torch.norm(g, 2) for g in g_actor_ppo]), 2).item()

    # Gradient of Ref KL loss w.r.t actor params
    g_actor_ref_kl = torch.autograd.grad(ref_kl_loss, actor_params, retain_graph=True)
    norm_actor_ref_kl = torch.norm(torch.stack([torch.norm(g, 2) for g in g_actor_ref_kl]), 2).item()

    # Inner product & cosine similarity between PPO gradient and Ref KL gradient
    flat_g_ppo = torch.cat([g.flatten() for g in g_actor_ppo])
    flat_g_ref = torch.cat([g.flatten() for g in g_actor_ref_kl])
    cos_sim = torch.dot(flat_g_ppo, flat_g_ref).item() / (norm_actor_ppo * norm_actor_ref_kl + 1e-8)

    print(f"Actor PPO Loss        : {actor_ppo_loss.item():.6f}")
    print(f"Reference KL Mean     : {ref_kl_loss.item():.6f}")
    print(f"Reference KL p95      : {torch.quantile(ref_kl_elementwise, 0.95).item():.6f}")
    print(f"Reference KL Max      : {torch.max(ref_kl_elementwise).item():.6f}")
    print(f"||g_actor_ppo||_2     : {norm_actor_ppo:.4f}")
    print(f"||g_actor_ref_kl||_2  : {norm_actor_ref_kl:.4f}")
    print(f"Gradient Cosine Sim   : {cos_sim:.4f}")

    # Evaluate beta sweep
    candidate_betas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
    print("\n" + "=" * 80)
    print(f"{'Beta':<10} | {'||beta * g_ref||_2':<20} | {'Ratio to ||g_ppo||':<20} | {'Reg Loss Contribution':<22}")
    print("-" * 80)

    for beta in candidate_betas:
        norm_reg = beta * norm_actor_ref_kl
        ratio = norm_reg / (norm_actor_ppo + 1e-8) * 100.0
        reg_loss = beta * ref_kl_loss.item()
        print(f"{beta:<10.3f} | {norm_reg:<20.4f} | {ratio:>18.2f}% | {reg_loss:<22.6f}")

    print("=" * 80)

if __name__ == '__main__':
    calibrate_beta()
