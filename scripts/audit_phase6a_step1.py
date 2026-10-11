"""
Phase 6A-R Audit Script: Replay exact Iteration 1 rollout and mini-batch updates.
Investigates:
1. Warm start state (weights, std, optimizer initial state, lr).
2. Advantage and returns distributions on rollout 1.
3. Separate Actor vs Critic gradient norms on mini-batch 1.
4. Exact sample-by-sample KL distribution on mini-batch 2 after 1 optimizer step.
5. Per-action dimension breakdown of KL.
6. Outlier analysis: who caused KL max and why?
"""

import isaacgym
import os
import sys
import torch
import numpy as np

WORKSPACE_ROOT = "/home/xy/IsaacTu"
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "rsl_rl"))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "legged_gym"))

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args

def audit_step1():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.seed = 1
    args.num_envs = 2048
    args.rl_device = 'cuda:0'
    args.sim_device = 'cuda:0'

    print("=" * 80)
    print("PHASE 6A-R AUDIT: REPLAYING ITERATION 1 ROLLOUT & UPDATE 1")
    print("=" * 80)

    # 1. Environment and runner creation
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    ppo = ppo_runner.alg
    ac = ppo.actor_critic
    opt = ppo.optimizer

    print(f"\n[1. Warm Start Inspection]")
    print(f"Actor parameters count : {sum(p.numel() for p in ac.actor.parameters())}")
    print(f"Critic parameters count: {sum(p.numel() for p in ac.critic.parameters())}")
    print(f"Action noise std       : {ac.std.detach().cpu().numpy()}")
    print(f"Optimizer initial LR   : {opt.param_groups[0]['lr']}")
    print(f"Optimizer state dict keys count: {len(opt.state_dict()['state'])} (should be 0 for fresh Adam)")

    # 2. Replay Rollout 1 (24 steps)
    print(f"\n[2. Collecting Rollout 1 (24 steps, 2048 envs = 49,152 transitions)...]")
    obs = env.get_observations()
    priv_obs = env.get_privileged_observations()
    critic_obs = priv_obs if priv_obs is not None else obs
    obs, critic_obs = obs.to(ppo.device), critic_obs.to(ppo.device)

    with torch.inference_mode():
        for i in range(ppo_runner.num_steps_per_env):
            actions = ppo.act(obs, critic_obs)
            obs, priv_obs, rewards, dones, infos = env.step(actions)
            critic_obs = priv_obs if priv_obs is not None else obs
            obs, critic_obs, rewards, dones = obs.to(ppo.device), critic_obs.to(ppo.device), rewards.to(ppo.device), dones.to(ppo.device)
            ppo.process_env_step(rewards, dones, infos)

        ppo.compute_returns(critic_obs)

    storage = ppo.storage
    adv = storage.advantages.flatten().cpu().numpy()
    ret = storage.returns.flatten().cpu().numpy()
    val = storage.values.flatten().cpu().numpy()
    rew = storage.rewards.flatten().cpu().numpy()
    dones_all = storage.dones.flatten().cpu().numpy()

    print(f"Transitions collected: {len(adv)}")
    print(f"Rewards    : mean={rew.mean():.4f}, std={rew.std():.4f}, min={rew.min():.4f}, max={rew.max():.4f}")
    print(f"Values     : mean={val.mean():.4f}, std={val.std():.4f}, min={val.min():.4f}, max={val.max():.4f}")
    print(f"Returns    : mean={ret.mean():.4f}, std={ret.std():.4f}, min={ret.min():.4f}, max={ret.max():.4f}")
    print(f"Advantages : mean={adv.mean():.4f}, std={adv.std():.4f}, min={adv.min():.4f}, max={adv.max():.4f}")
    print(f"Terminations (dones count): {np.sum(dones_all)} / {len(dones_all)} ({np.mean(dones_all)*100:.2f}%)")

    # 3. Mini-batch generation
    generator = storage.mini_batch_generator(ppo.num_mini_batches, ppo.num_learning_epochs)

    # Mini-batch 1
    batch_1 = next(generator)
    obs_b1, c_obs_b1, act_b1, tgt_val_b1, adv_b1, ret_b1, old_logp_b1, old_mu_b1, old_sigma_b1, _, _ = batch_1

    print(f"\n[3. Mini-batch 1 (Size: {obs_b1.shape[0]}) Before Optimizer Step 1]")
    with torch.no_grad():
        ppo.actor_critic.act(obs_b1)
        mu_b1 = ppo.actor_critic.action_mean
        sigma_b1 = ppo.actor_critic.action_std
        diff_mu_b1 = torch.norm(mu_b1 - old_mu_b1).item()
        print(f"||mu_b1 - old_mu_b1|| before any update: {diff_mu_b1:.8f} (should be ~0)")

    # Compute loss for Mini-batch 1
    ppo.actor_critic.act(obs_b1)
    act_logp_b1 = ppo.actor_critic.get_actions_log_prob(act_b1)
    val_b1 = ppo.actor_critic.evaluate(c_obs_b1)
    ent_b1 = ppo.actor_critic.entropy

    ratio_b1 = torch.exp(act_logp_b1 - torch.squeeze(old_logp_b1))
    surr_b1 = -torch.squeeze(adv_b1) * ratio_b1
    surr_clip_b1 = -torch.squeeze(adv_b1) * torch.clamp(ratio_b1, 1.0 - ppo.clip_param, 1.0 + ppo.clip_param)
    surr_loss_b1 = torch.max(surr_b1, surr_clip_b1).mean()

    val_clip_b1 = tgt_val_b1 + (val_b1 - tgt_val_b1).clamp(-ppo.clip_param, ppo.clip_param)
    val_losses_b1 = torch.max((val_b1 - ret_b1).pow(2), (val_clip_b1 - ret_b1).pow(2)).mean()

    total_loss_b1 = surr_loss_b1 + ppo.value_loss_coef * val_losses_b1 - ppo.entropy_coef * ent_b1.mean()

    opt.zero_grad()
    total_loss_b1.backward()

    # Separate gradient norm analysis
    actor_params = list(ac.actor.parameters())
    critic_params = list(ac.critic.parameters())
    std_param = [ac.std]

    actor_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in actor_params if p.grad is not None]), 2).item()
    critic_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in critic_params if p.grad is not None]), 2).item()
    std_grad_norm = torch.norm(ac.std.grad.detach(), 2).item()
    total_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in ac.parameters() if p.grad is not None]), 2).item()

    print(f"Surrogate Loss : {surr_loss_b1.item():.4f}")
    print(f"Value Loss     : {val_losses_b1.item():.4f}")
    print(f"Actor Grad Norm : {actor_grad_norm:.4f}")
    print(f"Critic Grad Norm: {critic_grad_norm:.4f}")
    print(f"Std Grad Norm   : {std_grad_norm:.4f}")
    print(f"Total Grad Norm : {total_grad_norm:.4f}")

    # Clip and step
    torch.nn.utils.clip_grad_norm_(ac.parameters(), ppo.max_grad_norm)
    opt.step()
    print(">>> Executed optimizer.step() for Mini-batch 1.")

    # 4. Now evaluate Mini-batch 2
    batch_2 = next(generator)
    obs_b2, c_obs_b2, act_b2, tgt_val_b2, adv_b2, ret_b2, old_logp_b2, old_mu_b2, old_sigma_b2, _, _ = batch_2

    print(f"\n[4. Mini-batch 2 (Size: {obs_b2.shape[0]}) Directly After 1 Optimizer Step]")
    with torch.no_grad():
        ac.act(obs_b2)
        mu_b2 = ac.action_mean
        sigma_b2 = ac.action_std

        # Exact formula from ppo.py
        kl_elementwise = torch.log(sigma_b2 / old_sigma_b2 + 1.e-5) + (torch.square(old_sigma_b2) + torch.square(old_mu_b2 - mu_b2)) / (2.0 * torch.square(sigma_b2)) - 0.5
        kl_samples = torch.sum(kl_elementwise, dim=-1)

        kl_np = kl_samples.cpu().numpy()
        print(f"KL Mean   : {kl_np.mean():.6f}")
        print(f"KL Median : {np.median(kl_np):.6f}")
        print(f"KL p90    : {np.percentile(kl_np, 90):.6f}")
        print(f"KL p95    : {np.percentile(kl_np, 95):.6f}")
        print(f"KL p99    : {np.percentile(kl_np, 99):.6f}")
        print(f"KL Max    : {kl_np.max():.6f}")
        print(f"Fraction of samples with KL > 0.02: {np.mean(kl_np > 0.02)*100:.2f}%")
        print(f"Fraction of samples with KL > 0.10: {np.mean(kl_np > 0.10)*100:.2f}%")
        print(f"Fraction of samples with KL > 1.00: {np.mean(kl_np > 1.00)*100:.2f}%")

        # Per-action dimension breakdown
        per_dim_kl = kl_elementwise.mean(dim=0).cpu().numpy()
        print(f"\nPer-action dimension mean KL:")
        dof_names = ['L_hip', 'L_knee', 'L_wheel', 'R_hip', 'R_knee', 'R_wheel']
        for name, val_k, s_val in zip(dof_names, per_dim_kl, sigma_b2[0].cpu().numpy()):
            print(f"  {name:10s} (sigma={s_val:.4f}): KL_mean = {val_k:.6f}")

        # Outlier sample investigation
        max_idx = int(np.argmax(kl_np))
        print(f"\nOutlier Sample #{max_idx} (Max KL = {kl_np[max_idx]:.4f}):")
        obs_max = obs_b2[max_idx].cpu().numpy()
        old_mu_max = old_mu_b2[max_idx].cpu().numpy()
        new_mu_max = mu_b2[max_idx].cpu().numpy()
        diff_mu_max = new_mu_max - old_mu_max
        adv_max = adv_b2[max_idx].item()
        print(f"  Advantage      : {adv_max:.4f}")
        print(f"  Old Mu         : {np.round(old_mu_max, 4)}")
        print(f"  New Mu         : {np.round(new_mu_max, 4)}")
        print(f"  Delta Mu       : {np.round(diff_mu_max, 4)}")
        print(f"  Action Std     : {np.round(sigma_b2[0].cpu().numpy(), 4)}")
        print(f"  Per-dim KL     : {np.round(kl_elementwise[max_idx].cpu().numpy(), 4)}")
        print(f"  Obs root_height: {obs_max[2]:.4f} (approx)")
        print(f"  Obs min/max/norm: min={obs_max.min():.4f}, max={obs_max.max():.4f}, norm={np.linalg.norm(obs_max):.4f}")

if __name__ == '__main__':
    audit_step1()
