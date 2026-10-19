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
from rsl_rl.algorithms import PPO
from legged_gym.envs import task_registry
from legged_gym.utils import get_args

def audit_update1():
    task_name = "myrobot_forward_jump_phase4"
    sys.argv = ["audit_u1", "--task", task_name, "--num_envs", "64"]
    args = get_args()
    args.headless = True
    args.seed = 1

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = 64
    env_cfg.seed = 1

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    champ_path = "evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    ckpt = torch.load(champ_path, map_location=env.device)

    # Make PPO runner components
    actor_critic = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    actor_critic.load_state_dict(ckpt["model_state_dict"])

    from legged_gym.utils import class_to_dict
    alg_dict = class_to_dict(train_cfg.algorithm)
    ppo = PPO(actor_critic, device=env.device, **alg_dict)
    ppo.init_storage(64, 140, [35], [None], [6])

    obs, _ = env.reset()

    # Step 1: Collect exact 140-step rollout using Champion policy
    print("Collecting 140 steps rollout with Champion policy...")
    with torch.inference_mode():
        for i in range(140):
            actions = ppo.act(obs, obs)
            next_obs, _, rewards, dones, infos = env.step(actions)
            ppo.process_env_step(rewards, dones, infos)
            obs = next_obs

    # Step 2: Compute returns
    obs = obs.clone()
    last_values = ppo.actor_critic.evaluate(obs).detach()
    ppo.compute_returns(obs)

    # Step 3: Inspect RolloutStorage contents
    storage = ppo.storage
    print("\n--- RolloutStorage Diagnostics ---")
    print(f"Rewards: mean={storage.rewards.mean().item():.4f}, min={storage.rewards.min().item():.4f}, max={storage.rewards.max().item():.4f}")
    print(f"Values : mean={storage.values.mean().item():.4f}, min={storage.values.min().item():.4f}, max={storage.values.max().item():.4f}")
    print(f"Returns: mean={storage.returns.mean().item():.4f}, min={storage.returns.min().item():.4f}, max={storage.returns.max().item():.4f}")
    print(f"Raw Adv: mean={storage.advantages.mean().item():.4f}, std={storage.advantages.std().item():.4f}, min={storage.advantages.min().item():.4f}, max={storage.advantages.max().item():.4f}")
    print(f"Dones sum: {storage.dones.sum().item()} (at step 139: {storage.dones[139].sum().item()})")

    # Step 4: Step through generator mini-batch 1 and mini-batch 2
    generator = storage.mini_batch_generator(4, 5) # 4 mini-batches, 5 epochs
    batch_idx = 0

    print("\n--- Step-by-Step Mini-Batch Audit ---")
    for b in generator:
        obs_b, critic_obs_b, act_b, target_val_b, adv_b, ret_b, old_log_p_b, old_mu_b, old_sig_b, _, _ = b
        batch_idx += 1

        ppo.actor_critic.act(obs_b)
        log_p = ppo.actor_critic.get_actions_log_prob(act_b)
        val = ppo.actor_critic.evaluate(critic_obs_b)
        mu = ppo.actor_critic.action_mean
        sig = ppo.actor_critic.action_std

        # KL
        kl = torch.sum(torch.log(sig / old_sig_b + 1e-5) + (old_sig_b**2 + (old_mu_b - mu)**2) / (2.0 * sig**2) - 0.5, axis=-1)
        kl_m = kl.mean().item()

        # Ratio
        ratio = torch.exp(log_p - old_log_p_b.squeeze(-1))
        clip_frac = torch.mean((torch.abs(ratio - 1.0) > 0.2).float()).item()

        surr = -adv_b.squeeze(-1) * ratio
        surr_clip = -adv_b.squeeze(-1) * torch.clamp(ratio, 0.8, 1.2)
        surr_loss = torch.max(surr, surr_clip).mean().item()

        val_loss = (ret_b - val).pow(2).mean().item()

        print(f"Batch {batch_idx}:")
        print(f"  KL mean={kl_m:.6f}, max={kl.max().item():.6f}, p95={torch.quantile(kl.float(), 0.95).item():.6f}")
        print(f"  Ratio: mean={ratio.mean().item():.6f}, min={ratio.min().item():.6f}, max={ratio.max().item():.6f}, clip_frac={clip_frac:.4f}")
        print(f"  Adv_batch: mean={adv_b.mean().item():.4f}, std={adv_b.std().item():.4f}, min={adv_b.min().item():.4f}, max={adv_b.max().item():.4f}")
        print(f"  Surr loss={surr_loss:.6f}, Val loss={val_loss:.6f}")

        # Check early stopping condition
        if kl_m > 0.04:
            print(f"  --> EARLY STOPPING WOULD TRIGGER HERE! (KL {kl_m:.6f} > 0.04)")
            break

        # Execute gradient step on batch 1
        loss = surr_loss + 1.0 * val_loss - 0.005 * ppo.actor_critic.entropy.mean()
        ppo.optimizer.zero_grad()
        loss_tensor = torch.max(surr, surr_clip).mean() + 1.0 * (val - ret_b).pow(2).mean() - 0.005 * ppo.actor_critic.entropy.mean()
        loss_tensor.backward()

        actor_params = [p for p in ppo.actor_critic.actor.parameters() if p.grad is not None]
        critic_params = [p for p in ppo.actor_critic.critic.parameters() if p.grad is not None]
        actor_gn = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in actor_params]), 2).item()
        critic_gn = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in critic_params]), 2).item()

        print(f"  Gradients: Actor grad norm={actor_gn:.4f}, Critic grad norm={critic_gn:.4f}")
        torch.nn.utils.clip_grad_norm_(ppo.actor_critic.parameters(), 1.0)
        ppo.optimizer.step()
        print(f"  Optimizer step executed.")

if __name__ == "__main__":
    audit_update1()
