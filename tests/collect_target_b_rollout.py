#!/usr/bin/env python3
# ==============================================================================
# Phase 4.7: Target B Rollout Collector
# Collects rollouts using frozen Champion Actor and computes exact Target B returns:
#   G_139 = r_139 + gamma * V_champ(s_140_pre)
#   G_t   = r_t + gamma * G_{t+1}
# Uses the updated MyRobotForwardJumpPhase4 with verified post_physics_step.
# ==============================================================================

import os
import sys
import argparse
import isaacgym
import torch
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args


def collect_rollout(seed, num_envs, out_path):
    task_name = "myrobot_forward_jump_phase4"
    sys.argv = ["collector", "--task", task_name, "--num_envs", str(num_envs), "--headless"]
    args = get_args()
    args.seed = seed

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.domain_rand.push_robots = False
    env_cfg.noise.add_noise = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # Load frozen Champion ActorCritic
    champ_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    ac = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(torch.load(champ_path, map_location=env.device)["model_state_dict"])
    ac.eval()

    obs, _ = env.reset()

    T = 140
    N = num_envs
    gamma = 0.99
    lam = 0.95

    obs_list = []
    val_champ_list = []
    act_list = []
    rew_list = []
    done_list = []
    terminal_obs = None
    post_reset_obs = None

    fwd_distances = torch.zeros(N, device=env.device)
    max_clearances = torch.zeros(N, device=env.device)
    touchdown_steps = torch.full((N,), -1, dtype=torch.long, device=env.device)

    with torch.no_grad():
        for step in range(T):
            obs_list.append(obs.clone())
            val = ac.evaluate(obs).squeeze(-1)
            val_champ_list.append(val.clone())

            act = ac.act_inference(obs)
            act_list.append(act.clone())

            next_obs, _, rew, dones, infos = env.step(act)
            rew_list.append(rew.clone())
            done_list.append(dones.clone())

            max_clearances = torch.maximum(max_clearances, env.max_flight_clearance)
            fwd_distances = torch.where(env.has_touched_down, env.final_flight_distance, fwd_distances)
            just_landed = env.one_shot_touchdown_event
            if just_landed.any():
                touchdown_steps[just_landed] = step

            if step == T - 1:
                assert "terminal_observation" in infos, "CRITICAL: infos missing terminal_observation at step 139!"
                assert "time_outs" in infos, "CRITICAL: infos missing time_outs at step 139!"
                terminal_obs = infos["terminal_observation"].clone()
                post_reset_obs = next_obs.clone()

            obs = next_obs

        # Evaluate terminal state values
        v_terminal_pre = ac.evaluate(terminal_obs).squeeze(-1)
        v_post_reset = ac.evaluate(post_reset_obs).squeeze(-1)

    obs_all = torch.stack(obs_list, dim=0)       # (140, N, 35)
    vals_champ = torch.stack(val_champ_list, dim=0) # (140, N)
    acts_all = torch.stack(act_list, dim=0)      # (140, N, 6)
    rews_all = torch.stack(rew_list, dim=0)      # (140, N)
    dones_all = torch.stack(done_list, dim=0)    # (140, N)

    # Compute Target B: Pre-Reset Bootstrap
    targets_b = torch.zeros_like(rews_all)
    running_g_b = v_terminal_pre.clone()
    for t in reversed(range(T)):
        running_g_b = rews_all[t] + gamma * running_g_b
        targets_b[t] = running_g_b

    # Compute Target C: Post-Reset Bootstrap (Phase 4.6 comparison baseline)
    targets_c = torch.zeros_like(rews_all)
    running_g_c = v_post_reset.clone()
    for t in reversed(range(T)):
        running_g_c = rews_all[t] + gamma * running_g_c
        targets_c[t] = running_g_c

    # Compute GAE Advantages for Target B
    adv_b = torch.zeros_like(rews_all)
    gae_b = torch.zeros(N, device=env.device)
    td_b = torch.zeros_like(rews_all)
    for t in reversed(range(T)):
        next_v = vals_champ[t+1] if t < T - 1 else v_terminal_pre
        delta = rews_all[t] + gamma * next_v - vals_champ[t]
        td_b[t] = delta
        gae_b = delta + gamma * lam * gae_b
        adv_b[t] = gae_b

    save_dict = {
        "seed": seed,
        "num_envs": num_envs,
        "obs": obs_all.cpu(),
        "actions": acts_all.cpu(),
        "rewards": rews_all.cpu(),
        "dones": dones_all.cpu(),
        "values_champ": vals_champ.cpu(),
        "targets_b": targets_b.cpu(),
        "targets_c": targets_c.cpu(),
        "adv_b": adv_b.cpu(),
        "td_b": td_b.cpu(),
        "v_terminal_pre": v_terminal_pre.cpu(),
        "v_post_reset": v_post_reset.cpu(),
        "terminal_obs": terminal_obs.cpu(),
        "post_reset_obs": post_reset_obs.cpu(),
        "fwd_distances": fwd_distances.cpu(),
        "max_clearances": max_clearances.cpu(),
        "touchdown_steps": touchdown_steps.cpu(),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(save_dict, out_path)
    print(f"\n[Rollout Collection Complete]")
    print(f"  Seed {seed} ({num_envs} envs x 140 steps = {num_envs * 140} transitions)")
    print(f"  V(s_140_pre) mean : {v_terminal_pre.mean().item():.4f}")
    print(f"  V(s_0_post)  mean : {v_post_reset.mean().item():.4f}")
    print(f"  Delta (Post - Pre): {v_post_reset.mean().item() - v_terminal_pre.mean().item():+.4f}")
    print(f"  Target B G_0 mean : {targets_b[0].mean().item():.4f}")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    collect_rollout(args.seed, args.num_envs, args.out)
