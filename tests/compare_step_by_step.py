#!/usr/bin/env python3
import os
import sys
import argparse
import json

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
import torch
from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.envs.myrobot.myrobot_dual_env import compute_exact_cylinder_clearance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--cmd_y", type=float, default=None)
    cli_args, _ = parser.parse_known_args()
    task_name = cli_args.task

    sys.argv = ["trace", "--task", task_name, "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = 42

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    if cli_args.cmd_y is not None:
        env_cfg.commands.ranges.lin_vel_y = [cli_args.cmd_y, cli_args.cmd_y]
    env_cfg.env.num_envs = 1
    env_cfg.seed = 42
    env_cfg.env.episode_length_s = 2.0
    env_cfg.noise.add_noise = False
    env_cfg.env.enable_camera_sensors = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    model_path = os.path.join(workspace_root, "evaluation_assets", "phase1_forward_jump", "models", "model_30_yaw_aligned.pt")
    ac = ActorCritic(
        num_actor_obs=env_cfg.env.num_observations,
        num_critic_obs=env_cfg.env.num_observations,
        num_actions=env_cfg.env.num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0
    ).to(env.device)
    ckpt = torch.load(model_path, map_location=env.device)
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    policy = ac.act_inference

    obs, _ = env.reset()
    trace = []

    for step in range(35):
        clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(env)
        root_z = env.root_states[0, 2].item()
        root_vz = env.root_states[0, 9].item()
        tau_knee_L = env.torques[0, env.left_knee_idx].item()
        tau_knee_R = env.torques[0, env.right_knee_idx].item()
        phase = env.jump_phase[0].item()

        with torch.no_grad():
            actions = policy(obs)

        trace.append({
            "step": step,
            "phase": round(phase, 4),
            "root_z": round(root_z, 4),
            "root_vz": round(root_vz, 3),
            "clr_min_cm": round(clr_min[0].item() * 100.0, 2),
            "tau_knee_L": round(tau_knee_L, 2),
            "tau_knee_R": round(tau_knee_R, 2),
            "action_knee_L": round(actions[0, 1].item(), 3),
            "action_knee_R": round(actions[0, 4].item(), 3),
            "obs_slot9_cmd_fwd": round(obs[0, 9].item(), 3),
            "obs_slot11_h": round(obs[0, 11].item(), 3),
            "obs_slot12_phi": round(obs[0, 12].item(), 3),
        })

        obs, _, _, _, _ = env.step(actions)

    with open(cli_args.out, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"Saved {task_name} trace to {cli_args.out}")

if __name__ == "__main__":
    main()
