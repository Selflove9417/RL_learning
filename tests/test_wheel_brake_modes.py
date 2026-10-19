#!/usr/bin/env python3
import os
import sys
import json
import argparse
import isaacgym
from isaacgym import gymapi, gymtorch
import torch
from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.envs.myrobot.myrobot_dual_env import compute_exact_cylinder_clearance

def run_single(seed, brake_k=0.0):
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["eval_diag", "--task", task_name, "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = seed

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.normalization.clip_actions = 100.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.env.episode_length_s = 2.8
    env_cfg.noise.add_noise = False
    env_cfg.env.enable_camera_sensors = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    ac = ActorCritic(
        num_actor_obs=env_cfg.env.num_observations,
        num_critic_obs=env_cfg.env.num_observations,
        num_actions=env_cfg.env.num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0
    ).to(env.device)
    model_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    ckpt = torch.load(model_path, map_location=env.device)
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    policy = ac.act_inference

    obs, _ = env.reset()
    total_steps = int(2.8 / env.dt)

    takeoff_step = None
    touchdown_step = None
    took_off = False
    landed = False
    max_clr = 0.0
    to_com_y = 0.0
    td_com_y = 0.0
    to_vy = 0.0
    to_vz = 0.0
    s5_vy = 0.0
    prep_max_w = 0.0
    prep_max_vy = 0.0

    for step in range(total_steps):
        true_com_pos, true_com_vel = env._compute_true_com()
        base_lin_vel = env.base_lin_vel[0].cpu().numpy()
        w_L = abs(env.dof_vel[0, env.left_wheel_idx].item())
        w_R = abs(env.dof_vel[0, env.right_wheel_idx].item())
        clr_min, _, _ = compute_exact_cylinder_clearance(env)
        clr_min_cm = clr_min[0].item() * 100.0

        is_air = bool(env.is_airborne[0].item())
        has_took = bool(env.has_taken_off[0].item())

        if step <= 5 and not took_off:
            prep_max_w = max(prep_max_w, max(w_L, w_R))
            prep_max_vy = max(prep_max_vy, abs(base_lin_vel[1]))
        if step == 5:
            s5_vy = true_com_vel[0, 1].item()

        if is_air and has_took:
            max_clr = max(max_clr, clr_min_cm)

        if is_air and has_took and not took_off:
            took_off = True
            takeoff_step = step
            to_com_y = true_com_pos[0, 1].item()
            to_vy = true_com_vel[0, 1].item()
            to_vz = true_com_vel[0, 2].item()

        if took_off and (not is_air) and (not landed):
            landed = True
            touchdown_step = step
            td_com_y = true_com_pos[0, 1].item()

        with torch.no_grad():
            actions = policy(obs)

        # DIAGNOSTIC INTERVENTION ONLY DURING STEP 0 TO 5
        if step <= 5 and brake_k > 0.0:
            actions_mod = actions.clone()
            w_L_vel = env.dof_vel[0, env.left_wheel_idx].item()
            w_R_vel = env.dof_vel[0, env.right_wheel_idx].item()
            actions_mod[0, env.left_wheel_idx] -= brake_k * w_L_vel
            actions_mod[0, env.right_wheel_idx] -= brake_k * w_R_vel
            obs, _, _, _, _ = env.step(actions_mod)
        else:
            obs, _, _, _, _ = env.step(actions)

    fwd_dist = (td_com_y - to_com_y) * 100.0 if took_off and landed else 0.0
    res = {
        "seed": int(seed),
        "brake_k": float(brake_k),
        "prep_max_w": round(float(prep_max_w), 1),
        "prep_max_vy": round(float(prep_max_vy), 3),
        "s5_vy": round(float(s5_vy), 3),
        "to_vy": round(float(to_vy), 3),
        "to_vz": round(float(to_vz), 3),
        "delta_vy": round(float(to_vy - s5_vy), 3),
        "max_clr": round(float(max_clr), 2),
        "fwd_dist": round(float(fwd_dist), 2)
    }
    print(json.dumps(res))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--k", type=float, default=0.0)
    args = parser.parse_args()
    run_single(args.seed, args.k)
