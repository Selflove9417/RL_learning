#!/usr/bin/env python3
import os
import sys
import argparse
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
    parser.add_argument("--task", type=str, default="myrobot_forward_jump")
    parser.add_argument("--cmd_y", type=float, default=None)
    parser.add_argument("--zero_dof_vel", action="store_true")
    cli_args, _ = parser.parse_known_args()

    task_name = cli_args.task
    cmd_y = cli_args.cmd_y
    zero_dof_vel = cli_args.zero_dof_vel

    sys.argv = ["diag", "--task", task_name, "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = 42

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = 1
    env_cfg.seed = 42
    env_cfg.env.episode_length_s = 2.0
    env_cfg.noise.add_noise = False
    env_cfg.env.enable_camera_sensors = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # Load model_30_yaw_aligned.pt
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
    if cmd_y is not None:
        env.commands[:, 1] = cmd_y
    if zero_dof_vel:
        env.dof_vel[:] = 0.0

    max_clr = 0.0
    takeoff_vz = 0.0
    takeoff_vy = 0.0
    took_off = False
    prep_vy = 0.0

    for step in range(100):
        if step == 5:
            prep_vy = env.base_lin_vel[0, 1].item()

        clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(env)
        clr_min_cm = clr_min[0].item() * 100.0

        is_air = bool(env.is_airborne[0].item())
        has_took = bool(env.has_taken_off[0].item())

        if is_air and has_took:
            max_clr = max(max_clr, clr_min_cm)
            if not took_off:
                took_off = True
                takeoff_vz = env.root_states[0, 9].item()
                takeoff_vy = env.root_states[0, 8].item()

        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _, _ = env.step(actions)

    print(f"RESULT: Task={task_name}, cmd_y={cmd_y}, zero_dof_vel={zero_dof_vel} => "
          f"Clearance={max_clr:.2f} cm, Takeoff_Vz={takeoff_vz:.3f} m/s, Takeoff_Vy={takeoff_vy:.3f} m/s, Prep_Vy={prep_vy:.3f} m/s")

if __name__ == "__main__":
    main()
