import os
import sys
import argparse
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
from legged_gym.envs.myrobot.myrobot_dual_env import compute_exact_cylinder_clearance

def eval_model(seed, model_path, clip_actions=100.0):
    sys.argv = ["test", "--task", "myrobot_forward_jump_static", "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = seed
    env_cfg, _ = task_registry.get_cfgs(name="myrobot_forward_jump_static")
    env_cfg.normalization.clip_actions = clip_actions
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.noise.add_noise = False

    env, _ = task_registry.make_env(name="myrobot_forward_jump_static", args=args, env_cfg=env_cfg)
    ac = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], "elu", 1.0).to(env.device)
    ckpt = torch.load(model_path, map_location=env.device)
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    policy = ac.act_inference
    obs, _ = env.reset()

    max_clr = 0.0
    took_off = False
    touchdown_step = None
    takeoff_com_y = 0.0
    touchdown_com_y = 0.0
    takeoff_vy = 0.0
    takeoff_vz = 0.0

    for step in range(120):
        clr_min, _, _ = compute_exact_cylinder_clearance(env)
        is_air = env.is_airborne[0].item()
        has_took = env.has_taken_off[0].item()
        true_com_pos, true_com_vel = env._compute_true_com()

        if is_air and has_took:
            max_clr = max(max_clr, clr_min[0].item() * 100.0)
            if not took_off:
                took_off = True
                takeoff_com_y = true_com_pos[0, 1].item()
                takeoff_vy = true_com_vel[0, 1].item()
                takeoff_vz = true_com_vel[0, 2].item()
        if took_off and (not is_air) and touchdown_step is None:
            touchdown_step = step
            touchdown_com_y = true_com_pos[0, 1].item()

        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _, _ = env.step(actions)

    fwd_dist = (touchdown_com_y - takeoff_com_y) * 100.0 if touchdown_step is not None else 0.0
    return max_clr, fwd_dist, takeoff_vz, takeoff_vy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--clip", type=float, default=100.0)
    args = parser.parse_args()
    clr, fwd, vz, vy = eval_model(args.seed, args.model, args.clip)
    print(f"EVAL: Seed={args.seed}, Model={os.path.basename(args.model)}, Clip={args.clip} => "
          f"Clr={clr:.2f} cm, FwdDist={fwd:.2f} cm, Takeoff_Vz={vz:.3f} m/s, Vy={vy:.3f} m/s")
