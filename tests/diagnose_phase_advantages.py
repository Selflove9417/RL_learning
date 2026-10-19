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

def main():
    task_name = "myrobot_forward_jump_phase4"
    sys.argv = ["eval_adv", "--task", task_name, "--num_envs", "64"]
    args = get_args()
    args.headless = True
    args.seed = 1

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = 64
    env_cfg.seed = 1
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    champ_path = "evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    ckpt = torch.load(champ_path, map_location=env.device)
    ac = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()

    obs, _ = env.reset()
    gamma = 0.99
    lam = 0.95

    vals = []
    rews = []
    dones_list = []

    with torch.no_grad():
        for step in range(140):
            v = ac.evaluate(obs).squeeze(-1)
            vals.append(v)
            act = ac.act_inference(obs)
            obs, _, rew, dones, _ = env.step(act)
            rews.append(rew.clone())
            dones_list.append(dones.clone())

    vals = torch.stack(vals, dim=0) # (140, 64)
    rews = torch.stack(rews, dim=0) # (140, 64)
    dones_t = torch.stack(dones_list, dim=0)

    adv = torch.zeros_like(rews)
    gae = torch.zeros(64, device=env.device)
    for t in reversed(range(140)):
        next_val = vals[t+1] if t < 139 else torch.zeros(64, device=env.device)
        next_not_done = 1.0 - dones_t[t].float()
        delta = rews[t] + gamma * next_val * next_not_done - vals[t]
        gae = delta + gamma * lam * next_not_done * gae
        adv[t] = gae

    norm_adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    phases = {
        "Prep (0-9)": (0, 10),
        "Launch (10-13)": (10, 14),
        "Ascent (14-21)": (14, 22),
        "Descent (22-31)": (22, 32),
        "Touchdown (32-39)": (32, 40),
        "Recovery (40-139)": (40, 140)
    }

    print("Phase-by-Phase Mean Normalized Advantage and Raw TD Delta:")
    for name, (s, e) in phases.items():
        p_adv = norm_adv[s:e].mean().item()
        p_raw_adv = adv[s:e].mean().item()
        p_rew = rews[s:e].mean().item()
        p_val = vals[s:e].mean().item()
        print(f"  {name:20s}: rew={p_rew:6.3f}, val={p_val:6.3f}, raw_adv={p_raw_adv:6.3f}, norm_adv={p_adv:+6.3f}")

if __name__ == "__main__":
    main()
