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
from legged_gym.envs import task_registry
from legged_gym.utils import get_args

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
cli_args, _ = parser.parse_known_args()
task_name = cli_args.task

sys.argv = ["test", "--task", task_name, "--num_envs", "1"]
args = get_args()
args.headless = True
args.seed = 42

env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
if task_name == "myrobot_forward_jump_static":
    env_cfg.commands.ranges.lin_vel_y = [0.3824, 0.3824]
env_cfg.env.num_envs = 1
env_cfg.seed = 42
env_cfg.noise.add_noise = False

env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
obs, _ = env.reset()

data = {
    "obs": obs[0].cpu().tolist(),
    "dof_pos": env.dof_pos[0].cpu().tolist(),
    "dof_vel": env.dof_vel[0].cpu().tolist(),
    "root_states": env.root_states[0].cpu().tolist(),
}
with open(cli_args.out, "w") as f:
    json.dump(data, f, indent=2)
print("Saved to", cli_args.out)
