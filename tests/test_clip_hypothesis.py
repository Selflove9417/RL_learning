import os
import sys
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

sys.argv = ["test", "--task", "myrobot_forward_jump_static", "--num_envs", "1"]
args = get_args()
args.headless = True
args.seed = 42

env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_forward_jump_static")
env_cfg.normalization.clip_actions = 100.0  # Phase 1 had 100.0
env_cfg.domain_rand.push_robots = False
env_cfg.env.num_envs = 1
env_cfg.seed = 42
env_cfg.noise.add_noise = False

env, _ = task_registry.make_env(name="myrobot_forward_jump_static", args=args, env_cfg=env_cfg)
model_path = os.path.join(workspace_root, "evaluation_assets", "phase1_forward_jump", "models", "model_30_yaw_aligned.pt")
ac = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], "elu", 1.0).to(env.device)
ckpt = torch.load(model_path, map_location=env.device)
ac.load_state_dict(ckpt["model_state_dict"])
ac.eval()
policy = ac.act_inference

obs, _ = env.reset()
max_clr = 0.0
takeoff_vz = 0.0
for step in range(100):
    clr_min, _, _ = compute_exact_cylinder_clearance(env)
    if env.is_airborne[0].item() and env.has_taken_off[0].item():
        max_clr = max(max_clr, clr_min[0].item() * 100.0)
        if takeoff_vz == 0.0:
            takeoff_vz = env.root_states[0, 9].item()
    with torch.no_grad():
        actions = policy(obs)
    obs, _, _, _, _ = env.step(actions)

print(f"MYSTERY SOLVED: static_env with clip_actions=100.0 => Max Clearance: {max_clr:.2f} cm, Takeoff Vz: {takeoff_vz:.3f} m/s")
