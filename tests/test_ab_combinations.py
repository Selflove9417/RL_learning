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

def evaluate_combo(task_name, override_cfg_type, label):
    sys.argv = ["test", "--task", task_name, "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = 42

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    if override_cfg_type == "static_cfg":
        from legged_gym.envs.myrobot.myrobot_forward_jump_static_config import MyRobotForwardJumpStaticCfg
        env_cfg = MyRobotForwardJumpStaticCfg()
    elif override_cfg_type == "fwd_cfg":
        from legged_gym.envs.myrobot.myrobot_forward_jump_config import MyRobotForwardJumpCfg
        env_cfg = MyRobotForwardJumpCfg()

    env_cfg.env.num_envs = 1
    env_cfg.seed = 42
    env_cfg.noise.add_noise = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    model_path = os.path.join(workspace_root, "evaluation_assets", "phase1_forward_jump", "models", "model_30_yaw_aligned.pt")
    ac = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], "elu", 1.0).to(env.device)
    ckpt = torch.load(model_path, map_location=env.device)
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    policy = ac.act_inference

    obs, _ = env.reset()
    max_clr = 0.0
    for step in range(100):
        clr_min, _, _ = compute_exact_cylinder_clearance(env)
        if env.is_airborne[0].item() and env.has_taken_off[0].item():
            max_clr = max(max_clr, clr_min[0].item() * 100.0)
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _, _ = env.step(actions)

    print(f"{label} => Max Clearance: {max_clr:.2f} cm")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "A"
    if mode == "A":
        # Env = myrobot_forward_jump (Phase 1 env), Cfg = static_cfg (Phase 2 cfg)
        evaluate_combo("myrobot_forward_jump", "static_cfg", "Env=Phase1_Env + Cfg=Phase2_Cfg")
    else:
        # Env = myrobot_forward_jump_static (Phase 2 env), Cfg = fwd_cfg (Phase 1 cfg)
        evaluate_combo("myrobot_forward_jump_static", "fwd_cfg", "Env=Phase2_Env + Cfg=Phase1_Cfg")
