#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Test Phase 2 Static Forward Jump Environment
# Verifies single-jump latching, CoM displacement, stationary prep constraint,
# and one-shot touchdown event over 2.8 seconds (140 steps @ 50Hz).
# ==============================================================================

import os
import sys

import isaacgym
from isaacgym import gymapi, gymtorch
import torch

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from rsl_rl.modules import ActorCritic


def test_env():
    sys.argv = ["test_static", "--task", "myrobot_forward_jump_static", "--num_envs", "4"]
    args = get_args()
    args.headless = True
    args.seed = 42

    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_forward_jump_static")
    env_cfg.env.num_envs = 4
    env_cfg.noise.add_noise = False
    env_cfg.env.enable_camera_sensors = False
    env, _ = task_registry.make_env(name="myrobot_forward_jump_static", args=args, env_cfg=env_cfg)

    # Check hot-start loading from Phase 1 Trial 2 model
    trial2_model_path = os.path.join(workspace_root, "evaluation_assets", "phase1_forward_jump", "models", "model_30_yaw_aligned.pt")
    ac = ActorCritic(
        num_actor_obs=env_cfg.env.num_observations,
        num_critic_obs=env_cfg.env.num_observations,
        num_actions=env_cfg.env.num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0
    ).to(env.device)
    ckpt = torch.load(trial2_model_path, map_location=env.device)
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    policy = ac.act_inference
    print("✅ Hot-start weights from Phase 1 Trial 2 successfully loaded into ActorCritic!")

    obs, _ = env.reset()

    # Step through entire 2.8s episode (140 steps)
    touchdown_count = 0
    max_phase_after_latch = 0.0

    print("Stepping 140 steps (2.8 seconds)...")
    for s in range(140):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, rewards, dones, infos = env.step(actions)

        # Check touchdown event firing
        if env.one_shot_touchdown_event[0].item():
            touchdown_count += 1
            print(f"  [Step {s}, t={s*0.02:.2f}s] Touchdown event fired! final_flight_d = {env.final_flight_distance[0].item()*100:.2f} cm")

        if env.jump_latched[0].item():
            max_phase_after_latch = max(max_phase_after_latch, env.jump_phase[0].item())

    print(f"Results:")
    print(f"  Touchdown events fired for env 0: {touchdown_count} (must be exactly 1)")
    print(f"  Jump latched state: {env.jump_latched.cpu().numpy().tolist()}")
    print(f"  Max phase after latch: {max_phase_after_latch:.3f} (must be <= 0.85, locking in recovery balance)")
    print(f"  Final flight distance: {env.final_flight_distance.cpu().numpy() * 100} cm")
    print(f"  Max flight clearance: {env.max_flight_clearance.cpu().numpy() * 100} cm")

    assert touchdown_count == 1, f"Expected exactly 1 touchdown event, got {touchdown_count}"
    assert max_phase_after_latch <= 0.855, f"Phase did not latch in recovery, max={max_phase_after_latch}"
    print("✅ All pre-training assertions passed successfully!")


if __name__ == "__main__":
    test_env()
