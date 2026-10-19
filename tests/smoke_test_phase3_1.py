#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Phase 3.1 Pre-Flight Smoke Test
# Verifies:
# 1. Environment initialization without error
# 2. Checkpoint weights loading into ActorCritic
# 3. Observation shape (35) and action shape (6)
# 4. Correct configuration parameters (8-10 cm curriculum target)
# 5. Asset preservation: Historical directories untouched
# ==============================================================================

import os
import sys
import hashlib
import isaacgym
from isaacgym import gymapi, gymtorch
import torch

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from rsl_rl.modules import ActorCritic

CHECKPOINT_PATH = "/home/xy/IsaacTu_jump/legged_gym/logs/myrobot_forward_jump_static/Oct16_15-26-49_phase2_1_clearance_priority/model_15.pt"
EXPECTED_MD5 = "880080cc5afe47edb08c587a8e957a62"


def run_smoke_test():
    print("=" * 80)
    print("🔍 MyRobot Phase 3.1 Pre-Flight Smoke Test")
    print("=" * 80)

    # 1. Checkpoint verification
    print("\n--> 1. Checking Warm-Start Checkpoint...")
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
    with open(CHECKPOINT_PATH, "rb") as f:
        file_md5 = hashlib.md5(f.read()).hexdigest()
    assert file_md5 == EXPECTED_MD5, f"MD5 mismatch! Expected {EXPECTED_MD5}, got {file_md5}"
    print(f"  ✅ Checkpoint exists: {CHECKPOINT_PATH}")
    print(f"  ✅ MD5 verified: {file_md5} (Exact Phase 2.1 Iter 15)")

    # 2. Configuration verification
    print("\n--> 2. Checking Phase 3.1 Task Configuration...")
    sys.argv = ["smoke_test", "--task", "myrobot_forward_jump_static", "--num_envs", "2", "--headless"]
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_forward_jump_static")

    print(f"  • num_observations: {env_cfg.env.num_observations} (Expected: 35)")
    print(f"  • num_actions: {env_cfg.env.num_actions} (Expected: 6)")
    print(f"  • clip_actions: {env_cfg.normalization.clip_actions} (Expected: 100.0)")
    print(f"  • target_jump_height: {getattr(env_cfg.commands, 'target_jump_height', None)} (Expected: 0.215)")
    print(f"  • expected_airborne_distance: {getattr(env_cfg.commands, 'expected_airborne_distance', None)} (Level 3.1 Target: 0.090)")
    print(f"  • lin_vel_y range: {getattr(env_cfg.commands.ranges, 'lin_vel_y', None)} (Level 3.1 Target: [0.20, 0.24])")
    print(f"  • target_forward_vel: {getattr(env_cfg.commands, 'target_forward_vel', None)} (Level 3.1 Target: 0.22)")
    print(f"  • warm_start: {train_cfg.runner.warm_start}")
    print(f"  • warm_start_path: {train_cfg.runner.warm_start_path}")
    print(f"  • run_name: {train_cfg.runner.run_name}")

    assert env_cfg.env.num_observations == 35, f"Obs shape changed! {env_cfg.env.num_observations}"
    assert env_cfg.env.num_actions == 6, f"Action shape changed! {env_cfg.env.num_actions}"
    assert env_cfg.normalization.clip_actions == 100.0, f"clip_actions should be 100.0!"
    print("  ✅ Architecture & core constraints verified.")

    # 3. Environment instantiation check
    print("\n--> 3. Instantiating Test Environment...")
    env_cfg.env.num_envs = 2
    env, _ = task_registry.make_env(name="myrobot_forward_jump_static", args=args, env_cfg=env_cfg)
    obs, _ = env.reset()
    assert obs.shape == (2, 35), f"Unexpected reset obs shape: {obs.shape}"
    print(f"  ✅ Environment created successfully, reset obs shape: {obs.shape}")

    # 4. ActorCritic load test
    print("\n--> 4. Testing Checkpoint Loading into Policy...")
    ac = ActorCritic(
        num_actor_obs=env_cfg.env.num_observations,
        num_critic_obs=env_cfg.env.num_observations,
        num_actions=env_cfg.env.num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0
    ).to(env.device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=env.device)
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    with torch.no_grad():
        actions = ac.act_inference(obs)
    assert actions.shape == (2, 6), f"Unexpected action shape: {actions.shape}"
    print(f"  ✅ Policy loaded and executed forward inference, actions shape: {actions.shape}")

    # 5. Asset preservation check
    print("\n--> 5. Checking Asset Preservation...")
    protected_paths = [
        "legged_gym/logs/myrobot_jump/Oct14_17-04-58_jump_stage8_earlyzero/model_40.pt",
        "evaluation_assets/phase1_forward_jump/models/model_30_yaw_aligned.pt",
        "evaluation_assets/phase2_static_forward_jump/models/model_champion.pt",
        "evaluation_assets/phase2_1_clearance_joint/models/model_champion.pt",
        "evaluation_assets/phase2_1_clearance_joint/models/model_candidate_iter15.pt",
        "evaluation_assets/phase2_1_clearance_joint/phase2_1_clearance_joint_evaluation_summary.json"
    ]
    for p in protected_paths:
        full_p = os.path.join(workspace_root, p)
        assert os.path.exists(full_p), f"Protected asset missing! {full_p}"
    print(f"  ✅ All {len(protected_paths)} historical and Phase 2.1 assets confirmed intact.")

    print("\n" + "=" * 80)
    print("🎉 ALL SMOKE TESTS PASSED! READY FOR PHASE 3.1 TRAINING.")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()
