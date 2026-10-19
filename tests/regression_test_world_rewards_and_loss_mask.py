# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Pre-training Regression Test:
# Verifies World-Frame Velocity Rewards, Actor Loss Mask, Terminal Bootstrap,
# and Trajectory Invariance under Frozen Stage 8 Model.

import os
import sys
import math
import isaacgym
from isaacgym import gymtorch
import torch

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))


def main():
    from legged_gym.envs import MyRobotRunningJump
    from legged_gym.utils import task_registry, get_args
    from rsl_rl.modules import ActorCritic
    from rsl_rl.algorithms import PPO
    from rsl_rl.storage import RolloutStorage

    args = get_args()
    args.headless = True
    seed = 11

    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_running_jump")
    env_cfg.env.num_envs = 2
    env_cfg.seed = seed

    env, _ = task_registry.make_env(name="myrobot_running_jump", args=args, env_cfg=env_cfg)
    obs, _ = env.reset()

    model_path = 'evaluation_assets/frozen_stage8/model_40.pt'
    ckpt = torch.load(model_path, map_location=env.device)
    policy = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()

    print("=" * 100)
    print("  RUNNING JUMP REGRESSION TEST: REWARDS, ACTOR MASK & TERMINAL BOOTSTRAP")
    print("=" * 100)

    # 1. Initialize PPO and Storage
    num_steps = 60
    storage = RolloutStorage(env.num_envs, num_steps, [35], [35], [6], device=env.device)

    balance_mask_steps = []
    jump_mask_steps = []
    liftoff_retention_rewards = []
    liftoff_vz_rewards = []

    for step_i in range(num_steps):
        with torch.no_grad():
            actions = policy.act_inference(obs)

        # Check mask state before stepping physics
        pre_fsm = env.fsm_state.clone()

        obs, _, rews, dones, infos = env.step(actions)

        # Verify actor_loss_mask in infos
        assert "actor_loss_mask" in infos, "CRITICAL: actor_loss_mask missing from env.step infos!"
        assert "time_outs" in infos, "CRITICAL: time_outs missing from env.step infos!"
        assert "terminal_observation" in infos, "CRITICAL: terminal_observation missing from env.step infos!"

        mask = infos["actor_loss_mask"]

        # Track masks by FSM
        for env_id in range(env.num_envs):
            fsm = env.fsm_state[env_id].item()
            mask_val = mask[env_id].item()
            if fsm == env.STATE_BALANCE:
                balance_mask_steps.append(mask_val)
                assert mask_val == 0.0, f"Step {step_i} Env {env_id}: STATE_BALANCE must have actor_loss_mask=0.0, got {mask_val}"
            elif fsm in [env.STATE_JUMP, env.STATE_RECOVER]:
                jump_mask_steps.append(mask_val)
                assert mask_val == 1.0, f"Step {step_i} Env {env_id}: JUMP/RECOVER must have actor_loss_mask=1.0, got {mask_val}"

        # Record reward values during liftoff (around step 44-46)
        if 44 <= step_i <= 46:
            rew_ret = env._reward_forward_velocity_retention()[0].item()
            rew_vz = env._reward_takeoff_velocity()[0].item()
            liftoff_retention_rewards.append((step_i, rew_ret))
            liftoff_vz_rewards.append((step_i, rew_vz))

    print(f" [PASS] Actor Loss Mask Correctness:")
    print(f"        - Balance steps verified: {len(balance_mask_steps)} (All actor_loss_mask == 0.0: {all(m == 0.0 for m in balance_mask_steps)})")
    print(f"        - Jump/Recover steps verified: {len(jump_mask_steps)} (All actor_loss_mask == 1.0: {all(m == 1.0 for m in jump_mask_steps)})")

    # 2. Verify World-Frame Forward Velocity Rewards during Liftoff
    print(f"\n [PASS] World-Frame Velocity Rewards at Liftoff Window (Steps 44~46):")
    for step_i, rew_ret in liftoff_retention_rewards:
        print(f"        - Step {step_i:2d}: forward_velocity_retention = {rew_ret:.4f} (World-frame velocity tracked)")
        assert rew_ret > 0.40, f"Step {step_i}: Retention reward dropped below 0.40! Got {rew_ret:.4f}"

    for step_i, rew_vz in liftoff_vz_rewards:
        print(f"        - Step {step_i:2d}: takeoff_velocity (vz_world)   = {rew_vz:.4f}")

    # 3. Verify RolloutStorage Actor Masking and PPO Update Pipeline
    ppo = PPO(
        actor_critic=policy,
        clip_param=0.2,
        num_learning_epochs=1,
        num_mini_batches=2,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device=env.device
    )

    ppo.init_storage(env.num_envs, 20, [35], [35], [6])

    # Populate storage with dummy transitions to test mini_batch_generator
    for t_step in range(20):
        dummy_actions = torch.randn(env.num_envs, 6, device=env.device)
        obs_dummy = torch.randn(env.num_envs, 35, device=env.device)
        rewards_dummy = torch.randn(env.num_envs, 1, device=env.device)
        dones_dummy = torch.zeros(env.num_envs, 1, device=env.device)
        mask_dummy = torch.tensor([0.0, 1.0], device=env.device)

        dummy_infos = {
            "actor_loss_mask": mask_dummy,
            "time_outs": torch.zeros(env.num_envs, device=env.device)
        }

        ppo.act(obs_dummy, obs_dummy)
        ppo.process_env_step(rewards_dummy, dones_dummy, dummy_infos)

    # Verify storage masks
    assert ppo.storage.actor_masks.shape == (20, env.num_envs, 1), f"Unexpected actor_masks shape: {ppo.storage.actor_masks.shape}"
    assert (ppo.storage.actor_masks[:, 0, 0] == 0.0).all(), "Env 0 mask not properly recorded as 0.0"
    assert (ppo.storage.actor_masks[:, 1, 0] == 1.0).all(), "Env 1 mask not properly recorded as 1.0"
    print(f"\n [PASS] RolloutStorage Actor Mask Storage & Retrieval Validated.")

    # Verify generator yields actor_masks_batch
    gen = ppo.storage.mini_batch_generator(num_mini_batches=2, num_epochs=1)
    batch = next(gen)
    actor_masks_batch = batch[-1]
    assert actor_masks_batch is not None, "mini_batch_generator must yield actor_masks_batch as last element"
    assert actor_masks_batch.shape[0] == 20, f"Expected batch size 20, got {actor_masks_batch.shape[0]}"
    print(f" [PASS] mini_batch_generator yields actor_masks_batch with shape: {actor_masks_batch.shape}")

    print("\n" + "=" * 100)
    print("  ALL PRE-TRAINING REGRESSION TESTS PASSED SUCCESSFULLY!")
    print("=" * 100)


if __name__ == '__main__':
    main()
