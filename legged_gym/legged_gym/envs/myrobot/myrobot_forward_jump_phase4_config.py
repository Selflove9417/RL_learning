# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# MyRobot Forward Jump Phase 4 Configuration
# Minimal Incremental PPO Curriculum (Expected Airborne Distance: 0.090 -> 0.100 m)

from legged_gym.envs.myrobot.myrobot_forward_jump_static_config import (
    MyRobotForwardJumpStaticCfg,
    MyRobotForwardJumpStaticCfgPPO,
)


class MyRobotForwardJumpPhase4Cfg(MyRobotForwardJumpStaticCfg):
    class env(MyRobotForwardJumpStaticCfg.env):
        num_envs = 64
        num_actions = 6
        num_observations = 35
        num_privileged_obs = None
        episode_length_s = 2.8  # 140 steps @ 50Hz (Single isolated jump: 0.64s jump + 2.16s recovery)
        zero_cmd_post_touchdown = True

    class commands(MyRobotForwardJumpStaticCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0
        heading_command = False
        # Jump command parameters
        init_jump_height = 0.215  # Target clearance: 21.5 cm
        target_jump_height = 0.215 # Final goal: 21.5 cm
        jump_cycle_time = 2.0     # 2.0 seconds per jump cycle
        target_forward_vel = 0.22 # Level 3.1 curriculum target forward speed: 0.22 m/s

        # PHASE 4.1 SINGLE-VARIABLE INCREMENT:
        # 0.090 m -> 0.100 m (+1.0 cm incremental curriculum target)
        expected_airborne_distance = 0.100

        class ranges:
            lin_vel_x = [0.0, 0.0]       # Strict zero lateral velocity
            lin_vel_y = [0.20, 0.24]     # Forward command along local Y centered at 0.22 m/s
            ang_vel_yaw = [0.0, 0.0]     # Strict zero yaw angular velocity
            heading = [0.0, 0.0]

    class init_state(MyRobotForwardJumpStaticCfg.init_state):
        pos = [0.0, 0.0, 0.397739]      # Tangent ground contact (zero bounce, zero penetration)
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]       # Strictly zero initial linear velocity
        ang_vel = [0.0, 0.0, 0.0]       # Strictly zero initial angular velocity


class MyRobotForwardJumpPhase4CfgPPO(MyRobotForwardJumpStaticCfgPPO):
    class algorithm(MyRobotForwardJumpStaticCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-4
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        enable_early_stopping = True
        early_stop_kl = 0.04
        max_grad_norm = 1.0

    class runner(MyRobotForwardJumpStaticCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 140 # 140 steps = exactly 1 synchronous episode per PPO iteration (64 envs * 140 = 8960 transitions)
        max_iterations = 5      # Exactly 5 iterations for Phase 4.4 Pilot
        save_interval = 1       # Save model_1.pt, model_2.pt, model_3.pt, model_4.pt, model_5.pt
        init_at_random_ep_len = False # Synchronous start at step 0 for all envs
        experiment_name = 'myrobot_forward_jump_phase4'
        run_name = 'phase4_4_pilot_sigmoid_5iter'
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        warm_start = True
        warm_start_path = '/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt'
        warm_start_optimizer = False
