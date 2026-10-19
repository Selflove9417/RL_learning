# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# MyRobot Running Jump Config (Dynamic In-Motion Takeoff & Continuous Cruising)

from legged_gym.envs.myrobot.myrobot_jump_config import MyRobotJumpCfg, MyRobotJumpCfgPPO


class MyRobotRunningJumpCfg(MyRobotJumpCfg):
    class env(MyRobotJumpCfg.env):
        num_envs = 64
        num_observations = 35
        num_privileged_obs = 35
        num_actions = 6
        episode_length_s = 2.8  # 140 steps per running jump cycle (40 cruise + 40 jump + 20 cushion + 40 resume)
        warmup_steps = 40       # 0.8s real Phase 15 cruising warmup before takeoff

    class commands(MyRobotJumpCfg.commands):
        curriculum = False
        target_forward_vel = 0.20     # Target 0.20 m/s for low-speed dynamic jump
        target_jump_height = 0.20     # 20cm clearance target
        jump_cycle_time = 0.80        # Temporal phase cycle length (40 steps = 0.80s)
        ranges = type("ranges", (), {
            "lin_vel_y": [0.20, 0.20], # Constant 0.20 m/s in Stage 1 (expandable to 0.35 and 0.50)
            "lin_vel_x": [0.0, 0.0],
            "ang_vel_yaw": [0.0, 0.0],
            "heading": [0.0, 0.0],
        })()

    class control(MyRobotJumpCfg.control):
        # High-power jump PD gains + wheel velocity damping
        control_type = 'P'
        stiffness = {'link_002_joint': 140.0, 'link_003_joint': 140.0,
                     'link_005_joint': 140.0, 'link_006_joint': 140.0}
        damping = {'link_002_joint': 3.5, 'link_003_joint': 3.5,
                   'link_005_joint': 3.5, 'link_006_joint': 3.5}
        action_scale = 1.0
        hip_action_scale = 0.70
        knee_action_scale = 0.70
        wheel_torque_scale = 10.0
        wheel_velocity_damping = 0.30
        decimation = 4

    class domain_rand(MyRobotJumpCfg.domain_rand):
        randomize_friction = False
        push_robots = False

    class rewards(MyRobotJumpCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.08

        class scales:
            # 1. Forward Momentum Retention (Rewarding vy ~ vy_cmd across the entire jump cycle)
            forward_velocity_retention = 2.5
            # 2. Vertical Push-off Impulse
            takeoff_velocity = 1.8
            # 3. Airborne Obstacle Clearance (>= 20 cm)
            jump_clearance = 2.2
            # 4. Touchdown Absorption Cushion
            landing_cushion = 1.5
            # 5. Post-touchdown Forward Tracking
            post_touchdown_forward = 2.0
            # 6. Attitude & Symmetry Stabilization
            pitch_stability = -1.2
            roll_stability = -1.5
            ang_vel_xy = -0.05
            # 7. Energy & Actuator Limits
            torques = -0.0001
            dof_acc = -2.5e-7
            torque_limits = -0.2


class MyRobotRunningJumpCfgPPO(MyRobotJumpCfgPPO):
    class algorithm(MyRobotJumpCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 5.0e-5
        schedule = 'fixed'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.015
        enable_early_stopping = True
        early_stop_kl = 0.03
        max_grad_norm = 1.0

    class runner(MyRobotJumpCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 140  # Matches episode_length_s = 2.8s (140 * 0.02)
        max_iterations = 3       # Up to 3 iterations for controlled exploration
        save_interval = 1
        experiment_name = 'myrobot_running_jump'
        run_name = 'running_jump_020ms_iter3_controlled'
        resume = False
        load_run = -1
        checkpoint = -1
        warm_start = True
        warm_start_path = 'legged_gym/logs/myrobot_running_jump/Oct19_20-39-35_running_jump_020ms_iter1/model_1.pt'
        warm_start_optimizer = False
        init_at_random_ep_len = False


class MyRobotRunningJump035Cfg(MyRobotRunningJumpCfg):
    class commands(MyRobotRunningJumpCfg.commands):
        target_forward_vel = 0.35     # Target 0.35 m/s
        target_jump_height = 0.20     # 20cm clearance target
        ranges = type("ranges", (), {
            "lin_vel_y": [0.35, 0.35],
            "lin_vel_x": [0.0, 0.0],
            "ang_vel_yaw": [0.0, 0.0],
            "heading": [0.0, 0.0],
        })()


class MyRobotRunningJump035CfgPPO(MyRobotRunningJumpCfgPPO):
    class runner(MyRobotRunningJumpCfgPPO.runner):
        experiment_name = 'myrobot_running_jump'
        run_name = 'running_jump_035ms_controlled'
        warm_start = True
        warm_start_path = 'legged_gym/logs/myrobot_running_jump/Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt'
        max_iterations = 3
        save_interval = 1


class MyRobotRunningJump050Cfg(MyRobotRunningJumpCfg):
    class commands(MyRobotRunningJumpCfg.commands):
        target_forward_vel = 0.50     # Target 0.50 m/s
        target_jump_height = 0.20     # 20cm clearance target
        ranges = type("ranges", (), {
            "lin_vel_y": [0.50, 0.50],
            "lin_vel_x": [0.0, 0.0],
            "ang_vel_yaw": [0.0, 0.0],
            "heading": [0.0, 0.0],
        })()


class MyRobotRunningJump050CfgPPO(MyRobotRunningJumpCfgPPO):
    class runner(MyRobotRunningJumpCfgPPO.runner):
        experiment_name = 'myrobot_running_jump'
        run_name = 'running_jump_050ms_controlled'
        warm_start = True
        warm_start_path = 'legged_gym/logs/myrobot_running_jump/Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt'
        max_iterations = 3
        save_interval = 1

