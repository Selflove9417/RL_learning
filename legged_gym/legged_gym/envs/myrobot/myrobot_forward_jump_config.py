# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# MyRobot Forward Jump PPO Configuration (Phase 1: Forward Jump Capability Development)

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class MyRobotForwardJumpCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 512
        num_actions = 6
        num_observations = 35
        num_privileged_obs = None
        episode_length_s = 6.0  # 300 steps @ 50Hz (3 full jump cycles)

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 6.0
        heading_command = False
        # Jump command parameters
        init_jump_height = 0.215  # Target clearance: 21.5 cm
        target_jump_height = 0.215 # Final goal: 21.5 cm
        jump_cycle_time = 2.0     # 2.0 seconds per jump cycle
        target_forward_vel = 0.40 # Target forward jump speed (m/s)
        expected_airborne_distance = 0.15 # Target authentic flight distance (m)

        class ranges:
            lin_vel_x = [0.0, 0.0]       # Strict zero lateral velocity
            lin_vel_y = [0.35, 0.45]     # Forward command along local Y (Phase 1 target ~0.40 m/s)
            ang_vel_yaw = [0.0, 0.0]     # Strict zero yaw angular velocity
            heading = [0.0, 0.0]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.40]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            'link_002_joint': 0.0,
            'link_003_joint': 0.0,
            'link_004_joint': 0.0,
            'link_005_joint': 0.0,
            'link_006_joint': 0.0,
            'link_007_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        # Jump Actuator PD Gains (stiff for explosive push-off, compliant landing)
        stiffness = {
            'link_002_joint': 180.0,
            'link_003_joint': 180.0,
            'link_005_joint': 180.0,
            'link_006_joint': 180.0,
            'link_004_joint': 0.0,
            'link_007_joint': 0.0,
        }
        damping = {
            'link_002_joint': 4.5,
            'link_003_joint': 4.5,
            'link_005_joint': 4.5,
            'link_006_joint': 4.5,
            'link_004_joint': 0.0,
            'link_007_joint': 0.0,
        }
        action_scale = 1.0
        decimation = 4  # 4 * 0.005s = 0.02s (50 Hz control frequency)

        # Action ranges for jump
        hip_action_scale = 0.6    # +/- 0.6 rad range for hip
        knee_action_scale = 0.8   # +/- 0.8 rad range for knee
        wheel_torque_scale = 10.0 # +/- 10.0 Nm for wheel ground drive

        # Wheel velocity damping
        wheel_velocity_damping = 0.5
        wheel_air_torque_limit = 3.0 # Maximum wheel torque in air for reaction stabilization

    class asset(LeggedRobotCfg.asset):
        # Uses the upgraded actuator limits URDF
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/myrobot/myrobot_urdf/robot_jump.urdf'
        name = "myrobot_forward_jump"
        foot_name = "link_004"
        terminate_after_contacts_on = ["base_link", "link_002", "link_003", "link_005", "link_006"]
        penalize_contacts_on = ["base_link", "link_002", "link_003", "link_005", "link_006"]
        self_collisions = 1
        flip_visual_attachments = False
        collapse_fixed_joints = True

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [2.0, 2.0, 1.0]
        lookat = [0.0, 0.0, 0.4]

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_torque_limit = 0.95
        only_positive_rewards = False
        tracking_sigma = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            # Preserved Stage 8 Jump Core Task Rewards
            jump_clearance = 8.0          # Reward for true bilateral wheel clearance tracking (target 21.5cm)
            feet_air_time = 1.5           # Reward for airborne duration with valid push-off
            takeoff_velocity = 4.0        # Reward explosive upward push-off velocity (~1.98 m/s)
            squat_prep = 2.5              # Reward for controlled pre-jump squat depth
            wheel_contact = 3.0           # Reward firm wheel contact during stance & recovery
            landing_cushion = 5.0         # Reward for active leg reach & touchdown cushioning
            recovery_balance = 5.0        # Reward for stable upright recovery post-landing

            # Incremental Forward Jump Task Rewards (Phase 1)
            takeoff_forward_velocity = 4.0        # Reward forward push-off velocity (vy ~ 0.40 m/s)
            airborne_forward_displacement = 5.0   # Reward authentic flight displacement strictly between takeoff & touchdown
            touchdown_forward_distance = 3.0      # One-shot touchdown bonus for achieved airborne jump distance

            # Attitude & Symmetry constraints
            bilateral_symmetry = -2.5     # Penalize asymmetric wheel clearance and joint disparity
            airborne_orientation = -3.0   # Penalize pitch & roll while in flight
            orientation = -1.5            # General ground attitude maintenance
            landing_impact = -0.01        # Penalize excessive peak contact force (>350N)
            termination = -30.0           # Severe penalty for premature crash/collapse

            # Smoothness & Energy
            action_rate = -0.01
            dof_acc = -2.5e-7
            torques = -0.00005
            collision = -10.0
            dof_pos_limits = -5.0

            # Zeroed obsolete antagonistic rewards & Post-landing zero velocity recovery
            lin_vel_z = 0.0
            base_height = 0.0
            tracking_lin_vel = 3.0        # Reward braking to stationary zero horizontal velocity post-landing
            tracking_ang_vel = 2.0        # Penalize yaw rotation to maintain forward heading
            stand_still = 0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.6, 1.2]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        push_robots = False


class MyRobotForwardJumpCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 3.0e-4
        num_learning_epochs = 5
        num_mini_batches = 4
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.015
        max_grad_norm = 1.0
        enable_early_stopping = False
        enable_ref_kl = False

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # 24 steps * 0.02s = 0.48s rollout per env
        max_iterations = 30     # Phase 1 minimal scale trial training (30 iterations)
        save_interval = 10      # Save checkpoints every 10 iterations
        run_name = 'fwd_jump_phase1_trial_yaw_aligned'
        experiment_name = 'myrobot_forward_jump'
        resume = False
        warm_start = True
        warm_start_optimizer = False # Strictly do not inherit old optimizer state (Requirement 4)
        warm_start_path = 'evaluation_assets/frozen_stage8/model_40.pt'
        load_run = -1
        checkpoint = -1
