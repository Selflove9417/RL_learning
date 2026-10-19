# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# MyRobot Forward Jump Static PPO Configuration (Phase 2: Minimal Static Forward Jump Curriculum)

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class MyRobotForwardJumpStaticCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 512
        num_actions = 6
        num_observations = 35
        num_privileged_obs = None
        episode_length_s = 2.8  # 140 steps @ 50Hz (Single isolated jump: 0.64s jump + 2.16s recovery)

    class commands(LeggedRobotCfg.commands):
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
        expected_airborne_distance = 0.090 # Target authentic flight distance: 9.0 cm (8-10 cm curriculum)

        class ranges:
            lin_vel_x = [0.0, 0.0]       # Strict zero lateral velocity
            lin_vel_y = [0.20, 0.24]     # Forward command along local Y centered at 0.22 m/s (8-10 cm band)
            ang_vel_yaw = [0.0, 0.0]     # Strict zero yaw angular velocity
            heading = [0.0, 0.0]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.40]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]       # Strictly zero initial linear velocity
        ang_vel = [0.0, 0.0, 0.0]       # Strictly zero initial angular velocity
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
        # Uses the verified robot_jump.urdf (Hip 140Nm, Knee 120Nm, Wheel 12Nm)
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/myrobot/myrobot_urdf/robot_jump.urdf'
        name = "myrobot_forward_jump_static"
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
            # Preserved & Defended Vertical Jump Core Task Rewards (Dominant)
            jump_clearance = 16.0         # Dominant reward for true bilateral wheel clearance tracking (target >= 21.5cm)
            takeoff_velocity = 8.0        # Explosive upward push-off velocity (~1.98 m/s)
            feet_air_time = 2.0           # Reward for airborne duration with valid push-off
            squat_prep = 2.5              # Reward for controlled pre-jump squat depth
            wheel_contact = 3.0           # Reward firm wheel contact during stance & recovery
            landing_cushion = 5.0         # Reward for active leg reach & touchdown cushioning
            recovery_balance = 5.0        # Reward for stable upright recovery post-landing

            # Phase 2.1 Clearance-First Static Forward Jump Curriculum Rewards (Gated & Centered)
            stationary_prep = 3.0                 # Suppress ground rolling / sliding during prep (phi in [0.0, 0.05])
            takeoff_forward_velocity = 3.0        # Forward push-off velocity centered at vy ~ 0.20 m/s gated by vz >= 0.8
            airborne_forward_displacement = 3.5   # Clearance-gated CoM flight displacement centered at 7.0 cm (>=20cm clr gate)
            touchdown_forward_distance = 2.0      # Clearance-gated ONE-SHOT touchdown bonus (>=20cm clr gate)
            tracking_lin_vel = 4.0                # Post-landing stationary braking reward
            tracking_ang_vel = 2.0                # Heading yaw velocity penalty
            bilateral_symmetry = -2.0             # Penalize bilateral clearance disparity & torque asymmetry

            # Safety & limit penalties
            torques = -0.0001
            dof_vel = -0.0001
            dof_acc = -1.5e-7
            collision = -5.0
            action_rate = -0.005
            dof_pos_limits = -2.0
            torque_limits = -1.0

    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 100.0


class MyRobotForwardJumpStaticCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
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
        max_grad_norm = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # 24 steps @ 50Hz = ~0.48s per rollout iteration
        max_iterations = 20     # 20 iterations short trial (Phase 3.1 Level 3.1 curriculum)
        save_interval = 5       # Save every 5 iterations
        experiment_name = 'myrobot_forward_jump_static'
        run_name = 'phase3_1_fwd_dist_curriculum_level1'
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        warm_start = True
        warm_start_path = '/home/xy/IsaacTu_jump/legged_gym/logs/myrobot_forward_jump_static/Oct16_15-26-49_phase2_1_clearance_priority/model_15.pt'
        warm_start_optimizer = False
