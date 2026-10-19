# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# MyRobot Obstacle Clearance Jump Configuration
# Defines true physical obstacle with collision properties and spatial/temporal triggering.

from legged_gym.envs.myrobot.myrobot_running_jump_config import (
    MyRobotRunningJumpCfg,
    MyRobotRunningJumpCfgPPO
)


class MyRobotObstacleJumpCfg(MyRobotRunningJumpCfg):
    """
    Configuration for true physical obstacle clearance jump task.
    Derived independently from MyRobotRunningJumpCfg, preserving baseline checkpoints.
    """

    class env(MyRobotRunningJumpCfg.env):
        num_envs = 64
        num_observations = 35
        num_privileged_obs = 35
        num_actions = 6
        episode_length_s = 4.0   # 200 steps (prevents timeout reset data pollution)
        warmup_steps = 40        # 0.8s Phase 15 cruising warmup
        continuous_mode = False

    class obstacle:
        # Initial low obstacle physical dimensions
        # Wheel radius = 0.070m (7.0cm). Height = 0.050m (71% of wheel radius).
        # Width (along Y travel axis) = 0.040m (4.0cm).
        # Length (along X lateral axis) = 1.000m (spans across both wheels).
        height = 0.050
        width = 0.040
        length = 1.000
        pos_x = 0.000
        pos_y = 0.300
        fixed = True
        friction = 1.0
        restitution = 0.0
        color = [0.85, 0.25, 0.15]  # High-visibility red-orange in Isaac Gym Viewer

    class trigger:
        # Trigger mode:
        # - "spatial": triggers when front wheel approaches obstacle within spatial_trigger_dist
        # - "step": legacy step-based trigger at step 40 (for model_3.pt baseline comparison)
        mode = "spatial"
        spatial_trigger_dist = 0.09   # Trigger takeoff 9 cm before obstacle front edge
        step_trigger = 40             # Step-based fallback (t = 0.80s)

    class viewer:
        # Fixed lateral side-view camera for clear observation of takeoff, hurdle, and landing
        pos = [1.2, 0.30, 0.30]
        lookat = [0.0, 0.30, 0.20]

    class commands(MyRobotRunningJumpCfg.commands):
        curriculum = False
        target_forward_vel = 0.35     # Baseline 0.35 m/s cruising velocity
        target_jump_height = 0.20     # Target 20cm clearance
        jump_cycle_time = 0.80
        ranges = type("ranges", (), {
            "lin_vel_y": [0.35, 0.35],
            "lin_vel_x": [0.0, 0.0],
            "ang_vel_yaw": [0.0, 0.0],
            "heading": [0.0, 0.0],
        })()

    class rewards(MyRobotRunningJumpCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.08

        class scales(MyRobotRunningJumpCfg.rewards.scales):
            # 1. Obstacle collision penalty (strong negative penalty for touching obstacle)
            obstacle_collision = -5.0
            # 2. Obstacle vertical clearance (rewards wheel height above obstacle top while crossing)
            obstacle_clearance = 3.0
            # 3. Successful complete clearance (bonus for landing past obstacle rear with zero collision)
            obstacle_cleared_success = 5.0
            # 4. Forward horizontal progress across obstacle
            forward_progress_across_obstacle = 2.5
            # 5. Continuous cruising after touchdown
            post_touchdown_forward = 2.5
            # 6. Forward velocity retention across the jump
            forward_velocity_retention = 2.5
            # 7. Takeoff vertical impulse
            takeoff_velocity = 1.8
            # 8. Landing compliance
            landing_cushion = 1.5
            # 9. In-air attitude stability
            pitch_stability = -1.2
            roll_stability = -1.5
            ang_vel_xy = -0.05
            # 10. Energy and actuator bounds
            torques = -0.0001
            dof_acc = -2.5e-7
            torque_limits = -0.2


class MyRobotObstacleJumpCfgPPO(MyRobotRunningJumpCfgPPO):
    class runner(MyRobotRunningJumpCfgPPO.runner):
        experiment_name = 'myrobot_obstacle_jump'
        run_name = 'obstacle_jump_curriculum_stage1'
        num_steps_per_env = 200
        max_iterations = 5
        save_interval = 1
        warm_start = True
        warm_start_path = 'legged_gym/logs/myrobot_running_jump/Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt'


class MyRobotObstacleJump020Cfg(MyRobotObstacleJumpCfg):
    class commands(MyRobotObstacleJumpCfg.commands):
        target_forward_vel = 0.20
        ranges = type("ranges", (), {
            "lin_vel_y": [0.20, 0.20],
            "lin_vel_x": [0.0, 0.0],
            "ang_vel_yaw": [0.0, 0.0],
            "heading": [0.0, 0.0],
        })()


class MyRobotObstacleJump020CfgPPO(MyRobotObstacleJumpCfgPPO):
    class runner(MyRobotObstacleJumpCfgPPO.runner):
        run_name = 'obstacle_jump_020ms_stage1'
