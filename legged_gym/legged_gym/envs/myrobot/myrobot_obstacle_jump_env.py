# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# MyRobot Obstacle Clearance Jump Environment
# Genuine physical obstacle with collision properties, spatial triggering, and contact validation.

import os
import math
import torch
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz, quat_rotate, quat_rotate_inverse, to_torch, get_axis_params, torch_rand_float

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.myrobot.myrobot_dual_env import (
    compute_exact_cylinder_clearance,
    STATE_NAMES
)
from legged_gym.envs.myrobot.myrobot_running_jump_env import MyRobotRunningJump


class MyRobotObstacleJump(MyRobotRunningJump):
    """
    Genuine Physical Obstacle Clearance Running Jump Environment.
    Features:
    1. Fixed Real Collision Obstacle:
       - Instantiated via gym.create_box with fix_base_link=True and physical collision shape.
       - No visual-only lines or phantom hurdles.
    2. Exact Geometry & Physical Contact Telemetry:
       - Full 3D wheel cylinder bounds vs obstacle box bounds.
       - Direct physical contact force and penetration detection.
    3. Spatial Triggering:
       - Distance-based takeoff timing dynamically chosen based on spatial distance to obstacle.
       - Backward-compatible with step-based trigger for frozen baseline evaluation.
    4. Strict Phase 2 Clearance Criteria:
       - Pre-jump forward cruise verified.
       - Liftoff before obstacle front edge.
       - Zero collision / penetration.
       - Touchdown past obstacle rear edge with verified margin.
       - Post-touchdown velocity recovery in dedicated observation window.
    """

    def _create_envs(self):
        """
        Creates environments with two physical actors per env:
        Actor 0: MyRobot (URDF)
        Actor 1: Obstacle Box (Physical collision box, fixed base)
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # Rigid body and DOF names
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = ["link_004", "link_007"]


        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # Create physical obstacle box asset
        box_opts = gymapi.AssetOptions()
        box_opts.fix_base_link = getattr(self.cfg.obstacle, 'fixed', True)
        box_opts.density = 2000.0
        box_opts.angular_damping = 0.0
        box_opts.linear_damping = 0.0
        box_opts.thickness = 0.001

        obs_h = getattr(self.cfg.obstacle, 'height', 0.050)
        obs_w = getattr(self.cfg.obstacle, 'width', 0.040)
        obs_l = getattr(self.cfg.obstacle, 'length', 1.000)

        # gym.create_box takes (sim, width_x, length_y, height_z)
        self.obstacle_box_asset = self.gym.create_box(
            self.sim,
            obs_l,  # X length across track
            obs_w,  # Y width along travel
            obs_h,  # Z height
            box_opts
        )

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = torch.tensor(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.obstacle_handles = []
        self.envs = []

        obs_px = getattr(self.cfg.obstacle, 'pos_x', 0.000)
        obs_py = getattr(self.cfg.obstacle, 'pos_y', 0.300)
        obs_pz = obs_h * 0.5  # Box center is at half-height above ground

        color_rgb = getattr(self.cfg.obstacle, 'color', [0.85, 0.25, 0.15])
        obstacle_color = gymapi.Vec3(color_rgb[0], color_rgb[1], color_rgb[2])

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

            # Actor 0: Robot
            robot_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, robot_handle, body_props, recomputeInertia=True)

            # Actor 1: Physical Obstacle Box
            obs_pose = gymapi.Transform()
            obs_pose.p = gymapi.Vec3(pos[0] + obs_px, pos[1] + obs_py, pos[2] + obs_pz)
            obstacle_handle = self.gym.create_actor(
                env_handle, self.obstacle_box_asset, obs_pose, "obstacle", i, 0, 0
            )

            # Set obstacle color to bright red/orange in viewer
            self.gym.set_rigid_body_color(
                env_handle, obstacle_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, obstacle_color
            )

            self.envs.append(env_handle)
            self.actor_handles.append(robot_handle)
            self.obstacle_handles.append(obstacle_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _init_buffers(self):
        """
        Initializes tensors correctly handling two actors per environment (Robot + Obstacle)
        without dimension mismatch in LeggedRobot base class.
        """
        # Joint and leg indices in robot
        self.left_hip_idx = self.dof_names.index("link_002_joint")
        self.left_knee_idx = self.dof_names.index("link_003_joint")
        self.right_hip_idx = self.dof_names.index("link_005_joint")
        self.right_knee_idx = self.dof_names.index("link_006_joint")
        self.left_wheel_idx = self.dof_names.index("link_004_joint")
        self.right_wheel_idx = self.dof_names.index("link_007_joint")

        self.wheel_indices = torch.tensor(
            [self.left_wheel_idx, self.right_wheel_idx],
            device=self.device, dtype=torch.long
        )
        self.leg_indices = torch.tensor(
            [self.left_hip_idx, self.left_knee_idx, self.right_hip_idx, self.right_knee_idx],
            device=self.device, dtype=torch.long
        )

        # Acquire raw GPU tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        _rb_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Re-map root state tensor for 2 actors per env
        self.all_root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, 2, 13)
        self.root_states = self.all_root_states[:, 0, :]           # (num_envs, 13) - Robot
        self.obstacle_root_states = self.all_root_states[:, 1, :]   # (num_envs, 13) - Obstacle
        self.base_quat = self.root_states[:, 3:7]

        # Re-map rigid body state tensor (7 robot bodies + 1 obstacle body = 8 bodies per env)
        self.all_rb_states = gymtorch.wrap_tensor(_rb_tensor).view(self.num_envs, self.num_bodies + 1, 13)
        self.rb_states = self.all_rb_states[:, :self.num_bodies, :]
        self.obstacle_rb_states = self.all_rb_states[:, self.num_bodies, :]

        # Re-map contact force tensor
        self.all_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, self.num_bodies + 1, 3)
        self.contact_forces = self.all_contact_forces[:, :self.num_bodies, :]
        self.obstacle_contact_forces = self.all_contact_forces[:, self.num_bodies, :]

        # DOF states (only robot has DOFs)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # Base vectors & kinematics
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # Joint offsets and default gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # Jump buffers
        self.wheel_radius = 0.070
        self.left_wheel_body = self.feet_indices[0].item()
        self.right_wheel_body = self.feet_indices[1].item()
        self.jump_cycle_time = getattr(self.cfg.commands, 'jump_cycle_time', 2.0)
        self.jump_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.jump_phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.target_jump_height = torch.full((self.num_envs,), getattr(self.cfg.commands, 'target_jump_height', 0.20), device=self.device, dtype=torch.float)
        self.wheel_clearance = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        self.bilateral_clearance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.is_airborne = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.air_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.max_flight_clearance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.peak_landing_impact = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.takeoff_vz = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.has_taken_off = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.filtered_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
        self.nominal_dof_pos = self.default_dof_pos.clone()

        # Action scales from control config
        self.hip_action_scale = getattr(self.cfg.control, 'hip_action_scale', 0.6)
        self.knee_action_scale = getattr(self.cfg.control, 'knee_action_scale', 0.8)
        self.wheel_torque_scale = getattr(self.cfg.control, 'wheel_torque_scale', 10.0)
        self.wheel_velocity_damping = getattr(self.cfg.control, 'wheel_velocity_damping', 0.5)
        self.wheel_air_torque_limit = getattr(self.cfg.control, 'wheel_air_torque_limit', 3.0)

        # Running jump buffers
        self.target_forward_vel = getattr(self.cfg.commands, 'target_forward_vel', 0.20)
        self.default_jump_height = getattr(self.cfg.commands, 'target_jump_height', 0.20)
        self.cycle_length = getattr(self.cfg.env, 'cycle_length', 140)
        self.jump_trigger_step = getattr(self.cfg.env, 'jump_trigger_step', 40)
        self.max_episode_length = int(round(self.max_episode_length_s / self.dt))
        self.commands[:, 1] = self.target_forward_vel

        self.takeoff_vy = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.touchdown_vy = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.flight_distance_y = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.velocity_retention_ratio = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.has_touched_down = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Obstacle parameters
        self.obstacle_h = getattr(self.cfg.obstacle, 'height', 0.050)
        self.obstacle_w = getattr(self.cfg.obstacle, 'width', 0.040)
        self.obstacle_l = getattr(self.cfg.obstacle, 'length', 1.000)
        self.obstacle_py = getattr(self.cfg.obstacle, 'pos_y', 0.300)
        self.trigger_mode = getattr(self.cfg.trigger, 'mode', 'spatial')
        self.spatial_trigger_dist = getattr(self.cfg.trigger, 'spatial_trigger_dist', 0.09)
        self.step_trigger = getattr(self.cfg.trigger, 'step_trigger', 40)

        # Obstacle collision and clearance tracking
        self.has_obstacle_collision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.peak_obstacle_contact_force = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.min_obstacle_clearance = torch.full((self.num_envs,), 1.0, device=self.device, dtype=torch.float)
        self.obstacle_cleared = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.is_crossing_obstacle = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.takeoff_margin_m = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.landing_margin_m = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.post_touchdown_vels = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.post_touchdown_samples = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # Configure viewer side-view camera if interactive
        if self.viewer is not None:
            c_pos = getattr(self.cfg.viewer, 'pos', [1.2, 0.30, 0.30])
            c_lookat = getattr(self.cfg.viewer, 'lookat', [0.0, 0.30, 0.20])
            cam_pos = gymapi.Vec3(c_pos[0], c_pos[1], c_pos[2])
            cam_target = gymapi.Vec3(c_lookat[0], c_lookat[1], c_lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _reset_dofs(self, env_ids):
        """
        Resets DOF positions and velocities of robot actors, using correct robot actor IDs (env_id * 2).
        """
        if len(env_ids) == 0:
            return
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.0

        robot_actor_ids = (env_ids * 2).to(device=self.device, dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(robot_actor_ids),
            len(robot_actor_ids)
        )

    def _reset_root_states(self, env_ids):
        """
        Resets only the robot actors (actor index = env_id * 2), keeping obstacles fixed.
        """
        if len(env_ids) == 0:
            return

        # Reset robot root states
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        self.root_states[env_ids, 7:13] = 0.0

        # Reset obstacle root states to nominal fixed poses
        obs_px = getattr(self.cfg.obstacle, 'pos_x', 0.000)
        obs_py = getattr(self.cfg.obstacle, 'pos_y', 0.300)
        self.obstacle_py = obs_py
        obs_pz = self.obstacle_h * 0.5
        self.obstacle_root_states[env_ids, 0] = self.env_origins[env_ids, 0] + obs_px
        self.obstacle_root_states[env_ids, 1] = self.env_origins[env_ids, 1] + obs_py
        self.obstacle_root_states[env_ids, 2] = self.env_origins[env_ids, 2] + obs_pz
        self.obstacle_root_states[env_ids, 3:6] = 0.0
        self.obstacle_root_states[env_ids, 6] = 1.0  # Quat W
        self.obstacle_root_states[env_ids, 7:13] = 0.0

        # Both robot and obstacle actor indices for reset on GPU
        robot_actor_ids = (env_ids * 2).to(device=self.device, dtype=torch.int32)
        obstacle_actor_ids = (env_ids * 2 + 1).to(device=self.device, dtype=torch.int32)
        all_reset_actor_ids = torch.cat([robot_actor_ids, obstacle_actor_ids], dim=0)

        all_root_contiguous = self.all_root_states.view(-1, 13).contiguous()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(all_root_contiguous),
            gymtorch.unwrap_tensor(all_reset_actor_ids),
            len(all_reset_actor_ids)
        )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # 1. Reset DOFs and Root States with correct multi-actor indexing
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # 2. Reset LeggedRobot base buffers
        self._resample_commands(env_ids)
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_root_vel[env_ids] = 0.0

        if "episode" in self.extras:
            for key in self.episode_sums.keys():
                self.episode_sums[key][env_ids] = 0.0

        # 3. Reset MyRobotJump buffers
        target_h = getattr(self.cfg.commands, 'target_jump_height', 0.20)
        self.target_jump_height[env_ids] = target_h
        self.filtered_actions[env_ids] = 0.0
        self.wheel_clearance[env_ids] = 0.0
        self.bilateral_clearance[env_ids] = 0.0
        self.is_airborne[env_ids] = False
        self.air_time[env_ids] = 0.0
        self.max_flight_clearance[env_ids] = 0.0
        self.peak_landing_impact[env_ids] = 0.0
        self.takeoff_vz[env_ids] = 0.0

        # 4. Reset MyRobotDualEnv & RunningJump buffers
        if hasattr(self, 'fsm_state'):
            self.fsm_state[env_ids] = self.STATE_BALANCE
            self.fsm_state_timer[env_ids] = 0.0
            self.stopping_stable_steps[env_ids] = 0
            self.gate_stable_steps[env_ids] = 0
            self.settle_stable_steps[env_ids] = 0

            self.bal_filtered_actions[env_ids] = 0.0
            self.pi_integral[env_ids] = 0.0
            self.pi_v_filt[env_ids] = 0.0

            self.current_kp[env_ids] = 40.0
            self.current_kd[env_ids] = 2.0
            self.has_taken_off[env_ids] = False
            self.has_touched_down[env_ids] = False
            self.takeoff_vy[env_ids] = 0.0
            self.touchdown_vy[env_ids] = 0.0
            self.takeoff_pos_y[env_ids] = 0.0
            self.touchdown_pos_y[env_ids] = 0.0
            self.flight_distance_y[env_ids] = 0.0
            self.velocity_retention_ratio[env_ids] = 0.0

            self.last_fsm_state[env_ids] = self.STATE_BALANCE
            self.handoff_prev_q_tgt[env_ids] = 0.0
            self.last_q_tgt[env_ids] = 0.0

            self.user_cmd_vy[env_ids] = self.target_forward_vel
            self.user_cmd_yaw[env_ids] = 0.0
            self.user_target_height[env_ids] = 0.39774
            self.actual_cmd_vy[env_ids] = 0.0
            self.actual_cmd_yaw[env_ids] = 0.0
            self.actual_cmd_height[env_ids] = 0.39774
            self.teleop_jump_requested[env_ids] = False

            self.takeoff_step[env_ids] = -1
            self.touchdown_step[env_ids] = -1
            self.handoff_step[env_ids] = -1
            self.abort_step[env_ids] = -1
            self.stopping_start_pos[env_ids] = 0.0

            for eid in env_ids.tolist():
                self.last_transition_event[eid] = "INIT_BALANCE"

        self.commands[env_ids, 1] = self.target_forward_vel

        # 5. Reset Obstacle tracking buffers
        self.has_obstacle_collision[env_ids] = False
        self.peak_obstacle_contact_force[env_ids] = 0.0
        self.min_obstacle_clearance[env_ids] = 1.0
        self.obstacle_cleared[env_ids] = False
        self.is_crossing_obstacle[env_ids] = False
        self.takeoff_margin_m[env_ids] = 0.0
        self.landing_margin_m[env_ids] = 0.0
        self.post_touchdown_vels[env_ids] = 0.0
        self.post_touchdown_samples[env_ids] = 0.0


    def compute_obstacle_metrics(self):
        """
        Computes analytical and physical interaction metrics with the obstacle:
        1. Wheel cylinder & chassis clearance above obstacle top.
        2. Collision force and geometric penetration.
        3. Real-time front/rear spatial margins.
        """
        # Obstacle bounds relative to environment origins
        obs_center_y = self.env_origins[:, 1] + self.obstacle_py
        obs_front_y = obs_center_y - 0.5 * self.obstacle_w
        obs_rear_y = obs_center_y + 0.5 * self.obstacle_w

        # Wheel positions
        wheel_L_pos = self.rb_states[:, self.left_wheel_body, 0:3]
        wheel_R_pos = self.rb_states[:, self.right_wheel_body, 0:3]
        R = self.wheel_radius  # 0.070 m

        # Lowest points of bilateral wheel cylinders
        clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(self)
        wheel_bottom_min_z = clr_min

        # Check if wheels overlap with obstacle along Y travel axis
        wheel_min_y = torch.minimum(wheel_L_pos[:, 1], wheel_R_pos[:, 1]) - R
        wheel_max_y = torch.maximum(wheel_L_pos[:, 1], wheel_R_pos[:, 1]) + R

        in_obstacle_y_zone = (wheel_max_y >= obs_front_y) & (wheel_min_y <= obs_rear_y)
        self.is_crossing_obstacle = in_obstacle_y_zone

        # Clearance above obstacle top: wheel_bottom - obstacle_height
        clearance_above_obs = wheel_bottom_min_z - self.obstacle_h
        in_zone_indices = in_obstacle_y_zone.nonzero(as_tuple=False).flatten()
        if len(in_zone_indices) > 0:
            self.min_obstacle_clearance[in_zone_indices] = torch.minimum(
                self.min_obstacle_clearance[in_zone_indices],
                clearance_above_obs[in_zone_indices]
            )

        # Direct physical contact detection
        obs_fn = torch.norm(self.obstacle_contact_forces, dim=-1)
        self.peak_obstacle_contact_force = torch.maximum(self.peak_obstacle_contact_force, obs_fn)

        # Contact force on robot bodies while in obstacle Y-zone
        robot_body_contact = torch.norm(self.contact_forces, dim=-1)
        for b_idx in range(self.num_bodies):
            if b_idx not in [self.left_wheel_body, self.right_wheel_body]:
                body_fn = robot_body_contact[:, b_idx]
                body_pos = self.rb_states[:, b_idx, 0:3]
                in_obs_reach = (body_pos[:, 1] >= (obs_front_y - 0.05)) & (body_pos[:, 1] <= (obs_rear_y + 0.05))
                hit_chassis = (body_fn > 5.0) & in_obs_reach
                if hit_chassis.any():
                    self.has_obstacle_collision |= hit_chassis

        # Geometric penetration: wheel cylinder below obstacle top while in zone
        geometric_collision = in_obstacle_y_zone & (clearance_above_obs < 0.001)
        force_collision = obs_fn > 2.0

        new_collision = geometric_collision | force_collision
        self.has_obstacle_collision |= new_collision

        # Touchdown rear margin: (wheel_center_y - R) - obs_rear_y
        touchdown_wheel_y = torch.minimum(wheel_L_pos[:, 1], wheel_R_pos[:, 1])
        rear_margin = (touchdown_wheel_y - R) - obs_rear_y
        self.landing_margin_m = rear_margin

        # Takeoff front margin at liftoff: obs_front_y - (takeoff_y + R)
        if self.has_taken_off.any():
            takeoff_wheel_y = self.takeoff_pos_y + R
            self.takeoff_margin_m = obs_front_y - takeoff_wheel_y

        cleared_clean = self.has_touched_down & (rear_margin > 0.005) & (~self.has_obstacle_collision)
        self.obstacle_cleared |= cleared_clean

        # Post-touchdown velocity tracking in observation window
        in_recovery_window = self.has_touched_down & (self.fsm_state_timer >= 0.15) & (self.fsm_state_timer <= 0.80)
        if in_recovery_window.any():
            v_fwd = self._get_world_forward_velocity()
            self.post_touchdown_vels[in_recovery_window] += v_fwd[in_recovery_window]
            self.post_touchdown_samples[in_recovery_window] += 1.0

    def step(self, actions):
        """
        Step simulation with spatial or step-based jump triggering and obstacle interaction.
        """
        # Triggering logic
        obs_center_y = self.env_origins[:, 1] + self.obstacle_py
        obs_front_y = obs_center_y - 0.5 * self.obstacle_w
        robot_front_y = self.root_states[:, 1] + self.wheel_radius
        dist_to_obs_front = obs_front_y - robot_front_y

        if self.trigger_mode == "spatial":
            # Spatial distance trigger: initiate takeoff when reaching optimal takeoff window
            in_balance = (self.fsm_state == self.STATE_BALANCE)
            is_cruising = self.root_states[:, 8] > 0.10
            reached_window = (dist_to_obs_front <= self.spatial_trigger_dist) & (dist_to_obs_front >= -0.02)
            has_warmed_up = self.episode_length_buf >= 20  # Minimum 0.4s to stabilize balance
            can_jump = in_balance & is_cruising & reached_window & has_warmed_up
            if can_jump.any():
                jump_ids = can_jump.nonzero(as_tuple=False).flatten()
                self.peak_obstacle_contact_force[jump_ids] = 0.0
                self.has_obstacle_collision[jump_ids] = False
                self.min_obstacle_clearance[jump_ids] = 1.0
                self.obstacle_cleared[jump_ids] = False
                self.is_crossing_obstacle[jump_ids] = False
                self.takeoff_margin_m[jump_ids] = 0.0
                self.landing_margin_m[jump_ids] = 0.0
                self.post_touchdown_vels[jump_ids] = 0.0
                self.post_touchdown_samples[jump_ids] = 0.0
                self.request_teleop_jump(can_jump)
        else:
            # Legacy step-based trigger for frozen model_3.pt baseline comparison
            step_in_cycle = self.episode_length_buf % self.cycle_length
            can_jump = (step_in_cycle == self.step_trigger) & (self.fsm_state == self.STATE_BALANCE)
            if can_jump.any():
                jump_ids = can_jump.nonzero(as_tuple=False).flatten()
                self.peak_obstacle_contact_force[jump_ids] = 0.0
                self.has_obstacle_collision[jump_ids] = False
                self.min_obstacle_clearance[jump_ids] = 1.0
                self.obstacle_cleared[jump_ids] = False
                self.is_crossing_obstacle[jump_ids] = False
                self.takeoff_margin_m[jump_ids] = 0.0
                self.landing_margin_m[jump_ids] = 0.0
                self.post_touchdown_vels[jump_ids] = 0.0
                self.post_touchdown_samples[jump_ids] = 0.0
                self.request_teleop_jump(can_jump)

        # Execute parent step (renders, steps dual policy, refreshes tensors, handles reset)
        obs, priv_obs, rews, dones, infos = super().step(actions)

        # In continuous mode, reposition the obstacle ahead of the robot whenever recovering to BALANCE
        continuous_mode = getattr(self.cfg.env, 'continuous_mode', False)
        if continuous_mode:
            recovered_to_balance = (self.fsm_state == self.STATE_BALANCE) & self.has_touched_down
            if recovered_to_balance.any():
                for idx in recovered_to_balance.nonzero(as_tuple=False).flatten().tolist():
                    robot_y = self.root_states[idx, 1].item()
                    lead_dist = getattr(self.cfg.obstacle, 'lead_dist', 0.60)
                    new_obs_py = robot_y + lead_dist
                    self.obstacle_py = new_obs_py
                    self.obstacle_root_states[idx, 1] = self.env_origins[idx, 1] + new_obs_py

                    obs_actor_id = torch.tensor([idx * 2 + 1], device=self.device, dtype=torch.int32)
                    all_root_contiguous = self.all_root_states.view(-1, 13).contiguous()
                    self.gym.set_actor_root_state_tensor_indexed(
                        self.sim,
                        gymtorch.unwrap_tensor(all_root_contiguous),
                        gymtorch.unwrap_tensor(obs_actor_id),
                        1
                    )
                    self.has_touched_down[idx] = False
                    self.has_taken_off[idx] = False

        # Update obstacle physics telemetry
        self.compute_obstacle_metrics()

        # Telemetry info dictionary
        mean_post_vel = torch.where(
            self.post_touchdown_samples > 0,
            self.post_touchdown_vels / torch.clamp(self.post_touchdown_samples, min=1.0),
            torch.zeros_like(self.post_touchdown_vels)
        )
        self.extras["obstacle_collision"] = self.has_obstacle_collision.float()
        self.extras["obstacle_cleared"] = self.obstacle_cleared.float()
        self.extras["min_obstacle_clearance"] = self.min_obstacle_clearance
        self.extras["peak_obstacle_contact_force"] = self.peak_obstacle_contact_force
        self.extras["takeoff_margin_m"] = self.takeoff_margin_m
        self.extras["landing_margin_m"] = self.landing_margin_m
        self.extras["mean_post_touchdown_vel"] = mean_post_vel

        return obs, priv_obs, rews, dones, infos

    # =========================================================================
    # Reward Functions: Explicitly Tailored for Physical Obstacle Clearance
    # =========================================================================

    def _reward_obstacle_collision(self):
        """
        Penalizes any collision with the physical obstacle (force or penetration).
        """
        collision_pen = torch.where(self.has_obstacle_collision, torch.ones_like(self.rew_buf), torch.zeros_like(self.rew_buf))
        force_pen = torch.clamp(self.peak_obstacle_contact_force / 100.0, 0.0, 2.0)
        return -(collision_pen + force_pen)

    def _reward_obstacle_clearance(self):
        """
        Rewards vertical wheel clearance above the top of the obstacle while passing over it.
        """
        clr_min, _, _ = compute_exact_cylinder_clearance(self)
        clr_above_obs = clr_min - self.obstacle_h
        target_margin = 0.15  # Desired 15cm margin above hurdle
        reward = torch.clamp(clr_above_obs / target_margin, 0.0, 1.5)
        return torch.where(self.is_crossing_obstacle & self.has_taken_off, reward, torch.zeros_like(reward))

    def _reward_obstacle_cleared_success(self):
        """
        Terminal / event bonus for completely and cleanly clearing the obstacle.
        """
        return torch.where(self.obstacle_cleared, torch.ones_like(self.rew_buf), torch.zeros_like(self.rew_buf))

    def _reward_forward_progress_across_obstacle(self):
        """
        Rewards forward horizontal velocity while airborne over the obstacle zone.
        Prevents learning purely vertical jumps that land on or before the hurdle.
        """
        v_fwd = self._get_world_forward_velocity()
        in_flight_zone = (self.fsm_state == self.STATE_JUMP) & self.has_taken_off
        target_v = self.actual_cmd_vy if hasattr(self, 'actual_cmd_vy') else self.commands[:, 1]
        vel_rew = torch.clamp(v_fwd / target_v, 0.0, 1.25)
        return torch.where(in_flight_zone, vel_rew, torch.zeros_like(vel_rew))
