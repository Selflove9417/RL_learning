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


# ---------------------------------------------------------------------------
# Contact-pair helpers (module level so they are unit-testable without PhysX)
# ---------------------------------------------------------------------------

# Root-relative Y of the WHOLE-BODY rear edge, measured on the standing robot with
# NO height ceiling: `rear_y - base_y = -0.2089 m`. Only used as a KINEMATIC
# cross-check against the geometric whole-body envelope (`whole_body_interval`),
# never as a substitute for it: on a pitching/airborne robot the true rear edge
# moves, which is exactly why the envelope is the authority.
_WHOLEBODY_REAR_OFFSET = -0.2089

def _read_local_pos(contact, slot):
    """Read `localPos{slot}` (a gymapi Vec3 struct) as a (3,) float tensor.

    The struct exposes `.x/.y/.z`, not `[0]/[1]/[2]`, and NumPy cannot cast it
    directly (a structured field of a structured array keeps its own dtype).
    """
    key = 'localPos%d' % slot
    field = contact[key]
    return torch.tensor([float(field['x']), float(field['y']), float(field['z'])],
                        dtype=torch.float)


def contact_point_world(contact, obstacle_root_state, obstacle_is_body0,
                        obstacle_body_index, device=None):
    """World position of the contact point ON THE OBSTACLE surface.

    Isaac Gym reports each pair as (body0, body1) in whatever order PhysX
    produced, together with `localPos0` in body0's frame and `localPos1` in
    body1's frame. The obstacle is a single-body actor whose root state is its
    authoritative pose (identity orientation here, but the maths below does not
    assume that), so:

        obstacle == body0  ->  p_world = p_obs + R_obs @ localPos0
        obstacle == body1  ->  p_world = p_obs + R_obs @ localPos1

    Picking the wrong slot silently yields the point expressed in the ROBOT
    body's frame, transformed as if it were the box's. That is not a small
    numerical error: on the observed pairs it turned every contact y into
    +/-0.02 (the box half-width) instead of 0.28/0.32 (the box faces).

    Args:
        contact: one element of `gym.get_env_rigid_contacts(...)`.
        obstacle_root_state: (7,) tensor [pos(3), quat_xyzw(4)] of the obstacle.
        obstacle_is_body0: True when the obstacle occupies `body0`.
        obstacle_body_index: body index of the obstacle (== env.num_bodies), used
            only to sanity-check that the caller's flag matches the record.
        device: torch device for the result.

    Returns:
        (numpy (3,) float32 world point, point_frame str) or (None, reason).
        `point_frame` is 'obstacle_local0' or 'obstacle_local1' so downstream
        logs can prove which slot was used.
    """
    body0 = int(contact['body0'])
    body1 = int(contact['body1'])
    # Cross-check the flag against the record itself; a mismatch means the caller
    # built the flag from a different pair.
    actual_is_body0 = (body0 == obstacle_body_index)
    actual_is_body1 = (body1 == obstacle_body_index)
    if not (actual_is_body0 or actual_is_body1):
        return None, 'obstacle_not_in_pair'
    if obstacle_is_body0 != actual_is_body0:
        return None, 'flag_mismatch'

    slot = 0 if actual_is_body0 else 1
    local = _read_local_pos(contact, slot)
    if device is not None:
        local = local.to(device)
    quat = obstacle_root_state[3:7]
    if device is not None:
        quat = quat.to(device)
    r_world = quat_rotate(quat.unsqueeze(0), local.unsqueeze(0))[0]
    pos = (obstacle_root_state[0:3].to(r_world.device) + r_world)
    return pos.detach().cpu().numpy(), ('obstacle_local0' if slot == 0
                                       else 'obstacle_local1')


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
        # Per-environment obstacle Y positions. The obstacle centre Y is derived from
        # the actual obstacle rigid-body pose every step, so this tensor is only the
        # authoritative record used for logging/reporting and for continuous
        # repositioning requests. It is NEVER a shared scalar.
        init_py = getattr(self.cfg.obstacle, 'pos_y', 0.300)
        self.obstacle_py_t = torch.full((self.num_envs,), init_py, device=self.device, dtype=torch.float)
        self.trigger_mode = getattr(self.cfg.trigger, 'mode', 'spatial')
        self.spatial_trigger_dist = getattr(self.cfg.trigger, 'spatial_trigger_dist', 0.09)
        self.step_trigger = getattr(self.cfg.trigger, 'step_trigger', 40)

        # Obstacle collision and clearance tracking
        self.has_obstacle_collision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.peak_obstacle_contact_force = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.min_obstacle_clearance = torch.full((self.num_envs,), 1.0, device=self.device, dtype=torch.float)
        self.obstacle_cleared = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.is_crossing_obstacle = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Signed distance from the robot's rear envelope to the box rear face.
        # Positive means the whole body is past the far face.
        self.robot_rear_margin = torch.full((self.num_envs,), -1.0, device=self.device, dtype=torch.float)
        self.takeoff_margin_m = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.landing_margin_m = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.post_touchdown_vels = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.post_touchdown_samples = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # --- TASK-OWNED liftoff / touchdown latches (defect B) ----------------
        # `has_taken_off`, `has_touched_down`, `takeoff_step` and `touchdown_step`
        # are all owned by myrobot_dual_env: its FSM clears them on every state
        # transition (`request_teleop_jump` at :331-336, STATE_JUMP re-entry at
        # :403, and three more sites), because for the DUAL task they mean "this
        # jump cycle". The obstacle task asks a different question -- "has this
        # OBSTACLE attempt seen a real liftoff/touchdown?" -- so it must not read
        # those flags. Measured failure before this latch existed: a robot that
        # had jumped (liftoff step 39, touchdown step 60) was marked
        # CROSSED_WITHOUT_LIFTOFF at step 83 because the FSM had cleared
        # `has_taken_off` while `robot_rear_margin` was legitimately +0.0535:
        #
        #   ctx ep_len=83 fsm=0 root_y=+0.3276 rb_base_y=+0.3276 taken_off=False
        #
        # These three are set once and are only cleared by `reset_idx` or by
        # `_begin_new_obstacle_cycle` -- never by an FSM transition.
        self.task_took_off = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.task_takeoff_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.float)
        self.task_touchdown_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.float)

        # --- SEPARATED passage latches (defect A, tier split) ----------------
        # The three questions the old single envelope tried to answer at once are
        # now three independent latches:
        #   wheel_pass_clean     -- per wheel: it got completely past the far face
        #                           without ever being below the box top while over
        #                           the footprint.  This is the "wheels cleared the
        #                           obstacle" tier (TIER-1).
        #   landing_safe         -- at the moment of touchdown, no wheel was inside
        #                           the box footprint.  Landing BEHIND the box is
        #                           safe; landing ON it is not.
        #   whole_body_passed    -- the whole machine's rear envelope is past the
        #                           far face.  Deliberately NOT required at
        #                           touchdown: a robot that lands just behind the
        #                           box still has its tail hanging over it.
        self.wheel_danger_seen = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.bool)
        self.wheel_pass_clean = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.bool)
        self.landing_safe = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.whole_body_passed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.whole_body_rear_margin = torch.full((self.num_envs,), -1.0, device=self.device, dtype=torch.float)

        # --- One-shot event flags -------------------------------------------
        # These are consumed by the reward function and cleared on the next step,
        # which turns the associated rewards into genuine event rewards instead of
        # a per-step constant.
        #
        # TWO TIERS (Phase B task 1). The Phase A code paid the full success reward
        # on a single `cleared` latch and called it done. On seed 15 that paid out at
        # step 118, two steps before the robot hit the box at step 120 -- the reward
        # was banked for an attempt that then failed. Success is therefore split:
        #
        #   geometric_pass_event : the rear envelope got past the box rear face.
        #                          PHYSICAL PROGRESS ONLY -- modest staged reward.
        #   clear_success_event  : the full mission succeeded -- real liftoff, full
        #                          passage, no obstacle contact at any time, a
        #                          landing, and a sustained forward cruise in the
        #                          recovery window that ended in NORMAL_HANDOFF.
        #
        # A collision at ANY point permanently disarms `success_armed`, so an
        # attempt that passes and then hits the box can never collect tier 2.
        self.collision_event = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.geometric_pass_event = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.clear_success_event = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Armed by a jump-over passage, cleared forever by any collision.
        self.success_armed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Paid at most once per episode.
        self.final_success_paid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.success_disarmed_reason = ["NONE"] * self.num_envs
        # Latches on the first frame of collision evidence, so the disarm reason is
        # written once and stays stable instead of being recomputed every step.
        self._collision_seen = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.task_failed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.failure_reason = ["NONE"] * self.num_envs
        # Number of completed jump cycles (used to make the success reward once-per-cycle)
        self.cycles_completed = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # Set for exactly one step after reset_idx, so consumers can tell a genuine
        # first-step state from residual pre-reset state.
        self.just_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Reset request latched by reset_idx, consumed by _update_task_metrics(post_reset=True)
        self._last_reset_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # --- Task-level contact / event telemetry ---------------------------
        # First obstacle contact of the CURRENT episode, with the responsible body.
        # `first_contact_force` is the per-pair contact FORCE in N, taken from
        # `RigidContact.lambda`; `first_contact_net_force` is the net force on the
        # whole obstacle body from the force tensor. Both are in Newtons.
        #
        # MEASURED, NOT ASSUMED (tests/test_contact_force_units.py): a box of
        # known mass settled on a plane reports lambda summed over its pairs equal
        # to m*g = 78.48 N with ratio 0.999995, and that sum is INVARIANT across
        # dt = 1/200, 1/100, 1/400 (relative spread 1.6e-5) while lambda/(m*g*dt)
        # tracks 1/dt exactly. A quantity that does not care about the timestep is
        # a force, not an impulse. An intermediate version of this file declared
        # lambda to be an impulse in N*s and "corrected" a factor 1/dt = 200 --
        # that correction was the actual bug, and the pre-existing `force` naming
        # had been right.
        self.first_contact_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.first_contact_body = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.first_contact_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.first_contact_force = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.first_contact_net_force = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.first_contact_source = ["none"] * self.num_envs
        # Which localPos slot the world point came from ('obstacle_local0' /
        # 'obstacle_local1' / 'body_origin' / 'none'), so a reversed pair order is
        # visible in the JSON instead of silently shifting the reported point.
        self.first_contact_frame = ["none"] * self.num_envs
        # Liftoff snapshots of the true geometric envelope.
        # `takeoff_rear_y` is what the clearance requirement is measured from: the
        # robot has crossed the box only once its REAR envelope passes the box rear
        # face, so the displacement it still owes is (box_rear - takeoff_rear).
        # Measuring from the front edge instead (as the legacy benchmarks did)
        # mixes two different points of the body and yields a number that can never
        # be compared against a root-frame flight distance.
        self.takeoff_front_y = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.takeoff_rear_y = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.takeoff_clearance_above_obs = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # Step on which the jump was COMMANDED (the trigger fired), which is a
        # different event from the actual liftoff. Kept separately so traces can be
        # cut into the phases the acceptance report needs: before first contact,
        # at trigger, at real liftoff, at box passage and after touchdown.
        self.trigger_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        # Step on which the rear envelope first passed the box rear face (the box
        # passage event), independent of whether the pass was clean.
        self.pass_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        # Contact-pair collection is only available on the CPU pipeline.
        self.contact_pairs_available = not bool(getattr(self.sim_params, 'use_gpu_pipeline', True))

        # --- pre-reset episode snapshot (authoritative evaluation record) ----
        # reset_idx clears the per-episode flags on the same step the episode ends,
        # so anything read after step() returns describes the NEXT episode. This
        # snapshot is written by _capture_episode_snapshot() while the finished
        # episode is still intact.
        self.episode_snapshot = {
            'valid': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            'reset_step': torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long),
            'has_obstacle_collision': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            'has_taken_off': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            'has_touched_down': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            'obstacle_cleared': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            'is_crossing_obstacle': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            'task_failed': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            'fell': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            'min_obstacle_clearance': torch.full((self.num_envs,), 1.0, device=self.device, dtype=torch.float),
            'peak_obstacle_contact_force': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'takeoff_margin_m': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'landing_margin_m': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'takeoff_pos_y': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'touchdown_pos_y': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'takeoff_step': torch.full((self.num_envs,), -1, device=self.device, dtype=torch.float),
            'touchdown_step': torch.full((self.num_envs,), -1, device=self.device, dtype=torch.float),
            'task_took_off': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            'wheel_pass_clean': torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.bool),
            'wheel_danger_seen': torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.bool),
            'landing_safe': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            'whole_body_passed': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            'whole_body_rear_margin': torch.full((self.num_envs,), -1.0, device=self.device, dtype=torch.float),
            'first_contact_step': torch.full((self.num_envs,), -1, device=self.device, dtype=torch.float),
            'first_contact_body': torch.full((self.num_envs,), -1, device=self.device, dtype=torch.float),
            'first_contact_force': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'first_contact_net_force': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'first_contact_pos': torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float),
            'takeoff_front_y': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'takeoff_rear_y': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'takeoff_clearance_above_obs': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'trigger_step': torch.full((self.num_envs,), -1, device=self.device, dtype=torch.float),
            'pass_step': torch.full((self.num_envs,), -1, device=self.device, dtype=torch.float),
            'cycles_completed': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'root_y': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'root_z': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'forward_vel': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            # Geometry of the terminating frame. The trace row an evaluator reads
            # after step() returns describes the NEXT episode whenever that step
            # ended the episode, so "at first contact" / "at touchdown" would show
            # a pristine stand-still pose (y=0.0, gap=0.0979) instead of the impact
            # frame. These fields preserve the real one.
            'front_y': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'rear_y': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'obstacle_front_y': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'wheel_gap_min': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'wheel_clearance_min': torch.full((self.num_envs,), 1.0, device=self.device, dtype=torch.float),
            # Attitude of the terminating frame (see `_last_geometry`). Kept so an
            # impact row can report the real attitude instead of a calm pose.
            'roll_rad': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'pitch_rad': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            # Handoff clock placeholders. The real buffers live in
            # `init_dual_policies` (after `_init_buffers`), so they cannot be
            # referenced here; `_capture_episode_snapshot` copies them at capture
            # time. -1 means "never happened", matching the source buffers.
            'handoff_step': torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long),
            'abort_step': torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long),
            'failure_reason': ["NONE"] * self.num_envs,
            'first_contact_source': ["none"] * self.num_envs,
            'first_contact_frame': ["none"] * self.num_envs,
            'last_transition_event': ["INIT_BALANCE"] * self.num_envs,
            # --- recovery evidence (task 2) -----------------------------------
            # The scalar summary is not enough to decide whether a cruise was
            # sustained: an evaluator that only sees a mean cannot tell a steady
            # 0.70x from a spike followed by a roll-back. The RAW velocity curve
            # and the run counters ride along so the verdict is reproducible
            # from the data rather than from a summary statistic.
            #
            # `recovery_curve` is NOT allocated here on purpose: the ring buffer
            # lives in `init_dual_policies`, which runs after `_init_buffers`.
            # `_capture_episode_snapshot` copies the live curve at capture time,
            # so the snapshot can never be sized differently from the source.
            'recovery_best_hold_steps': torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
            'recovery_regression_steps': torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
            'recovery_grounded_steps': torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
            'recovery_outcome': ["NONE"] * self.num_envs,
            'recovered_sustained': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
        }

        # --- URDF-derived geometric envelopes -------------------------------
        from legged_gym.envs.myrobot.obstacle_geometry import (
            register_link_points, register_link_edges)
        urdf_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        register_link_points(self, urdf_path)
        # Hull EDGES are needed for the exact half-space clip: the forwardmost
        # point of a link sliced at the box-top plane is where an edge crosses the
        # plane, which is not a vertex (defect A, see link_y_extents_clipped).
        register_link_edges(self, urdf_path)

        # Configure viewer side-view camera if interactive
        if self.viewer is not None:
            c_pos = getattr(self.cfg.viewer, 'pos', [1.2, 0.30, 0.30])
            c_lookat = getattr(self.cfg.viewer, 'lookat', [0.0, 0.30, 0.20])
            cam_pos = gymapi.Vec3(c_pos[0], c_pos[1], c_pos[2])
            cam_target = gymapi.Vec3(c_lookat[0], c_lookat[1], c_lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ---------------------------------------------------------------------
    # Backwards-compatible single-scalar view of the obstacle position
    # ---------------------------------------------------------------------
    @property
    def obstacle_py(self):
        """Env-0 obstacle centre Y (read-only compatibility view).

        Historic scripts read `env.obstacle_py` as a scalar. The authoritative
        per-environment source is `obstacle_py_t`; this property exists only so
        old diagnostics/visibility code keeps working and can never be used to
        mutate the simulation state of more than one environment.
        """
        return float(self.obstacle_py_t[0].item()) if self.num_envs > 0 else 0.300

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

        # Reset obstacle root states to nominal fixed poses.
        # NOTE: only the envs in `env_ids` are touched, and the authoritative
        # per-environment record is `obstacle_py_t`. There is no shared scalar that
        # could leak one environment's repositioning into another.
        obs_px = getattr(self.cfg.obstacle, 'pos_x', 0.000)
        obs_py = getattr(self.cfg.obstacle, 'pos_y', 0.300)
        self.obstacle_py_t[env_ids] = obs_py
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

        # (episode_sums are published/zeroed by _finalize_episode_stats at the end
        # of this method, which does not depend on extras["episode"] existing.)

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
        self.robot_rear_margin[env_ids] = -1.0
        self.takeoff_margin_m[env_ids] = 0.0
        self.landing_margin_m[env_ids] = 0.0
        self.post_touchdown_vels[env_ids] = 0.0
        self.post_touchdown_samples[env_ids] = 0.0
        # Task-owned latches: an Episode Reset is one of the only two legal
        # places these are cleared (the other is _begin_new_obstacle_cycle).
        self.task_took_off[env_ids] = False
        self.task_takeoff_step[env_ids] = -1
        self.task_touchdown_step[env_ids] = -1
        # Defect A layer latches. Also reset-free except here and at
        # _begin_new_obstacle_cycle: a pass that happened is a fact about the
        # attempt and must not be re-derived from a later FSM state.
        self.wheel_danger_seen[env_ids] = False
        self.wheel_pass_clean[env_ids] = False
        self.landing_safe[env_ids] = False
        self.whole_body_passed[env_ids] = False
        self.whole_body_rear_margin[env_ids] = -1.0

        # 6. Reset one-shot event / task-state buffers.
        #    This must happen for EVERY reset, unconditionally: a stale event flag
        #    or a stale `task_failed` would otherwise be replayed on the new episode.
        self.collision_event[env_ids] = False
        self.geometric_pass_event[env_ids] = False
        self.clear_success_event[env_ids] = False
        self.success_armed[env_ids] = False
        self.final_success_paid[env_ids] = False
        self._collision_seen[env_ids] = False
        self.task_failed[env_ids] = False
        self.cycles_completed[env_ids] = 0.0
        self.just_reset[env_ids] = True
        self._last_reset_mask[env_ids] = True
        self.first_contact_step[env_ids] = -1
        self.first_contact_body[env_ids] = -1
        self.first_contact_pos[env_ids] = 0.0
        self.first_contact_force[env_ids] = 0.0
        self.first_contact_net_force[env_ids] = 0.0
        self.takeoff_front_y[env_ids] = 0.0
        self.takeoff_rear_y[env_ids] = 0.0
        self.takeoff_clearance_above_obs[env_ids] = 0.0
        self.trigger_step[env_ids] = -1
        self.pass_step[env_ids] = -1
        for eid in env_ids.tolist():
            self.failure_reason[eid] = "NONE"
            self.first_contact_source[eid] = "none"
            self.first_contact_frame[eid] = "none"
            self.success_disarmed_reason[eid] = "NONE"

        # 7. Base-class episode bookkeeping.
        #    `extras["episode"]` is created by LeggedRobot.reset_idx(), which this
        #    class does not call, so the historic `if "episode" in self.extras`
        #    guard silently skipped the episode_sums zeroing on the very first
        #    reset.
        self._finalize_episode_stats(env_ids)

        # 8. Publish the reset poses into the tensor views.
        #    `_reset_root_states` and `_reset_dofs` write through
        #    set_actor_root_state_tensor_indexed / set_dof_state_tensor_indexed,
        #    but Isaac Gym does NOT refresh the state tensors afterwards. Without
        #    this refresh, `rb_states` still holds the PREVIOUS episode's pose, so
        #    the post-reset `_update_task_metrics()` hook would measure a robot
        #    that is still standing at the box (front_y 0.28, wheels at ground
        #    level) and report a phantom -0.05 m clearance plus a
        #    CROSSED_WITHOUT_LIFTOFF failure for a robot that just teleported home.
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

    def _finalize_episode_stats(self, env_ids):
        """Publish per-episode reward sums and zero them for the reset envs.

        Mirrors LeggedRobot.reset_idx()'s episode bookkeeping without relying on
        `extras["episode"]` having been created by a super() call.
        """
        if not hasattr(self, 'episode_sums') or len(self.episode_sums) == 0:
            return
        if 'episode' not in self.extras:
            self.extras['episode'] = {}
        max_len_s = self.max_episode_length_s if self.max_episode_length_s > 0 else 1.0
        for key in self.episode_sums.keys():
            self.extras['episode']['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / max_len_s
            self.episode_sums[key][env_ids] = 0.0


    def _geometric_wheel_intersection(self):
        """EXACT wheel/box intersection, independent of PhysX contact records.

        Returns (mask, gaps) where `mask` is an (N,) bool tensor that is True for
        environments whose wheel collision volume genuinely overlaps the obstacle
        box, and `gaps` is the (N, 2) signed capsule-to-box gap.

        This is deliberately a SEPARATE evidence source from
        `_collect_obstacle_contacts()`:

          * it does not depend on the contact list existing (PhysX only publishes
            contact records after a simulate() step, so a state written directly
            into the tensors has no records until physics runs);
          * it does not depend on the GPU/CPU pipeline;
          * it is exact for the wheel<->box pair, because the box is axis aligned
            and the wheel is a capsule whose axis and centre follow `rb_states`.

        It CANNOT speak for the non-wheel links: those are triangle meshes for
        which no penetration depth is available. A chassis strike is therefore
        only reported once PhysX publishes a non-zero contact force or the
        obstacle reports a net force; see `_collect_obstacle_contacts()`.
        """
        from legged_gym.envs.myrobot import obstacle_geometry as geo
        gaps = geo.wheel_box_signed_gap(self)
        return torch.min(gaps, dim=1).values < 0.0, gaps

    def _collect_obstacle_contacts(self):
        """Attribute genuine obstacle IMPACT to the responsible robot body.

        Isaac Gym's contact list is a speculative-contact list, not an impact
        list. A `RigidContact` record exists as soon as two collision volumes come
        within `contact_offset` (20 mm here) of each other, long before anything
        touches. Empirically, a genuine near-miss produced:

            body0=7(obstacle) body1=3(wheel) lambda=0.0
            minDist=20.000 mm  initialOverlap=True  normal=(0, +0.96, -0.269)

        while the analytic capsule-to-box gap at that exact frame was +9.885 mm.
        Treating bare record existence as a collision charged a full
        `OBSTACLE_COLLISION` episode failure for a clean pass 1 cm clear of the
        box. A record is therefore only accepted as an impact when the solver
        actually pushed, i.e. `lambda > FORCE_EPS_N`. The solver allocates a
        contact force exactly for the pairs it separates, so a non-zero lambda
        is positive proof of contact; a zero lambda is proof of nothing.

        UNITS (this was got wrong in BOTH directions before it was measured)
        -------------------------------------------------------------------
        `RigidContact.lambda` is the contact FORCE in N, not an impulse in N*s.
        This is measured, not assumed -- `tests/test_contact_force_units.py`
        settles a box of known mass (8.0 kg, m*g = 78.48 N) on a plane and reads
        lambda summed over its four contact records: 78.4796 N, i.e.
        lambda/(m*g) = 0.999995. Repeating at dt = 1/200, 1/100, 1/400 leaves
        that sum unchanged (relative spread 1.6e-5) while lambda/(m*g*dt) tracks
        1/dt exactly (199.999 / 100.000 / 400.004, i.e. 1/dt). A quantity that
        does not depend on the timestep cannot be an impulse.

        History, so the next reader does not repeat it: the field was originally
        called `force` (correct), then "corrected" to an impulse in N*s with a
        claimed factor-1/dt = 200 unit error (wrong -- that change WAS the
        error), and is now `force_n` again, backed by the measurement above.

        A per-pair force and the net force on the whole obstacle body are
        different quantities and get different keys: `force_n` (this pair's
        lambda) and `net_force_n` (norm of `obstacle_contact_forces`, all pairs
        summed). Both are Newtons.

        COORDINATES (this used to be wrong too)
        ---------------------------------------
        `localPos0` is expressed in the frame of `body0` and `localPos1` in the
        frame of `body1`. The obstacle may appear in EITHER slot depending on the
        order PhysX happened to report the pair, so the point on the obstacle
        surface must be taken from the slot that actually holds the obstacle:

            obstacle == body0  ->  world = p_obs + R_obs @ localPos0
            obstacle == body1  ->  world = p_obs + R_obs @ localPos1

        Using `localPos0` unconditionally (as the first version did) silently
        produced the ROBOT-frame point whenever the order was reversed, and in
        the observed order it produced numbers like y == +/-0.02 for every
        contact -- the box's own half-width -- which read as plausible world
        coordinates but were not.

        `contact_point_world()` implements that choice as a pure function so the
        reversed-order case is unit-testable without PhysX.

        Genuine geometric overlap of a wheel with the box is handled separately and
        independently by `_geometric_wheel_intersection()`, because PhysX publishes
        no records at all for a state that was written into the tensors without a
        simulate() step.

        Returns a list of dicts with keys:
            env_idx, body_idx, world_pos, force_n, net_force_n,
            body_force_n, point_frame, source
        `source` is 'contact_pair' (a real impact pair) or 'force_attribution'
        (the obstacle reported force, no pair identity). The latter CANNOT name a
        contact pair and must never be reported as one; for it `force_n` is
        None because no per-pair lambda is attributable.
        """
        # A settled mass reports m*g of distributed force; a single recorded pair
        # carries a share well above 1e-3 N whenever the solver actually pushed.
        # Zero (or numerically zero) lambda is a speculative proximity record.
        FORCE_EPS_N = 1e-3

        found = []
        if self.contact_pairs_available:
            for env_idx in range(self.num_envs):
                contacts = self.gym.get_env_rigid_contacts(self.envs[env_idx])
                if contacts is None or len(contacts) == 0:
                    continue
                for contact in contacts:
                    body0 = int(contact['body0'])
                    body1 = int(contact['body1'])
                    # The obstacle actor occupies body index == num_bodies.
                    if body0 == self.num_bodies:
                        other, obstacle_is_body0 = body1, True
                    elif body1 == self.num_bodies:
                        other, obstacle_is_body0 = body0, False
                    else:
                        continue
                    if other < 0:
                        # obstacle against a static/shared collider, no robot body
                        continue
                    lam = float(contact['lambda'])
                    if lam <= FORCE_EPS_N:
                        # Speculative proximity record; the solver pushed nothing.
                        continue
                    world_pos, point_frame = contact_point_world(
                        contact, self.obstacle_root_states[env_idx, 0:7],
                        obstacle_is_body0=obstacle_is_body0,
                        obstacle_body_index=self.num_bodies)
                    if world_pos is None:
                        continue
                    found.append({
                        'env_idx': env_idx,
                        'body_idx': other,
                        'world_pos': world_pos,
                        'force_n': lam,
                        'net_force_n': float(torch.norm(
                            self.obstacle_contact_forces[env_idx]).item()),
                        'body_force_n': float(torch.norm(
                            self.contact_forces[env_idx, other]).item()),
                        'point_frame': point_frame,
                        'source': 'contact_pair',
                    })
        if found:
            return found

        # Fallback: force attribution over robot bodies (no pair identity).
        # `obstacle_contact_forces` is the obstacle's own net force, so a non-zero
        # value proves something pushed the box. It is still NOT a contact pair,
        # and no per-pair force can be attributed from it.
        obstacle_fn = torch.norm(self.obstacle_contact_forces, dim=-1)
        body_fn = torch.norm(self.contact_forces, dim=-1)
        for env_idx in range(self.num_envs):
            if obstacle_fn[env_idx].item() <= 2.0:
                continue
            per_body = body_fn[env_idx]
            best = int(torch.argmax(per_body).item())
            if per_body[best].item() <= 2.0:
                continue
            pos = self.rb_states[env_idx, best, 0:3].detach().cpu().numpy()
            found.append({
                'env_idx': env_idx,
                'body_idx': best,
                'world_pos': pos,
                'force_n': None,
                'net_force_n': float(obstacle_fn[env_idx].item()),
                'body_force_n': float(per_body[best].item()),
                'point_frame': 'body_origin',
                'source': 'force_attribution',
            })
        return found

    def compute_obstacle_metrics(self, post_reset=False):
        """Refresh all obstacle interaction metrics for the CURRENT physics state.

        Args:
            post_reset: True when called immediately after `reset_idx`. The
                contact-force tensor still holds the PREVIOUS episode's values
                (Isaac Gym refreshes it after physics, not after a state write),
                so contact evidence must be skipped entirely. Reporting it would
                make every freshly reset env appear to have collided.

        Geometry discipline:
          * obstacle interval and each wheel's interval are EXACT closed forms
            (axis-aligned box; analytic cylinder/capsule transformed by rb_states);
          * non-wheel links use a conservative support-point outer bound from the
            URDF meshes, which is an UPPER BOUND and is never reported as a
            penetration depth;
          * true geometric intersection is only claimed when a real rigid-body
            contact pair exists (CPU pipeline). Elsewhere the collision flags are
            labelled as force/containment evidence.
        """
        from legged_gym.envs.myrobot import obstacle_geometry as geo

        # --- exact obstacle interval, per environment (no shared scalar) -----
        obs_front_y, obs_rear_y, obs_top_z = geo.obstacle_y_interval(self)

        # --- LAYER 1: approach envelope (height-limited, no root fallback) ----
        # Only geometry that can physically reach the box counts: anything above
        # the obstacle top plane passes over it.  Without this ceiling the
        # base_link mesh (which sits ~0.4 m up) reports a front edge 0.22 m ahead
        # of the wheels and the spatial trigger fires immediately at reset.
        #
        # Defect A: this envelope may NOT be reused for a "cleared the box"
        # decision. Its rear edge is the rear of the LOW parts (measured +0.0466 m
        # relative to the root, i.e. AHEAD of the root), not the rear of the
        # robot (measured -0.2089 m). And it no longer falls back to the root Y
        # when the robot is airborne; `could_reach` says so explicitly.
        approach = geo.approach_envelope(self)
        approach_front_y = approach['front_y']
        robot_front_y = approach_front_y
        could_reach = approach['could_reach']
        approach_rear_y = approach['rear_y']

        # --- exact per-wheel passage facts (LAYER 2) --------------------------
        wheel_state = geo.wheel_crossing_state(self)
        wheel_iv = approach['exact_wheels']
        wheel_in_zone_each = wheel_state['overlaps_footprint']

        # --- vertical clearance of the wheels above the obstacle top ---------
        # EXACT: the lowest point of the part of each wheel that is horizontally
        # over the box footprint, minus the box top. `+inf` when the wheel cannot
        # reach the footprint at all. See `wheel_clearance_above_box_top` for why
        # `wheel_bottom_z - box_top_z` is NOT usable here.
        wheel_clearances = geo.wheel_clearance_above_box_top(self)

        # --- containment evidence: Y-interval overlap (NOT penetration) ------
        # Two world Y intervals [lo, hi] overlap iff hi_a >= lo_b and lo_a <= hi_b.
        # This is a projection test on a conservative envelope. It says nothing
        # about whether solids actually touch, and it must never be used as a
        # collision test: on seed 15 the right wheel's front edge reached 0.2825
        # (box front 0.28) while the analytic wheel/box gap was still +0.43 mm.
        # Its only job is to tell "some part of the robot is above the box
        # footprint", which gates the clearance reward.
        wheel_in_zone = wheel_in_zone_each.any(dim=1)
        self.is_crossing_obstacle = wheel_in_zone

        # --- clearance above the box top, for wheels actually over it --------
        # A wheel outside the footprint contributes nothing: it has no clearance
        # relative to a box it is not above. Such wheels yield +inf, filtered out.
        masked = torch.where(wheel_in_zone_each, wheel_clearances,
                             torch.full_like(wheel_clearances, float('inf')))
        clr_in_zone = torch.min(masked, dim=1).values
        real = wheel_in_zone & torch.isfinite(clr_in_zone)
        if real.any():
            idx = real.nonzero(as_tuple=False).flatten()
            self.min_obstacle_clearance[idx] = torch.minimum(
                self.min_obstacle_clearance[idx], clr_in_zone[idx])

        # --- LAYER 2 latches: per-wheel safe passage + safe landing -----------
        # `danger` is the only predicate that means "this wheel could hit the box
        # while over it": the wheel's Y interval overlaps the footprint AND its
        # lowest point is below the box top. Overlap alone is proximity.
        self.wheel_danger_seen |= wheel_state['danger']
        clean_now = wheel_state['passed_rear'] & (~self.wheel_danger_seen)
        self.wheel_pass_clean |= clean_now
        landing_now = (self.has_touched_down & (~self.landing_safe)
                       & (~wheel_state['overlaps_footprint'].any(dim=1)))
        if landing_now.any():
            self.landing_safe |= landing_now

        # --- LAYER 3: whole-body envelope (full hull, no height clip) --------
        # `robot_rear_margin` is documented as "the whole body is past the far
        # face" and is now actually whole-body. It is the only honest "the robot
        # crossed the obstacle line" measure and it is what the plough-through
        # detector needs: a robot that walks into the box never gets its tail past
        # the far face without leaving the ground.
        whole = geo.whole_body_interval(self)
        robot_whole_rear_y = whole['rear_y']
        robot_rear_y = robot_whole_rear_y
        self.whole_body_rear_margin = robot_whole_rear_y - obs_rear_y
        self.robot_rear_margin = self.whole_body_rear_margin
        whole_now = self.robot_rear_margin > 0.005
        self.whole_body_passed |= whole_now
        # Box passage: recorded once per cycle, and independent of whether the
        # pass was clean, so a trace can be segmented at "the robot is past the
        # box" without implying success.
        just_passed = whole_now & (self.pass_step < 0)
        if just_passed.any():
            self.pass_step[just_passed] = self.common_step_counter

        # --- contact evidence: per-cycle independent peak ---------------------
        if post_reset:
            # Stale tensors from the finished episode -- see the method docstring.
            contacts = []
            obs_fn = None
            wheel_intersect = torch.zeros(self.num_envs, dtype=torch.bool,
                                          device=self.device)
            gaps = None
        else:
            obs_fn = torch.norm(self.obstacle_contact_forces, dim=-1)
            self.peak_obstacle_contact_force = torch.maximum(
                self.peak_obstacle_contact_force, obs_fn)
            contacts = self._collect_obstacle_contacts()
            # Independent EXACT geometric evidence for the wheel<->box pair.
            wheel_intersect, gaps = self._geometric_wheel_intersection()

        new_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for rec in (contacts or []):
            env_idx = rec['env_idx']
            new_collision[env_idx] = True
            if self.first_contact_step[env_idx].item() < 0:
                self.first_contact_step[env_idx] = self.common_step_counter
                self.first_contact_body[env_idx] = rec['body_idx']
                # Both fields are Newtons. `force_n` is the impulse solver's
                # lambda for THIS pair; `net_force_n` is the net force acting on
                # the whole obstacle body (all of its pairs summed). They answer
                # different questions -- "how hard did this body hit" versus "how
                # hard is the box being pushed overall" -- so they stay separate
                # keys, but they share a unit and a magnitude scale.
                self.first_contact_force[env_idx] = (
                    rec['force_n'] if rec['force_n'] is not None else 0.0)
                self.first_contact_net_force[env_idx] = rec['net_force_n']
                self.first_contact_pos[env_idx] = torch.as_tensor(
                    rec['world_pos'], device=self.device, dtype=torch.float)
                self.first_contact_source[env_idx] = rec['source']
                self.first_contact_frame[env_idx] = rec['point_frame']
        if not post_reset and wheel_intersect.any():
            # Exact capsule/box overlap. Attribute it to the deeper of the two
            # wheels; `gaps` holds a signed distance, so the smaller value is the
            # more deeply involved wheel.
            for env_idx in wheel_intersect.nonzero(as_tuple=False).flatten().tolist():
                new_collision[env_idx] = True
                if self.first_contact_step[env_idx].item() < 0:
                    deeper = int(torch.argmin(gaps[env_idx]).item())
                    body_idx = (self.left_wheel_body if deeper == 0
                                else self.right_wheel_body)
                    self.first_contact_step[env_idx] = self.common_step_counter
                    self.first_contact_body[env_idx] = body_idx
                    # No contact pair exists for a purely geometric overlap, so
                    # there is no per-pair lambda: only the net force is known.
                    self.first_contact_net_force[env_idx] = float(
                        torch.norm(self.obstacle_contact_forces[env_idx]).item())
                    self.first_contact_pos[env_idx] = self.rb_states[
                        env_idx, body_idx, 0:3].detach()
                    self.first_contact_source[env_idx] = "analytic_geometry"
        if new_collision.any():
            self.has_obstacle_collision |= new_collision
            self.collision_event |= new_collision

        # --- task-owned liftoff / touchdown latches (defect B) ----------------
        # The FSM clears `has_taken_off`/`has_touched_down` on its own transitions,
        # so the obstacle task observes them as EDGES and latches its own copy.
        # `task_took_off` can therefore only ever be cleared by reset_idx or by
        # _begin_new_obstacle_cycle -- never by a JUMP/RECOVER/BALANCE transition.
        took_off_now = self.has_taken_off & (~self.task_took_off)
        if took_off_now.any():
            self.task_took_off |= took_off_now
            self.task_takeoff_step[took_off_now] = float(self.common_step_counter)
        touched_down_now = (self.has_touched_down & self.task_took_off
                            & (self.task_touchdown_step < 0))
        if touched_down_now.any():
            self.task_touchdown_step[touched_down_now] = float(self.common_step_counter)

        # --- liftoff snapshot of the true envelope ---------------------------
        # Keyed on the TASK-OWNED rising edge, not on the FSM flag. The old form
        # (`has_taken_off & takeoff_step == common_step_counter`) silently missed
        # the snapshot whenever the FSM had already reset `takeoff_step` to -1.
        fresh_liftoff = took_off_now
        if fresh_liftoff.any():
            idx = fresh_liftoff.nonzero(as_tuple=False).flatten()
            self.takeoff_front_y[idx] = robot_front_y[idx]
            self.takeoff_rear_y[idx] = robot_rear_y[idx]
            # +inf means "no wheel is over the box footprint at liftoff", which is
            # the normal case for a takeoff before the box. Store 0 rather than inf
            # so downstream JSON stays finite and unambiguous.
            lift_clr = torch.where(torch.isfinite(clr_in_zone[idx]), clr_in_zone[idx],
                                   torch.zeros_like(clr_in_zone[idx]))
            self.takeoff_clearance_above_obs[idx] = lift_clr
            self.takeoff_margin_m[idx] = obs_front_y[idx] - robot_front_y[idx]

        # --- Clearing evaluation: TIER 1, geometric passage -------------------
        # `rear_margin > 0.005` proves the whole robot's Y envelope is 5 mm past
        # the box rear face. This is PHYSICAL PROGRESS, not mission success: on
        # seed 15 the robot got here at step 118 and hit the box at step 120.
        rear_margin = robot_rear_y - obs_rear_y
        if self.has_touched_down.any():
            idx = self.has_touched_down.nonzero(as_tuple=False).flatten()
            self.landing_margin_m[idx] = rear_margin[idx]

        # Passage requires: real liftoff (otherwise the robot walked/ploughed
        # through the box line), a real touchdown, no contact so far, and the rear
        # envelope past the far face. All three state predicates come from the
        # TASK-OWNED latches: the FSM's `has_taken_off`/`has_touched_down` are
        # cleared on state transitions and would drop a legitimate pass on the
        # floor whenever the handoff happened to land on the same step.
        pass_clean = (self.task_took_off & (self.task_touchdown_step >= 0)
                      & (rear_margin > 0.005)
                      & (~self.has_obstacle_collision) & (~self.obstacle_cleared))
        if pass_clean.any():
            self.obstacle_cleared |= pass_clean
            self.geometric_pass_event |= pass_clean
            self.success_armed |= pass_clean

        # --- Clearing evaluation: TIER 2, full mission success ----------------
        # Every clause is a separate physical fact, and the reward is paid on the
        # step all of them are simultaneously true, at most once per episode:
        #   1. real liftoff            -> has_taken_off / takeoff_step latched
        #   2. full passage            -> obstacle_cleared (rear envelope past face)
        #   3. no obstacle contact     -> has_obstacle_collision must be False
        #   4. successful landing      -> touchdown_step latched, not a fall
        #   5. sustained forward cruise in the recovery window, ending in NORMAL_HANDOFF
        # Clauses 3 and 5 are the ones that the Phase A event omitted, which is
        # exactly why a pass-then-collide attempt could still collect the reward.
        recovered_sustained = self._recovery_succeeded()
        mission_success = self._mission_success_mask(recovered_sustained)
        if mission_success.any():
            self.clear_success_event |= mission_success
            self.final_success_paid |= mission_success

        # --- post-touchdown recovery velocity window -------------------------
        in_recovery_window = (self.has_touched_down & (self.fsm_state_timer >= 0.15)
                              & (self.fsm_state_timer <= 0.80))
        if in_recovery_window.any():
            v_fwd = self._get_world_forward_velocity()
            self.post_touchdown_vels[in_recovery_window] += v_fwd[in_recovery_window]
            self.post_touchdown_samples[in_recovery_window] += 1.0

        # --- geometry of this frame, kept for _capture_episode_snapshot ------
        # The snapshot runs one step later (right before reset_idx), by which time
        # these locals are gone. Storing them here is what lets the terminating
        # frame be reported accurately instead of as the post-reset stand-still.
        # `wheel_gap_min` needs a live PhysX-derived gap, so on the post-reset
        # pass (stale tensors) it is zeroed rather than carried over.
        clr_min_finite = torch.where(
            torch.isfinite(clr_in_zone), clr_in_zone,
            torch.full_like(clr_in_zone, 1.0)).detach()
        if post_reset:
            gap_min = torch.zeros_like(obs_front_y)
        else:
            gap_min = torch.min(gaps, dim=1).values.detach()
        # Attitude/velocity of the same frame ride along so that a terminating
        # trace row can report the REAL impact attitude. Without them the row is
        # rebuilt from `root_y`/`root_z` alone and a 40-degree impact pitch would
        # read as a calm upright pose.
        r_t, p_t, _ = get_euler_xyz(self.base_quat)
        roll_t = torch.atan2(torch.sin(p_t), torch.cos(p_t))
        pitch_t = torch.atan2(torch.sin(r_t), torch.cos(r_t))
        self._last_geometry = {
            'front_y': robot_front_y.detach(),
            'rear_y': robot_rear_y.detach(),
            'obstacle_front_y': obs_front_y.detach(),
            'wheel_gap_min': gap_min,
            'wheel_clearance_min': clr_min_finite,
            'roll_rad': roll_t.detach(),
            'pitch_rad': pitch_t.detach(),
            'forward_vel': self.root_states[:, 8].detach(),
        }

        # --- THREE-LAYER per-step record (defect A) ---------------------------
        # The task book requires that every step publish, separately:
        #   * the wheel front/rear envelopes (layer 2, exact analytic),
        #   * the root displacement relative to the box rear face (a KINEMATIC
        #     fact, never a substitute for the geometric envelope),
        #   * the real PhysX contact state (layer independent of any envelope),
        #   * the final whole-body passage state (layer 3).
        # Keeping them in one dict makes the separation auditable: a caller can no
        # longer read "cleared" and be unable to tell which of the three it meant.
        root_rear_offset = self.root_states[:, 1] + _WHOLEBODY_REAR_OFFSET
        self.obstacle_layers = {
            # (a) approach: height-limited, no root fallback
            'approach_front_y': approach_front_y.detach(),
            'approach_rear_y': approach_rear_y.detach(),
            'approach_could_reach': could_reach.detach(),
            # (b) wheels: exact per-wheel front/rear + safety
            'wheel_front_y': wheel_state['interval_front'].detach(),
            'wheel_rear_y': wheel_state['interval_rear'].detach(),
            'wheel_bottom_z': wheel_state['bottom_z'].detach(),
            'wheel_over_footprint': wheel_state['overlaps_footprint'].detach(),
            'wheel_above_top': wheel_state['above_top'].detach(),
            'wheel_danger': wheel_state['danger'].detach(),
            'wheel_pass_clean': self.wheel_pass_clean.detach(),
            'landing_safe': self.landing_safe.detach(),
            # (c) whole body: full hull, no clip
            'whole_body_front_y': whole['front_y'].detach(),
            'whole_body_rear_y': robot_whole_rear_y.detach(),
            'whole_body_rear_margin': self.whole_body_rear_margin.detach(),
            'whole_body_passed': self.whole_body_passed.detach(),
            # kinematic cross-check: root rear face estimate, NOT an envelope
            'root_estimated_rear_y': root_rear_offset.detach(),
            'root_estimated_margin': (root_rear_offset - obs_rear_y).detach(),
            # real contact evidence, independent of every envelope above
            'has_obstacle_collision': self.has_obstacle_collision.detach(),
            'peak_obstacle_contact_force_N': self.peak_obstacle_contact_force.detach(),
            'box_front_y': obs_front_y.detach(),
            'box_rear_y': obs_rear_y.detach(),
            'box_top_z': obs_top_z.detach(),
        }

    # =========================================================================
    # Task hooks consumed by MyRobotRunningJump.step()
    # =========================================================================

    def _recovery_succeeded(self):
        """True only for a genuinely recovered forward cruise.

        Requires ALL of:
          * the RECOVER state exited with the NORMAL_HANDOFF event -- a
            TIMEOUT_HANDOFF or RECOVER_ABORT is explicitly NOT a success and must
            never be counted as one;
          * a consecutive qualifying run of at least RECOVERY_HOLD_STEPS steps
            (`recovery_best_hold_steps`, which survives later disturbances);
          * forward velocity never ran backwards for a sustained stretch;
          * both wheels were loaded at some point during the recovery window
            (`recovery_grounded_steps > 0`), i.e. the gait actually re-established
            ground contact rather than the robot balancing on one wheel.
        """
        event_ok = [v == "NORMAL_HANDOFF" for v in getattr(self, 'recovery_outcome',
                                                          ["NONE"] * self.num_envs)]
        hold = getattr(self, 'recovery_best_hold_steps',
                       torch.zeros(self.num_envs, device=self.device, dtype=torch.long))
        regress = getattr(self, 'recovery_regression_steps',
                          torch.zeros(self.num_envs, device=self.device, dtype=torch.long))
        grounded_steps = getattr(self, 'recovery_grounded_steps',
                                 torch.zeros(self.num_envs, device=self.device, dtype=torch.long))
        hold_ok = hold >= getattr(self, 'RECOVERY_HOLD_STEPS', 10)
        regress_ok = regress <= 1
        grounded_ok = grounded_steps > 0
        event_mask = torch.tensor(event_ok, device=self.device, dtype=torch.bool)
        return event_mask & hold_ok & regress_ok & grounded_ok

    def _mission_success_mask(self, recovered_sustained=None):
        """TIER 2: the full mission actually succeeded.

        Extracted from `compute_obstacle_metrics` so the counter-example tests can
        drive the gate directly instead of having to fake a whole physics episode.

        Each clause is one physical fact and each is load-bearing:

          success_armed        -- a clean geometric passage happened first, and no
                                  collision has occurred at any point (the arming
                                  latch is revoked on collision, see
                                  `_update_task_metrics`)
          obstacle_cleared     -- the rear envelope is past the far face
          no collision         -- a jump that clips the box is a failed jump, so
                                  the `~has_obstacle_collision` clause is checked
                                  again here rather than trusted from the latch
          not task_failed      -- geometric plough-through also disqualifies
          takeoff/touchdown    -- real liftoff AND real landing were latched by the
                                  TASK, not by the FSM; a pass with
                                  `task_touchdown_step < 0` means the episode
                                  ended before the robot came down. Reading the
                                  FSM's `takeoff_step` here was wrong twice over:
                                  `request_teleop_jump` resets it to -1 on every
                                  new jump request, and the dual task owns it.
          recovered_sustained  -- a genuine sustained forward cruise followed the
                                  landing (see `_recovery_succeeded`)
          not already paid     -- at most one payment per episode
        """
        if recovered_sustained is None:
            recovered_sustained = self._recovery_succeeded()
        return (self.success_armed
                & self.obstacle_cleared
                & (~self.has_obstacle_collision)
                & (~self.task_failed)
                & (self.task_takeoff_step >= 0)
                & (self.task_touchdown_step >= 0)
                & recovered_sustained
                & (~self.final_success_paid))

    def _update_task_metrics(self, post_reset=False):
        """Refresh obstacle metrics inside the parent step pipeline.

        post_reset=False -> called AFTER physics/FSM and BEFORE termination/reward.
        post_reset=True  -> called AFTER reset_idx. Deliberately evaluates NOTHING;
                            see the early return below.

        Why the post-reset pass cannot evaluate geometry
        -----------------------------------------------
        `reset_idx` already zeroes every task buffer (``min_obstacle_clearance``
        -> 1.0, ``robot_rear_margin`` -> -1.0, ``pass_step`` -> -1, all event
        flags -> False), so there is nothing left for this hook to clean up. What
        it *would* do is re-derive those buffers from ``rb_states`` -- and on the
        GPU pipeline ``rb_states`` does NOT yet describe the freshly reset
        episode. `set_actor_root_state_tensor` followed by both
        `refresh_actor_root_state_tensor` and `refresh_rigid_body_state_tensor`
        leaves ``rb_states`` at the PREVIOUS episode's pose until the next
        `simulate()`; the CPU pipeline publishes it immediately. Measured with
        2 envs on the terminal pose ``base_y = 0.62`` (robot fully past the box):

            after reset_idx:  root y = 0.0000   rb_states base y = 0.6201   (GPU)
            after reset_idx:  root y = -0.0000  rb_states base y = -0.0000  (CPU)

        Evaluating that stale pose marks the brand-new episode as
        ``CROSSED_WITHOUT_LIFTOFF`` (``robot_rear_margin > 0.005`` while
        ``has_taken_off`` is False), which then terminates the fresh episode on
        its very first step -- silently destroying one episode per success on
        GPU. Measured before this guard: ``post_reset: task_failed=[True, True]
        reason=['CROSSED_WITHOUT_LIFTOFF', 'CROSSED_WITHOUT_LIFTOFF']``.

        Nothing is lost by skipping: the first real `step()` recomputes every one
        of these values from live physics before termination and reward run.
        """
        if post_reset:
            # Consume the latch so a later non-post_reset pass is not mislabelled.
            self.just_reset |= self._last_reset_mask
            self._last_reset_mask[:] = False
            return

        # one-shot events live for exactly one reward evaluation
        self.collision_event[:] = False
        self.geometric_pass_event[:] = False
        self.clear_success_event[:] = False
        self.compute_obstacle_metrics(post_reset=False)

        # A collision at ANY time permanently disarms tier-2 success for this
        # episode. Without this, a pass-then-collide attempt would keep
        # `success_armed` set and could still collect the full reward later in the
        # same episode -- precisely the seed 15 defect (cleared at 118, collided at
        # 120). The geometric passage itself stays recorded as tier 1; only the
        # mission-success eligibility is revoked.
        # Record why tier-2 success was lost, exactly once per episode, on the
        # first collision. "AFTER_PASSAGE" is the seed-15 signature (geometric
        # passage first, impact later); "BEFORE_PASSAGE" is the head-on case.
        first_collision = self.has_obstacle_collision & ~self._collision_seen
        if first_collision.any():
            self._collision_seen |= first_collision
            for idx in first_collision.nonzero(as_tuple=False).flatten().tolist():
                self.success_disarmed_reason[idx] = (
                    "COLLISION_AFTER_PASSAGE" if bool(self.obstacle_cleared[idx].item())
                    else "COLLISION_BEFORE_PASSAGE")
        self.success_armed &= ~self.has_obstacle_collision
        self._update_task_outcome()

    def _task_failure_mask(self):
        """Terminate on mission failure.

        Rules (see task spec item 1.5/4.5):
          * ANY obstacle contact is a violation of the clearance task and must end
            the episode -- a jumped-over obstacle is never allowed to be touched,
            not even by a glancing wheel strike.
          * Hard instability (already covered by the fall test) is not duplicated.
          * `task_failed` latches failures detected by the geometric envelope so a
            chassis strike detected without a contact pair still terminates.
        """
        return self.has_obstacle_collision | self.task_failed

    def _capture_episode_snapshot(self):
        """Freeze the terminating episode's outcome before reset_idx wipes it.

        `reset_idx` zeroes `has_obstacle_collision`, `has_taken_off`,
        `has_touched_down`, `obstacle_cleared`, `min_obstacle_clearance` and
        `failure_reason`, and it runs on the same step the episode ends. An
        evaluator reading those buffers after `step()` returns therefore sees the
        FRESH episode rather than the one that just finished -- the historic
        `tests/benchmark_obstacle_jump.py` reported `touchdown_pos_y=0.0` and
        `flight_dist=-0.106 m` for exactly this reason (both were post-reset reads).

        This snapshot is the authoritative record of the finished episode.
        """
        snap = getattr(self, 'episode_snapshot', None)
        if snap is None:
            return
        reset_mask = self.reset_buf
        if not bool(reset_mask.any()):
            return
        tensors = {
            'has_obstacle_collision': self.has_obstacle_collision,
            'has_taken_off': self.has_taken_off,
            'has_touched_down': self.has_touched_down,
            'obstacle_cleared': self.obstacle_cleared,
            'is_crossing_obstacle': self.is_crossing_obstacle,
            'task_failed': self.task_failed,
            'fell': getattr(self, 'fall_buf',
                            torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)),
            'min_obstacle_clearance': self.min_obstacle_clearance,
            'peak_obstacle_contact_force': self.peak_obstacle_contact_force,
            'takeoff_margin_m': self.takeoff_margin_m,
            'landing_margin_m': self.landing_margin_m,
            'takeoff_pos_y': self.takeoff_pos_y,
            'touchdown_pos_y': self.touchdown_pos_y,
            # These two are the TASK-OWNED latches, not the FSM's. The FSM's
            # `takeoff_step`/`touchdown_step` are reset to -1 by
            # `request_teleop_jump` on every new jump request, so a multi-jump
            # episode would report -1 for a jump that plainly happened.
            'takeoff_step': self.task_takeoff_step.clone(),
            'touchdown_step': self.task_touchdown_step.clone(),
            'task_took_off': self.task_took_off.clone(),
            # Defect A: the three layer latches are recorded separately in the
            # episode snapshot. A single "cleared" flag cannot tell a reader
            # whether the wheels got past, whether the landing was clear, or
            # whether the whole body finally left the box region.
            'wheel_pass_clean': self.wheel_pass_clean.clone(),
            'wheel_danger_seen': self.wheel_danger_seen.clone(),
            'landing_safe': self.landing_safe.clone(),
            'whole_body_passed': self.whole_body_passed.clone(),
            'whole_body_rear_margin': self.whole_body_rear_margin.clone(),
            'first_contact_step': self.first_contact_step.float(),
            'first_contact_body': self.first_contact_body.float(),
            'first_contact_force': self.first_contact_force,
            'first_contact_net_force': self.first_contact_net_force,
            'takeoff_front_y': self.takeoff_front_y,
            'takeoff_rear_y': self.takeoff_rear_y,
            'takeoff_clearance_above_obs': self.takeoff_clearance_above_obs,
            'trigger_step': self.trigger_step.float(),
            'pass_step': self.pass_step.float(),
            'cycles_completed': self.cycles_completed,
            'root_y': self.root_states[:, 1],
            'root_z': self.root_states[:, 2],
            'forward_vel': self.root_states[:, 8],
            # Handoff clock (task 4). `handoff_step` is stamped when the FSM
            # leaves RECOVER for BALANCE, `abort_step` when it leaves by the
            # abort branch. Both are needed to answer "how long after landing did
            # the balance policy take over" -- and, crucially, to show that a
            # TIMEOUT_HANDOFF is a takeover by ABANDONING the recovery objective,
            # not a recovered cruise.
            'handoff_step': self.handoff_step,
            'abort_step': self.abort_step,
        }
        # Geometry of the terminating frame, captured by compute_obstacle_metrics
        # on the pre-reset pass. Without these, an evaluator's "at first contact"
        # / "at touchdown" rows would silently be the next episode's stand-still.
        for name, value in getattr(self, '_last_geometry', {}).items():
            tensors[name] = value
        for name, tensor in tensors.items():
            snap[name][reset_mask] = tensor.detach()[reset_mask]
        # The three-layer geometry of the terminating frame, kept as its own
        # tensor dict rather than merged into `snap`: these are per-row quantities
        # an evaluator must be able to read for the impact frame specifically. A
        # reader that only had the post-reset `env.obstacle_layers` would see the
        # fresh episode's stand-still geometry in the row that is supposed to
        # document the crash.
        layers = getattr(self, 'obstacle_layers', None)
        if layers:
            snap.setdefault('obstacle_layers', {})
            for name, tensor in layers.items():
                if torch.is_tensor(tensor):
                    snap['obstacle_layers'].setdefault(
                        name, torch.zeros_like(tensor))
                    snap['obstacle_layers'][name][reset_mask] = \
                        tensor.detach().clone()[reset_mask]
        snap['first_contact_pos'][reset_mask] = self.first_contact_pos.detach()[reset_mask]
        # Recovery evidence (task 2). The ring buffer is copied as a whole row,
        # NaN padding included, so the evaluator can reconstruct the raw curve
        # in time order instead of trusting a mean.
        curve = getattr(self, 'recovery_curve', None)
        if curve is not None:
            snap['recovery_curve'] = curve.detach().clone()
            head = getattr(self, 'recovery_curve_head', None)
            if head is not None:
                snap['recovery_curve_head'] = head.detach().clone()
        snap['recovery_best_hold_steps'][reset_mask] = \
            self.recovery_best_hold_steps[reset_mask]
        snap['recovery_regression_steps'][reset_mask] = \
            self.recovery_regression_steps[reset_mask]
        snap['recovery_grounded_steps'][reset_mask] = \
            self.recovery_grounded_steps[reset_mask]
        snap['recovered_sustained'][reset_mask] = self._recovery_succeeded()[reset_mask]
        for idx in reset_mask.nonzero(as_tuple=False).flatten().tolist():
            snap['failure_reason'][idx] = self.failure_reason[idx]
            snap['first_contact_source'][idx] = self.first_contact_source[idx]
            snap['first_contact_frame'][idx] = self.first_contact_frame[idx]
            snap['last_transition_event'][idx] = self.last_transition_event[idx]
            snap['recovery_outcome'][idx] = self.recovery_outcome[idx]
        snap['valid'][reset_mask] = True
        snap['reset_step'][reset_mask] = self.common_step_counter

    def _update_task_outcome(self):
        """Latch mission-level failure reasons from the freshly computed metrics.

        Two distinct failures are recognised:

        * ``OBSTACLE_COLLISION`` -- positive collision evidence (a contact pair
          carrying a real contact force, or an exact analytic wheel/box overlap).
        * ``CROSSED_WITHOUT_LIFTOFF`` -- the robot's whole envelope ended up past
          the box rear face without ever leaving the ground. That means the robot
          went through the obstacle line instead of over it (ploughing straight
          into the fence, or physics letting it pass), which is a task failure
          even when no single contact force was recorded.

        Note on what is deliberately NOT used here: an earlier version tripped
        this flag on ``overlap_below_top`` (Y-interval overlap while a wheel sat
        at or below the box top). That condition is true whenever the robot is
        simply *near* the box with its wheels down, which is the normal state
        right before a jump -- on seed 15 it fired while the analytic wheel/box
        gap was still +0.43 mm. Proximity is not penetration.

        Note on the liftoff predicate (defect B): this reads ``task_took_off``,
        the obstacle task's own per-cycle latch, NOT the FSM's ``has_taken_off``.
        The two disagree exactly where it matters. Measured false positive before
        the change -- a robot that had lifted off at step 39 and touched down at
        step 60 was failed at step 83:

            ctx ep_len=83 fsm=0 root_y=+0.3276 rb_base_y=+0.3276 obs_rear=+0.3200
            >> LATCH step=83 rear_margin=+0.0535 taken_off=False collision=False

        The pose and the margin were both genuine; only ``has_taken_off`` was
        stale, because the FSM had returned to another state and cleared it.
        """
        if not hasattr(self, 'task_failed'):
            return
        ploughed = (self.robot_rear_margin > 0.005) & (~self.task_took_off)
        new_fail = (self.has_obstacle_collision | ploughed) & (~self.task_failed)
        if new_fail.any():
            for idx in new_fail.nonzero(as_tuple=False).flatten().tolist():
                if self.has_obstacle_collision[idx]:
                    self.failure_reason[idx] = "OBSTACLE_COLLISION"
                else:
                    self.failure_reason[idx] = "CROSSED_WITHOUT_LIFTOFF"
            self.task_failed |= new_fail

    def step(self, actions):
        """
        Step simulation with spatial or step-based jump triggering and obstacle interaction.

        Sequence:
          1. evaluate the spatial trigger against the TRUE geometric envelope;
          2. delegate to MyRobotRunningJump.step(), which refreshes the task metrics
             via _update_task_metrics() BEFORE termination and reward;
          3. apply continuous-mode obstacle repositioning;
          4. publish telemetry.
        """
        from legged_gym.envs.myrobot import obstacle_geometry as geo

        # --- Trigger geometry: use the real front of the robot ----------------
        # Historic code used `root_states[:,1] + wheel_radius`, which omits the
        # ~0.114 m longitudinal offset of the wheel centres and the forward extent
        # of the legs/chassis. That underestimated the true front by 9-11 cm and
        # made the robot strike the obstacle before the jump was ever commanded.
        #
        # This is LAYER 1, the approach envelope, and nothing else: it is limited
        # to geometry that can actually reach the box top, and it never falls back
        # to the root Y. If `could_reach` is False nothing is low enough to hit the
        # box, so the distance is undefined and the trigger simply holds.
        obs_front_y, obs_rear_y, obs_top_z = geo.obstacle_y_interval(self)
        approach = geo.approach_envelope(self)
        robot_front_y = approach['front_y']
        dist_to_obs_front = obs_front_y - robot_front_y
        could_reach = approach['could_reach']

        if self.trigger_mode == "spatial":
            # Spatial distance trigger: initiate takeoff when reaching optimal takeoff window
            in_balance = (self.fsm_state == self.STATE_BALANCE)
            is_cruising = self.root_states[:, 8] > 0.10
            reached_window = ((dist_to_obs_front <= self.spatial_trigger_dist)
                              & (dist_to_obs_front >= -0.02) & could_reach)
            has_warmed_up = self.episode_length_buf >= 20  # Minimum 0.4s to stabilize balance
            can_jump = in_balance & is_cruising & reached_window & has_warmed_up
            if can_jump.any():
                # Order matters: _begin_new_obstacle_cycle clears the per-cycle
                # buffers including trigger_step, so the stamp must follow it.
                self._begin_new_obstacle_cycle(can_jump)
                self.trigger_step[can_jump] = self.common_step_counter
                self.request_teleop_jump(can_jump)
        else:
            # Legacy step-based trigger for frozen model_3.pt baseline comparison
            step_in_cycle = self.episode_length_buf % self.cycle_length
            can_jump = (step_in_cycle == self.step_trigger) & (self.fsm_state == self.STATE_BALANCE)
            if can_jump.any():
                self._begin_new_obstacle_cycle(can_jump)
                self.trigger_step[can_jump] = self.common_step_counter
                self.request_teleop_jump(can_jump)

        # Execute parent step (renders, steps dual policy, refreshes tensors, handles reset)
        obs, priv_obs, rews, dones, infos = super().step(actions)

        # In continuous mode, reposition the obstacle ahead of the robot whenever recovering to BALANCE
        continuous_mode = getattr(self.cfg.env, 'continuous_mode', False)
        if continuous_mode:
            recovered_to_balance = (self.fsm_state == self.STATE_BALANCE) & self.has_touched_down
            if recovered_to_balance.any():
                self._reposition_obstacles(recovered_to_balance)

        # Telemetry info dictionary
        mean_post_vel = torch.where(
            self.post_touchdown_samples > 0,
            self.post_touchdown_vels / torch.clamp(self.post_touchdown_samples, min=1.0),
            torch.zeros_like(self.post_touchdown_vels)
        )
        self._publish_obstacle_extras(mean_post_vel)

        return obs, priv_obs, rews, dones, infos

    def _begin_new_obstacle_cycle(self, env_mask):
        """Start a fresh per-cycle obstacle bookkeeping set for `env_mask`.

        Per-cycle means the peak force, clearance and first-contact record are all
        independent between attempts; nothing is carried over from a previous
        cycle, so a peak can never be re-used as if it belonged to this attempt.
        """
        self.peak_obstacle_contact_force[env_mask] = 0.0
        self.has_obstacle_collision[env_mask] = False
        self.min_obstacle_clearance[env_mask] = 1.0
        self.obstacle_cleared[env_mask] = False
        self.is_crossing_obstacle[env_mask] = False
        self.robot_rear_margin[env_mask] = -1.0
        self.takeoff_margin_m[env_mask] = 0.0
        self.landing_margin_m[env_mask] = 0.0
        self.post_touchdown_vels[env_mask] = 0.0
        self.post_touchdown_samples[env_mask] = 0.0
        # A new obstacle attempt starts with "has not left the ground yet".
        self.task_took_off[env_mask] = False
        self.task_takeoff_step[env_mask] = -1
        self.task_touchdown_step[env_mask] = -1
        self.wheel_danger_seen[env_mask] = False
        self.wheel_pass_clean[env_mask] = False
        self.landing_safe[env_mask] = False
        self.whole_body_passed[env_mask] = False
        self.whole_body_rear_margin[env_mask] = -1.0
        self.takeoff_front_y[env_mask] = 0.0
        self.takeoff_rear_y[env_mask] = 0.0
        self.takeoff_clearance_above_obs[env_mask] = 0.0
        self.trigger_step[env_mask] = -1
        self.pass_step[env_mask] = -1
        self.collision_event[env_mask] = False
        self.geometric_pass_event[env_mask] = False
        self.clear_success_event[env_mask] = False
        # A new attempt re-arms tier-2 success: the previous attempt's collision
        # must not poison the next one. Only an in-episode collision disarms.
        self.success_armed[env_mask] = False
        self.final_success_paid[env_mask] = False
        self._collision_seen[env_mask] = False
        self.task_failed[env_mask] = False
        self.just_reset[env_mask] = False
        for idx in env_mask.nonzero(as_tuple=False).flatten().tolist():
            self.failure_reason[idx] = "NONE"
            self.success_disarmed_reason[idx] = "NONE"

    def _reposition_obstacles(self, env_mask):
        """Move the obstacle ahead of each selected environment independently.

        Each environment owns its own obstacle pose in `obstacle_root_states`, and
        only its own actor index is written, so reposiitoning one environment can
        never move another environment's obstacle.
        """
        idxs = env_mask.nonzero(as_tuple=False).flatten()
        if len(idxs) == 0:
            return
        lead_dist = getattr(self.cfg.obstacle, 'lead_dist', 0.60)
        robot_y = self.root_states[idxs, 1]
        new_obs_py = robot_y + lead_dist

        prev_cleared = self.obstacle_cleared[idxs].clone()
        prev_cycles = self.cycles_completed[idxs].clone()
        self.obstacle_py_t[idxs] = new_obs_py
        self.obstacle_root_states[idxs, 1] = self.env_origins[idxs, 1] + new_obs_py
        self.obstacle_root_states[idxs, 0] = self.env_origins[idxs, 0] + getattr(
            self.cfg.obstacle, 'pos_x', 0.0)
        self.obstacle_root_states[idxs, 2] = self.env_origins[idxs, 2] + self.obstacle_h * 0.5
        self.obstacle_root_states[idxs, 3:6] = 0.0
        self.obstacle_root_states[idxs, 6] = 1.0
        self.obstacle_root_states[idxs, 7:13] = 0.0

        obs_actor_ids = (idxs * 2 + 1).to(device=self.device, dtype=torch.int32)
        all_root_contiguous = self.all_root_states.view(-1, 13).contiguous()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(all_root_contiguous),
            gymtorch.unwrap_tensor(obs_actor_ids),
            len(obs_actor_ids)
        )

        # Repositioning is a NEW crossing opportunity: clear the per-attempt state
        # so the next attempt is measured independently. The completed-cycle count
        # only advances when the previous attempt actually cleared cleanly.
        self.cycles_completed[idxs] = prev_cycles + prev_cleared.float()
        self.has_touched_down[idxs] = False
        self.has_taken_off[idxs] = False
        self._begin_new_obstacle_cycle(idxs)

        # Root states were rewritten for these actors; refresh so the next step
        # reads the new pose instead of a stale tensor.
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def _publish_obstacle_extras(self, mean_post_vel):
        self.extras["obstacle_collision"] = self.has_obstacle_collision.float()
        self.extras["obstacle_cleared"] = self.obstacle_cleared.float()
        self.extras["min_obstacle_clearance"] = self.min_obstacle_clearance
        self.extras["peak_obstacle_contact_force"] = self.peak_obstacle_contact_force
        self.extras["takeoff_margin_m"] = self.takeoff_margin_m
        self.extras["landing_margin_m"] = self.landing_margin_m
        self.extras["mean_post_touchdown_vel"] = mean_post_vel
        self.extras["obstacle_task_failed"] = self.task_failed.float()
        self.extras["obstacle_collision_event"] = self.collision_event.float()
        # Tier-1 (geometric progress) and tier-2 (full mission success) are
        # published under DIFFERENT keys so no consumer can mistake one for the
        # other.
        self.extras["obstacle_geometric_pass_event"] = self.geometric_pass_event.float()
        self.extras["obstacle_final_success_event"] = self.clear_success_event.float()
        self.extras["obstacle_success_armed"] = self.success_armed.float()
        self.extras["obstacle_final_success_paid"] = self.final_success_paid.float()
        self.extras["recovery_outcome"] = list(self.recovery_outcome)
        self.extras["first_contact_body"] = self.first_contact_body.to(torch.float)
        self.extras["first_contact_step"] = self.first_contact_step.to(torch.float)
        self.extras["obstacle_py"] = self.obstacle_py_t.clone()
        self.extras["cycles_completed"] = self.cycles_completed
        self.extras["robot_rear_margin"] = self.robot_rear_margin
        self.extras["obstacle_trigger_step"] = self.trigger_step.to(torch.float)
        self.extras["obstacle_pass_step"] = self.pass_step.to(torch.float)
        # Defect A: publish the three layers under distinct keys. A consumer that
        # wants "did the wheels get over it" reads wheel_pass_clean; "was the
        # landing clear" reads landing_safe; "has the whole machine left the box
        # region" reads whole_body_passed. No single flag is overloaded.
        self.extras["wheel_pass_clean"] = self.wheel_pass_clean.float()
        self.extras["wheel_danger_seen"] = self.wheel_danger_seen.float()
        self.extras["landing_safe"] = self.landing_safe.float()
        self.extras["whole_body_passed"] = self.whole_body_passed.float()
        self.extras["whole_body_rear_margin"] = self.whole_body_rear_margin
        self.extras["obstacle_layers"] = getattr(self, 'obstacle_layers', {})
        # Defect B: the task-owned liftoff latch, distinct from the FSM's
        # `has_taken_off`, which the dual policy clears on state transitions.
        self.extras["task_took_off"] = self.task_took_off.float()
        self.extras["task_takeoff_step"] = self.task_takeoff_step
        self.extras["task_touchdown_step"] = self.task_touchdown_step

    # =========================================================================
    # Reward Functions: Explicitly Tailored for Physical Obstacle Clearance
    # =========================================================================

    def _reward_obstacle_collision(self):
        """Penalty magnitude for obstacle contact.

        CONTRACT: reward functions in this codebase return a VALUE that is later
        multiplied by `cfg.rewards.scales[name]`. Sign is therefore owned entirely
        by the config scale (which is negative here). Returning a negative value
        from the function AND a negative scale double-negates the penalty into a
        reward, which is exactly the defect this replaces.

        Returns
        -------
        value >= 0 : larger means a more severe collision, so the negative scale
                     always produces a LOWER (more punitive) total contribution.
        """
        # Base penalty: any contact at all (event-driven, so it is charged once the
        # contact is detected and then sustained while contact persists).
        collision_pen = self.has_obstacle_collision.to(self.rew_buf.dtype)
        # Severity term: peak contact force normalised to a 2.0 cap.
        force_pen = torch.clamp(self.peak_obstacle_contact_force / 100.0, 0.0, 2.0)
        return collision_pen + force_pen

    def _reward_obstacle_clearance(self):
        """Vertical wheel clearance above the obstacle top while passing over it."""
        clr_min, _, _ = compute_exact_cylinder_clearance(self)
        clr_above_obs = clr_min - self.obstacle_h
        target_margin = 0.15  # Desired 15cm margin above hurdle
        reward = torch.clamp(clr_above_obs / target_margin, 0.0, 1.5)
        return torch.where(self.is_crossing_obstacle & self.has_taken_off, reward,
                           torch.zeros_like(reward))

    def _reward_obstacle_cleared_success(self):
        """ONE-SHOT bonus for FULL mission success (tier 2 only).

        Two levels of "success" exist in this environment and they are paid
        differently on purpose:

        * `geometric_pass_event` -- the rear envelope got past the box rear face.
          That is physical progress, and it is paid separately and modestly by
          `_reward_obstacle_geometric_pass`. It is NOT success.
        * `clear_success_event` (this function) -- full mission success: real
          liftoff, complete passage, no obstacle contact at any point, a landing,
          and a sustained forward cruise in the recovery window that ended in
          NORMAL_HANDOFF. Paid at most once per episode.

        The Phase A version paid this on the `cleared` latch alone. On seed 15 that
        banked the bonus at step 118 and the robot hit the box at step 120 -- the
        reward was collected for an attempt that immediately failed. Any collision
        now permanently disarms this event (`success_armed`), and the latch itself
        is edge-triggered so it can never be paid on consecutive steps.
        """
        return self.clear_success_event.to(self.rew_buf.dtype)

    def _reward_obstacle_geometric_pass(self):
        """Staged bonus for getting the whole body past the box (tier 1).

        Deliberately small relative to `obstacle_cleared_success`: passing the box
        without recovering a forward cruise, or colliding later, must not be as
        valuable as a genuine cleared jump. This is the only reward the geometric
        passage pays.
        """
        return self.geometric_pass_event.to(self.rew_buf.dtype)

    def _reward_forward_progress_across_obstacle(self):
        """Forward horizontal velocity reward while airborne over the obstacle zone."""
        v_fwd = self._get_world_forward_velocity()
        in_flight_zone = (self.fsm_state == self.STATE_JUMP) & self.has_taken_off
        target_v = self.actual_cmd_vy if hasattr(self, 'actual_cmd_vy') else self.commands[:, 1]
        vel_rew = torch.clamp(v_fwd / target_v, 0.0, 1.25)
        return torch.where(in_flight_zone, vel_rew, torch.zeros_like(vel_rew))
