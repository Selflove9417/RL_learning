# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# MyRobot Forward Jump Environment (Phase 1: Forward Jump Capability Development)

import math
import torch
from isaacgym import gymapi, gymtorch
from legged_gym.envs.myrobot.myrobot_jump_env import MyRobotJump


class MyRobotForwardJump(MyRobotJump):
    """
    Dedicated Forward Jump Environment for MyRobot dual-wheel-legged robot.
    Features:
    - Retains Stage 8 35-dim observation and 6-dim action space for 100% hot-start compatibility.
    - Accurately aligns robot forward direction: local Y axis (vy) is longitudinal forward motion.
    - True bilateral wheel clearance tracking (target >= 21.5 cm).
    - Authentic flight forward displacement measurement (takeoff to touchdown strictly).
    - Non-conflicting rewards for:
        1. Stationary squat prep (phi in [0.02, 0.12])
        2. Forward push-off velocity generation (phi in [0.10, 0.24])
        3. Real airborne forward displacement & orientation stability (phi in [0.20, 0.40])
        4. Touchdown cushion and impact absorption (phi in [0.28, 0.50])
        5. Stable post-landing braking and stationary recovery balance (phi in [0.45, 1.00])
    """

    def _init_buffers(self):
        super()._init_buffers()

        # Forward jump kinematics & telemetry tracking buffers
        self.takeoff_vy = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.takeoff_pos_y = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.touchdown_pos_y = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.flight_distance_y = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.final_flight_distance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.has_touched_down = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.expected_airborne_distance = getattr(self.cfg.commands, 'expected_airborne_distance', 0.15)
        self.target_forward_vel = getattr(self.cfg.commands, 'target_forward_vel', 0.40)

        # Initialize commands: forward command along local Y
        self.commands[:, 0] = 0.0
        self.commands[:, 1] = self.target_forward_vel
        self.commands[:, 2] = 0.0

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        if hasattr(self.cfg.commands, 'ranges') and hasattr(self.cfg.commands.ranges, 'lin_vel_y'):
            vy_min, vy_max = self.cfg.commands.ranges.lin_vel_y
            if vy_max > vy_min:
                self.commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (vy_max - vy_min) + vy_min
            else:
                self.commands[env_ids, 1] = vy_min
        else:
            self.commands[env_ids, 1] = self.target_forward_vel
        self.commands[env_ids, 0] = 0.0
        self.commands[env_ids, 2] = 0.0

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        super().reset_idx(env_ids)

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Set forward velocity command along local Y (the true forward axis in MyRobot)
        if hasattr(self.cfg.commands, 'ranges') and hasattr(self.cfg.commands.ranges, 'lin_vel_y'):
            vy_min, vy_max = self.cfg.commands.ranges.lin_vel_y
            if vy_max > vy_min:
                self.commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (vy_max - vy_min) + vy_min
            else:
                self.commands[env_ids, 1] = vy_min
        else:
            self.commands[env_ids, 1] = self.target_forward_vel

        # Lateral (X) and yaw velocity commands strictly zero
        self.commands[env_ids, 0] = 0.0
        self.commands[env_ids, 2] = 0.0

        # Reset forward trajectory trackers
        self.takeoff_vy[env_ids] = 0.0
        self.takeoff_pos_y[env_ids] = 0.0
        self.touchdown_pos_y[env_ids] = 0.0
        self.flight_distance_y[env_ids] = 0.0
        self.final_flight_distance[env_ids] = 0.0
        self.has_touched_down[env_ids] = False

    def compute_observations(self):
        """
        35-dimensional deployable observation vector (consistent with Stage 8 architecture):
        - base_lin_vel * 2.0 (3: [vx*2, vy*2, vz*2])
        - base_ang_vel * 0.25 (3)
        - projected_gravity (3)
        - jump_cmds (4: [cmd_fwd * scale, cmd_yaw * scale, norm_target_height, jump_phase])
            * In MyRobot, local Y is forward direction (vy).
            * Slot 9 receives forward command along local Y (commands[:, 1] * scale).
            * Slot 10 receives yaw command (commands[:, 2] * scale = 0).
            * Slot 11 receives normalized target jump height.
            * Slot 12 receives jump phase.
        - obs_dof_pos * 1.0 (6: wheels zeroed)
        - dof_vel * 0.05 (6)
        - filtered_actions (6)
        - contact_state (2: normalized contact force [0, 1])
        - phase_clock (2: sin(2pi*phase), cos(2pi*phase))
        Total = 3 + 3 + 3 + 4 + 6 + 6 + 6 + 2 + 2 = 35 dims.
        """
        obs_dof_pos = (self.dof_pos - self.default_dof_pos).clone()
        obs_dof_pos[:, self.wheel_indices] = 0.0

        # Forward command along local Y (aligning balance controller and jump policy)
        cmd_fwd = self.commands[:, 1]
        jump_cmds = torch.stack([
            cmd_fwd * self.commands_scale[1],
            self.commands[:, 2] * self.commands_scale[2],
            (self.target_jump_height - 0.10) / 0.10,
            self.jump_phase,
        ], dim=-1)

        # Contact state
        fn_L = torch.clamp(self.contact_forces[:, self.left_wheel_body, 2] / 100.0, 0.0, 1.0)
        fn_R = torch.clamp(self.contact_forces[:, self.right_wheel_body, 2] / 100.0, 0.0, 1.0)
        contact_state = torch.stack([fn_L, fn_R], dim=-1)

        # Phase clock
        phi = 2.0 * math.pi * self.jump_phase
        phase_clock = torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)

        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            jump_cmds,
            obs_dof_pos * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.filtered_actions,
            contact_state,
            phase_clock,
        ), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def step(self, actions):
        # Action smoothing
        clip_actions = self.cfg.normalization.clip_actions
        clipped = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.last_actions[:] = self.filtered_actions
        self.filtered_actions[:] = 0.80 * clipped + 0.20 * self.last_actions

        # Update jump timer and phase
        self.jump_timer += self.dt
        cycle_done = self.jump_timer >= self.jump_cycle_time
        self.jump_timer[cycle_done] -= self.jump_cycle_time
        self.jump_phase[:] = self.jump_timer / self.jump_cycle_time

        # Step physics via base class
        res = super(MyRobotJump, self).step(actions)

        # Refresh kinematics, clearance, airborne & contacts
        self._refresh_physical_state(update_has_taken_off=True)

        # Reset cycle trackers at periodic cycle boundary
        if cycle_done.any():
            self.max_flight_clearance[cycle_done] = 0.0
            self.peak_landing_impact[cycle_done] = 0.0
            self.has_taken_off[cycle_done] = False
            self.takeoff_vy[cycle_done] = 0.0
            self.takeoff_pos_y[cycle_done] = 0.0
            self.touchdown_pos_y[cycle_done] = 0.0
            self.flight_distance_y[cycle_done] = 0.0
            self.final_flight_distance[cycle_done] = 0.0
            self.has_touched_down[cycle_done] = False

        return res

    def _refresh_physical_state(self, update_has_taken_off=True):
        """
        Refresh rigid body states and calculate authentic wheel clearance,
        airborne flight window, takeoff forward/vertical velocity, and flight displacement.
        """
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        pos_L = self.rb_states[:, self.left_wheel_body, 2]
        pos_R = self.rb_states[:, self.right_wheel_body, 2]
        clr_L = torch.clamp(pos_L - self.wheel_radius, min=0.0)
        clr_R = torch.clamp(pos_R - self.wheel_radius, min=0.0)
        self.wheel_clearance[:, 0] = clr_L
        self.wheel_clearance[:, 1] = clr_R
        self.bilateral_clearance[:] = torch.minimum(clr_L, clr_R)

        fn_L = self.contact_forces[:, self.left_wheel_body, 2]
        fn_R = self.contact_forces[:, self.right_wheel_body, 2]

        # Authentic airborne condition: clearance > 10mm on both wheels and contact forces < 2.0N
        now_airborne = (clr_L > 0.010) & (clr_R > 0.010) & (fn_L < 2.0) & (fn_R < 2.0)

        # Takeoff detection: in push-off window with positive vz and airborne
        if update_has_taken_off:
            in_pushoff = (self.jump_phase >= 0.10) & (self.jump_phase <= 0.28)
            upward_launch = (self.base_lin_vel[:, 2] > 0.8) & now_airborne
            self.has_taken_off |= (in_pushoff & upward_launch)

        # Detect the exact takeoff transition instant
        just_took_off = now_airborne & (~self.is_airborne)
        if just_took_off.any():
            self.takeoff_vz[just_took_off] = self.base_lin_vel[just_took_off, 2]
            self.takeoff_vy[just_took_off] = self.base_lin_vel[just_took_off, 1]
            self.takeoff_pos_y[just_took_off] = self.root_states[just_took_off, 1]

        # Detect the exact touchdown transition instant
        just_landed = (~now_airborne) & self.is_airborne & self.has_taken_off
        if just_landed.any():
            self.touchdown_pos_y[just_landed] = self.root_states[just_landed, 1]
            self.final_flight_distance[just_landed] = self.touchdown_pos_y[just_landed] - self.takeoff_pos_y[just_landed]
            self.has_touched_down[just_landed] = True

        # While airborne, flight_distance_y tracks authentic displacement from takeoff point
        self.flight_distance_y = torch.where(
            now_airborne,
            self.root_states[:, 1] - self.takeoff_pos_y,
            self.flight_distance_y
        )

        self.is_airborne[:] = now_airborne
        self.air_time = torch.where(now_airborne, self.air_time + self.dt, torch.zeros_like(self.air_time))

        # Max flight clearance
        self.max_flight_clearance = torch.where(
            now_airborne,
            torch.maximum(self.max_flight_clearance, self.bilateral_clearance),
            self.max_flight_clearance
        )

        # Landing impact tracking
        touchdown_impact = (~now_airborne) & (fn_L > 10.0)
        self.peak_landing_impact = torch.where(
            touchdown_impact,
            torch.maximum(fn_L, fn_R),
            self.peak_landing_impact
        )

    # ==================== INCREMENTAL FORWARD JUMP REWARDS ====================

    def _reward_takeoff_forward_velocity(self):
        """
        Rewards actively generating forward push-off velocity along local Y axis
        during push-off and early ascent window (phi in [0.10, 0.24]).
        Tuned to match the commanded forward speed (~0.40 m/s).
        """
        in_launch = (self.jump_phase >= 0.10) & (self.jump_phase <= 0.24)
        vy = self.base_lin_vel[:, 1]
        target_vy = self.commands[:, 1]

        # Positive velocity progression towards target
        fwd_speed = torch.clamp(vy / torch.clamp(target_vy, min=0.1), min=-0.5, max=1.2)
        # Tracking precision bell curve
        vel_tracking = torch.exp(-torch.square(vy - target_vy) / (2.0 * (0.15 ** 2)))
        reward = 0.5 * fwd_speed + 0.5 * vel_tracking
        return reward * in_launch.float()

    def _reward_airborne_forward_displacement(self):
        """
        STRICT REQUIREMENT 6:
        Rewards forward displacement strictly accumulated between authentic takeoff and touchdown.
        Zero on ground, zero without authentic push-off.
        """
        valid_flight = self.is_airborne & self.has_taken_off & (self.jump_phase <= 0.40)
        d_air = torch.clamp(self.flight_distance_y, min=0.0)
        target_d = torch.clamp(self.commands[:, 1] * 0.38, min=0.05)  # Expected flight distance ~15 cm
        progress = torch.clamp(d_air / target_d, min=0.0, max=1.2)
        return progress * valid_flight.float()

    def _reward_touchdown_forward_distance(self):
        """
        Rewards achieved final takeoff-to-touchdown forward distance upon landing.
        Active during landing window (phi in [0.28, 0.45]).
        """
        in_landing = (self.jump_phase >= 0.28) & (self.jump_phase <= 0.45) & self.has_taken_off
        touchdown_active = (~self.is_airborne) & in_landing
        target_d = torch.clamp(self.commands[:, 1] * 0.38, min=0.05)
        achieved_ratio = torch.clamp(self.flight_distance_y / target_d, min=0.0, max=1.2)
        return achieved_ratio * touchdown_active.float()

    def _reward_tracking_lin_vel(self):
        """
        Post-landing recovery braking reward (phi in [0.40, 1.0]):
        Once landed and cushioned, rewards braking the robot to a stationary zero-velocity balance.
        Prevents uncontrolled rolling after forward jump landing.
        """
        fn_L = self.contact_forces[:, self.left_wheel_body, 2]
        fn_R = self.contact_forces[:, self.right_wheel_body, 2]
        firm_grounded = (fn_L > 20.0) & (fn_R > 20.0) & (~self.is_airborne)
        cushioned = (self.base_lin_vel[:, 2] > -0.3)
        in_ground_recovery = (self.jump_phase >= 0.40) & self.has_taken_off & firm_grounded & cushioned

        # Target post-landing velocity is 0.0 (stationary balance)
        lin_vel_error = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        tracking = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        return tracking * in_ground_recovery.float()

    def _reward_tracking_ang_vel(self):
        """
        Penalizes yaw angular velocity throughout the jump cycle to maintain forward heading.
        """
        ang_vel_error = torch.square(self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / 0.10)

    def _reward_bilateral_symmetry(self):
        """Penalizes difference in wheel clearance, left/right joint disparity, and yaw angular velocity."""
        clr_diff = torch.abs(self.wheel_clearance[:, 0] - self.wheel_clearance[:, 1])
        hip_diff = torch.abs(self.dof_pos[:, self.left_hip_idx] - self.dof_pos[:, self.right_hip_idx])
        knee_diff = torch.abs(self.dof_pos[:, self.left_knee_idx] - self.dof_pos[:, self.right_knee_idx])
        yaw_rate = torch.abs(self.base_ang_vel[:, 2])
        return clr_diff * 5.0 + 0.5 * (hip_diff + knee_diff) + 1.0 * yaw_rate
