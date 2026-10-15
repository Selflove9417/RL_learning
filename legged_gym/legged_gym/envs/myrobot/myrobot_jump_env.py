# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# MyRobot Jump Environment (Phase 2: 20cm Jump Capability Development)

import math
import torch
from isaacgym import gymapi, gymtorch
from legged_gym.envs.base.legged_robot import LeggedRobot


class MyRobotJump(LeggedRobot):
    """
    Dedicated Jump Environment for MyRobot dual-wheel-legged robot.
    Features:
    - Upgraded actuator limits (Hip 140Nm / 12rad/s, Knee 120Nm / 14rad/s, Wheel 12Nm / 40rad/s)
    - True bilateral wheel clearance tracking via rigid body states
    - Temporal jump cycle with phase-clock observations
    - Jump-tailored action space (direct high-power leg PD targets + wheel reaction stabilization)
    - Grounded / airborne state-aware rewards and zero-latency landing reconnection
    """

    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props, env_id)
        wheel_names = ["link_004_joint", "link_007_joint"]
        for i, name in enumerate(self.dof_names):
            if name in wheel_names:
                props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
                props["stiffness"][i] = 0.0
                props["damping"][i] = 0.0
        return props

    def _init_buffers(self):
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

        super()._init_buffers()

        # Wheel rigid body handles
        wheel_body_names = ["link_004", "link_007"]
        self.feet_indices = torch.tensor(
            [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], b_name) for b_name in wheel_body_names],
            device=self.device, dtype=torch.long
        )

        # Acquire rigid body state tensor for true 3D wheel clearance tracking
        _rb_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_tensor).view(self.num_envs, -1, 13)

        self.wheel_radius = 0.070  # meters
        self.left_wheel_body = self.feet_indices[0].item()
        self.right_wheel_body = self.feet_indices[1].item()

        # Jump cycle & timing buffers
        self.jump_cycle_time = getattr(self.cfg.commands, 'jump_cycle_time', 2.0)
        self.jump_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.jump_phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.target_jump_height = torch.full(
            (self.num_envs,),
            getattr(self.cfg.commands, 'target_jump_height', 0.20),
            device=self.device, dtype=torch.float
        )

        # Clearance & flight buffers
        self.wheel_clearance = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        self.bilateral_clearance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.is_airborne = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.air_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.max_flight_clearance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.peak_landing_impact = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.takeoff_vz = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.has_taken_off = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Action buffers
        self.filtered_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)

        # Action scales
        self.hip_action_scale = getattr(self.cfg.control, 'hip_action_scale', 0.6)
        self.knee_action_scale = getattr(self.cfg.control, 'knee_action_scale', 0.8)
        self.wheel_torque_scale = getattr(self.cfg.control, 'wheel_torque_scale', 10.0)
        self.wheel_velocity_damping = getattr(self.cfg.control, 'wheel_velocity_damping', 0.5)
        self.wheel_air_torque_limit = getattr(self.cfg.control, 'wheel_air_torque_limit', 3.0)

        # Nominal standing angles
        self.nominal_dof_pos = self.default_dof_pos.clone()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        super().reset_idx(env_ids)

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Always start jump cycle at beginning (phase 0.0) for stable squat preparation
        self.jump_timer[env_ids] = 0.0
        self.jump_phase[env_ids] = 0.0
        self.has_taken_off[env_ids] = False

        # Target jump height curriculum
        min_h = getattr(self.cfg.commands, 'init_jump_height', 0.20)
        max_h = getattr(self.cfg.commands, 'target_jump_height', 0.20)
        if max_h > min_h:
            self.target_jump_height[env_ids] = torch.rand_like(self.target_jump_height[env_ids]) * (max_h - min_h) + min_h
        else:
            self.target_jump_height[env_ids] = max_h

        # Reset initial base velocities to zero for clean stance initiation without drop shock
        self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

        self.filtered_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.wheel_clearance[env_ids] = 0.0
        self.bilateral_clearance[env_ids] = 0.0
        self.is_airborne[env_ids] = False
        self.air_time[env_ids] = 0.0
        self.max_flight_clearance[env_ids] = 0.0
        self.peak_landing_impact[env_ids] = 0.0
        self.takeoff_vz[env_ids] = 0.0

    def compute_observations(self):
        """
        35-dimensional deployable observation vector:
        - base_lin_vel * 2.0 (3)
        - base_ang_vel * 0.25 (3)
        - projected_gravity (3)
        - commands (4: cmd_vx, cmd_yaw, target_jump_height, jump_phase)
        - obs_dof_pos (6: joint angles relative to nominal, wheels zeroed)
        - dof_vel * 0.05 (6)
        - filtered_actions (6)
        - contact_state (2: normalized normal force [0, 1])
        - phase_clock (2: sin(2pi*phase), cos(2pi*phase))
        Total = 3 + 3 + 3 + 4 + 6 + 6 + 6 + 2 + 2 = 35
        """
        obs_dof_pos = (self.dof_pos - self.default_dof_pos).clone()
        obs_dof_pos[:, self.wheel_indices] = 0.0

        # Commands: vx=0, yaw=0, target_h, phase
        jump_cmds = torch.stack([
            self.commands[:, 0] * self.commands_scale[0],
            self.commands[:, 2] * self.commands_scale[2],
            (self.target_jump_height - 0.10) / 0.10,  # normalized target height [-0.0, 1.0]
            self.jump_phase,
        ], dim=-1)

        # Contact state: normalized contact force
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

    def _get_noise_scale_vec(self, cfg):
        self.add_noise = self.cfg.noise.add_noise
        noise_vec = torch.zeros_like(self.obs_buf[0])
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:13] = 0.0 # commands & phase
        noise_vec[13:19] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[13 + self.left_wheel_idx] = 0.0
        noise_vec[13 + self.right_wheel_idx] = 0.0
        noise_vec[19:25] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[25:31] = 0.0 # actions
        noise_vec[31:33] = 0.02 # contact noise
        noise_vec[33:35] = 0.0 # phase clock
        return noise_vec

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
        res = super().step(actions)

        # Refresh kinematics, clearance, airborne & contacts
        self._refresh_physical_state(update_has_taken_off=True)

        # Reset flight trackers at cycle reset
        self.max_flight_clearance[cycle_done] = 0.0
        self.peak_landing_impact[cycle_done] = 0.0
        self.has_taken_off[cycle_done] = False

        return res

    def _refresh_physical_state(self, update_has_taken_off=True):
        """
        Refresh rigid body state tensor from PhysX and update bilateral wheel clearance,
        airborne status, takeoff velocity, air time, and landing impact trackers.
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

        # Airborne tracking: clearance > 10mm and contact forces < 2.0N
        now_airborne = (clr_L > 0.010) & (clr_R > 0.010) & (fn_L < 2.0) & (fn_R < 2.0)

        # Takeoff detection: in push-off window with positive vz and airborne
        if update_has_taken_off:
            in_pushoff = (self.jump_phase >= 0.10) & (self.jump_phase <= 0.28)
            upward_launch = (self.base_lin_vel[:, 2] > 0.8) & now_airborne
            self.has_taken_off |= (in_pushoff & upward_launch)

        just_took_off = now_airborne & (~self.is_airborne)
        self.takeoff_vz[just_took_off] = self.base_lin_vel[just_took_off, 2]

        self.is_airborne[:] = now_airborne
        self.air_time = torch.where(now_airborne, self.air_time + self.dt, torch.zeros_like(self.air_time))

        # Max flight clearance
        self.max_flight_clearance = torch.where(
            now_airborne,
            torch.maximum(self.max_flight_clearance, self.bilateral_clearance),
            self.max_flight_clearance
        )

        # Landing impact tracking
        just_landed = (~now_airborne) & (fn_L > 10.0)
        self.peak_landing_impact = torch.where(
            just_landed,
            torch.maximum(fn_L, fn_R),
            self.peak_landing_impact
        )

    def _compute_torques(self, actions):
        """
        Computes actuator torques for jump:
        - Leg joint targets: nominal angle + scaled policy action
        - High-bandwidth PD with URDF effort limits (Hip 140Nm, Knee 120Nm)
        - Wheel torque: ground drive feedforward or mid-air reaction pitch stabilization
        """
        torques = torch.zeros_like(actions)

        # Hip targets: nominal + a_hip * 0.6 rad
        hip_targets_L = self.default_dof_pos[:, self.left_hip_idx] + self.filtered_actions[:, self.left_hip_idx] * self.hip_action_scale
        hip_targets_R = self.default_dof_pos[:, self.right_hip_idx] + self.filtered_actions[:, self.right_hip_idx] * self.hip_action_scale

        # Knee targets: nominal + a_knee * 0.8 rad
        knee_targets_L = self.default_dof_pos[:, self.left_knee_idx] + self.filtered_actions[:, self.left_knee_idx] * self.knee_action_scale
        knee_targets_R = self.default_dof_pos[:, self.right_knee_idx] + self.filtered_actions[:, self.right_knee_idx] * self.knee_action_scale

        # Clamping position targets within safe limits
        hip_targets_L = torch.clamp(hip_targets_L, -1.40, 1.40)
        hip_targets_R = torch.clamp(hip_targets_R, -1.40, 1.40)
        knee_targets_L = torch.clamp(knee_targets_L, -1.40, 1.40)
        knee_targets_R = torch.clamp(knee_targets_R, -1.40, 1.40)

        # Leg PD computation
        torques[:, self.left_hip_idx] = (
            self.p_gains[self.left_hip_idx] * (hip_targets_L - self.dof_pos[:, self.left_hip_idx])
            - self.d_gains[self.left_hip_idx] * self.dof_vel[:, self.left_hip_idx]
        )
        torques[:, self.right_hip_idx] = (
            self.p_gains[self.right_hip_idx] * (hip_targets_R - self.dof_pos[:, self.right_hip_idx])
            - self.d_gains[self.right_hip_idx] * self.dof_vel[:, self.right_hip_idx]
        )
        torques[:, self.left_knee_idx] = (
            self.p_gains[self.left_knee_idx] * (knee_targets_L - self.dof_pos[:, self.left_knee_idx])
            - self.d_gains[self.left_knee_idx] * self.dof_vel[:, self.left_knee_idx]
        )
        torques[:, self.right_knee_idx] = (
            self.p_gains[self.right_knee_idx] * (knee_targets_R - self.dof_pos[:, self.right_knee_idx])
            - self.d_gains[self.right_knee_idx] * self.dof_vel[:, self.right_knee_idx]
        )

        # Wheel control: ground torque feedforward vs mid-air reaction stabilization
        for widx in [self.left_wheel_idx, self.right_wheel_idx]:
            act_w = self.filtered_actions[:, widx]
            tau_ground = act_w * self.wheel_torque_scale - self.wheel_velocity_damping * self.dof_vel[:, widx]
            tau_air = torch.clamp(act_w * self.wheel_air_torque_limit - 0.3 * self.dof_vel[:, widx], -self.wheel_air_torque_limit, self.wheel_air_torque_limit)
            torques[:, widx] = torch.where(self.is_airborne, tau_air, tau_ground)

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def check_termination(self):
        """Termination conditions: non-wheel contact (>15.0N), severe tilt, or fall."""
        if len(self.termination_contact_indices) > 0:
            contact_reset = torch.any(
                torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 15.0,
                dim=1
            )
        else:
            contact_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Tilt reset: pitch or roll > 53 degrees
        tilt_reset = self.projected_gravity[:, 2] > -0.60
        # Height collapse reset: root height < 0.16m (real ground impact)
        height_reset = self.root_states[:, 2] < 0.16

        self.reset_buf = contact_reset | tilt_reset | height_reset
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    # ==================== JUMP REWARD FUNCTIONS ====================

    def _reward_jump_clearance(self):
        """
        Rewards achieving true bilateral wheel clearance up to target height (0.20m).
        STRICT: Only awarded during ascent and apex window (phi <= 0.28) with authentic push-off.
        Does NOT reward maintaining clearance during descent, allowing legs to extend for landing.
        """
        h_cmd = torch.clamp(self.target_jump_height, min=0.05)
        h = self.bilateral_clearance

        # Must be airborne from an authentic push-off during ascent/apex window
        in_apex = (self.jump_phase <= 0.28)
        valid_jump = self.is_airborne & self.has_taken_off & in_apex

        tracking = torch.where(
            h <= h_cmd,
            h / h_cmd,
            torch.exp(-torch.square(h - h_cmd) / (2.0 * (0.04 ** 2)))
        )
        overshoot = torch.clamp(h - (h_cmd + 0.04), min=0.0) / 0.05
        reward = tracking - 1.5 * torch.square(overshoot)
        return torch.clamp(reward, min=-1.0, max=1.2) * valid_jump.float()

    def _reward_feet_air_time(self):
        """Rewards duration where both wheels are airborne cleanly during flight window."""
        in_flight = (self.jump_phase <= 0.40)
        return (self.is_airborne & self.has_taken_off & in_flight).float()

    def _reward_takeoff_velocity(self):
        """
        Rewards vertical push-off velocity tuned for the desired jump height (~1.98 m/s).
        Active during push-off window phi in [0.12, 0.22].
        """
        in_pushoff = (self.jump_phase >= 0.12) & (self.jump_phase <= 0.22)
        vz = self.base_lin_vel[:, 2]

        upward = torch.clamp(vz / 2.0, min=0.0, max=1.0)
        excess = torch.clamp(vz - 2.2, min=0.0)
        return (upward - torch.square(excess)) * in_pushoff.float()

    def _reward_squat_prep(self):
        """
        Rewards controlled squat posture prior to push-off (phi in [0.02, 0.12]):
        - Lower base height to ~0.34m for effective stroke length
        - Keep upright orientation without forward/backward tipping
        """
        in_squat = (self.jump_phase >= 0.02) & (self.jump_phase <= 0.12)
        h_err = torch.square(self.root_states[:, 2] - 0.34)
        pitch_err = torch.square(self.projected_gravity[:, 1])
        roll_err = torch.square(self.projected_gravity[:, 0])
        squat_quality = torch.exp(-h_err / 0.005) * torch.exp(-(pitch_err + roll_err) / 0.01)
        return squat_quality * in_squat.float()

    def _reward_wheel_contact(self):
        """
        Rewards firm wheel ground contact during pre-jump stance (phi < 0.12)
        and post-landing recovery (phi > 0.42). Prohibits premature lifting/collapsing.
        """
        in_stance = (self.jump_phase < 0.12) | (self.jump_phase > 0.42)
        fn_L = self.contact_forces[:, self.left_wheel_body, 2]
        fn_R = self.contact_forces[:, self.right_wheel_body, 2]
        grounded = (fn_L > 20.0) & (fn_R > 20.0)
        return grounded.float() * in_stance.float()

    def _reward_landing_cushion(self):
        """
        Rewards touchdown cushioning and leg reach during landing window (phi in [0.26, 0.55]).
        - In descent before touchdown: strongly reward extending legs downward to meet the ground
        - Upon touchdown contact: reward active knee flexion and downward velocity damping
        """
        in_landing = (self.jump_phase >= 0.26) & (self.jump_phase <= 0.55) & self.has_taken_off

        # Pre-touchdown leg reach: in descent, extend wheels down so wheels hit ground first
        in_descent = self.is_airborne & in_landing & (self.base_lin_vel[:, 2] < -0.2)
        leg_len_L = self.root_states[:, 2] - self.rb_states[:, self.left_wheel_body, 2]
        leg_len_R = self.root_states[:, 2] - self.rb_states[:, self.right_wheel_body, 2]
        leg_reach = torch.clamp((torch.minimum(leg_len_L, leg_len_R) - 0.15) / 0.20, 0.0, 1.0)
        reach_bonus = leg_reach * in_descent.float()

        # Touchdown contact active: wheels touched ground
        touchdown = (~self.is_airborne) & in_landing
        h_cushion = torch.exp(-torch.square(self.root_states[:, 2] - 0.35) / 0.008)
        vz_damping = torch.exp(-torch.square(self.base_lin_vel[:, 2]) / 0.2)
        touchdown_bonus = (0.5 * h_cushion + 0.5 * vz_damping) * touchdown.float()

        return touchdown_bonus * 2.0 + reach_bonus * 1.5

    def _reward_bilateral_symmetry(self):
        """Penalizes difference in wheel clearance and left/right joint angle disparity."""
        clr_diff = torch.abs(self.wheel_clearance[:, 0] - self.wheel_clearance[:, 1])
        hip_diff = torch.abs(self.dof_pos[:, self.left_hip_idx] - self.dof_pos[:, self.right_hip_idx])
        knee_diff = torch.abs(self.dof_pos[:, self.left_knee_idx] - self.dof_pos[:, self.right_knee_idx])
        return clr_diff * 5.0 + 0.5 * (hip_diff + knee_diff)

    def _reward_airborne_orientation(self):
        """Penalizes pitch and roll while airborne to ensure clean, upright flight."""
        pitch_err = torch.square(self.projected_gravity[:, 1])
        roll_err = torch.square(self.projected_gravity[:, 0])
        return (pitch_err + roll_err) * self.is_airborne.float()

    def _reward_landing_impact(self):
        """Penalizes excessive landing impact forces (> 350 N)."""
        fn_max = torch.maximum(
            self.contact_forces[:, self.left_wheel_body, 2],
            self.contact_forces[:, self.right_wheel_body, 2]
        )
        excess = torch.clamp(fn_max - 350.0, min=0.0) / 150.0
        return torch.square(excess)

    def _reward_recovery_balance(self):
        """Rewards stable upright balance post-landing (phi in [0.48, 1.0])."""
        in_recovery = (self.jump_phase >= 0.48)
        h_err = torch.square(self.root_states[:, 2] - 0.38)
        pitch_err = torch.square(self.projected_gravity[:, 1])
        roll_err = torch.square(self.projected_gravity[:, 0])
        vel_err = torch.square(self.base_lin_vel[:, 0]) + torch.square(self.base_lin_vel[:, 1]) + torch.square(self.base_ang_vel[:, 0])
        balance_bonus = torch.exp(-h_err / 0.005) * torch.exp(-(pitch_err + roll_err) / 0.01) * torch.exp(-vel_err / 0.1)
        return balance_bonus * in_recovery.float()

    def _reward_tracking_lin_vel(self):
        """
        Phase 2.4: Optimized Post-Landing Ground Zero-Velocity Recovery Reward.
        Activates strictly after landing touchdown and initial impact cushioning,
        when bilateral wheel contact is firmly established.
        - Must have taken off (has_taken_off)
        - Must be past flight phase (jump_phase >= 0.35)
        - Both wheels firmly grounded (fn_L > 20N, fn_R > 20N, not airborne)
        - Vertical impact cushioned (vz > -0.3 m/s)
        """
        fn_L = self.contact_forces[:, self.left_wheel_body, 2]
        fn_R = self.contact_forces[:, self.right_wheel_body, 2]
        firm_grounded = (fn_L > 20.0) & (fn_R > 20.0) & (~self.is_airborne)
        cushioned = (self.base_lin_vel[:, 2] > -0.3)
        in_ground_recovery = (self.jump_phase >= 0.35) & self.has_taken_off & firm_grounded & cushioned

        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        tracking = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        return tracking * in_ground_recovery.float()


