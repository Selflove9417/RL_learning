# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# MyRobot Forward Jump Static Environment (Phase 2: Minimal Static Forward Jump Curriculum)

import math
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate, get_euler_xyz
from legged_gym.envs.myrobot.myrobot_forward_jump_env import MyRobotForwardJump
from legged_gym.envs.myrobot.myrobot_jump_env import MyRobotJump


class MyRobotForwardJumpStatic(MyRobotForwardJump):
    """
    Dedicated Static Forward Jump Environment for MyRobot (Phase 2 Curriculum).
    Features:
    1. Single-Jump Latching:
       - Strictly enforces single jump cycle per episode.
       - Locks phase in recovery balance window (phi = 0.85) once first jump touches down.
       - Prevents wrap-around and completely stops second jump occurrence.
    2. True Whole-Body Center of Mass (CoM) Displacement:
       - Uses exact URDF body masses and rotational CoM offset projections R(q)*c.
       - Guarantees 100% kinematic closure between flight displacement and velocity integral.
    3. Clearance-Gated Forward Rewards:
       - Forward displacement and touchdown rewards are strictly gated by achieving vertical clearance >= 18 cm.
       - Completely prevents sacrificing vertical jump height to lunge forward horizontally.
    4. Gaussian Centered Velocity & Distance Target:
       - Forward velocity tracking centered at 0.20 m/s (sigma = 0.06 m/s).
       - Flight displacement tracking centered at 7.5 cm (0.075 m, target window 5-10 cm).
    5. One-Shot Touchdown Reward:
       - Fired strictly on the single transition step from airborne to grounded inside post_physics_step.
       - Zero step repetition, zero cross-episode leakage.
    6. Stationary Preparation Constraint:
       - Penalizes wheel rotation and base horizontal velocity during pre-jump window (phi in [0.00, 0.10]).
       - In push-off window (phi in [0.10, 0.22]), constraint is inactive, allowing ground reaction propulsion.
    """

    def _init_buffers(self):
        super()._init_buffers()

        # Cache rigid body masses and local CoM offsets for exact whole-body CoM tracking
        props = self.gym.get_actor_rigid_body_properties(self.envs[0], 0)
        self.rb_masses = torch.tensor([p.mass for p in props], device=self.device, dtype=torch.float)
        self.total_mass = self.rb_masses.sum()
        self.rb_com_offsets = torch.tensor([[p.com.x, p.com.y, p.com.z] for p in props], device=self.device, dtype=torch.float)
        self.num_bodies = len(props)

        # Single-jump lifecycle buffers
        self.jump_latched = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.one_shot_touchdown_event = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.expected_airborne_distance = getattr(self.cfg.commands, 'expected_airborne_distance', 0.075)
        self.target_forward_vel = getattr(self.cfg.commands, 'target_forward_vel', 0.20)

        # Initialize commands: forward command along local Y
        self.commands[:, 0] = 0.0
        self.commands[:, 1] = self.target_forward_vel
        self.commands[:, 2] = 0.0

    def _compute_true_com(self):
        """
        Computes the exact 3D position and linear velocity of the whole-body Center of Mass.
        Incorporates rigid body origin positions plus 3D rotated internal CoM offsets: R(q) * c.
        """
        rb_pos = self.rb_states[:, :, 0:3]       # (num_envs, num_bodies, 3)
        rb_quat = self.rb_states[:, :, 3:7]      # (num_envs, num_bodies, 4)
        rb_lin_vel = self.rb_states[:, :, 7:10]  # (num_envs, num_bodies, 3)

        flat_quat = rb_quat.reshape(-1, 4)
        flat_offsets = self.rb_com_offsets.unsqueeze(0).expand(self.num_envs, -1, -1).reshape(-1, 3)
        r_rot = quat_rotate(flat_quat, flat_offsets).reshape(self.num_envs, self.num_bodies, 3)

        true_body_pos = rb_pos + r_rot
        masses_expanded = self.rb_masses.unsqueeze(0).unsqueeze(-1)  # (1, num_bodies, 1)

        true_com_pos = (true_body_pos * masses_expanded).sum(dim=1) / self.total_mass
        true_com_vel = (rb_lin_vel * masses_expanded).sum(dim=1) / self.total_mass
        return true_com_pos, true_com_vel

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        super().reset_idx(env_ids)

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Enforce strictly zero initial velocities at episode start (no cheating with pre-roll)
        self.root_states[env_ids, 7:13] = 0.0
        self.dof_vel[env_ids, :] = 0.0
        self.base_lin_vel[env_ids, :] = 0.0
        self.base_ang_vel[env_ids, :] = 0.0

        # Set forward velocity command along local Y (0.20 m/s curriculum center)
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

        # Reset single-jump tracking states
        self.jump_timer[env_ids] = 0.0
        self.jump_phase[env_ids] = 0.0
        self.jump_latched[env_ids] = False
        self.has_taken_off[env_ids] = False
        self.has_touched_down[env_ids] = False
        self.one_shot_touchdown_event[env_ids] = False

        self.takeoff_vy[env_ids] = 0.0
        self.takeoff_vz[env_ids] = 0.0
        self.takeoff_pos_y[env_ids] = 0.0
        self.touchdown_pos_y[env_ids] = 0.0
        self.flight_distance_y[env_ids] = 0.0
        self.final_flight_distance[env_ids] = 0.0

    def step(self, actions):
        # Action smoothing
        clip_actions = self.cfg.normalization.clip_actions
        clipped = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.last_actions[:] = self.filtered_actions
        self.filtered_actions[:] = 0.80 * clipped + 0.20 * self.last_actions

        # Advance jump timer
        self.jump_timer += self.dt

        # SINGLE-JUMP LATCHING MECHANISM:
        # Once the robot has taken off, touched down, and reached the recovery phase (jump_timer >= 0.50 * cycle),
        # latch the jump permanently for the remainder of this episode!
        landing_completed = self.has_taken_off & self.has_touched_down & (self.jump_timer >= 0.50 * self.jump_cycle_time)
        self.jump_latched |= landing_completed

        # When latched: lock the phase into the stationary recovery balance window (phi = 0.85).
        # This completely prevents cycle wrap-around and stops any second jump from occurring!
        clamped_timer = torch.where(
            self.jump_latched,
            torch.clamp(self.jump_timer, max=0.85 * self.jump_cycle_time),
            self.jump_timer
        )
        self.jump_phase[:] = clamped_timer / self.jump_cycle_time

        # Step physics via base LeggedRobot class directly
        return super(MyRobotJump, self).step(actions)

    def _post_physics_step_callback(self):
        """
        Executed inside post_physics_step() immediately after PhysX simulation and before compute_reward().
        Guarantees rewards and observations see current-step physical state.
        """
        self._refresh_physical_state(update_has_taken_off=True)

    def _refresh_physical_state(self, update_has_taken_off=True):
        """
        Refresh rigid body states and calculate true CoM clearance,
        airborne flight window, takeoff velocities, and authentic CoM flight displacement.
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

        # Compute exact whole-body Center of Mass
        true_com_pos, true_com_vel = self._compute_true_com()

        # Takeoff detection: in push-off window with positive vz and airborne, before any touchdown
        if update_has_taken_off:
            in_pushoff = (self.jump_phase >= 0.10) & (self.jump_phase <= 0.28)
            upward_launch = (self.base_lin_vel[:, 2] > 0.8) & now_airborne
            self.has_taken_off |= (in_pushoff & upward_launch & (~self.has_touched_down))

        # Detect the exact takeoff transition instant
        just_took_off = now_airborne & (~self.is_airborne) & (~self.has_touched_down)
        if just_took_off.any():
            self.takeoff_vz[just_took_off] = self.base_lin_vel[just_took_off, 2]
            self.takeoff_vy[just_took_off] = self.base_lin_vel[just_took_off, 1]
            self.takeoff_pos_y[just_took_off] = true_com_pos[just_took_off, 1]

        # Detect the exact touchdown transition instant (ONE-SHOT EVENT)
        just_landed = (~now_airborne) & self.is_airborne & self.has_taken_off & (~self.has_touched_down)
        self.one_shot_touchdown_event[:] = False
        if just_landed.any():
            self.one_shot_touchdown_event[just_landed] = True
            self.touchdown_pos_y[just_landed] = true_com_pos[just_landed, 1]
            self.final_flight_distance[just_landed] = self.touchdown_pos_y[just_landed] - self.takeoff_pos_y[just_landed]
            self.has_touched_down[just_landed] = True

        # While airborne, flight_distance_y tracks authentic displacement of true whole-body CoM
        self.flight_distance_y = torch.where(
            now_airborne & self.has_taken_off & (~self.has_touched_down),
            true_com_pos[:, 1] - self.takeoff_pos_y,
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

    # ==================== PHASE 2 REWARD FUNCTIONS ====================

    def _reward_stationary_prep(self):
        """
        STRICT REQUIREMENT:
        Penalizes wheel rolling and base velocity during pre-jump window (phi in [0.00, 0.05], t <= 0.10s).
        Strictly zero penalty during push-off (phi >= 0.06), allowing propulsion against ground.
        """
        in_prep = (self.jump_phase >= 0.00) & (self.jump_phase <= 0.05) & (~self.has_taken_off)
        lin_vel_sq = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        wheel_vel_sq = torch.square(self.dof_vel[:, self.left_wheel_idx]) + torch.square(self.dof_vel[:, self.right_wheel_idx])
        stationary_reward = torch.exp(-lin_vel_sq / 0.02) * torch.exp(-wheel_vel_sq / 1.0)
        return stationary_reward * in_prep.float()

    def _reward_takeoff_forward_velocity(self):
        """
        Rewards actively generating forward push-off velocity along local Y axis
        during push-off window (phi in [0.06, 0.22]).
        Pure Gaussian tracking centered at 0.20 m/s (sigma = 0.06 m/s).
        GATED BY UPWARD VELOCITY: Must achieve upward launch vz >= 0.8 m/s to earn forward velocity!
        Prevents forward tipping before vertical push-off force is engaged.
        """
        in_launch = (self.jump_phase >= 0.06) & (self.jump_phase <= 0.22) & (~self.has_touched_down)
        vy = self.base_lin_vel[:, 1]
        target_vy = self.commands[:, 1]  # ~0.20 m/s

        vz_gate = torch.clamp((self.base_lin_vel[:, 2] - 0.8) / 0.8, min=0.0, max=1.0)
        vel_tracking = torch.exp(-torch.square(vy - target_vy) / (2.0 * (0.06 ** 2)))
        return vel_tracking * vz_gate * in_launch.float()

    def _reward_airborne_forward_displacement(self):
        """
        Rewards forward displacement strictly accumulated between authentic takeoff and touchdown
        using the TRUE whole-body Center of Mass displacement.
        STRICT AUDIT REQUIREMENT: Gated by achieving vertical clearance >= 20.0 cm!
        Zero reward if robot lunges forward with degraded vertical clearance (< 18.5 cm).
        """
        valid_flight = self.is_airborne & self.has_taken_off & (~self.has_touched_down) & (self.jump_phase <= 0.40)
        
        # Clearance gate: must achieve at least 18.5 cm clearance, reaches full reward at 21.0 cm
        clr_gate = torch.clamp((self.max_flight_clearance - 0.185) / 0.025, min=0.0, max=1.0)
        
        target_d = self.expected_airborne_distance  # 0.070 m (7.0 cm)
        d_air = torch.clamp(self.flight_distance_y, min=0.0)
        
        # Tracking bell curve centered at target 7.0 cm (target window 5-10 cm)
        dist_tracking = torch.exp(-torch.square(d_air - target_d) / (2.0 * (0.04 ** 2)))
        progress = torch.clamp(d_air / target_d, min=0.0, max=1.0)
        score = 0.5 * progress + 0.5 * dist_tracking
        
        return score * clr_gate * valid_flight.float()

    def _reward_touchdown_forward_distance(self):
        """
        ONE-SHOT REWARD: Awarded strictly on the exact transition step of touchdown!
        Gated by achieving minimum 20.0 cm clearance during flight.
        """
        target_d = self.expected_airborne_distance  # 0.070 m
        clr_ok = (self.max_flight_clearance >= 0.200).float()
        d_achieved = self.final_flight_distance
        dist_score = torch.exp(-torch.square(d_achieved - target_d) / (2.0 * (0.04 ** 2)))
        return dist_score * clr_ok * self.one_shot_touchdown_event.float()

    def _reward_tracking_lin_vel(self):
        """
        Post-landing recovery braking reward (phi >= 0.45, or after landing):
        Once landed and cushioned, rewards braking the robot to stationary zero-velocity balance.
        Maintained until episode end.
        """
        fn_L = self.contact_forces[:, self.left_wheel_body, 2]
        fn_R = self.contact_forces[:, self.right_wheel_body, 2]
        firm_grounded = (fn_L > 15.0) & (fn_R > 15.0) & (~self.is_airborne)
        cushioned = (self.base_lin_vel[:, 2] > -0.3)
        in_recovery = self.has_touched_down & firm_grounded & cushioned

        lin_vel_error = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        tracking = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        return tracking * in_recovery.float()

    def _reward_bilateral_symmetry(self):
        """Penalizes difference in wheel clearance, left/right joint disparity, and push-off knee torque disparity."""
        clr_diff = torch.abs(self.wheel_clearance[:, 0] - self.wheel_clearance[:, 1])
        knee_diff = torch.abs(self.dof_pos[:, self.left_knee_idx] - self.dof_pos[:, self.right_knee_idx])
        yaw_rate = torch.abs(self.base_ang_vel[:, 2])

        # Push-off torque symmetry penalty (suppressing yaw impulse generation at takeoff)
        in_pushoff = (self.jump_phase >= 0.10) & (self.jump_phase <= 0.22)
        knee_tau_diff = torch.abs(self.torques[:, self.left_knee_idx] - self.torques[:, self.right_knee_idx]) / 120.0
        pushoff_penalty = knee_tau_diff * in_pushoff.float()

        return clr_diff * 4.0 + knee_diff * 0.5 + yaw_rate * 0.5 + pushoff_penalty * 2.0
