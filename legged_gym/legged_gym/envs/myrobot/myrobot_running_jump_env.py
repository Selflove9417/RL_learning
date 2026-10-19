# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# MyRobot Running Jump Environment (Dynamic In-Motion Takeoff & Continuous Cruising)

import os
import math
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse
from rsl_rl.modules import ActorCritic
from legged_gym.envs.myrobot.myrobot_dual_env import (
    MyRobotDualEnv,
    compute_exact_cylinder_clearance,
    compute_leg_ik,
    STATE_NAMES
)


class MyRobotRunningJump(MyRobotDualEnv):
    """
    Dedicated Dynamic Running Jump Environment for MyRobot.
    Architecture:
    1. Continuous Cruising (Phase 15 Balance Policy):
       - Robot physically accelerates to and maintains cruising speed (0.20 ~ 0.50 m/s).
       - Full inverted-pendulum balance and wheel speed synchronization.
    2. Dynamic In-Motion Takeoff (Jump Policy):
       - Zero stopping / zero braking condition before jump.
       - Direct transition into explosive leg extension while rolling forward.
       - Forward wheel drive feedforward preserves horizontal momentum without pitch tipping.
    3. Airborne Flight & Clearance:
       - Bilateral wheel cylinder clearance exceeds >= 20 cm (apex height ~ 0.75m).
       - In-air body attitude stabilization (pitch & roll regulation).
    4. Compliant Touchdown Cushion & Continuous Forward Resumption:
       - Compliant impact absorption without smashing into the ground.
       - Smooth handoff back to Phase 15 with forward cruising velocity intact!
       - Continuous multi-cycle running jump capability without environment reset.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # Initialize dual policies from trained assets
        self.init_dual_policies()
        self.user_cmd_vy[:] = self.target_forward_vel
        self.actual_cmd_vy[:] = self.target_forward_vel

    def _init_buffers(self):
        super()._init_buffers()
        self.target_forward_vel = getattr(self.cfg.commands, 'target_forward_vel', 0.20)
        self.default_jump_height = getattr(self.cfg.commands, 'target_jump_height', 0.20)
        self.cycle_length = getattr(self.cfg.env, 'cycle_length', 140) # 140 steps = 2.8s per running jump cycle
        self.jump_trigger_step = getattr(self.cfg.env, 'jump_trigger_step', 40) # Jump triggers at t = 0.80s

        self.commands[:, 1] = self.target_forward_vel

        self.takeoff_vy = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.touchdown_vy = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.flight_distance_y = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.velocity_retention_ratio = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.has_taken_off = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.has_touched_down = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.is_airborne = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.max_flight_clearance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        # Initialize teleoperation command targets to cruising speed
        self.user_cmd_vy[env_ids] = self.target_forward_vel
        self.actual_cmd_vy[env_ids] = self.target_forward_vel
        self.user_cmd_yaw[env_ids] = 0.0
        self.actual_cmd_yaw[env_ids] = 0.0
        self.user_target_height[env_ids] = 0.39774
        self.actual_cmd_height[env_ids] = 0.39774
        self.commands[env_ids, 1] = self.target_forward_vel

        # Dynamic jump trackers
        self.takeoff_vy[env_ids] = 0.0
        self.touchdown_vy[env_ids] = 0.0
        self.flight_distance_y[env_ids] = 0.0
        self.velocity_retention_ratio[env_ids] = 0.0
        self.filtered_actions[env_ids] = 0.0
        self.has_taken_off[env_ids] = False
        self.has_touched_down[env_ids] = False
        self.is_airborne[env_ids] = False
        self.wheel_clearance[env_ids] = 0.0
        self.bilateral_clearance[env_ids] = 0.0
        self.max_flight_clearance[env_ids] = 0.0

    def compute_observations(self):
        """
        35-dim observation vector (100% compatible with ActorCritic architecture):
        - base_lin_vel * 2.0 (3: [vx*2, vy*2, vz*2])
        - base_ang_vel * 0.25 (3)
        - projected_gravity (3)
        - jump_cmds (4: [cmd_fwd * 2.0, cmd_yaw * 0.25, norm_h, jump_phase])
          * Slot 9 receives forward command (cmd_fwd = actual_cmd_vy) with NO ZeroCmd!
        - obs_dof_pos (6: joints relative to nominal, wheels zeroed)
        - dof_vel * 0.05 (6)
        - filtered_actions (6)
        - contact_state (2: normalized contact forces)
        - phase_clock (2: sin(2pi*phase), cos(2pi*phase))
        """
        obs_dof_pos = (self.dof_pos - self.default_dof_pos).clone()
        obs_dof_pos[:, self.wheel_indices] = 0.0

        norm_h = (self.default_jump_height - 0.20) / 0.10
        cmd_fwd = self.actual_cmd_vy if hasattr(self, 'actual_cmd_vy') else self.commands[:, 1]
        cmd_yaw = self.actual_cmd_yaw if hasattr(self, 'actual_cmd_yaw') else self.commands[:, 2]

        jump_cmds = torch.stack([
            cmd_fwd * 2.0,
            cmd_yaw * 0.25,
            torch.full_like(cmd_fwd, norm_h),
            self.jump_phase
        ], dim=-1)

        phi = 2.0 * math.pi * self.jump_phase
        phase_clock = torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)

        fn_L = torch.clamp(self.contact_forces[:, self.left_wheel_body, 2:3] / 150.0, 0.0, 1.0)
        fn_R = torch.clamp(self.contact_forces[:, self.right_wheel_body, 2:3] / 150.0, 0.0, 1.0)
        contact_state = torch.cat([fn_L, fn_R], dim=-1)

        self.obs_buf = torch.cat([
            self.base_lin_vel * 2.0,
            self.base_ang_vel * 0.25,
            self.projected_gravity,
            jump_cmds,
            obs_dof_pos * 1.0,
            self.dof_vel * 0.05,
            self.filtered_actions,
            contact_state,
            phase_clock,
        ], dim=-1)

        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = self.obs_buf.clone()

    def step(self, actions):
        """
        Step simulation with unified action execution for train.py and play.py.
        1. Cruising phase (STATE_BALANCE):
           Robot is balanced and accelerated by frozen Phase 15.
        2. Jump phase (STATE_JUMP / STATE_RECOVER):
           Actuators are directly driven by incoming external `actions` from PPO Runner / play.py!
        3. Timing and lifecycle:
           render -> simulate physics -> refresh -> check termination -> compute reward -> reset -> compute obs.
        """
        # Periodic auto-trigger for running jump cycle
        step_in_cycle = self.common_step_counter % self.cycle_length
        if step_in_cycle == self.jump_trigger_step:
            can_jump = (self.fsm_state == self.STATE_BALANCE)
            if can_jump.any():
                self.request_teleop_jump(can_jump)

        # 1. Render for interactive viewer if active
        self.render()

        # 2. Execute dual-policy physics step without internal post_physics_step
        self.step_dual_policy(actions=actions, call_post_physics_step=False)

        # 3. Refresh simulation sensors and base kinematics
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 4. Refresh kinematics, clearance, airborne & contacts
        self._refresh_physical_state(update_has_taken_off=False)
        clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(self)
        self.wheel_clearance[:, 0] = clr_L
        self.wheel_clearance[:, 1] = clr_R
        self.bilateral_clearance[:] = clr_min

        # 5. Check termination
        r_raw, p_raw, _ = get_euler_xyz(self.base_quat)
        pitch = torch.atan2(torch.sin(p_raw), torch.cos(p_raw))
        roll = torch.atan2(torch.sin(r_raw), torch.cos(r_raw))

        fall = (self.root_states[:, 2] < 0.20) | (torch.abs(pitch) > math.radians(45.0)) | (torch.abs(roll) > math.radians(40.0))
        timeout = self.episode_length_buf >= self.max_episode_length

        self.reset_buf = fall | timeout
        self.time_out_buf = timeout & (~fall)

        # 6. Compute reward for (s_t, a_t) -> s_{t+1} transition BEFORE reset
        self.compute_reward()

        # 7. Reset terminated envs
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # 8. Compute observations for next step AFTER reset
        self.compute_observations()

        # 9. Update last step buffers
        clip_actions = getattr(self.cfg.normalization, 'clip_actions', 100.0)
        self.last_actions[:] = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    # =========================================================================
    # Reward Functions: Explicitly Tailored for In-Motion Running Jump
    # =========================================================================

    def _reward_forward_velocity_retention(self):
        """Rewards keeping forward velocity v_y ~ v_y^cmd across the ENTIRE jump cycle."""
        cmd_vy = self.actual_cmd_vy if hasattr(self, 'actual_cmd_vy') else self.commands[:, 1]
        err_vy = self.base_lin_vel[:, 1] - cmd_vy
        sigma = self.cfg.rewards.tracking_sigma
        return torch.exp(-torch.square(err_vy) / (2.0 * sigma**2))

    def _reward_takeoff_velocity(self):
        """Rewards vertical explosive impulse during push-off window (phi in [0.08, 0.24])."""
        is_push = (self.jump_phase >= 0.08) & (self.jump_phase <= 0.24)
        vz = self.base_lin_vel[:, 2]
        return torch.where(is_push, torch.clamp(vz, 0.0, 1.2), torch.zeros_like(vz))

    def _reward_jump_clearance(self):
        """Rewards ground clearance exceeding >= 20 cm while in flight."""
        target = self.cfg.commands.target_jump_height
        clr_min, _, _ = compute_exact_cylinder_clearance(self)
        reward = torch.clamp(clr_min / target, 0.0, 1.25)
        return torch.where(self.is_airborne & self.has_taken_off, reward, torch.zeros_like(reward))

    def _reward_landing_cushion(self):
        """Rewards compliant impact absorption after touchdown without smashing into ground."""
        height = self.root_states[:, 2]
        cushion_rew = torch.clamp((height - 0.30) / 0.10, 0.0, 1.0)
        return torch.where(self.has_touched_down, cushion_rew, torch.zeros_like(cushion_rew))

    def _reward_post_touchdown_forward(self):
        """Rewards immediately tracking forward velocity after landing."""
        cmd_vy = self.actual_cmd_vy if hasattr(self, 'actual_cmd_vy') else self.commands[:, 1]
        err_vy = self.base_lin_vel[:, 1] - cmd_vy
        ret = torch.exp(-torch.square(err_vy) / (2.0 * (self.cfg.rewards.tracking_sigma**2)))
        return torch.where(self.has_touched_down, ret, torch.zeros_like(ret))

    def _reward_pitch_stability(self):
        """Penalizes pitch deviation and angular pitch rate."""
        _, p_raw, _ = get_euler_xyz(self.base_quat)
        pitch = torch.atan2(torch.sin(p_raw), torch.cos(p_raw))
        return -torch.square(pitch) - 0.05 * torch.square(self.base_ang_vel[:, 0])

    def _reward_roll_stability(self):
        """Penalizes roll deviation and roll angular rate."""
        r_raw, _, _ = get_euler_xyz(self.base_quat)
        roll = torch.atan2(torch.sin(r_raw), torch.cos(r_raw))
        return -torch.square(roll) - 0.05 * torch.square(self.base_ang_vel[:, 1])

    def _reward_ang_vel_xy(self):
        return -torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

    def _reward_torques(self):
        return -torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_acc(self):
        return -torch.sum(torch.square((self.dof_vel - self.last_dof_vel) / self.dt), dim=-1)

    def _reward_torque_limits(self):
        return -torch.sum(torch.clamp(torch.abs(self.torques) - 0.90 * self.torque_limits, min=0.0), dim=-1)
