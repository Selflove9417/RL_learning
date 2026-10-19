# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# MyRobot Running Jump Environment (Dynamic In-Motion Takeoff & Continuous Cruising)

import math
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse
from rsl_rl.modules import ActorCritic
from legged_gym.envs.myrobot.myrobot_jump_env import MyRobotJump
from legged_gym.envs.myrobot.myrobot_dual_env import compute_exact_cylinder_clearance, compute_leg_ik


class MyRobotRunningJump(MyRobotJump):
    """
    Dedicated Dynamic Running Jump Environment for MyRobot.
    Features:
    1. Continuous Cruising Warmup:
       - Phase 15 balance controller physically accelerates the robot to cruising speed
         (0.20 m/s ~ 0.50 m/s) in reset_idx() with real wheel traction and body pitch equilibrium.
       - Zero artificial root state hacking; authentic physical momentum.
    2. Dynamic In-Motion Takeoff:
       - Zero stopping / zero braking condition before jump.
       - Jump Actor initiates leg push-off while rolling forward at full speed.
       - Forward wheel bias prevents pitch-up tipping and maintains forward drive.
    3. Airborne Flight & Clearance:
       - Preserves horizontal momentum, achieving wide forward ballistic flight (>= 30-50 cm).
       - Bilateral wheel cylinder clearance exceeds >= 20 cm.
       - In-air body attitude stabilization (pitch & roll regulation).
    4. Touchdown Cushion & Continuous Forward Resumption:
       - NO ZeroCmd! Forward velocity command remains v_y^cmd throughout and after landing.
       - Compliance cushion absorbs vertical impact while wheels maintain cruising drive.
    """

    def _init_buffers(self):
        super()._init_buffers()

        # 1. Load frozen Phase 15 Balance Policy for autonomous cruising warmup
        bal_model_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase15/models/phase15_m130.pt"
        self.ac_balance = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(self.device)
        ckpt_bal = torch.load(bal_model_path, map_location=self.device)
        self.ac_balance.load_state_dict(ckpt_bal["model_state_dict"])
        self.ac_balance.eval()
        for param in self.ac_balance.parameters():
            param.requires_grad = False

        # Standing height and leg IK constants
        self.h_ctrl = 0.39774
        self.norm_h_val = (self.h_ctrl - 0.40) / 0.15
        self.norm_h_tensor = torch.full((self.num_envs, 1), self.norm_h_val, device=self.device, dtype=torch.float)
        self.q1_ik, self.q2_ik = compute_leg_ik(torch.tensor([self.h_ctrl], device=self.device))

        # 2. Dynamic Jump Trackers
        self.target_forward_vel = getattr(self.cfg.commands, 'target_forward_vel', 0.20)
        self.commands[:, 0] = 0.0
        self.commands[:, 1] = self.target_forward_vel
        self.commands[:, 2] = 0.0

        self.takeoff_vy = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.takeoff_pos_y = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.touchdown_vy = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.touchdown_pos_y = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.flight_distance_y = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.velocity_retention_ratio = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.has_touched_down = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Warmup buffers
        self.warmup_steps = getattr(self.cfg.env, 'warmup_steps', 40)
        self.bal_filtered_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self.pi_integral = torch.zeros(self.num_envs, device=self.device)
        self.pi_v_filt = torch.zeros(self.num_envs, device=self.device)

        # Episode steps
        self.max_episode_length = int(round(self.max_episode_length_s / self.dt))

    def _ensure_mutable_tensors(self):
        if not torch.is_inference_mode_enabled():
            for name in [
                'max_flight_clearance', 'takeoff_vy', 'takeoff_pos_y',
                'touchdown_vy', 'touchdown_pos_y', 'flight_distance_y',
                'velocity_retention_ratio', 'has_taken_off', 'has_touched_down',
                'is_airborne', 'wheel_clearance', 'bilateral_clearance',
                'jump_timer', 'jump_phase', 'filtered_actions',
                'bal_filtered_actions', 'pi_integral', 'pi_v_filt'
            ]:
                val = getattr(self, name, None)
                if val is not None and isinstance(val, torch.Tensor) and val.is_inference():
                    setattr(self, name, val.clone())

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._ensure_mutable_tensors()

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Sample or set forward velocity command
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

        # Place robot at tangent ground contact stance (z = 0.397739)
        self.root_states[env_ids, 0] = self.env_origins[env_ids, 0]
        self.root_states[env_ids, 1] = self.env_origins[env_ids, 1]
        self.root_states[env_ids, 2] = 0.397739
        self.root_states[env_ids, 3:6] = 0.0
        self.root_states[env_ids, 6] = 1.0  # Unit quaternion
        self.root_states[env_ids, 7:13] = 0.0

        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # ------------------------------------------------------------------
        # Authentic Cruising Warmup: Phase 15 drives the robot to v_y^cmd
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Authentic Cruising Warmup: Phase 15 drives the robot smoothly to v_y^cmd
        # ------------------------------------------------------------------
        self.bal_filtered_actions[env_ids] = 0.0
        actual_cmd_vy = torch.zeros(len(env_ids), device=self.device)
        step_max_vy = 0.8 * self.dt  # Smooth acceleration ramp: 0.016 m/s per step

        warmup_n = self.warmup_steps
        for _ in range(warmup_n):
            delta_vy = self.commands[env_ids, 1] - actual_cmd_vy
            actual_cmd_vy += torch.clamp(delta_vy, -step_max_vy, step_max_vy)

            bal_cmds = torch.zeros(len(env_ids), 3, device=self.device)
            bal_cmds[:, 1] = actual_cmd_vy

            dof_pos_bal = (self.dof_pos[env_ids] - self.default_dof_pos).clone()
            dof_pos_bal[:, self.wheel_indices] = 0.0

            bal_obs = torch.cat([
                self.base_lin_vel[env_ids] * 2.0,
                self.base_ang_vel[env_ids] * 0.25,
                self.projected_gravity[env_ids],
                bal_cmds * torch.tensor([2.0, 2.0, 0.25], device=self.device),
                dof_pos_bal * 1.0,
                self.dof_vel[env_ids] * 0.05,
                self.bal_filtered_actions[env_ids],
                self.norm_h_tensor[env_ids]
            ], dim=-1)

            with torch.no_grad():
                act_bal_raw = self.ac_balance.act_inference(bal_obs)
            self.bal_filtered_actions[env_ids] = 0.80 * act_bal_raw + 0.20 * self.bal_filtered_actions[env_ids]

            tgt_hip_bal_L = self.default_dof_pos[0, self.left_hip_idx] + self.q1_ik + 0.25 * 0.7 * self.bal_filtered_actions[env_ids, self.left_hip_idx]
            tgt_knee_bal_L = self.default_dof_pos[0, self.left_knee_idx] + self.q2_ik + 0.25 * 0.5 * self.bal_filtered_actions[env_ids, self.left_knee_idx]
            tgt_hip_bal_R = self.default_dof_pos[0, self.right_hip_idx] + self.q1_ik + 0.25 * 0.7 * self.bal_filtered_actions[env_ids, self.right_hip_idx]
            tgt_knee_bal_R = self.default_dof_pos[0, self.right_knee_idx] + self.q2_ik + 0.25 * 0.5 * self.bal_filtered_actions[env_ids, self.right_knee_idx]

            base_tau_L = 7.0 * torch.clamp(self.bal_filtered_actions[env_ids, self.left_wheel_idx], -1.0, 1.0) - 0.3 * self.dof_vel[env_ids, self.left_wheel_idx]
            base_tau_R = 7.0 * torch.clamp(self.bal_filtered_actions[env_ids, self.right_wheel_idx], -1.0, 1.0) - 0.3 * self.dof_vel[env_ids, self.right_wheel_idx]
            wheel_speed_error = self.dof_vel[env_ids, self.left_wheel_idx] - self.dof_vel[env_ids, self.right_wheel_idx]
            delta_tau_omega = torch.clamp(0.10 * wheel_speed_error, -1.5, 1.5)

            tau_limit_w = self.torque_limits[self.left_wheel_idx].item()
            tau_c = 0.5 * (base_tau_L + base_tau_R)
            tau_d = 0.5 * (base_tau_R - base_tau_L) + delta_tau_omega

            tau_d_clamped = torch.clamp(tau_d, -tau_limit_w, tau_limit_w)
            tau_c_limit = torch.clamp(torch.tensor([tau_limit_w], device=self.device) - torch.abs(tau_d_clamped), min=0.0)
            tau_c_allocated = torch.clamp(tau_c, -tau_c_limit, tau_c_limit)

            tau_w_L = tau_c_allocated - tau_d_clamped
            tau_w_R = tau_c_allocated + tau_d_clamped

            kp_leg = 40.0
            kd_leg = 2.0

            for _ in range(self.cfg.control.decimation):
                torques = torch.zeros(self.num_envs, self.num_actions, device=self.device)
                torques[env_ids, self.left_hip_idx] = kp_leg * (tgt_hip_bal_L - self.dof_pos[env_ids, self.left_hip_idx]) - kd_leg * self.dof_vel[env_ids, self.left_hip_idx]
                torques[env_ids, self.right_hip_idx] = kp_leg * (tgt_hip_bal_R - self.dof_pos[env_ids, self.right_hip_idx]) - kd_leg * self.dof_vel[env_ids, self.right_hip_idx]
                torques[env_ids, self.left_knee_idx] = kp_leg * (tgt_knee_bal_L - self.dof_pos[env_ids, self.left_knee_idx]) - kd_leg * self.dof_vel[env_ids, self.left_knee_idx]
                torques[env_ids, self.right_knee_idx] = kp_leg * (tgt_knee_bal_R - self.dof_pos[env_ids, self.right_knee_idx]) - kd_leg * self.dof_vel[env_ids, self.right_knee_idx]
                torques[env_ids, self.left_wheel_idx] = tau_w_L
                torques[env_ids, self.right_wheel_idx] = tau_w_R
                torques = torch.clip(torques, -self.torque_limits, self.torque_limits)

                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self.gym.refresh_dof_state_tensor(self.sim)

            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
            self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
            self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
            self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])

        # Initialize jump episode trackers
        self.jump_timer[env_ids] = 0.0
        self.jump_phase[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.filtered_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0

        self.takeoff_vy[env_ids] = self.base_lin_vel[env_ids, 1]
        self.takeoff_pos_y[env_ids] = self.root_states[env_ids, 1]
        self.touchdown_vy[env_ids] = 0.0
        self.touchdown_pos_y[env_ids] = 0.0
        self.flight_distance_y[env_ids] = 0.0
        self.velocity_retention_ratio[env_ids] = 0.0

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
          * Slot 9 receives forward command (cmd_fwd = v_y^cmd) with NO ZeroCmd!
        - obs_dof_pos (6: joints relative to nominal, wheels zeroed)
        - dof_vel * 0.05 (6)
        - filtered_actions (6)
        - contact_state (2: normalized contact forces)
        - phase_clock (2: sin(2pi*phase), cos(2pi*phase))
        """
        obs_dof_pos = (self.dof_pos - self.default_dof_pos).clone()
        obs_dof_pos[:, self.wheel_indices] = 0.0

        norm_h = (self.target_jump_height - 0.20) / 0.10
        jump_cmds = torch.stack([
            self.commands[:, 1] * 2.0,
            self.commands[:, 2] * 0.25,
            norm_h,
            self.jump_phase
        ], dim=-1)

        phase_sin = torch.sin(2.0 * math.pi * self.jump_phase).unsqueeze(-1)
        phase_cos = torch.cos(2.0 * math.pi * self.jump_phase).unsqueeze(-1)

        fn_L = torch.clamp(self.contact_forces[:, self.left_wheel_body, 2:3] / 150.0, 0.0, 1.0)
        fn_R = torch.clamp(self.contact_forces[:, self.right_wheel_body, 2:3] / 150.0, 0.0, 1.0)
        contact_state = torch.cat([fn_L, fn_R], dim=-1)

        self.obs_buf = torch.cat([
            self.base_lin_vel * 2.0,
            self.base_ang_vel * 0.25,
            self.projected_gravity,
            jump_cmds,
            obs_dof_pos,
            self.dof_vel * 0.05,
            self.filtered_actions,
            contact_state,
            phase_sin,
            phase_cos
        ], dim=-1)

    def reset(self):
        """
        Clean reset without dummy step(zeros), preventing drop shock before warmup.
        """
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        return self.obs_buf, self.privileged_obs_buf

    def _compute_torques(self, actions):
        """
        Compute joint torques combining leg PD extension and dynamic wheel forward feedforward.
        """
        actions_scaled = actions.clone()
        actions_scaled[:, [self.left_hip_idx, self.right_hip_idx]] *= self.hip_action_scale
        actions_scaled[:, [self.left_knee_idx, self.right_knee_idx]] *= self.knee_action_scale

        # Low-pass filter actions
        self.filtered_actions[:] = 0.80 * actions_scaled + 0.20 * self.filtered_actions

        # Leg PD targets
        target_pos = self.nominal_dof_pos.repeat(self.num_envs, 1)
        target_pos[:, self.leg_indices] += self.filtered_actions[:, self.leg_indices]

        torques = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        for i in [self.left_hip_idx, self.right_hip_idx]:
            kp = self.cfg.control.stiffness['link_002_joint']
            kd = self.cfg.control.damping['link_002_joint']
            torques[:, i] = kp * (target_pos[:, i] - self.dof_pos[:, i]) - kd * self.dof_vel[:, i]

        for i in [self.left_knee_idx, self.right_knee_idx]:
            kp = self.cfg.control.stiffness['link_003_joint']
            kd = self.cfg.control.damping['link_003_joint']
            torques[:, i] = kp * (target_pos[:, i] - self.dof_pos[:, i]) - kd * self.dof_vel[:, i]

        # Dynamic Wheel Torques: Output torque + forward speed tracking damping
        target_wheel_vel = self.commands[:, 1] / self.wheel_radius
        for idx in [self.left_wheel_idx, self.right_wheel_idx]:
            tau_raw = self.wheel_torque_scale * actions_scaled[:, idx]
            vel_err = self.dof_vel[:, idx] - target_wheel_vel
            tau_damping = - self.wheel_velocity_damping * vel_err
            torques[:, idx] = tau_raw + tau_damping

        # Clamp to physical torque limits
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.jump_timer += self.dt
        self.jump_phase[:] = torch.clamp(self.jump_timer / self.jump_cycle_time, 0.0, 1.0)

        # Clearances
        clr_min, clr_L, clr_R = compute_exact_cylinder_clearance(self)
        self.wheel_clearance[:, 0] = clr_L
        self.wheel_clearance[:, 1] = clr_R
        self.bilateral_clearance[:] = clr_min

        # Airborne detection
        fn_L = self.contact_forces[:, self.left_wheel_body, 2]
        fn_R = self.contact_forces[:, self.right_wheel_body, 2]
        is_air = (fn_L < 15.0) & (fn_R < 15.0) & (clr_min > 0.015)
        self.is_airborne[:] = is_air

        # Liftoff event
        liftoff_now = is_air & (self.base_lin_vel[:, 2] > 0.25) & (~self.has_taken_off)
        if liftoff_now.any():
            env_ids = liftoff_now.nonzero(as_tuple=False).flatten()
            self.has_taken_off[env_ids] = True
            self.takeoff_vy[env_ids] = self.base_lin_vel[env_ids, 1]
            self.takeoff_pos_y[env_ids] = self.root_states[env_ids, 1]

        # Flight clearance accumulation
        flight_mask = self.has_taken_off & is_air
        if flight_mask.any():
            self.max_flight_clearance[flight_mask] = torch.maximum(
                self.max_flight_clearance[flight_mask],
                clr_min[flight_mask] * 100.0
            )

        # Touchdown detection
        touchdown_now = self.has_taken_off & (~is_air) & (~self.has_touched_down) & (self.jump_phase >= 0.35)
        if touchdown_now.any():
            td_ids = touchdown_now.nonzero(as_tuple=False).flatten()
            self.has_touched_down[td_ids] = True
            self.touchdown_vy[td_ids] = self.base_lin_vel[td_ids, 1]
            self.touchdown_pos_y[td_ids] = self.root_states[td_ids, 1]
            self.flight_distance_y[td_ids] = (self.touchdown_pos_y[td_ids] - self.takeoff_pos_y[td_ids]) * 100.0
            self.velocity_retention_ratio[td_ids] = self.touchdown_vy[td_ids] / torch.clamp(self.takeoff_vy[td_ids], min=0.05)

        # Check reset conditions (instability / fall / episode timeout)
        self.check_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()

    def check_termination(self):
        # Fall / tipping detection
        r_raw, p_raw, _ = get_euler_xyz(self.base_quat)
        pitch = torch.atan2(torch.sin(p_raw), torch.cos(p_raw))
        roll = torch.atan2(torch.sin(r_raw), torch.cos(r_raw))

        fall = (self.root_states[:, 2] < 0.20) | (torch.abs(pitch) > math.radians(45.0)) | (torch.abs(roll) > math.radians(40.0))
        timeout = self.episode_length_buf >= self.max_episode_length

        self.reset_buf = fall | timeout
        self.time_out_buf = timeout & (~fall)

    # =========================================================================
    # Reward Functions: Explicitly Tailored for In-Motion Running Jump
    # =========================================================================

    def _reward_forward_velocity_retention(self):
        """
        Rewards keeping forward velocity v_y ~ v_y^cmd across the ENTIRE jump cycle.
        Completely penalizes braking before jump or dragging speed in mid-air.
        """
        err_vy = self.base_lin_vel[:, 1] - self.commands[:, 1]
        sigma = self.cfg.rewards.tracking_sigma
        return torch.exp(-torch.square(err_vy) / (2.0 * sigma**2))

    def _reward_takeoff_velocity(self):
        """
        Rewards vertical explosive impulse during push-off window (phi in [0.08, 0.24]).
        """
        is_push = (self.jump_phase >= 0.08) & (self.jump_phase <= 0.24)
        vz = self.base_lin_vel[:, 2]
        return torch.where(is_push, torch.clamp(vz, 0.0, 1.2), torch.zeros_like(vz))

    def _reward_jump_clearance(self):
        """
        Rewards ground clearance exceeding >= 20 cm while in flight.
        """
        target = self.cfg.commands.target_jump_height
        clr_min, _, _ = compute_exact_cylinder_clearance(self)
        reward = torch.clamp(clr_min / target, 0.0, 1.25)
        return torch.where(self.is_airborne & self.has_taken_off, reward, torch.zeros_like(reward))

    def _reward_landing_cushion(self):
        """
        Rewards compliant impact absorption after touchdown without smashing into ground.
        """
        height = self.root_states[:, 2]
        cushion_rew = torch.clamp((height - 0.30) / 0.10, 0.0, 1.0)
        return torch.where(self.has_touched_down, cushion_rew, torch.zeros_like(cushion_rew))

    def _reward_post_touchdown_forward(self):
        """
        Rewards immediately tracking forward velocity after landing.
        """
        err_vy = self.base_lin_vel[:, 1] - self.commands[:, 1]
        ret = torch.exp(-torch.square(err_vy) / (2.0 * (self.cfg.rewards.tracking_sigma**2)))
        return torch.where(self.has_touched_down, ret, torch.zeros_like(ret))

    def _reward_pitch_stability(self):
        """
        Penalizes pitch deviation and angular pitch rate.
        """
        _, p_raw, _ = get_euler_xyz(self.base_quat)
        pitch = torch.atan2(torch.sin(p_raw), torch.cos(p_raw))
        return -torch.square(pitch) - 0.05 * torch.square(self.base_ang_vel[:, 0])

    def _reward_roll_stability(self):
        """
        Penalizes roll deviation and roll angular rate.
        """
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
