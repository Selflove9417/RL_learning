# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# MyRobot Forward Jump Phase 4 Environment
# Incorporates:
# 1. Vectorized Phase 15 Balance Policy Settling inside reset_idx()
# 2. Synchronous Episodic Termination (guaranteeing clean RolloutStorage boundaries)
# 3. Aerial Descent Hip Compensation (+0.10 rad) in step()
# 4. Strict distribution matching with Phase 3.2-V evaluation

import os
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse, get_euler_xyz
from rsl_rl.modules import ActorCritic
from legged_gym.envs.myrobot.myrobot_forward_jump_static_env import MyRobotForwardJumpStatic
from legged_gym.envs.myrobot.myrobot_jump_env import MyRobotJump
from legged_gym.envs.myrobot.myrobot_dual_env import compute_leg_ik


class MyRobotForwardJumpPhase4(MyRobotForwardJumpStatic):
    def _init_buffers(self):
        super()._init_buffers()

        # Load frozen Phase 15 Balance Policy for autonomous stationary settling
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
        self.q1_ik, self.q2_ik = compute_leg_ik(torch.tensor([self.h_ctrl], device=self.device))

        # Fix Float32 precision ceiling issue: 2.8 / 0.02 = 140.0000031 -> ceil gave 141
        # Explicitly round to integer 140 steps to match runner.num_steps_per_env exactly
        self.max_episode_length = int(round(self.max_episode_length_s / self.dt))

        # Dedicated buffer for true pre-reset terminal observations (Target B)
        self.terminal_obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)

        # Reset flag
        self._initial_reset_done = False

    def compute_observations(self):
        super().compute_observations()
        if getattr(self.cfg.env, "zero_cmd_post_touchdown", True):
            if hasattr(self, "has_touched_down"):
                self.obs_buf[self.has_touched_down, 9] = 0.0

    def reset(self):
        """
        Overrides BaseTask.reset() to eliminate the legacy dummy self.step(zeros) step,
        guaranteeing clean balance settling handoff without normal force drop.
        """
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, self.privileged_obs_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # 1. Place robots in nominal ground contact pose (tangent, zero penetration)
        self.root_states[env_ids, 0:3] = self.env_origins[env_ids]
        self.root_states[env_ids, 2] += 0.397739
        self.root_states[env_ids, 3:6] = 0.0
        self.root_states[env_ids, 6] = 1.0
        self.root_states[env_ids, 7:13] = 0.0

        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])

        # 2. Vectorized Phase 15 Balance Policy State-Gated Settling
        # Strictly matches Phase 3.2-V eval_seed_full:
        # |b_vy| <= 0.03, |c_vy| <= 0.03, |b_vz| <= 0.05,
        # fn_L in [20, 180], fn_R in [20, 180],
        # |pitch_deg| <= 2.5, |roll_deg| <= 3.5,
        # consecutive stable steps >= 5, step >= 15.
        # Max settle steps: 75 steps (1.5s timeout).
        max_settle_steps = int(1.5 / self.dt)
        stable_consecutive = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        settle_step_reached = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        stable_reached = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        norm_h_tensor = torch.full((len(env_ids), 1), self.norm_h_val, device=self.device)
        bal_filtered_actions = torch.zeros(len(env_ids), 6, device=self.device)
        pi_integral = torch.zeros(len(env_ids), device=self.device)
        pi_v_filt = torch.zeros(len(env_ids), device=self.device)

        tau_limit_w = self.torque_limits[self.left_wheel_idx]
        kp_leg = 40.0
        kd_leg = 2.0

        for step in range(max_settle_steps):
            dof_pos_bal = (self.dof_pos[env_ids] - self.default_dof_pos).clone()
            dof_pos_bal[:, self.wheel_indices] = 0.0

            v_fwd = self.base_lin_vel[env_ids, 1]
            pi_v_filt = 0.85 * pi_v_filt + 0.15 * v_fwd
            err_v = 0.0 - pi_v_filt
            pi_integral = torch.clamp(pi_integral + err_v * self.dt, -0.075, 0.075)
            cmd_trim = torch.clamp(0.35 * err_v + 0.60 * pi_integral, -0.045, 0.045)

            bal_cmds = torch.zeros(len(env_ids), 3, device=self.device)
            bal_cmds[:, 1] = cmd_trim

            bal_obs = torch.cat([
                self.base_lin_vel[env_ids] * 2.0,
                self.base_ang_vel[env_ids] * 0.25,
                self.projected_gravity[env_ids],
                bal_cmds * torch.tensor([2.0, 2.0, 0.25], device=self.device),
                dof_pos_bal * 1.0,
                self.dof_vel[env_ids] * 0.05,
                bal_filtered_actions,
                norm_h_tensor
            ], dim=-1)

            with torch.no_grad():
                act_bal_raw = self.ac_balance.act_inference(bal_obs)
            bal_filtered_actions = 0.80 * act_bal_raw + 0.20 * bal_filtered_actions

            tgt_hip_bal_L = self.default_dof_pos[:, self.left_hip_idx] + self.q1_ik + 0.25 * 0.7 * bal_filtered_actions[:, self.left_hip_idx]
            tgt_knee_bal_L = self.default_dof_pos[:, self.left_knee_idx] + self.q2_ik + 0.25 * 0.5 * bal_filtered_actions[:, self.left_knee_idx]
            tgt_hip_bal_R = self.default_dof_pos[:, self.right_hip_idx] + self.q1_ik + 0.25 * 0.7 * bal_filtered_actions[:, self.right_hip_idx]
            tgt_knee_bal_R = self.default_dof_pos[:, self.right_knee_idx] + self.q2_ik + 0.25 * 0.5 * bal_filtered_actions[:, self.right_knee_idx]

            base_tau_L = 7.0 * torch.clamp(bal_filtered_actions[:, self.left_wheel_idx], -1.0, 1.0) - 0.3 * self.dof_vel[env_ids, self.left_wheel_idx]
            base_tau_R = 7.0 * torch.clamp(bal_filtered_actions[:, self.right_wheel_idx], -1.0, 1.0) - 0.3 * self.dof_vel[env_ids, self.right_wheel_idx]
            wheel_speed_error = self.dof_vel[env_ids, self.left_wheel_idx] - self.dof_vel[env_ids, self.right_wheel_idx]
            delta_tau_omega = torch.clamp(0.10 * wheel_speed_error, -1.5, 1.5)

            tau_c = 0.5 * (base_tau_L + base_tau_R)
            tau_d = 0.5 * (base_tau_R - base_tau_L) + delta_tau_omega

            tau_d_clamped = torch.clamp(tau_d, -tau_limit_w, tau_limit_w)
            tau_c_limit = torch.clamp(tau_limit_w - torch.abs(tau_d_clamped), min=0.0)
            tau_c_allocated = torch.clamp(tau_c, -tau_c_limit, tau_c_limit)

            tau_w_L = tau_c_allocated - tau_d_clamped
            tau_w_R = tau_c_allocated + tau_d_clamped

            for _ in range(self.cfg.control.decimation):
                torques = torch.zeros(self.num_envs, self.num_actions, device=self.device)
                torques[env_ids, self.left_hip_idx] = kp_leg * (tgt_hip_bal_L - self.dof_pos[env_ids, self.left_hip_idx]) - kd_leg * self.dof_vel[env_ids, self.left_hip_idx]
                torques[env_ids, self.right_hip_idx] = kp_leg * (tgt_hip_bal_R - self.dof_pos[env_ids, self.right_hip_idx]) - kd_leg * self.dof_vel[env_ids, self.right_hip_idx]
                torques[env_ids, self.left_knee_idx] = kp_leg * (tgt_knee_bal_L - self.dof_pos[env_ids, self.left_knee_idx]) - kd_leg * self.dof_vel[env_ids, self.left_knee_idx]
                torques[env_ids, self.right_knee_idx] = kp_leg * (tgt_knee_bal_R - self.dof_pos[env_ids, self.right_knee_idx]) - kd_leg * self.dof_vel[env_ids, self.right_knee_idx]
                torques[env_ids, self.left_wheel_idx] = tau_w_L
                torques[env_ids, self.right_wheel_idx] = tau_w_R
                torques[env_ids] = torch.clip(torques[env_ids], -self.torque_limits, self.torque_limits)

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

            # Check Phase 3.2-V stability criteria
            b_vy = self.base_lin_vel[env_ids, 1]
            _, c_vel = self._compute_true_com()
            c_vy = c_vel[env_ids, 1]
            b_vz = self.base_lin_vel[env_ids, 2]
            fn_L = self.contact_forces[env_ids, self.left_wheel_body, 2]
            fn_R = self.contact_forces[env_ids, self.right_wheel_body, 2]
            r_raw, p_raw, _ = get_euler_xyz(self.base_quat[env_ids])
            pitch_deg = torch.atan2(torch.sin(r_raw), torch.cos(r_raw)) * 180.0 / 3.141592653589793
            roll_deg = torch.atan2(torch.sin(p_raw), torch.cos(p_raw)) * 180.0 / 3.141592653589793

            is_stable_now = (
                (b_vy.abs() <= 0.03) & (c_vy.abs() <= 0.03) & (b_vz.abs() <= 0.05) &
                (fn_L >= 20.0) & (fn_L <= 180.0) & (fn_R >= 20.0) & (fn_R <= 180.0) &
                (pitch_deg.abs() <= 2.5) & (roll_deg.abs() <= 3.5)
            )

            stable_consecutive = torch.where(is_stable_now, stable_consecutive + 1, torch.zeros_like(stable_consecutive))
            newly_stable = (stable_consecutive >= 5) & (step >= 15) & (~stable_reached)
            if newly_stable.any():
                settle_step_reached[newly_stable] = step + 1
                stable_reached |= newly_stable

            # If all environments in env_ids reached stable gate, finish settling
            if stable_reached.all():
                break

        if not stable_reached.all():
            settle_step_reached[~stable_reached] = max_settle_steps
            failed_envs = env_ids[~stable_reached]
            print(f"[SETTLE TIMEOUT] {len(failed_envs)} envs timed out at {max_settle_steps} steps. Re-damping states to nominal rest.")
            self.root_states[failed_envs, 2] = self.env_origins[failed_envs, 2] + 0.397739
            self.root_states[failed_envs, 7:13] = 0.0
            self.dof_pos[failed_envs] = self.default_dof_pos
            self.dof_vel[failed_envs] = 0.0
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)

        # Store settling telemetry for diagnostic audit
        self.last_settle_step_reached = settle_step_reached.clone()
        self.last_stable_reached = stable_reached.clone()
        self.last_handoff_b_vy = b_vy.clone()
        self.last_handoff_c_vy = c_vy.clone()
        self.last_handoff_b_vz = b_vz.clone()
        self.last_handoff_fn_L = fn_L.clone()
        self.last_handoff_fn_R = fn_R.clone()
        self.last_handoff_pitch = pitch_deg.clone()
        self.last_handoff_roll = roll_deg.clone()

        # 3. Clean Jump Phase Handoff at Step 0
        self.jump_timer[env_ids] = 0.0
        self.jump_phase[env_ids] = 0.0
        self.jump_latched[env_ids] = False
        self.has_taken_off[env_ids] = False
        self.has_touched_down[env_ids] = False
        self.one_shot_touchdown_event[env_ids] = False
        self.filtered_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.time_out_buf[env_ids] = False

        # Fill extras for logging
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.

        self.takeoff_vy[env_ids] = 0.0
        self.takeoff_vz[env_ids] = 0.0
        self.takeoff_pos_y[env_ids] = 0.0
        self.touchdown_pos_y[env_ids] = 0.0
        self.flight_distance_y[env_ids] = 0.0
        self.final_flight_distance[env_ids] = 0.0
        self.max_flight_clearance[env_ids] = 0.0
        self.is_airborne[env_ids] = False
        self.air_time[env_ids] = 0.0

        # Resample commands along local Y
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

        # Compute initial observation for Step 0 of Jump
        self._refresh_physical_state(update_has_taken_off=False)
        self.compute_observations()

    def post_physics_step(self):
        """
        Synchronous post-physics processing with verified Target B terminal observation capture.
        Captures pre-reset terminal observation s_{140}^{pre} BEFORE reset_idx executes.
        Transmits extras['time_outs'] and extras['terminal_observation'] to RSL-RL.
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Prepare base quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # Check terminations and compute rewards based on genuine pre-reset physics
        self.check_termination()
        self.compute_reward()

        # Cleanly pop any previous step's timeout and terminal observation flags
        self.extras.pop("time_outs", None)
        self.extras.pop("terminal_observation", None)

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            # 1. Compute observation on the TRUE pre-reset terminal state s_{t+1}
            self.compute_observations()
            self.terminal_obs_buf[env_ids] = self.obs_buf[env_ids].clone()

            # 2. Package timeout flag and terminal observation into extras
            self.extras["time_outs"] = self.time_out_buf.clone()
            self.extras["terminal_observation"] = self.terminal_obs_buf.clone()

            # 3. Perform synchronous reset & balance settling
            self.reset_idx(env_ids)

        # Compute observations for the next step (s_0' for reset envs, s_{t+1} for running envs)
        self.compute_observations()

    def check_termination(self):
        """
        Synchronous episodic termination:
        All environments reset strictly together when episode_length_buf >= max_episode_length.
        This preserves complete GPU batch synchronization and avoids mid-episode simulation splitting.
        """
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf = self.time_out_buf.clone()

    def step(self, actions):
        """
        Jump Step:
        Applies aerial descent hip compensation (+0.10 rad) matching Phase 3.2-V deployment pipeline.
        """
        # Aerial descent compensation (+0.10 rad to hips)
        executed_actions = actions.clone()
        comp_mask = self.is_airborne & (self.jump_phase >= 0.22)
        if comp_mask.any():
            executed_actions[comp_mask, self.left_hip_idx] += 0.10
            executed_actions[comp_mask, self.right_hip_idx] += 0.10

        # Action smoothing
        clip_actions = self.cfg.normalization.clip_actions
        clipped = torch.clip(executed_actions, -clip_actions, clip_actions).to(self.device)
        self.last_actions[:] = self.filtered_actions
        self.filtered_actions[:] = 0.80 * clipped + 0.20 * self.last_actions

        # Advance jump timer
        self.jump_timer += self.dt

        # Single-jump latching
        landing_completed = self.has_taken_off & self.has_touched_down & (self.jump_timer >= 0.50 * self.jump_cycle_time)
        self.jump_latched |= landing_completed

        clamped_timer = torch.where(
            self.jump_latched,
            torch.clamp(self.jump_timer, max=0.85 * self.jump_cycle_time),
            self.jump_timer
        )
        self.jump_phase[:] = clamped_timer / self.jump_cycle_time

        # Step physics via base LeggedRobot class directly (skipping MyRobotJump duplicate timer/filtering)
        return super(MyRobotJump, self).step(executed_actions)

    # =========================================================================
    # Phase 4.4 Monotonic Forward Distance Reward Candidate 1 (Sigmoid)
    # Score(d) = Sigmoid((d - 0.110) / 0.025)
    # Strictly non-decreasing, non-saturating around baseline, no overshoot penalty
    # =========================================================================

    def _reward_airborne_forward_displacement(self):
        """
        Rewards forward displacement accumulated between takeoff and touchdown.
        Monotonic Sigmoid formulation centered at 11.0 cm (scale 2.5 cm).
        STRICT REQUIREMENT: Clearance gate >= 18.5 cm (reaches full gate at 21.0 cm).
        """
        valid_flight = self.is_airborne & self.has_taken_off & (~self.has_touched_down) & (self.jump_phase <= 0.40)
        clr_gate = torch.clamp((self.max_flight_clearance - 0.185) / 0.025, min=0.0, max=1.0)
        d_air = torch.clamp(self.flight_distance_y, min=0.0)
        score = torch.sigmoid((d_air - 0.110) / 0.025)
        return score * clr_gate * valid_flight.float()

    def _reward_touchdown_forward_distance(self):
        """
        ONE-SHOT REWARD: Awarded strictly on the exact transition step of touchdown!
        Monotonic Sigmoid formulation centered at 11.0 cm (scale 2.5 cm).
        Gated by achieving minimum 20.0 cm clearance during flight.
        """
        clr_ok = (self.max_flight_clearance >= 0.200).float()
        dist_score = torch.sigmoid((self.final_flight_distance - 0.110) / 0.025)
        return dist_score * clr_ok * self.one_shot_touchdown_event.float()
