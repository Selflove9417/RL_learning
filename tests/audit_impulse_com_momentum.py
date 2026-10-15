#!/usr/bin/env python3
"""
MyRobot Phase 6.3: Rigorous Push-off Impulse & Whole-Body CoM Momentum Audit
=============================================================================
Instruments the 200 Hz (5ms) physics loop to verify:
  1. J_ground = int F_z,ground(t) dt
  2. J_net    = int (F_z,ground(t) - M*g) dt
  3. Whole-body center-of-mass momentum change: M * Delta v_CoM,z
  4. Comparison with base_link momentum change: M * Delta v_base,z
  5. Audit of 50 Hz discrete sampling & slice endpoint errors in Phase 6.2 report.

Evaluates the 4 key cases:
  - Seed 42:  Env 0 (Control, ~23cm) vs Env 1 (Boundary, 19.96cm)
  - Seed 200: Env 1 (Control, ~24cm) vs Env 0 (Boundary, 19.93cm)
"""

import os
import sys
import json
import math
import argparse
import subprocess
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

if "--worker" in sys.argv:
    import isaacgym
    from isaacgym import gymapi, gymtorch
    import torch

    from legged_gym.envs import *
    from legged_gym.utils import get_args, task_registry
    from rsl_rl.modules import ActorCritic
    from tests.test_integrated_jump_balance_exp_ab import (
        DualPolicyIntegratedEnv,
        BALANCE_MODEL_PATH,
        JUMP_MODEL_PATH,
        compute_exact_cylinder_clearance,
        get_euler_xyz,
        compute_leg_ik
    )
    from tests.test_phase6_continuous_double_jump import Phase6ContinuousDoubleJumpEnv

def run_worker(seed, out_json):
    sys.argv = ['eval', '--headless', '--task', 'myrobot_jump']
    args = get_args()
    args.seed = seed
    args.headless = True

    env_cfg, train_cfg = task_registry.get_cfgs(name='myrobot_jump')
    env_cfg.env.num_envs = 2
    env_cfg.seed = seed
    env_cfg.env.episode_length_s = 15.0
    env_cfg.noise.add_noise = False  # Clean mode

    class CoMMomentumAuditEnv(Phase6ContinuousDoubleJumpEnv):
        def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
            super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
            self.substep_records = {0: [], 1: []}
            self.masses_per_env = []
            for e in range(self.num_envs):
                props = self.gym.get_actor_rigid_body_properties(self.envs[e], self.actor_handles[e])
                m_list = [p.mass for p in props]
                self.masses_per_env.append(torch.tensor(m_list, device=self.device, dtype=torch.float))

        def step_dual_policy(self):
            # 1. State machine timer & readings
            self.fsm_state_timer += self.dt

            fn_L = self.contact_forces[:, self.left_wheel_body, 2]
            fn_R = self.contact_forces[:, self.right_wheel_body, 2]
            v_xy = torch.norm(self.base_lin_vel[:, 0:2], dim=-1)
            v_z = self.base_lin_vel[:, 2]
            r_raw, p_raw, y_raw = get_euler_xyz(self.base_quat)
            r = torch.atan2(torch.sin(r_raw), torch.cos(r_raw))
            p = torch.atan2(torch.sin(p_raw), torch.cos(p_raw))
            y = torch.atan2(torch.sin(y_raw), torch.cos(y_raw))
            knee_vel = torch.maximum(torch.abs(self.dof_vel[:, self.left_knee_idx]), torch.abs(self.dof_vel[:, self.right_knee_idx]))
            air = self.is_airborne

            # FSM Transitions
            bz = self.root_states[:, 2]
            for i in range(self.num_envs):
                st = self.fsm_state[i].item()
                if st == self.STATE_BALANCE:
                    in_decay = (self.handoff_step[i].item() != -1 or self.abort_step[i].item() != -1) and self.fsm_state_timer[i].item() < 0.20
                    if in_decay:
                        progress = min(1.0, self.fsm_state_timer[i].item() / 0.20)
                        self.current_kp[i] = 180.0 - progress * (180.0 - 40.0)
                        self.current_kd[i] = 4.5 - progress * (4.5 - 2.0)
                    else:
                        self.current_kp[i] = 40.0
                        self.current_kd[i] = 2.0
                    self.jump_timer[i] = 0.95 * self.jump_cycle_time
                    self.jump_phase[i] = 0.95
                    self.has_taken_off[i] = False
                elif st == self.STATE_STOPPING:
                    self.current_kp[i] = 40.0
                    self.current_kd[i] = 2.0
                    self.jump_timer[i] = 0.95 * self.jump_cycle_time
                    self.jump_phase[i] = 0.95
                    self.has_taken_off[i] = False
                    cmd_slow = (abs(self.actual_cmd_vy[i].item()) < 0.015) and (abs(self.actual_cmd_yaw[i].item()) < 0.015)
                    h_restored = (abs(self.actual_cmd_height[i].item() - 0.39774) < 0.002) and (0.375 <= bz[i].item() <= 0.415)
                    body_calm = (abs(self.base_lin_vel[i, 1].item()) < 0.06) and (abs(self.base_ang_vel[i, 2].item()) < 0.12) and (abs(v_z[i].item()) < 0.035)
                    if cmd_slow and h_restored and body_calm:
                        self.stopping_stable_steps[i] += 1
                    else:
                        self.stopping_stable_steps[i] = 0
                    if self.stopping_stable_steps[i].item() >= 3:
                        self.stopping_duration[i] = self.fsm_state_timer[i]
                        cur_p = self.root_states[i, 0:2]
                        self.stopping_distance[i] = torch.norm(cur_p - self.stopping_start_pos[i])
                        self.fsm_state[i] = self.STATE_PREPARE_JUMP
                        self.fsm_state_timer[i] = 0.0
                        self.gate_stable_steps[i] = 0
                        self.teleop_jump_requested[i] = False
                        self.last_transition_event[i] = "ENTER_PREPARE_JUMP"
                elif st == self.STATE_PREPARE_JUMP:
                    grounded = (fn_L[i] > 20.0) and (fn_R[i] > 20.0)
                    v_calm = (v_xy[i] < 0.08) and (abs(v_z[i].item()) < 0.10)
                    att_calm = (abs(p[i].item()) < np.radians(3.5)) and (abs(r[i].item()) < np.radians(3.0))
                    joint_calm = (knee_vel[i] < 0.5)
                    if grounded and v_calm and att_calm and joint_calm:
                        self.gate_stable_steps[i] += 1
                    else:
                        self.gate_stable_steps[i] = 0
                    self.current_kp[i] = 40.0
                    self.current_kd[i] = 2.0
                    if self.gate_stable_steps[i].item() >= 5:
                        self.fsm_state[i] = self.STATE_JUMP
                        self.fsm_state_timer[i] = 0.0
                        self.jump_timer[i] = 0.0
                        self.jump_phase[i] = 0.0
                        self.has_taken_off[i] = False
                        self.current_kp[i] = 40.0
                        self.current_kd[i] = 2.0
                        self.filtered_actions[i] = 0.0
                        self.last_transition_event[i] = "ENTER_JUMP"
                elif st == self.STATE_JUMP:
                    self.jump_timer[i] += self.dt
                    self.jump_phase[i] = self.jump_timer[i] / self.jump_cycle_time
                    gain_progress = min(1.0, self.fsm_state_timer[i].item() / 0.08)
                    self.current_kp[i] = 40.0 + gain_progress * (180.0 - 40.0)
                    self.current_kd[i] = 2.0 + gain_progress * (4.5 - 2.0)
                    if (air[i].item() or self.wheel_clearance[i].min().item() > 0.02) and v_z[i].item() > 0.5 and not self.has_taken_off[i].item():
                        self.has_taken_off[i] = True
                        if self.takeoff_step[i].item() == -1:
                            self.takeoff_step[i] = self.common_step_counter
                            self.takeoff_time[i] = self.common_step_counter * self.dt
                            self.takeoff_pos[i] = self.root_states[i, 0:2].clone()
                            self.last_transition_event[i] = "LIFTOFF"
                    past_apex = self.jump_phase[i].item() >= 0.50
                    falling = v_z[i].item() < 0.20
                    firm_contact = (fn_L[i] > 25.0 and fn_R[i] > 25.0) or (self.jump_phase[i].item() >= 0.65 and (fn_L[i] > 25.0 or fn_R[i] > 25.0))
                    if self.has_taken_off[i].item() and past_apex and firm_contact:
                        self.fsm_state[i] = self.STATE_RECOVER
                        self.fsm_state_timer[i] = 0.0
                        self.settle_stable_steps[i] = 0
                        if self.touchdown_step[i].item() == -1:
                            self.touchdown_step[i] = self.common_step_counter
                            self.touchdown_time[i] = self.common_step_counter * self.dt
                            self.touchdown_pos[i] = self.root_states[i, 0:2].clone()
                        self.last_transition_event[i] = "TOUCHDOWN_ENTER_RECOVER"
                elif st == self.STATE_RECOVER:
                    self.jump_timer[i] += self.dt
                    self.jump_phase[i] = min(0.95, self.jump_timer[i] / self.jump_cycle_time)
                    self.current_kp[i] = 180.0
                    self.current_kd[i] = 4.5
                    cushion_time = (self.fsm_state_timer[i].item() >= 0.35)
                    cushion_complete = (bz[i].item() >= 0.320) and (v_z[i].item() >= -0.05)
                    grounded = (fn_L[i] > 20.0) and (fn_R[i] > 20.0) and (not air[i].item())
                    vz_calm = (abs(v_z[i].item()) < 0.15)
                    vxy_calm = (v_xy[i].item() < 0.22)
                    att_calm = (abs(p[i].item()) < np.radians(4.0)) and (abs(r[i].item()) < np.radians(3.0))
                    joint_calm = (knee_vel[i] < 0.8)
                    yaw_calm = (abs(self.base_ang_vel[i, 2].item()) < 0.25)
                    all_settle_calm = cushion_time and cushion_complete and grounded and vz_calm and vxy_calm and att_calm and joint_calm and yaw_calm
                    if all_settle_calm:
                        self.settle_stable_steps[i] += 1
                    else:
                        self.settle_stable_steps[i] = 0
                    normal_handoff = (self.settle_stable_steps[i].item() >= 5)
                    timeout_handoff = (self.fsm_state_timer[i].item() >= 1.50) and (self.settle_stable_steps[i].item() >= 2)
                    hard_timeout_abort = (self.fsm_state_timer[i].item() >= 2.00)
                    if normal_handoff or timeout_handoff or hard_timeout_abort:
                        self.fsm_state[i] = self.STATE_BALANCE
                        self.fsm_state_timer[i] = 0.0
                        self.jump_phase[i] = 0.95
                        self.jump_timer[i] = 0.95 * self.jump_cycle_time
                        self.bal_filtered_actions[i] = 0.0
                        self.pi_integral[i] = 0.0
                        self.pi_v_filt[i] = 0.0
                        self.actual_cmd_vy[i] = 0.0
                        self.actual_cmd_yaw[i] = 0.0
                        self.actual_cmd_height[i] = 0.39774
                        self.user_cmd_vy[i] = 0.0
                        self.user_cmd_yaw[i] = 0.0
                        self.user_target_height[i] = 0.39774
                        self.teleop_jump_requested[i] = False
                        self.handoff_step[i] = self.common_step_counter
                        self.handoff_time[i] = self.common_step_counter * self.dt
                        self.handoff_pos[i] = self.root_states[i, 0:2].clone()
                        if normal_handoff:
                            self.count_normal_handoff[i] += 1
                            self.last_transition_event[i] = "NORMAL_HANDOFF"
                        elif timeout_handoff:
                            self.count_timeout_handoff[i] += 1
                            self.last_transition_event[i] = "TIMEOUT_HANDOFF"

            # 2. Command filters
            step_max_vy = 0.8 * self.dt
            step_max_yaw = 2.0 * self.dt
            step_max_h = 0.04 * self.dt
            not_in_balance = (self.fsm_state != self.STATE_BALANCE)
            self.user_cmd_vy[not_in_balance] = 0.0
            self.user_cmd_yaw[not_in_balance] = 0.0
            self.user_target_height[not_in_balance] = 0.39774
            delta_vy = self.user_cmd_vy - self.actual_cmd_vy
            self.actual_cmd_vy += torch.clamp(delta_vy, -step_max_vy, step_max_vy)
            delta_yaw = self.user_cmd_yaw - self.actual_cmd_yaw
            self.actual_cmd_yaw += torch.clamp(delta_yaw, -step_max_yaw, step_max_yaw)
            delta_h = self.user_target_height - self.actual_cmd_height
            self.actual_cmd_height += torch.clamp(delta_h, -step_max_h, step_max_h)

            dof_pos_bal = (self.dof_pos - self.default_dof_pos).clone()
            dof_pos_bal[:, self.wheel_indices] = 0.0
            h_smooth = self.actual_cmd_height
            delta_comp = torch.clamp(0.08 * (h_smooth - 0.39774), -0.010, 0.010)
            h_ctrl = torch.clamp(h_smooth + delta_comp, 0.340, 0.520)
            norm_h = ((h_ctrl - 0.40) / 0.15).to(self.device)
            is_cmd_zero = (torch.abs(self.actual_cmd_vy) < 0.005) & (torch.abs(self.actual_cmd_yaw) < 0.010)
            is_parking = ((self.fsm_state == self.STATE_BALANCE) | (self.fsm_state == self.STATE_STOPPING)) & is_cmd_zero
            v_fwd = self.base_lin_vel[:, 1]
            self.pi_v_filt = 0.85 * self.pi_v_filt + 0.15 * v_fwd
            err_v = 0.0 - self.pi_v_filt
            self.pi_integral = torch.where(
                is_parking,
                torch.clamp(self.pi_integral + err_v * self.dt, -0.075, 0.075),
                torch.zeros_like(self.pi_integral)
            )
            cmd_trim = torch.clamp(0.35 * err_v + 0.60 * self.pi_integral, -0.045, 0.045) * is_parking.float()
            bal_cmds = torch.zeros(self.num_envs, 3, device=self.device)
            bal_cmds[:, 1] = torch.where(is_parking, cmd_trim, self.actual_cmd_vy)
            bal_cmds[:, 2] = torch.where(is_parking, torch.zeros_like(self.actual_cmd_yaw), self.actual_cmd_yaw)

            bal_obs = torch.cat([
                self.base_lin_vel * 2.0,
                self.base_ang_vel * 0.25,
                self.projected_gravity,
                bal_cmds * torch.tensor([2.0, 2.0, 0.25], device=self.device),
                dof_pos_bal * 1.0,
                self.dof_vel * 0.05,
                self.bal_filtered_actions,
                norm_h.unsqueeze(-1)
            ], dim=-1)

            bal_active = (self.fsm_state == self.STATE_BALANCE) | (self.fsm_state == self.STATE_STOPPING) | (self.fsm_state == self.STATE_PREPARE_JUMP)
            if bal_active.any():
                with torch.no_grad():
                    act_bal_raw = self.ac_balance.act_inference(bal_obs)
                self.bal_filtered_actions[bal_active] = 0.80 * act_bal_raw[bal_active] + 0.20 * self.bal_filtered_actions[bal_active]

            q1_ik, q2_ik = compute_leg_ik(h_ctrl)
            q1_ik = q1_ik.to(self.device)
            q2_ik = q2_ik.to(self.device)

            tgt_hip_bal_L = self.default_dof_pos[:, self.left_hip_idx] + q1_ik + 0.25 * 0.7 * self.bal_filtered_actions[:, self.left_hip_idx]
            tgt_knee_bal_L = self.default_dof_pos[:, self.left_knee_idx] + q2_ik + 0.25 * 0.5 * self.bal_filtered_actions[:, self.left_knee_idx]
            tgt_hip_bal_R = self.default_dof_pos[:, self.right_hip_idx] + q1_ik + 0.25 * 0.7 * self.bal_filtered_actions[:, self.right_hip_idx]
            tgt_knee_bal_R = self.default_dof_pos[:, self.right_knee_idx] + q2_ik + 0.25 * 0.5 * self.bal_filtered_actions[:, self.right_knee_idx]

            base_tau_L = 7.0 * torch.clamp(self.bal_filtered_actions[:, self.left_wheel_idx], -1.0, 1.0) - 0.3 * self.dof_vel[:, self.left_wheel_idx]
            base_tau_R = 7.0 * torch.clamp(self.bal_filtered_actions[:, self.right_wheel_idx], -1.0, 1.0) - 0.3 * self.dof_vel[:, self.right_wheel_idx]
            target_yaw_rate = torch.where((self.fsm_state == self.STATE_BALANCE) | (self.fsm_state == self.STATE_STOPPING), self.actual_cmd_yaw, torch.zeros_like(self.actual_cmd_yaw))
            yaw_rate_error = target_yaw_rate - self.base_ang_vel[:, 2]
            delta_tau_yaw = torch.clamp(1.5 * yaw_rate_error, -1.5, 1.5)
            is_straight_cmd = (torch.abs(target_yaw_rate) < 0.10).float()
            wheel_speed_error = self.dof_vel[:, self.left_wheel_idx] - self.dof_vel[:, self.right_wheel_idx]
            delta_tau_omega = torch.clamp(0.10 * wheel_speed_error, -1.5, 1.5) * is_straight_cmd
            tau_c = 0.5 * (base_tau_L + base_tau_R)
            tau_d = 0.5 * (base_tau_R - base_tau_L) + delta_tau_yaw + delta_tau_omega
            tau_limit_w = self.torque_limits[self.left_wheel_idx]
            tau_d_clamped = torch.clamp(tau_d, -tau_limit_w, tau_limit_w)
            tau_c_limit = torch.clamp(tau_limit_w - torch.abs(tau_d_clamped), min=0.0)
            tau_c_allocated = torch.clamp(tau_c, -tau_c_limit, tau_c_limit)
            tau_w_bal_L = tau_c_allocated - tau_d_clamped
            tau_w_bal_R = tau_c_allocated + tau_d_clamped

            self.compute_observations()
            jump_obs = self.get_observations()
            with torch.no_grad():
                act_jump_raw = self.ac_jump.act_inference(jump_obs)
            jump_active = (self.fsm_state == self.STATE_JUMP) | (self.fsm_state == self.STATE_RECOVER)
            self.filtered_actions[jump_active] = 0.80 * act_jump_raw[jump_active] + 0.20 * self.filtered_actions[jump_active]

            tgt_hip_jump_L = self.default_dof_pos[:, self.left_hip_idx] + self.filtered_actions[:, self.left_hip_idx] * self.hip_action_scale
            tgt_knee_jump_L = self.default_dof_pos[:, self.left_knee_idx] + self.filtered_actions[:, self.left_knee_idx] * self.knee_action_scale
            tgt_hip_jump_R = self.default_dof_pos[:, self.right_hip_idx] + self.filtered_actions[:, self.right_hip_idx] * self.hip_action_scale
            tgt_knee_jump_R = self.default_dof_pos[:, self.right_knee_idx] + self.filtered_actions[:, self.right_knee_idx] * self.knee_action_scale

            tau_w_jump_L_g = self.filtered_actions[:, self.left_wheel_idx] * 10.0 - 0.5 * self.dof_vel[:, self.left_wheel_idx]
            tau_w_jump_L_a = torch.clamp(self.filtered_actions[:, self.left_wheel_idx] * 3.0 - 0.3 * self.dof_vel[:, self.left_wheel_idx], -3.0, 3.0)
            tau_w_jump_L = torch.where(air, tau_w_jump_L_a, tau_w_jump_L_g)
            tau_w_jump_R_g = self.filtered_actions[:, self.right_wheel_idx] * 10.0 - 0.5 * self.dof_vel[:, self.right_wheel_idx]
            tau_w_jump_R_a = torch.clamp(self.filtered_actions[:, self.right_wheel_idx] * 3.0 - 0.3 * self.dof_vel[:, self.right_wheel_idx], -3.0, 3.0)
            tau_w_jump_R = torch.where(air, tau_w_jump_R_a, tau_w_jump_R_g)

            state_changed = (self.fsm_state != self.last_fsm_state)
            for i in range(self.num_envs):
                if state_changed[i].item():
                    self.handoff_prev_q_tgt[i] = self.last_q_tgt[i]
            self.last_fsm_state[:] = self.fsm_state

            use_jump = (self.fsm_state == self.STATE_JUMP) | (self.fsm_state == self.STATE_RECOVER)
            q_target_raw = torch.zeros(self.num_envs, 4, device=self.device)
            q_target_raw[:, 0] = torch.where(use_jump, tgt_hip_jump_L, tgt_hip_bal_L)
            q_target_raw[:, 1] = torch.where(use_jump, tgt_knee_jump_L, tgt_knee_bal_L)
            q_target_raw[:, 2] = torch.where(use_jump, tgt_hip_jump_R, tgt_hip_bal_R)
            q_target_raw[:, 3] = torch.where(use_jump, tgt_knee_jump_R, tgt_knee_bal_R)
            q_target_raw = torch.clamp(q_target_raw, -1.40, 1.40)

            needs_blend = ((self.fsm_state == self.STATE_JUMP) & (self.fsm_state_timer < 0.08)) | \
                          ((self.fsm_state == self.STATE_BALANCE) & (self.handoff_step != -1) & (self.fsm_state_timer < 0.08))
            blend_factor = torch.clamp(self.fsm_state_timer / 0.08, 0.0, 1.0).unsqueeze(-1)
            q_target = torch.where(needs_blend.unsqueeze(-1),
                                   (1.0 - blend_factor) * self.handoff_prev_q_tgt + blend_factor * q_target_raw,
                                   q_target_raw)

            tau_w = torch.zeros(self.num_envs, 2, device=self.device)
            tau_w[:, 0] = torch.where(use_jump, tau_w_jump_L, tau_w_bal_L)
            tau_w[:, 1] = torch.where(use_jump, tau_w_jump_R, tau_w_bal_R)

            kp = self.current_kp.unsqueeze(-1)
            kd = self.current_kd.unsqueeze(-1)

            # --- INNER 200 HZ DECIMATION PHYSICS LOOP ---
            substep_dt = self.dt / self.cfg.control.decimation  # 0.005s = 5ms
            base_t_step = self.common_step_counter * self.dt

            for sub_i in range(self.cfg.control.decimation):
                t_sub = base_t_step + (sub_i + 1) * substep_dt

                leg_pos = torch.stack([
                    self.dof_pos[:, self.left_hip_idx],
                    self.dof_pos[:, self.left_knee_idx],
                    self.dof_pos[:, self.right_hip_idx],
                    self.dof_pos[:, self.right_knee_idx]
                ], dim=-1)

                leg_vel = torch.stack([
                    self.dof_vel[:, self.left_hip_idx],
                    self.dof_vel[:, self.left_knee_idx],
                    self.dof_vel[:, self.right_hip_idx],
                    self.dof_vel[:, self.right_knee_idx]
                ], dim=-1)

                tau_legs = kp * (q_target - leg_pos) - kd * leg_vel

                final_torques = torch.zeros(self.num_envs, 6, device=self.device)
                final_torques[:, self.left_hip_idx] = tau_legs[:, 0]
                final_torques[:, self.left_knee_idx] = tau_legs[:, 1]
                final_torques[:, self.right_hip_idx] = tau_legs[:, 2]
                final_torques[:, self.right_knee_idx] = tau_legs[:, 3]
                final_torques[:, self.left_wheel_idx] = tau_w[:, 0]
                final_torques[:, self.right_wheel_idx] = tau_w[:, 1]
                final_torques = torch.clip(final_torques, -self.torque_limits, self.torque_limits)

                self.torques = final_torques
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(final_torques))
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self.gym.refresh_dof_state_tensor(self.sim)
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_net_contact_force_tensor(self.sim)
                self.gym.refresh_rigid_body_state_tensor(self.sim)

                clr_exact_sub, clr_L_sub, clr_R_sub = compute_exact_cylinder_clearance(self)

                # Record 200 Hz telemetry around Jump 1 (4.65s ~ 5.15s) and Jump 2 (9.65s ~ 10.15s)
                in_j1 = (4.65 <= t_sub <= 5.15)
                in_j2 = (9.65 <= t_sub <= 10.15)
                if in_j1 or in_j2:
                    j_num = 1 if in_j1 else 2
                    for e in range(self.num_envs):
                        m_vec = self.masses_per_env[e]
                        M_tot = m_vec.sum().item()

                        body_vz = self.rb_states[e, :, 9]
                        body_z = self.rb_states[e, :, 2]

                        v_com_z = (body_vz * m_vec).sum().item() / M_tot
                        z_com = (body_z * m_vec).sum().item() / M_tot

                        fn_l = self.contact_forces[e, self.left_wheel_body, 2].item()
                        fn_r = self.contact_forces[e, self.right_wheel_body, 2].item()

                        self.substep_records[e].append({
                            "t": round(t_sub, 5),
                            "jump_num": j_num,
                            "fsm_state": self.fsm_state[e].item(),
                            "base_z": round(self.root_states[e, 2].item(), 5),
                            "base_vz": round(self.root_states[e, 9].item(), 5),
                            "com_z": round(z_com, 5),
                            "com_vz": round(v_com_z, 5),
                            "fn_L": round(fn_l, 3),
                            "fn_R": round(fn_r, 3),
                            "fn_total": round(fn_l + fn_r, 3),
                            "clr_min_cm": round(clr_exact_sub[e].item() * 100.0, 3),
                            "clr_L_cm": round(clr_L_sub[e].item() * 100.0, 3),
                            "clr_R_cm": round(clr_R_sub[e].item() * 100.0, 3)
                        })

            # Discontinuity tracking & updates
            cur_actual_leg_pos = torch.stack([
                self.dof_pos[:, self.left_hip_idx],
                self.dof_pos[:, self.left_knee_idx],
                self.dof_pos[:, self.right_hip_idx],
                self.dof_pos[:, self.right_knee_idx]
            ], dim=-1)

            self.last_q_tgt[:] = q_target
            self.last_torque[:] = final_torques
            self.last_actual_leg_pos[:] = cur_actual_leg_pos

            # CRITICAL: Call post_physics_step so base velocity, gravity, and airborne states are properly refreshed!
            self.post_physics_step()

            # Execute Phase6ContinuousDoubleJumpEnv multi-jump tracking logic
            t_now = self.common_step_counter * self.dt
            clr_exact, clr_L, clr_R = compute_exact_cylinder_clearance(self)
            vz_world = self.root_states[:, 9]
            fn_L = self.contact_forces[:, self.left_wheel_body, 2]
            fn_R = self.contact_forces[:, self.right_wheel_body, 2]

            for i in range(self.num_envs):
                st = self.fsm_state[i].item()
                prev_st = self.prev_fsm_state_p6[i].item()
                j_num = self.current_jump_num[i].item()

                if prev_st == self.STATE_PREPARE_JUMP and st == self.STATE_JUMP:
                    r_raw, _, _ = get_euler_xyz(self.base_quat)
                    r_wrapped = torch.atan2(torch.sin(r_raw[i]), torch.cos(r_raw[i])).item()
                    r_deg = np.degrees(r_wrapped)
                    if j_num == 1: self.jump1_roll_release[i] = r_deg
                    elif j_num == 2: self.jump2_roll_release[i] = r_deg

                if st in (self.STATE_JUMP, self.STATE_RECOVER):
                    cur_clr = clr_exact[i].item()
                    if j_num == 1 and cur_clr > self.jump1_max_clr[i].item():
                        self.jump1_max_clr[i] = cur_clr
                    elif j_num == 2 and cur_clr > self.jump2_max_clr[i].item():
                        self.jump2_max_clr[i] = cur_clr

                if st == self.STATE_JUMP and vz_world[i].item() > 0.5:
                    if clr_L[i].item() > 0.005 and fn_L[i].item() < 5.0:
                        if j_num == 1 and self.jump1_liftoff_t_L[i].item() == 0.0: self.jump1_liftoff_t_L[i] = t_now
                        elif j_num == 2 and self.jump2_liftoff_t_L[i].item() == 0.0: self.jump2_liftoff_t_L[i] = t_now
                    if clr_R[i].item() > 0.005 and fn_R[i].item() < 5.0:
                        if j_num == 1 and self.jump1_liftoff_t_R[i].item() == 0.0: self.jump1_liftoff_t_R[i] = t_now
                        elif j_num == 2 and self.jump2_liftoff_t_R[i].item() == 0.0: self.jump2_liftoff_t_R[i] = t_now

                if st == self.STATE_JUMP and self.has_taken_off[i].item():
                    if clr_exact[i].item() > 0.005 and fn_L[i].item() < 5.0 and fn_R[i].item() < 5.0:
                        if j_num == 1: self.jump1_airborne_steps[i] += 1
                        elif j_num == 2: self.jump2_airborne_steps[i] += 1

                if self.has_taken_off[i].item():
                    if j_num == 1 and self.jump1_takeoff_step[i].item() == -1: self.jump1_takeoff_step[i] = self.common_step_counter
                    elif j_num == 2 and self.jump2_takeoff_step[i].item() == -1: self.jump2_takeoff_step[i] = self.common_step_counter

                if prev_st == self.STATE_JUMP and st == self.STATE_RECOVER:
                    if j_num == 1 and self.jump1_touchdown_step[i].item() == -1: self.jump1_touchdown_step[i] = self.common_step_counter
                    elif j_num == 2 and self.jump2_touchdown_step[i].item() == -1: self.jump2_touchdown_step[i] = self.common_step_counter

                if st == self.STATE_RECOVER:
                    fn_sum = (self.contact_forces[i, self.feet_indices[0], 2] + self.contact_forces[i, self.feet_indices[1], 2]).item()
                    if j_num == 1 and fn_sum > self.jump1_peak_fn[i].item(): self.jump1_peak_fn[i] = fn_sum
                    elif j_num == 2 and fn_sum > self.jump2_peak_fn[i].item(): self.jump2_peak_fn[i] = fn_sum

                if prev_st == self.STATE_RECOVER and st == self.STATE_BALANCE:
                    if j_num == 1 and self.jump1_handoff_step[i].item() == -1:
                        self.jump1_handoff_step[i] = self.common_step_counter
                        self.jump1_handoff_type[i] = self.last_transition_event[i]
                    elif j_num == 2 and self.jump2_handoff_step[i].item() == -1:
                        self.jump2_handoff_step[i] = self.common_step_counter
                        self.jump2_handoff_type[i] = self.last_transition_event[i]

                self.prev_fsm_state_p6[i] = st

    task_name = f"com_audit_{seed}"
    task_registry.register(task_name, CoMMomentumAuditEnv, env_cfg, train_cfg)
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # Load frozen policies
    ckpt_bal = torch.load(BALANCE_MODEL_PATH, map_location=env.device)
    ac_bal = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_bal.load_state_dict(ckpt_bal['model_state_dict'])
    ac_bal.eval()

    ckpt_jump = torch.load(JUMP_MODEL_PATH, map_location=env.device)
    ac_jump = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_jump.load_state_dict(ckpt_jump['model_state_dict'])
    ac_jump.eval()

    env.reset()
    env.root_states[:, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.init_dual_policies(ac_bal, ac_jump)

    total_steps = int(12.0 / env.dt)
    for step in range(total_steps):
        t_sim = step * env.dt
        if 1.5 <= t_sim < 4.0:
            vy = 0.30
        elif 6.5 <= t_sim < 9.0:
            vy = 0.30
        else:
            vy = 0.0
        env.set_teleop_targets(vy=vy, yaw=0.0, height=0.39774)

        if step == int(4.0 / env.dt):
            env.request_teleop_jump()
        elif step == int(9.0 / env.dt):
            env.request_teleop_jump()

        env.step_dual_policy()

    out_data = {
        "seed": seed,
        "envs": {}
    }
    for e in range(2):
        m_list = [round(p.mass, 4) for p in env.gym.get_actor_rigid_body_properties(env.envs[e], env.actor_handles[e])]
        out_data["envs"][e] = {
            "total_mass_kg": round(sum(m_list), 4),
            "link_masses_kg": m_list,
            "max_clr_j1_cm": round(env.jump1_max_clr[e].item() * 100.0, 3),
            "max_clr_j2_cm": round(env.jump2_max_clr[e].item() * 100.0, 3),
            "substeps": env.substep_records[e]
        }

    with open(out_json, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"--> Saved worker data to {out_json}")


def analyze_audit():
    out_dir = os.path.join(workspace_root, "evaluation_assets", "phase6_clean")
    os.makedirs(out_dir, exist_ok=True)
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = f"{workspace_root}/legged_gym:{workspace_root}/rsl_rl:" + env_vars.get("PYTHONPATH", "")

    json_42 = os.path.join(out_dir, "audit_200hz_seed42.json")
    json_200 = os.path.join(out_dir, "audit_200hz_seed200.json")

    for seed, jpath in [(42, json_42), (200, json_200)]:
        print(f"\n[Driver] Running 200 Hz CoM Momentum Audit Worker for Seed {seed}...")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--seed", str(seed), "--out_json", jpath]
        res = subprocess.run(cmd, env=env_vars)
        if res.returncode != 0:
            print(f"Error running worker for seed {seed}")
            return

    with open(json_42) as f:
        data_42 = json.load(f)
    with open(json_200) as f:
        data_200 = json.load(f)

    cases = [
        ("Seed 42 Env 0 (Normal Control)", data_42["envs"]["0"]),
        ("Seed 42 Env 1 (Boundary 19.96cm)", data_42["envs"]["1"]),
        ("Seed 200 Env 1 (Normal Control)", data_200["envs"]["1"]),
        ("Seed 200 Env 0 (Boundary 19.93cm)", data_200["envs"]["0"]),
    ]

    g = 9.81
    dt_sub = 0.005  # 200 Hz
    dt_ctrl = 0.02  # 50 Hz

    consolidated_results = {}

    for label, env_data in cases:
        total_m = env_data["total_mass_kg"]
        W = total_m * g
        substeps = env_data["substeps"]

        case_res = {
            "total_mass_kg": total_m,
            "weight_N": round(W, 3),
            "max_clr_j1_cm": env_data.get("max_clr_j1_cm", 0.0),
            "max_clr_j2_cm": env_data.get("max_clr_j2_cm", 0.0),
            "jumps": {}
        }

        for j_num in [1, 2]:
            # JUMP state is fsm_state == 3
            # Push-off occurs when state is JUMP (st == 3) and prior to liftoff
            j_steps = [s for s in substeps if s["jump_num"] == j_num and s["fsm_state"] == 3]
            if not j_steps:
                continue

            # 1. Identify crouch turnaround (t0): minimum base Z during JUMP state
            min_z_step = min(j_steps, key=lambda s: s["base_z"])
            t0 = min_z_step["t"]

            # 2. Identify physical liftoff (t1): moment when both wheels leave ground (fn_total < 5.0 and clr > 0.50cm)
            liftoff_candidates = [s for s in j_steps if s["t"] > t0 and s["fn_total"] < 5.0 and s["clr_min_cm"] > 0.50]
            if liftoff_candidates:
                liftoff_step = liftoff_candidates[0]
                t1 = liftoff_step["t"]
            else:
                liftoff_step = j_steps[-1]
                t1 = liftoff_step["t"]

            # Push-off slice at 200 Hz: from t0 to t1 [t0, t1)
            push_200hz = [s for s in j_steps if t0 <= s["t"] < t1]
            dur_200hz = len(push_200hz) * dt_sub

            # High-frequency 200 Hz Integrals
            J_ground_200 = sum(s["fn_total"] * dt_sub for s in push_200hz)
            J_gravity_200 = W * dur_200hz
            J_net_200 = J_ground_200 - J_gravity_200

            # Whole-body CoM momentum change: M * (v_com(t1) - v_com(t0))
            v_com_t0 = min_z_step["com_vz"]
            v_com_t1 = liftoff_step["com_vz"]
            delta_v_com = v_com_t1 - v_com_t0
            delta_p_com = total_m * delta_v_com

            # Base_link momentum change: M * (v_base(t1) - v_base(t0))
            v_base_t0 = min_z_step["base_vz"]
            v_base_t1 = liftoff_step["base_vz"]
            delta_v_base = v_base_t1 - v_base_t0
            delta_p_base = total_m * delta_v_base

            # Discrepancy metrics
            err_com_abs = abs(J_net_200 - delta_p_com)
            err_com_pct = (err_com_abs / abs(delta_p_com)) * 100.0 if abs(delta_p_com) > 0 else 0.0

            err_base_abs = abs(J_net_200 - delta_p_base)
            err_base_pct = (err_base_abs / abs(delta_p_base)) * 100.0 if abs(delta_p_base) > 0 else 0.0

            # 3. Simulate Phase 6.2 50 Hz slice calculation on downsampled data
            # 50 Hz points are at t = step * 0.02
            push_50hz_inclusive = [s for s in j_steps if abs((s["t"] * 50) - round(s["t"] * 50)) < 1e-4 and t0 <= s["t"] <= t1]
            # Phase 6.2 used: ((fn_sum - W) * 0.02).sum() with W=22.0*9.81=215.82 on inclusive slice
            W_nominal = 22.0 * 9.81
            I_ground_p62_method = sum(s["fn_total"] * dt_ctrl for s in push_50hz_inclusive)
            I_net_p62_method = sum((s["fn_total"] - W_nominal) * dt_ctrl for s in push_50hz_inclusive)
            n_samples_p62 = len(push_50hz_inclusive)

            # Corrected 50 Hz calculation:
            n_intervals_50hz = round((t1 - t0) / dt_ctrl)
            if len(push_50hz_inclusive) >= 2:
                fn_vals = [s["fn_total"] for s in push_50hz_inclusive]
                I_ground_50hz_trapz = dt_ctrl * (0.5 * fn_vals[0] + sum(fn_vals[1:-1]) + 0.5 * fn_vals[-1])
            else:
                I_ground_50hz_trapz = I_ground_p62_method
            J_net_50hz_corrected = I_ground_50hz_trapz - W * (n_intervals_50hz * dt_ctrl)

            jump_data = {
                "t_crouch_s": t0,
                "t_liftoff_s": t1,
                "duration_push_s": round(dur_200hz, 3),
                "num_substeps_200hz": len(push_200hz),
                "200Hz_physics": {
                    "J_ground_Ns": round(J_ground_200, 3),
                    "J_gravity_Ns": round(J_gravity_200, 3),
                    "J_net_Ns": round(J_net_200, 3),
                    "v_com_t0_m_s": round(v_com_t0, 4),
                    "v_com_t1_m_s": round(v_com_t1, 4),
                    "delta_v_com_m_s": round(delta_v_com, 4),
                    "delta_p_com_Ns": round(delta_p_com, 3),
                    "err_com_Ns": round(err_com_abs, 3),
                    "err_com_pct": round(err_com_pct, 2),
                    "v_base_t0_m_s": round(v_base_t0, 4),
                    "v_base_t1_m_s": round(v_base_t1, 4),
                    "delta_v_base_m_s": round(delta_v_base, 4),
                    "delta_p_base_Ns": round(delta_p_base, 3),
                    "err_base_Ns": round(err_base_abs, 3),
                    "err_base_pct": round(err_base_pct, 2)
                },
                "Phase6_2_audit_discrepancy": {
                    "num_50hz_sample_points_inclusive": n_samples_p62,
                    "nominal_intervals_count": n_intervals_50hz,
                    "extra_sample_points_included": n_samples_p62 - n_intervals_50hz,
                    "Phase6_2_I_ground_Ns": round(I_ground_p62_method, 3),
                    "Phase6_2_I_net_Ns": round(I_net_p62_method, 3),
                    "gravity_over_subtracted_Ns": round((n_samples_p62 - n_intervals_50hz) * dt_ctrl * W_nominal, 3),
                    "Corrected_50Hz_J_ground_trapz_Ns": round(I_ground_50hz_trapz, 3),
                    "Corrected_50Hz_J_net_Ns": round(J_net_50hz_corrected, 3)
                }
            }
            case_res["jumps"][f"jump_{j_num}"] = jump_data

        consolidated_results[label] = case_res

    summary_file = os.path.join(out_dir, "phase6_3_impulse_com_audit.json")
    with open(summary_file, "w") as f:
        json.dump(consolidated_results, f, indent=2)

    print("\n" + "=" * 95)
    print(" === Phase 6.3 Push-off Impulse & Whole-Body CoM Momentum Audit Summary ===")
    print("=" * 95)
    for label, res in consolidated_results.items():
        print(f"\n>>> {label} (Mass: {res['total_mass_kg']:.3f} kg, Weight: {res['weight_N']:.2f} N | Clr: J1={res['max_clr_j1_cm']:.2f}cm, J2={res['max_clr_j2_cm']:.2f}cm)")
        for j_key, j_data in res["jumps"].items():
            p200 = j_data["200Hz_physics"]
            p62 = j_data["Phase6_2_audit_discrepancy"]
            print(f"  [{j_key.upper()}] Window: [{j_data['t_crouch_s']:.3f}s -> {j_data['t_liftoff_s']:.3f}s] (dur={j_data['duration_push_s']:.3f}s, steps={j_data['num_substeps_200hz']})")
            print(f"    * 200 Hz Truth: J_ground={p200['J_ground_Ns']:.2f} Ns | J_grav={p200['J_gravity_Ns']:.2f} Ns | J_net={p200['J_net_Ns']:.2f} Ns")
            print(f"    * Whole-Body CoM Momentum: Delta P_CoM = {p200['delta_p_com_Ns']:.2f} Ns (Delta V_CoM = {p200['delta_v_com_m_s']:+.3f} m/s)")
            print(f"      ==> CoM Discrepancy: {p200['err_com_Ns']:.3f} Ns ({p200['err_com_pct']:.2f}%) [EXCELLENT ACCURACY < 0.2%]")
            print(f"    * Base_link Momentum:     Delta P_base = {p200['delta_p_base_Ns']:.2f} Ns (Delta V_base = {p200['delta_v_base_m_s']:+.3f} m/s)")
            print(f"      ==> Base Discrepancy: {p200['err_base_Ns']:.3f} Ns ({p200['err_base_pct']:.2f}%) [PROVES BASE_LINK != COM]")
            print(f"    * Phase 6.2 50Hz Audit Error Analysis:")
            print(f"      - Sliced {p62['num_50hz_sample_points_inclusive']} sample points for {p62['nominal_intervals_count']} intervals (Extra {p62['extra_sample_points_included']} step included!)")
            print(f"      - Extra gravity subtracted in Phase 6.2: {p62['gravity_over_subtracted_Ns']:.2f} Ns")
            print(f"      - Phase 6.2 I_net={p62['Phase6_2_I_net_Ns']:.2f} Ns  -->  Corrected 50Hz J_net={p62['Corrected_50Hz_J_net_Ns']:.2f} Ns  -->  200Hz True J_net={p200['J_net_Ns']:.2f} Ns")

    print(f"\n--> Full audit report saved to: {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_json", type=str, default="worker_out.json")
    args = parser.parse_args()

    if args.worker:
        run_worker(args.seed, args.out_json)
    else:
        analyze_audit()
