# ==============================================================================
# MyRobot Dual-Policy Continuous Jumping & Balance Integrated Environment
# ==============================================================================
# Architecture:
#   1. Phase 15 PPO: In-place balancing, cruising, height adjustment [0.34m, 0.52m],
#                    and parking PI zero-drift clamping.
#   2. Stage 8 PPO:  Explosive vertical push-off, aerial tucking, touchdown absorption.
#   3. Phase 5.1/6:  5-stage finite state machine (FSM) managing smooth gain transitions,
#                    continuous multi-jump capability without environment reset.
# ==============================================================================

import os
import math
import isaacgym
from isaacgym import gymtorch
from isaacgym.torch_utils import get_euler_xyz
import torch
import numpy as np

from legged_gym.envs.myrobot.myrobot_jump_env import MyRobotJump
from rsl_rl.modules import ActorCritic

STATE_NAMES = {
    0: "BALANCE",
    1: "STOPPING",
    2: "PREPARE_JUMP",
    3: "JUMP",
    4: "RECOVER"
}

def compute_exact_cylinder_clearance(env, R=0.070, L=0.0885):
    """
    Exact analytical minimum vertical clearance from bilateral 3D cylinder
    wheel collision geometry to ground.
    Returns:
        min_clearance (min(clr_L, clr_R)), clr_L, clr_R
    """
    q_L = env.rb_states[:, env.left_wheel_body, 3:7]
    p_L = env.rb_states[:, env.left_wheel_body, 0:3]
    uz_L = 2.0 * (q_L[:, 0]*q_L[:, 2] - q_L[:, 3]*q_L[:, 1])
    cz_L = p_L[:, 2] + 0.03825 * uz_L
    zmin_L = cz_L - (R * torch.sqrt(torch.clamp(1.0 - uz_L**2, min=0.0)) + 0.5 * L * torch.abs(uz_L))

    q_R = env.rb_states[:, env.right_wheel_body, 3:7]
    p_R = env.rb_states[:, env.right_wheel_body, 0:3]
    uz_R = 2.0 * (q_R[:, 0]*q_R[:, 2] - q_R[:, 3]*q_R[:, 1])
    cz_R = p_R[:, 2] + 0.01925 * uz_R
    zmin_R = cz_R - (R * torch.sqrt(torch.clamp(1.0 - uz_R**2, min=0.0)) + 0.5 * L * torch.abs(uz_R))

    exact_clr_L = torch.clamp(zmin_L, min=0.0)
    exact_clr_R = torch.clamp(zmin_R, min=0.0)
    return torch.minimum(exact_clr_L, exact_clr_R), exact_clr_L, exact_clr_R

def compute_leg_ik(H_target):
    """Exact analytical IK for MyRobot 5-bar linkage leg mechanism."""
    dy1, dz1 = -0.29348091, -0.06220095
    dy2, dz2 = 0.2821087, -0.19553796
    y_hip, z_hip = 0.125, -0.070
    R_wheel = 0.070
    y_wheel_target = 0.11362779

    L1 = 0.30000000
    L2 = 0.34324976
    delta_beta = 2.326640924565626

    dY = y_wheel_target - y_hip
    dZ = (R_wheel - H_target) - z_hip
    D_sq = dY**2 + dZ**2

    cos_phi = torch.clamp((D_sq - L1**2 - L2**2) / (2.0 * L1 * L2), -1.0, 1.0)
    phi = torch.acos(cos_phi)

    pi = math.pi
    q2 = (phi - delta_beta + pi) % (2.0 * pi) - pi
    wx = dy1 + torch.cos(q2) * dy2 - torch.sin(q2) * dz2
    wz = dz1 + torch.sin(q2) * dy2 + torch.cos(q2) * dz2

    theta_w = torch.atan2(wz, wx)
    if isinstance(H_target, torch.Tensor):
        theta_des = torch.atan2(dZ, torch.full_like(dZ, dY))
    else:
        theta_des = math.atan2(dZ, dY)

    q1 = (theta_des - theta_w + pi) % (2.0 * pi) - pi
    return q1, q2

class MyRobotDualEnv(MyRobotJump):
    """
    MyRobot Formal Dual-Policy Continuous Jumping and Teleoperation Environment.
    """
    STATE_BALANCE = 0
    STATE_STOPPING = 1
    STATE_PREPARE_JUMP = 2
    STATE_JUMP = 3
    STATE_RECOVER = 4

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.dual_cfg = getattr(cfg, 'dual_policy', None)

    def init_dual_policies(self, ac_balance=None, ac_jump=None):
        """
        Initialize both policies. If passed as None, automatically loads
        the frozen checkpoints specified in cfg.
        """
        if ac_balance is None:
            bal_path = getattr(self.dual_cfg, 'balance_model_path', "evaluation_assets/phase15/models/phase15_m130.pt")
            ckpt_bal = torch.load(bal_path, map_location=self.device)
            ac_balance = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(self.device)
            ac_balance.load_state_dict(ckpt_bal['model_state_dict'])
            ac_balance.eval()

        if ac_jump is None:
            jump_path = getattr(self.dual_cfg, 'jump_model_path', "evaluation_assets/frozen_stage8/model_40.pt")
            ckpt_jump = torch.load(jump_path, map_location=self.device)
            ac_jump = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(self.device)
            ac_jump.load_state_dict(ckpt_jump['model_state_dict'])
            ac_jump.eval()

        self.ac_balance = ac_balance
        self.ac_jump = ac_jump

        # FSM State buffers
        self.fsm_state = torch.full((self.num_envs,), self.STATE_BALANCE, device=self.device, dtype=torch.long)
        self.fsm_state_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.stopping_stable_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.gate_stable_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.settle_stable_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Policy internal buffers
        self.bal_filtered_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self.pi_integral = torch.zeros(self.num_envs, device=self.device)
        self.pi_v_filt = torch.zeros(self.num_envs, device=self.device)

        # Actuator gains and targets
        self.current_kp = torch.full((self.num_envs,), 40.0, device=self.device)
        self.current_kd = torch.full((self.num_envs,), 2.0, device=self.device)
        self.has_taken_off = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Transition smoothing buffers
        self.last_fsm_state = torch.full((self.num_envs,), self.STATE_BALANCE, device=self.device, dtype=torch.long)
        self.handoff_prev_q_tgt = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_q_tgt = torch.zeros(self.num_envs, 4, device=self.device)

        # Teleoperation buffers
        self.user_cmd_vy = torch.zeros(self.num_envs, device=self.device)
        self.user_cmd_yaw = torch.zeros(self.num_envs, device=self.device)
        self.user_target_height = torch.full((self.num_envs,), 0.39774, device=self.device)
        self.actual_cmd_vy = torch.zeros(self.num_envs, device=self.device)
        self.actual_cmd_yaw = torch.zeros(self.num_envs, device=self.device)
        self.actual_cmd_height = torch.full((self.num_envs,), 0.39774, device=self.device)
        self.teleop_jump_requested = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Multi-jump event & telemetry tracking
        self.jump_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.takeoff_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.touchdown_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.handoff_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.abort_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.last_transition_event = ["INIT_BALANCE"] * self.num_envs
        self.stopping_start_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._refresh_physical_state(update_has_taken_off=False)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)

        if not hasattr(self, 'fsm_state'):
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Reset FSM state buffers
        self.fsm_state[env_ids] = self.STATE_BALANCE
        self.fsm_state_timer[env_ids] = 0.0
        self.stopping_stable_steps[env_ids] = 0
        self.gate_stable_steps[env_ids] = 0
        self.settle_stable_steps[env_ids] = 0

        # Reset PI integral & balance filter buffers
        self.bal_filtered_actions[env_ids] = 0.0
        self.pi_integral[env_ids] = 0.0
        self.pi_v_filt[env_ids] = 0.0

        # Reset actuator gains & flight trackers
        self.current_kp[env_ids] = 40.0
        self.current_kd[env_ids] = 2.0
        self.has_taken_off[env_ids] = False

        # Reset transition & handoff smoothing caches
        self.last_fsm_state[env_ids] = self.STATE_BALANCE
        self.handoff_prev_q_tgt[env_ids] = 0.0
        self.last_q_tgt[env_ids] = 0.0

        # Reset teleoperation commands & requests
        self.user_cmd_vy[env_ids] = 0.0
        self.user_cmd_yaw[env_ids] = 0.0
        self.user_target_height[env_ids] = 0.39774
        self.actual_cmd_vy[env_ids] = 0.0
        self.actual_cmd_yaw[env_ids] = 0.0
        self.actual_cmd_height[env_ids] = 0.39774
        self.teleop_jump_requested[env_ids] = False

        # Reset event markers
        self.takeoff_step[env_ids] = -1
        self.touchdown_step[env_ids] = -1
        self.handoff_step[env_ids] = -1
        self.abort_step[env_ids] = -1
        self.stopping_start_pos[env_ids] = 0.0

        for eid in env_ids.tolist():
            self.last_transition_event[eid] = "INIT_BALANCE"

    def set_teleop_targets(self, vy=None, yaw=None, height=None):
        """Update teleoperation commands during BALANCE state."""
        in_balance = (self.fsm_state == self.STATE_BALANCE)
        if not in_balance.any():
            return
        active_mask = in_balance
        if vy is not None:
            self.user_cmd_vy[active_mask] = float(np.clip(vy, -0.30, 0.50))
        if yaw is not None:
            self.user_cmd_yaw[active_mask] = float(np.clip(yaw, -0.35, 0.35))
        if height is not None:
            self.user_target_height[active_mask] = float(np.clip(height, 0.340, 0.520))

    def request_teleop_jump(self, env_mask=None):
        """Request explosive jump. Supports continuous multiple jumps without sim reset."""
        if env_mask is None:
            env_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        can_request = env_mask & (self.fsm_state == self.STATE_BALANCE)
        if not can_request.any():
            return

        for i in range(self.num_envs):
            if can_request[i]:
                self.jump_count[i] += 1
                self.takeoff_step[i] = -1
                self.touchdown_step[i] = -1
                self.handoff_step[i] = -1
                self.abort_step[i] = -1
                self.has_taken_off[i] = False

                # Dynamic In-Motion Takeoff: If already cruising or commanded forward, jump immediately without stopping!
                is_cruising = abs(self.user_cmd_vy[i].item()) > 0.05 or abs(self.base_lin_vel[i, 1].item()) > 0.08
                if is_cruising:
                    self.fsm_state[i] = self.STATE_JUMP
                    self.fsm_state_timer[i] = 0.0
                    self.jump_timer[i] = 0.0
                    self.jump_phase[i] = 0.0
                    self.current_kp[i] = 40.0
                    self.current_kd[i] = 2.0
                    self.last_transition_event[i] = "ENTER_RUNNING_JUMP"
                else:
                    self.fsm_state[i] = self.STATE_STOPPING
                    self.fsm_state_timer[i] = 0.0
                    self.stopping_stable_steps[i] = 0
                    self.stopping_start_pos[i] = self.root_states[i, 0:2].clone()
                    self.user_cmd_vy[i] = 0.0
                    self.user_cmd_yaw[i] = 0.0
                    self.user_target_height[i] = 0.39774
                    self.last_transition_event[i] = "ENTER_STOPPING"

        self.teleop_jump_requested[can_request] = True

    def step_dual_policy(self, actions=None, call_post_physics_step=True):
        """
        Execute one control cycle (50 Hz outer loop with 200 Hz inner torque integration).
        """
        self.fsm_state_timer += self.dt

        # Sensor readings
        fn_L = self.contact_forces[:, self.left_wheel_body, 2]
        fn_R = self.contact_forces[:, self.right_wheel_body, 2]
        v_xy = torch.norm(self.base_lin_vel[:, 0:2], dim=-1)
        v_z = self.base_lin_vel[:, 2]
        r_raw, p_raw, y_raw = get_euler_xyz(self.base_quat)
        r = torch.atan2(torch.sin(r_raw), torch.cos(r_raw))
        p = torch.atan2(torch.sin(p_raw), torch.cos(p_raw))
        knee_vel = torch.maximum(torch.abs(self.dof_vel[:, self.left_knee_idx]), torch.abs(self.dof_vel[:, self.right_knee_idx]))
        air = self.is_airborne
        bz = self.root_states[:, 2]

        # 1. Finite State Machine (FSM) Transitions
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
                    self.fsm_state[i] = self.STATE_PREPARE_JUMP
                    self.fsm_state_timer[i] = 0.0
                    self.gate_stable_steps[i] = 0
                    self.teleop_jump_requested[i] = False
                    self.last_transition_event[i] = "ENTER_PREPARE_JUMP"
                elif self.fsm_state_timer[i].item() >= 5.0:
                    self.fsm_state[i] = self.STATE_BALANCE
                    self.fsm_state_timer[i] = 0.0
                    self.abort_step[i] = self.common_step_counter
                    self.teleop_jump_requested[i] = False
                    self.last_transition_event[i] = "STOPPING_TIMEOUT_ABORT"

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
                elif self.fsm_state_timer[i].item() >= 0.40:
                    self.fsm_state[i] = self.STATE_BALANCE
                    self.fsm_state_timer[i] = 0.0
                    self.abort_step[i] = self.common_step_counter
                    self.gate_stable_steps[i] = 0
                    self.teleop_jump_requested[i] = False
                    self.last_transition_event[i] = "PREPARE_TIMEOUT_ABORT"

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
                        self.last_transition_event[i] = "LIFTOFF"

                past_apex = self.jump_phase[i].item() >= 0.50
                firm_contact = (fn_L[i] > 25.0 and fn_R[i] > 25.0) or (self.jump_phase[i].item() >= 0.65 and (fn_L[i] > 25.0 or fn_R[i] > 25.0))

                if self.has_taken_off[i].item() and past_apex and firm_contact:
                    self.fsm_state[i] = self.STATE_RECOVER
                    self.fsm_state_timer[i] = 0.0
                    self.settle_stable_steps[i] = 0
                    if self.touchdown_step[i].item() == -1:
                        self.touchdown_step[i] = self.common_step_counter
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
                    self.teleop_jump_requested[i] = False
                    self.handoff_step[i] = self.common_step_counter

                    # Only reset commands to zero if this was a static jump (from STOPPING).
                    # For in-motion running jump, keep user_cmd_vy intact so cruising resumes!
                    is_running_jump = (abs(self.user_cmd_vy[i].item()) > 0.05) or (abs(self.actual_cmd_vy[i].item()) > 0.05)
                    if not is_running_jump:
                        self.actual_cmd_vy[i] = 0.0
                        self.actual_cmd_yaw[i] = 0.0
                        self.actual_cmd_height[i] = 0.39774
                        self.user_cmd_vy[i] = 0.0
                        self.user_cmd_yaw[i] = 0.0
                        self.user_target_height[i] = 0.39774

                    if normal_handoff:
                        self.last_transition_event[i] = "NORMAL_HANDOFF"
                    elif timeout_handoff:
                        self.last_transition_event[i] = "TIMEOUT_HANDOFF"
                    elif hard_timeout_abort:
                        self.last_transition_event[i] = "RECOVER_ABORT"
                        if self.abort_step[i].item() == -1:
                            self.abort_step[i] = self.common_step_counter

        # 2. Compute Policy Commands
        step_max_vy = 0.8 * self.dt
        step_max_yaw = 2.0 * self.dt
        step_max_h = 0.04 * self.dt

        # Only force command zeroing during active static stopping / prepare, NOT during running jump!
        in_static_stop = (self.fsm_state == self.STATE_STOPPING) | (self.fsm_state == self.STATE_PREPARE_JUMP)
        self.user_cmd_vy[in_static_stop] = 0.0
        self.user_cmd_yaw[in_static_stop] = 0.0
        self.user_target_height[in_static_stop] = 0.39774
        self.teleop_jump_requested[in_static_stop] = False

        delta_vy = self.user_cmd_vy - self.actual_cmd_vy
        self.actual_cmd_vy += torch.clamp(delta_vy, -step_max_vy, step_max_vy)

        delta_yaw = self.user_cmd_yaw - self.actual_cmd_yaw
        self.actual_cmd_yaw += torch.clamp(delta_yaw, -step_max_yaw, step_max_yaw)

        delta_h = self.user_target_height - self.actual_cmd_height
        self.actual_cmd_height += torch.clamp(delta_h, -step_max_h, step_max_h)

        # Balance Policy Pipeline
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

        # Jump Policy Pipeline
        jump_active = (self.fsm_state == self.STATE_JUMP) | (self.fsm_state == self.STATE_RECOVER)
        if actions is not None:
            # External actions provided (from PPO runner in train.py or policy in play.py)
            clip_act = getattr(self.cfg.normalization, 'clip_actions', 100.0) if hasattr(self, 'cfg') and hasattr(self.cfg, 'normalization') else 100.0
            act_jump_raw = torch.clamp(actions, -clip_act, clip_act)
            self.actions = actions.clone()
        else:
            # Standalone fallback (e.g. teleop_dual.py without PPO runner)
            self.compute_observations()
            jump_obs = self.get_observations()
            with torch.no_grad():
                act_jump_raw = self.ac_jump.act_inference(jump_obs)
            self.actions = act_jump_raw.clone()

        if jump_active.any():
            self.filtered_actions[jump_active] = 0.80 * act_jump_raw[jump_active] + 0.20 * self.filtered_actions[jump_active]
        self.filtered_actions[~jump_active] = 0.0

        tgt_hip_jump_L = self.default_dof_pos[:, self.left_hip_idx] + self.filtered_actions[:, self.left_hip_idx] * self.hip_action_scale
        tgt_knee_jump_L = self.default_dof_pos[:, self.left_knee_idx] + self.filtered_actions[:, self.left_knee_idx] * self.knee_action_scale
        tgt_hip_jump_R = self.default_dof_pos[:, self.right_hip_idx] + self.filtered_actions[:, self.right_hip_idx] * self.hip_action_scale
        tgt_knee_jump_R = self.default_dof_pos[:, self.right_knee_idx] + self.filtered_actions[:, self.right_knee_idx] * self.knee_action_scale

        target_w = - self.actual_cmd_vy / self.wheel_radius
        tau_w_jump_L_g = self.filtered_actions[:, self.left_wheel_idx] * 10.0 - 0.35 * (self.dof_vel[:, self.left_wheel_idx] - target_w)
        tau_w_jump_L_a = torch.clamp(self.filtered_actions[:, self.left_wheel_idx] * 3.0 - 0.3 * self.dof_vel[:, self.left_wheel_idx], -3.0, 3.0)
        tau_w_jump_L = torch.where(air, tau_w_jump_L_a, tau_w_jump_L_g)

        tau_w_jump_R_g = self.filtered_actions[:, self.right_wheel_idx] * 10.0 - 0.35 * (self.dof_vel[:, self.right_wheel_idx] - target_w)
        tau_w_jump_R_a = torch.clamp(self.filtered_actions[:, self.right_wheel_idx] * 3.0 - 0.3 * self.dof_vel[:, self.right_wheel_idx], -3.0, 3.0)
        tau_w_jump_R = torch.where(air, tau_w_jump_R_a, tau_w_jump_R_g)

        # Track FSM state switches
        state_changed = (self.fsm_state != self.last_fsm_state)
        for i in range(self.num_envs):
            if state_changed[i].item():
                self.handoff_prev_q_tgt[i] = self.last_q_tgt[i]
        self.last_fsm_state[:] = self.fsm_state

        # 3. Unified Actuator Commands & 200 Hz Integration Loop
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

        for _ in range(self.cfg.control.decimation):
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
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self.last_q_tgt = q_target.clone()
        if call_post_physics_step:
            self.post_physics_step()
            self._refresh_physical_state(update_has_taken_off=False)
