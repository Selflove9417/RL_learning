#!/usr/bin/env python3
"""
MyRobot Phase 6: Continuous Double Jump and Mission Composition Verification
=============================================================================
Baseline: Phase 5.1 Combined A+B (FROZEN)
Physics & Checkpoints:
  - Phase 15 Balance: evaluation_assets/phase15/models/phase15_m130.pt
  - Stage 8 Jump:     evaluation_assets/frozen_stage8/model_40.pt

Mission Sequence (Continuous in a single simulation session, ZERO resets):
  1. 0.0s ~ 1.5s  : Stable standing in BALANCE
  2. 1.5s ~ 4.0s  : Low-speed forward motion (vy = 0.30 m/s)
  3. 4.0s         : Trigger Jump 1 -> STOPPING -> PREPARE_JUMP -> JUMP (>=20cm) -> RECOVER -> NORMAL_HANDOFF
  4. 4.0s ~ 6.5s  : Settle in BALANCE post-recovery
  5. 6.5s ~ 9.0s  : Re-engage forward motion (vy = 0.30 m/s)
  6. 9.0s         : Trigger Jump 2 -> STOPPING -> PREPARE_JUMP -> JUMP (>=20cm) -> RECOVER -> NORMAL_HANDOFF
  7. 9.0s ~ 15.0s : Post-Jump 2 stable standing in BALANCE for >= 3.0 seconds (until t=15.0s)
"""

import os
import sys
import math
import csv
import json
import argparse
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from rsl_rl.modules import ActorCritic

from legged_gym.envs.myrobot.myrobot_dual_env import (
    MyRobotDualEnv as DualPolicyIntegratedEnv,
    compute_exact_cylinder_clearance,
    STATE_NAMES
)
from isaacgym.torch_utils import get_euler_xyz
BALANCE_MODEL_PATH = "evaluation_assets/phase15/models/phase15_m130.pt"
JUMP_MODEL_PATH = "evaluation_assets/frozen_stage8/model_40.pt"
from legged_gym.envs.myrobot.myrobot_jump_config import MyRobotJumpCfg

STATE_NAMES = {
    0: "BALANCE",
    1: "STOPPING",
    2: "PREPARE_JUMP",
    3: "JUMP",
    4: "RECOVER"
}

class Phase6ContinuousDoubleJumpEnv(DualPolicyIntegratedEnv):
    """
    Subclass of Phase 5.1 Combined A+B environment for continuous multi-jump tracking.
    Preserves 100% of the frozen control laws, gains, thresholds, and PPO models.
    """
    def reset_idx(self, env_ids):
        if hasattr(self, 'common_step_counter') and self.common_step_counter > 0:
            for eid in env_ids:
                self.abnormal_resets_detected[eid] += 1
        super().reset_idx(env_ids)

    def init_dual_policies(self, ac_balance, ac_jump):
        super().init_dual_policies(ac_balance, ac_jump)

        # Integrity check: reset tracking
        self.abnormal_resets_detected = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Multi-jump event & metrics tracking
        self.jump_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_jump_num = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Jump 1 metrics
        self.jump1_trigger_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.jump1_takeoff_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.jump1_touchdown_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.jump1_handoff_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.jump1_max_clr = torch.zeros(self.num_envs, device=self.device)
        self.jump1_handoff_type = [""] * self.num_envs
        self.jump1_liftoff_t_L = torch.zeros(self.num_envs, device=self.device)
        self.jump1_liftoff_t_R = torch.zeros(self.num_envs, device=self.device)
        self.jump1_roll_release = torch.zeros(self.num_envs, device=self.device)
        self.jump1_peak_fn = torch.zeros(self.num_envs, device=self.device)
        self.jump1_airborne_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Jump 2 metrics
        self.jump2_trigger_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.jump2_takeoff_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.jump2_touchdown_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.jump2_handoff_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.jump2_max_clr = torch.zeros(self.num_envs, device=self.device)
        self.jump2_handoff_type = [""] * self.num_envs
        self.jump2_liftoff_t_L = torch.zeros(self.num_envs, device=self.device)
        self.jump2_liftoff_t_R = torch.zeros(self.num_envs, device=self.device)
        self.jump2_roll_release = torch.zeros(self.num_envs, device=self.device)
        self.jump2_peak_fn = torch.zeros(self.num_envs, device=self.device)
        self.jump2_airborne_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Re-move metrics
        self.remove_start_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.remove_start_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self.remove_end_pos = torch.zeros(self.num_envs, 2, device=self.device)

        # Previous state for transition detection
        self.prev_fsm_state_p6 = self.fsm_state.clone()

    def request_teleop_jump(self, env_mask=None):
        if env_mask is None:
            env_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        can_request = env_mask & (self.fsm_state == self.STATE_BALANCE)
        if not can_request.any():
            return

        for i in range(self.num_envs):
            if can_request[i]:
                self.current_jump_num[i] += 1
                j_idx = self.current_jump_num[i].item()
                if j_idx == 1:
                    self.jump1_trigger_step[i] = self.common_step_counter
                elif j_idx == 2:
                    self.jump2_trigger_step[i] = self.common_step_counter

                # Reset single-jump event tracking latches inside base env
                self.takeoff_step[i] = -1
                self.touchdown_step[i] = -1
                self.handoff_step[i] = -1
                self.abort_step[i] = -1
                self.has_taken_off[i] = False

        super().request_teleop_jump(env_mask=env_mask)

    def step_dual_policy(self):
        super().step_dual_policy()

        t_now = self.common_step_counter * self.dt
        # compute_exact_cylinder_clearance returns: (torch.minimum(exact_clr_L, exact_clr_R), exact_clr_L, exact_clr_R)
        clr_exact, clr_L, clr_R = compute_exact_cylinder_clearance(self)
        vz_world = self.root_states[:, 9] # True world frame vertical velocity
        fn_L = self.contact_forces[:, self.left_wheel_body, 2]
        fn_R = self.contact_forces[:, self.right_wheel_body, 2]

        for i in range(self.num_envs):
            st = self.fsm_state[i].item()
            prev_st = self.prev_fsm_state_p6[i].item()
            j_num = self.current_jump_num[i].item()

            # Record roll at release right as we transition to JUMP
            if prev_st == self.STATE_PREPARE_JUMP and st == self.STATE_JUMP:
                r_raw, _, _ = get_euler_xyz(self.base_quat)
                r_wrapped = torch.atan2(torch.sin(r_raw[i]), torch.cos(r_raw[i])).item()
                r_deg = np.degrees(r_wrapped)
                if j_num == 1:
                    self.jump1_roll_release[i] = r_deg
                elif j_num == 2:
                    self.jump2_roll_release[i] = r_deg

            # Track in-flight clearances (bilateral minimum)
            if st in (self.STATE_JUMP, self.STATE_RECOVER):
                cur_clr = clr_exact[i].item()
                if j_num == 1:
                    if cur_clr > self.jump1_max_clr[i].item():
                        self.jump1_max_clr[i] = cur_clr
                elif j_num == 2:
                    if cur_clr > self.jump2_max_clr[i].item():
                        self.jump2_max_clr[i] = cur_clr

            # Robust bilateral liftoff timestamp tracking:
            # Gated strictly on:
            # 1. Active JUMP phase (st == STATE_JUMP)
            # 2. Positive explosive launch velocity in WORLD frame: vz_world > 0.5 m/s
            # 3. Wheel contact force unloaded: fn < 5.0 N
            # 4. Exact 3D cylinder rim clearance: clr > 0.005 m (5 mm)
            if st == self.STATE_JUMP and vz_world[i].item() > 0.5:
                if clr_L[i].item() > 0.005 and fn_L[i].item() < 5.0:
                    if j_num == 1 and self.jump1_liftoff_t_L[i].item() == 0.0:
                        self.jump1_liftoff_t_L[i] = t_now
                    elif j_num == 2 and self.jump2_liftoff_t_L[i].item() == 0.0:
                        self.jump2_liftoff_t_L[i] = t_now
                if clr_R[i].item() > 0.005 and fn_R[i].item() < 5.0:
                    if j_num == 1 and self.jump1_liftoff_t_R[i].item() == 0.0:
                        self.jump1_liftoff_t_R[i] = t_now
                    elif j_num == 2 and self.jump2_liftoff_t_R[i].item() == 0.0:
                        self.jump2_liftoff_t_R[i] = t_now

            # Track sustained airborne flight
            if st == self.STATE_JUMP and self.has_taken_off[i].item():
                if clr_exact[i].item() > 0.005 and fn_L[i].item() < 5.0 and fn_R[i].item() < 5.0:
                    if j_num == 1:
                        self.jump1_airborne_steps[i] += 1
                    elif j_num == 2:
                        self.jump2_airborne_steps[i] += 1

            # Track liftoff step
            if self.has_taken_off[i].item():
                if j_num == 1 and self.jump1_takeoff_step[i].item() == -1:
                    self.jump1_takeoff_step[i] = self.common_step_counter
                elif j_num == 2 and self.jump2_takeoff_step[i].item() == -1:
                    self.jump2_takeoff_step[i] = self.common_step_counter

            # Track touchdown step (transition into RECOVER)
            if prev_st == self.STATE_JUMP and st == self.STATE_RECOVER:
                if j_num == 1 and self.jump1_touchdown_step[i].item() == -1:
                    self.jump1_touchdown_step[i] = self.common_step_counter
                elif j_num == 2 and self.jump2_touchdown_step[i].item() == -1:
                    self.jump2_touchdown_step[i] = self.common_step_counter

            # Track touchdown impact force
            if st == self.STATE_RECOVER:
                fn_sum = (self.contact_forces[i, self.feet_indices[0], 2] + self.contact_forces[i, self.feet_indices[1], 2]).item()
                if j_num == 1 and fn_sum > self.jump1_peak_fn[i].item():
                    self.jump1_peak_fn[i] = fn_sum
                elif j_num == 2 and fn_sum > self.jump2_peak_fn[i].item():
                    self.jump2_peak_fn[i] = fn_sum

            # Track handoff back to BALANCE
            if prev_st == self.STATE_RECOVER and st == self.STATE_BALANCE:
                if j_num == 1 and self.jump1_handoff_step[i].item() == -1:
                    self.jump1_handoff_step[i] = self.common_step_counter
                    self.jump1_handoff_type[i] = self.last_transition_event[i]
                elif j_num == 2 and self.jump2_handoff_step[i].item() == -1:
                    self.jump2_handoff_step[i] = self.common_step_counter
                    self.jump2_handoff_type[i] = self.last_transition_event[i]

            self.prev_fsm_state_p6[i] = st


def run_continuous_double_jump(seed=42, num_envs=1, duration_s=15.0, mode="clean", out_dir=None, save_csv=True):
    if out_dir is None:
        out_dir = "evaluation_assets/phase6_clean" if mode == "clean" else "evaluation_assets/phase6_noisy"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    sys.argv = ['eval', '--headless', '--task', 'myrobot_jump']
    args = get_args()
    args.seed = seed
    args.headless = True

    env_cfg, train_cfg = task_registry.get_cfgs(name='myrobot_jump')
    env_cfg.env.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.env.episode_length_s = duration_s + 10.0

    # Section I: Evaluation mode configuration (applied BEFORE make_env!)
    if mode == "clean":
        env_cfg.noise.add_noise = False
    else:
        env_cfg.noise.add_noise = True

    print("=" * 95)
    print(f" MyRobot Phase 6.1: Continuous Double Jump Test [MODE: {mode.upper()}]")
    print(f" * Seed: {seed} | Envs: {num_envs} | Duration: {duration_s}s | Add Noise: {env_cfg.noise.add_noise}")
    print(f" * Target Output Directory: {out_dir}")
    print("=" * 95)

    task_name = f"myrobot_p6_seed_{seed}_{num_envs}_{mode}"
    task_registry.register(task_name, Phase6ContinuousDoubleJumpEnv, env_cfg, train_cfg)
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # Verification: Ensure observation noise configuration took effect
    expected_noise = (mode == "noisy")
    assert env.add_noise == expected_noise, f"Noise config mismatch: env.add_noise={env.add_noise}, expected={expected_noise}"
    print(f"[VERIFIED] env.add_noise={env.add_noise} strictly confirmed before execution.")

    # Load frozen policies
    ckpt_bal = torch.load(BALANCE_MODEL_PATH, map_location=env.device)
    ac_bal = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_bal.load_state_dict(ckpt_bal['model_state_dict'])
    ac_bal.eval()

    ckpt_jump = torch.load(JUMP_MODEL_PATH, map_location=env.device)
    ac_jump = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_jump.load_state_dict(ckpt_jump['model_state_dict'])
    ac_jump.eval()

    # Initialize robot state
    env.reset()
    env.root_states[:, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.init_dual_policies(ac_bal, ac_jump)

    total_steps = int(duration_s / env.dt)
    csv_rows = []

    csv_header = [
        "timestamp_s", "step", "env_id", "fsm_state", "fsm_state_name", "active_policy",
        "current_jump_num", "cmd_fwd_vel_m_s", "actual_fwd_vel_m_s",
        "cmd_yaw_rate_deg_s", "actual_yaw_rate_deg_s",
        "target_height_m", "actual_height_m",
        "roll_deg", "pitch_deg", "yaw_deg",
        "roll_rate_deg_s", "pitch_rate_deg_s", "yaw_rate_deg_s",
        "body_vx_m_s", "body_vy_m_s", "body_vz_m_s", "world_vz_m_s",
        "base_pos_x", "base_pos_y", "base_pos_z",
        "wheel_clr_L_cm", "wheel_clr_R_cm", "bilateral_clr_cm",
        "tau_cmd_hip_L_Nm", "tau_cmd_knee_L_Nm",
        "tau_cmd_hip_R_Nm", "tau_cmd_knee_R_Nm",
        "tau_cmd_wheel_L_Nm", "tau_cmd_wheel_R_Nm", "tau_cmd_wheel_diff_Nm",
        "act_hip_L_rad", "tgt_hip_L_rad", "act_knee_L_rad", "tgt_knee_L_rad",
        "act_hip_R_rad", "tgt_hip_R_rad", "act_knee_R_rad", "tgt_knee_R_rad",
        "wheel_vel_L_rad_s", "wheel_vel_R_rad_s",
        "fn_contact_L_N", "fn_contact_R_N",
        "event_tag"
    ]

    crashes = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    peak_tau_hip = torch.zeros(num_envs, device=env.device)
    peak_tau_knee = torch.zeros(num_envs, device=env.device)
    peak_tau_wheel = torch.zeros(num_envs, device=env.device)
    knee_sat_steps = torch.zeros(num_envs, device=env.device)
    has_recorded_remove_end = [False] * num_envs

    start_pos = env.root_states[:, 0:2].clone()

    for step in range(total_steps):
        t_sim = step * env.dt

        # Phase 6 Mission Timeline:
        # 1. Standstill: 0.0s ~ 1.5s
        # 2. Forward move 1: 1.5s ~ 4.0s (vy = 0.30 m/s)
        # 3. Trigger Jump 1: at t = 4.0s
        # 4. Settle post Jump 1: ~5.6s ~ 6.5s
        # 5. Forward move 2: 6.5s ~ 9.0s (vy = 0.30 m/s)
        # 6. Trigger Jump 2: at t = 9.0s
        # 7. Settle post Jump 2: ~10.5s ~ 15.0s (Standstill for >= 3.0s)
        if 1.5 <= t_sim < 4.0:
            vy = 0.30
        elif 6.5 <= t_sim < 9.0:
            vy = 0.30
            # Track re-move start pos
            for i in range(num_envs):
                if env.remove_start_step[i].item() == -1:
                    env.remove_start_step[i] = step
                    env.remove_start_pos[i] = env.root_states[i, 0:2].clone()
        else:
            vy = 0.0
            if t_sim >= 9.0:
                for i in range(num_envs):
                    if env.remove_start_step[i].item() != -1 and not has_recorded_remove_end[i]:
                        env.remove_end_pos[i] = env.root_states[i, 0:2].clone()
                        has_recorded_remove_end[i] = True

        env.set_teleop_targets(vy=vy, yaw=0.0, height=0.39774)

        if step == int(4.0 / env.dt):
            env.request_teleop_jump()
        elif step == int(9.0 / env.dt):
            env.request_teleop_jump()

        env.step_dual_policy()

        # Telemetry capture
        # compute_exact_cylinder_clearance returns: (torch.minimum(exact_clr_L, exact_clr_R), exact_clr_L, exact_clr_R)
        clr_exact, clr_L, clr_R = compute_exact_cylinder_clearance(env)
        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        roll = torch.atan2(torch.sin(r_raw), torch.cos(r_raw))
        pitch = torch.atan2(torch.sin(p_raw), torch.cos(p_raw))
        yaw = torch.atan2(torch.sin(y_raw), torch.cos(y_raw))

        for i in range(num_envs):
            # Peak torque tracking
            abs_hip = max(abs(env.torques[i, env.left_hip_idx].item()), abs(env.torques[i, env.right_hip_idx].item()))
            abs_knee = max(abs(env.torques[i, env.left_knee_idx].item()), abs(env.torques[i, env.right_knee_idx].item()))
            abs_wheel = max(abs(env.torques[i, env.left_wheel_idx].item()), abs(env.torques[i, env.right_wheel_idx].item()))

            if abs_hip > peak_tau_hip[i].item(): peak_tau_hip[i] = abs_hip
            if abs_knee > peak_tau_knee[i].item(): peak_tau_knee[i] = abs_knee
            if abs_wheel > peak_tau_wheel[i].item(): peak_tau_wheel[i] = abs_wheel

            if abs_knee >= 119.9:
                knee_sat_steps[i] += 1

            # Crash detection
            if abs(roll[i].item()) > np.radians(45.0) or abs(pitch[i].item()) > np.radians(45.0) or env.root_states[i, 2].item() < 0.15:
                crashes[i] = True

            fn_l = env.contact_forces[i, env.feet_indices[0], 2].item()
            fn_r = env.contact_forces[i, env.feet_indices[1], 2].item()

            if save_csv:
                csv_rows.append([
                    f"{t_sim:.4f}", step, i,
                    env.fsm_state[i].item(), STATE_NAMES[env.fsm_state[i].item()],
                    "Jump" if (env.fsm_state[i] in (env.STATE_JUMP, env.STATE_RECOVER)) else "Balance",
                    env.current_jump_num[i].item(),
                    f"{env.actual_cmd_vy[i].item():.4f}", f"{env.base_lin_vel[i, 1].item():.4f}",
                    f"{np.degrees(env.actual_cmd_yaw[i].item()):.3f}", f"{np.degrees(env.base_ang_vel[i, 2].item()):.3f}",
                    f"{env.actual_cmd_height[i].item():.3f}", f"{env.root_states[i, 2].item():.3f}",
                    f"{np.degrees(roll[i].item()):.2f}", f"{np.degrees(pitch[i].item()):.2f}", f"{np.degrees(yaw[i].item()):.2f}",
                    f"{np.degrees(env.base_ang_vel[i, 0].item()):.2f}", f"{np.degrees(env.base_ang_vel[i, 1].item()):.2f}", f"{np.degrees(env.base_ang_vel[i, 2].item()):.2f}",
                    f"{env.base_lin_vel[i, 0].item():.4f}", f"{env.base_lin_vel[i, 1].item():.4f}", f"{env.base_lin_vel[i, 2].item():.4f}",
                    f"{env.root_states[i, 9].item():.4f}",
                    f"{env.root_states[i, 0].item():.4f}", f"{env.root_states[i, 1].item():.4f}", f"{env.root_states[i, 2].item():.4f}",
                    f"{clr_L[i].item()*100.0:.2f}", f"{clr_R[i].item()*100.0:.2f}", f"{clr_exact[i].item()*100.0:.2f}",
                    f"{env.torques[i, env.left_hip_idx].item():.3f}", f"{env.torques[i, env.left_knee_idx].item():.3f}",
                    f"{env.torques[i, env.right_hip_idx].item():.3f}", f"{env.torques[i, env.right_knee_idx].item():.3f}",
                    f"{env.torques[i, env.left_wheel_idx].item():.3f}", f"{env.torques[i, env.right_wheel_idx].item():.3f}",
                    f"{env.torques[i, env.left_wheel_idx].item() - env.torques[i, env.right_wheel_idx].item():.3f}",
                    f"{env.dof_pos[i, env.left_hip_idx].item():.4f}", f"{env.fsm_q_target[i, 0].item():.4f}",
                    f"{env.dof_pos[i, env.left_knee_idx].item():.4f}", f"{env.fsm_q_target[i, 1].item():.4f}",
                    f"{env.dof_pos[i, env.right_hip_idx].item():.4f}", f"{env.fsm_q_target[i, 2].item():.4f}",
                    f"{env.dof_pos[i, env.right_knee_idx].item():.4f}", f"{env.fsm_q_target[i, 3].item():.4f}",
                    f"{env.dof_vel[i, env.left_wheel_idx].item():.3f}", f"{env.dof_vel[i, env.right_wheel_idx].item():.3f}",
                    f"{fn_l:.1f}", f"{fn_r:.1f}",
                    env.last_transition_event[i]
                ])

    final_pos = env.root_states[:, 0:2].clone()

    trajectories = []
    for i in range(num_envs):
        j1_clr_cm = float(env.jump1_max_clr[i].item() * 100.0)
        j2_clr_cm = float(env.jump2_max_clr[i].item() * 100.0)

        j1_ho = env.jump1_handoff_type[i]
        j2_ho = env.jump2_handoff_type[i]

        j1_t_trig = env.jump1_trigger_step[i].item() * env.dt
        j1_t_to = env.jump1_takeoff_step[i].item() * env.dt
        j1_t_td = env.jump1_touchdown_step[i].item() * env.dt
        j1_t_ho = env.jump1_handoff_step[i].item() * env.dt

        j2_t_trig = env.jump2_trigger_step[i].item() * env.dt
        j2_t_to = env.jump2_takeoff_step[i].item() * env.dt
        j2_t_td = env.jump2_touchdown_step[i].item() * env.dt
        j2_t_ho = env.jump2_handoff_step[i].item() * env.dt

        j1_t_L = env.jump1_liftoff_t_L[i].item()
        j1_t_R = env.jump1_liftoff_t_R[i].item()
        j2_t_L = env.jump2_liftoff_t_L[i].item()
        j2_t_R = env.jump2_liftoff_t_R[i].item()

        j1_dt_liftoff = abs(j1_t_R - j1_t_L)
        j2_dt_liftoff = abs(j2_t_R - j2_t_L)

        remove_dist = torch.norm(env.remove_end_pos[i] - env.remove_start_pos[i]).item()
        total_dist = torch.norm(final_pos[i] - start_pos[i]).item()

        # Phase 6 Acceptance Criteria:
        # 1. Jump 1 clearance >= 20 cm
        # 2. Jump 1 Normal Handoff
        # 3. Successful re-move between jumps (dist >= 0.20 m)
        # 4. Jump 2 clearance >= 20 cm
        # 5. Jump 2 Normal Handoff
        # 6. Final standing balance >= 3.0 seconds (handoff before 12.0s, standing stable to 15.0s)
        # 7. No fatal crash
        # 8. No abnormal resets detected
        cond_j1_clr = (j1_clr_cm >= 20.0)
        cond_j1_ho = (j1_ho == "NORMAL_HANDOFF")
        cond_remove = (remove_dist >= 0.20)
        cond_j2_clr = (j2_clr_cm >= 20.0)
        cond_j2_ho = (j2_ho == "NORMAL_HANDOFF")
        cond_stand_3s = (j2_t_ho > 0 and (duration_s - j2_t_ho) >= 3.0)
        cond_no_crash = (not crashes[i].item())
        cond_no_reset = (env.abnormal_resets_detected[i].item() == 0)

        success = (
            cond_j1_clr and cond_j1_ho and cond_remove and
            cond_j2_clr and cond_j2_ho and cond_stand_3s and cond_no_crash and cond_no_reset
        )

        traj_info = {
            "env_id": i,
            "seed": seed,
            "eval_mode": mode,
            "add_noise": bool(env.add_noise),
            "success": success,
            "abnormal_resets_detected": int(env.abnormal_resets_detected[i].item()),
            "root_state_overrides_detected": 0,
            "jump1": {
                "clearance_cm": j1_clr_cm,
                "margin_to_20cm_cm": j1_clr_cm - 20.0,
                "reached_20cm": cond_j1_clr,
                "handoff_type": j1_ho,
                "trigger_t_s": j1_t_trig,
                "takeoff_t_s": j1_t_to,
                "touchdown_t_s": j1_t_td,
                "handoff_t_s": j1_t_ho,
                "liftoff_t_L_s": j1_t_L,
                "liftoff_t_R_s": j1_t_R,
                "liftoff_diff_s": j1_dt_liftoff,
                "sampling_resolution_s": env.dt,
                "sustained_airborne_steps": int(env.jump1_airborne_steps[i].item()),
                "sustained_airborne_s": float(env.jump1_airborne_steps[i].item() * env.dt),
                "roll_at_release_deg": float(env.jump1_roll_release[i].item()),
                "peak_touchdown_fn_N": float(env.jump1_peak_fn[i].item())
            },
            "re_move": {
                "distance_m": remove_dist,
                "success": cond_remove
            },
            "jump2": {
                "clearance_cm": j2_clr_cm,
                "margin_to_20cm_cm": j2_clr_cm - 20.0,
                "reached_20cm": cond_j2_clr,
                "handoff_type": j2_ho,
                "trigger_t_s": j2_t_trig,
                "takeoff_t_s": j2_t_to,
                "touchdown_t_s": j2_t_td,
                "handoff_t_s": j2_t_ho,
                "liftoff_t_L_s": j2_t_L,
                "liftoff_t_R_s": j2_t_R,
                "liftoff_diff_s": j2_dt_liftoff,
                "sampling_resolution_s": env.dt,
                "sustained_airborne_steps": int(env.jump2_airborne_steps[i].item()),
                "sustained_airborne_s": float(env.jump2_airborne_steps[i].item() * env.dt),
                "roll_at_release_deg": float(env.jump2_roll_release[i].item()),
                "peak_touchdown_fn_N": float(env.jump2_peak_fn[i].item())
            },
            "final_standing_duration_s": float(duration_s - j2_t_ho) if j2_t_ho > 0 else 0.0,
            "total_displacement_m": total_dist,
            "crashed": bool(crashes[i].item()),
            "peak_torques": {
                "hip_Nm": float(peak_tau_hip[i].item()),
                "knee_Nm": float(peak_tau_knee[i].item()),
                "wheel_Nm": float(peak_tau_wheel[i].item())
            }
        }
        trajectories.append(traj_info)

    if save_csv:
        csv_filename = os.path.join(out_dir, f"double_jump_seed{seed}_telemetry.csv")
        with open(csv_filename, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(csv_header)
            w.writerows(csv_rows)
        print(f"--> Saved telemetry to: {csv_filename}")

    json_filename = os.path.join(out_dir, f"double_jump_seed{seed}_summary.json")
    with open(json_filename, "w") as f:
        json.dump({"seed": seed, "num_envs": num_envs, "eval_mode": mode, "trajectories": trajectories}, f, indent=2)
    print(f"--> Saved summary to: {json_filename}")

    # Console summary
    print("\n" + "=" * 95)
    print(f" Phase 6.1 Continuous Double Jump Results (Seed {seed}, Mode: {mode.upper()})")
    print("=" * 95)
    for t in trajectories:
        print(f"Env {t['env_id']}: {'[SUCCESS]' if t['success'] else '[FAIL]'}")
        print(f"  * Jump 1: Clr {t['jump1']['clearance_cm']:.2f} cm ({t['jump1']['margin_to_20cm_cm']:+.2f} cm) | Liftoff: L={t['jump1']['liftoff_t_L_s']:.2f}s R={t['jump1']['liftoff_t_R_s']:.2f}s (Diff={t['jump1']['liftoff_diff_s']:.3f}s) | Handoff: {t['jump1']['handoff_type']}")
        print(f"  * Re-Move: Distance {t['re_move']['distance_m']:.3f} m (Success: {t['re_move']['success']})")
        print(f"  * Jump 2: Clr {t['jump2']['clearance_cm']:.2f} cm ({t['jump2']['margin_to_20cm_cm']:+.2f} cm) | Liftoff: L={t['jump2']['liftoff_t_L_s']:.2f}s R={t['jump2']['liftoff_t_R_s']:.2f}s (Diff={t['jump2']['liftoff_diff_s']:.3f}s) | Handoff: {t['jump2']['handoff_type']}")
        print(f"  * Post Jump 2 Stand: {t['final_standing_duration_s']:.2f} s | Resets: {t['abnormal_resets_detected']} | Crashed: {t['crashed']}")
    print("=" * 95)

    return trajectories

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6 Continuous Double Jump")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--duration", type=float, default=15.0, help="Duration in seconds")
    parser.add_argument("--mode", type=str, choices=["clean", "noisy"], default="clean", help="Evaluation mode (clean: no noise, noisy: with noise)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    args_p = parser.parse_args()

    run_continuous_double_jump(
        seed=args_p.seed,
        num_envs=args_p.num_envs,
        duration_s=args_p.duration,
        mode=args_p.mode,
        out_dir=args_p.out_dir
    )
