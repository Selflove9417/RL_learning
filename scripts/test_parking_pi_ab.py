"""
Phase 15 Step 2, 3 & 4: Parking Velocity Closed-Loop PI Compensator A/B Validation
Evaluates model_130.pt with:
  [A] Compensation OFF (Pure Policy Baseline)
  [B] Compensation ON (Parking-Only Smooth PI Speed Trim with Anti-Windup & Smooth Mode Switching)

Validation Checks:
1. Steady-state 20s parking hold across 6 heights: [0.34, 0.38, 0.42, 0.46, 0.50, 0.52] m.
   - Quantifies 20s net displacement, mean vy, std vy, pitch stability, and wheel torque.
2. Dynamic transition test (start-stop-restart) at H = 0.34m:
   - Settle 5s in parking (PI engaged) -> Accel to +0.30 m/s for 4s (PI disengaged) -> Brake to 0 m/s for 5s (PI re-engaged).
   - Verifies zero step-out bump, instantaneous integral reset, and smooth parking re-lock.
"""

import os
import sys
import json
import math
import numpy as np

# isaacgym must be imported before torch
import isaacgym
from isaacgym import gymapi, gymtorch
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from rsl_rl.modules import ActorCritic


class ParkingPIController:
    """
    Parking-Only Velocity PI Compensator with Smooth Mode Switching & Anti-Windup.
    Only trims the policy velocity command when in parking mode.
    """
    def __init__(self, dt=0.02, Kp=0.35, Ki=0.60, trim_max=0.045, deadband=0.003):
        self.dt = dt
        self.Kp = Kp
        self.Ki = Ki
        self.trim_max = trim_max
        self.deadband = deadband

        # State buffers (vectorized for num_envs)
        self.integral = None
        self.filtered_vy = None
        self.park_settle_counter = None
        self.alpha_lpf = 0.15 # ~1.3 Hz cutoff at dt=0.02s
        self.i_max = self.trim_max / max(1e-4, self.Ki)

    def init_buffers(self, num_envs, device):
        self.integral = torch.zeros(num_envs, device=device, dtype=torch.float)
        self.filtered_vy = torch.zeros(num_envs, device=device, dtype=torch.float)
        self.park_settle_counter = torch.zeros(num_envs, device=device, dtype=torch.long)

    def reset_env(self, env_ids=None):
        if env_ids is None:
            self.integral.zero_()
            self.filtered_vy.zero_()
            self.park_settle_counter.zero_()
        else:
            self.integral[env_ids] = 0.0
            self.filtered_vy[env_ids] = 0.0
            self.park_settle_counter[env_ids] = 0

    def compute_trim(self, actual_vy, user_cmd_vy, user_cmd_yaw, is_elevating=False, is_grounded=True):
        """
        Computes velocity trim delta_v_trim (added to policy's cmd_vy).
        Returns delta_v_trim (tensor of shape [num_envs]) and is_active mask.
        """
        # Low-pass filter actual linear velocity
        self.filtered_vy = (1.0 - self.alpha_lpf) * self.filtered_vy + self.alpha_lpf * actual_vy

        # Check parking condition
        is_zero_cmd = (torch.abs(user_cmd_vy) < 0.005) & (torch.abs(user_cmd_yaw) < 0.010)
        is_slow = torch.abs(self.filtered_vy) < 0.12 # not actively in dynamic braking
        
        can_engage = is_zero_cmd & is_slow
        if isinstance(is_elevating, torch.Tensor):
            can_engage = can_engage & (~is_elevating)
        elif is_elevating:
            can_engage = torch.zeros_like(can_engage)

        if isinstance(is_grounded, torch.Tensor):
            can_engage = can_engage & is_grounded
        elif not is_grounded:
            can_engage = torch.zeros_like(can_engage)

        can_engage = can_engage.bool()

        # Update settle counter: require at least 15 steps (~0.3s) of stable zero cmd
        self.park_settle_counter = torch.where(
            can_engage,
            self.park_settle_counter + 1,
            torch.zeros_like(self.park_settle_counter)
        )
        is_active = (self.park_settle_counter >= 15).bool()

        # Error: target (0.0) - actual_vy
        err_v = -self.filtered_vy

        # Freeze integration in deadband
        in_deadband = torch.abs(err_v) < self.deadband
        err_int = torch.where(in_deadband, torch.zeros_like(err_v), err_v)

        # Anti-windup integration
        new_integral = torch.clamp(self.integral + err_int * self.dt, -self.i_max, self.i_max)
        self.integral = torch.where(is_active, new_integral, torch.zeros_like(self.integral))

        # PI output
        p_term = self.Kp * err_v
        i_term = self.Ki * self.integral
        trim_raw = p_term + i_term
        trim_clamped = torch.clamp(trim_raw, -self.trim_max, self.trim_max)

        # Only apply when active
        delta_v_trim = torch.where(is_active, trim_clamped, torch.zeros_like(trim_clamped))
        return delta_v_trim, is_active


def run_ab_evaluation():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    model_path = "legged_gym/logs/myrobot/Oct14_00-28-55_phase15_height_034_060_exploration/model_130.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False
    env.max_episode_length = 3500

    loaded = torch.load(model_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    pol = ac.act_inference

    rad_to_deg = 180.0 / np.pi
    dt = env.dt
    hold_duration_s = 20.0
    hold_steps = int(hold_duration_s / dt) # 1000 steps
    warmup_steps = 150

    test_heights = [0.340, 0.380, 0.420, 0.460, 0.500, 0.520]

    pi_controller = ParkingPIController(dt=dt, Kp=0.35, Ki=0.60, trim_max=0.045, deadband=0.003)
    pi_controller.init_buffers(args.num_envs, env.device)

    results_baseline = {}
    results_pi = {}

    print(f"\n================ STEP 4: A/B TESTING (OFF vs ON) ON model_130.pt ================")
    print(f"PI Params: Kp={pi_controller.Kp}, Ki={pi_controller.Ki}, trim_max={pi_controller.trim_max} m/s")
    print(f"Evaluating 20.0s hold across heights: {test_heights}\n")

    for mode in ["OFF", "ON"]:
        enable_pi = (mode == "ON")
        mode_records = {}
        print(f"--- Running Mode: Parking Compensation [{mode}] ---")

        for h_user in test_heights:
            torch.manual_seed(42)
            np.random.seed(42)
            env.reset()
            pi_controller.reset_env()

            delta_h = float(np.clip(0.08 * (h_user - 0.39774), -0.010, 0.010))
            h_ctrl_val = round(h_user + delta_h, 4)

            # Warmup phase: transition height gradually
            h_current = 0.39774
            step_dh = 0.04 * dt
            for _ in range(warmup_steps):
                if h_current < h_ctrl_val:
                    h_current = min(h_current + step_dh, h_ctrl_val)
                elif h_current > h_ctrl_val:
                    h_current = max(h_current - step_dh, h_ctrl_val)

                env.commands[:, 0] = 0.0
                env.commands[:, 1] = 0.0
                env.commands[:, 2] = 0.0
                env.height_cmd_buf[:] = h_current
                if hasattr(env, 'target_height_buf'):
                    env.target_height_buf[:] = h_ctrl_val

                env.compute_observations()
                obs = env.get_observations()
                with torch.no_grad():
                    actions = pol(obs)
                env.step(actions)

            # Start of 20s hold measurement
            start_y = env.root_states[:, 1].clone().cpu().numpy()
            vy_history = []
            pitch_history = []
            actual_h_history = []
            wheel_tau_history = []
            trim_history = []

            for step in range(hold_steps):
                actual_vy_tensor = env.base_lin_vel[:, 1]
                fn_L = env.contact_forces[:, env.feet_indices[0], 2]
                fn_R = env.contact_forces[:, env.feet_indices[1], 2]
                is_grounded = (fn_L > 10.0) & (fn_R > 10.0)

                user_cmd_vy = torch.zeros(args.num_envs, device=env.device)
                user_cmd_yaw = torch.zeros(args.num_envs, device=env.device)

                if enable_pi:
                    v_trim, is_act = pi_controller.compute_trim(
                        actual_vy=actual_vy_tensor,
                        user_cmd_vy=user_cmd_vy,
                        user_cmd_yaw=user_cmd_yaw,
                        is_elevating=False,
                        is_grounded=is_grounded
                    )
                else:
                    v_trim = torch.zeros_like(user_cmd_vy)

                # Feed effective command into env observations
                env.commands[:, 0] = 0.0
                env.commands[:, 1] = user_cmd_vy + v_trim
                env.commands[:, 2] = user_cmd_yaw
                env.height_cmd_buf[:] = h_ctrl_val
                if hasattr(env, 'target_height_buf'):
                    env.target_height_buf[:] = h_ctrl_val

                env.compute_observations()
                obs = env.get_observations()
                with torch.no_grad():
                    actions = pol(obs)
                env.step(actions)

                vy = env.base_lin_vel[:, 1].cpu().numpy()
                proj_gy = env.projected_gravity[:, 1].cpu().numpy()
                pitch_deg = np.arcsin(np.clip(proj_gy, -1.0, 1.0)) * rad_to_deg
                act_h = env.root_states[:, 2].cpu().numpy()
                torques = env.torques.cpu().numpy()
                wheel_tau = 0.5 * (np.abs(torques[:, env.left_wheel_idx]) + np.abs(torques[:, env.right_wheel_idx]))

                vy_history.append(vy)
                pitch_history.append(pitch_deg)
                actual_h_history.append(act_h)
                wheel_tau_history.append(wheel_tau)
                trim_history.append(v_trim.cpu().numpy())

            end_y = env.root_states[:, 1].clone().cpu().numpy()
            disp_y = end_y - start_y

            # Steady-state window: last 10 seconds (steps 500~1000)
            ss_vy = np.mean(vy_history[500:])
            ss_vy_std = np.std(vy_history[500:])
            mean_pitch = np.mean(pitch_history[500:])
            std_pitch = np.std(pitch_history[500:])
            mean_disp_y = np.mean(disp_y)
            mean_wheel_tau = np.mean(wheel_tau_history)
            max_wheel_tau = np.max(wheel_tau_history)
            mean_trim = np.mean(trim_history[500:])

            mode_records[f"H_{h_user:.3f}m"] = {
                "target_H": h_user,
                "disp_20s_y_m": float(mean_disp_y),
                "steady_vy_mps": float(ss_vy),
                "steady_vy_std": float(ss_vy_std),
                "pitch_mean_deg": float(mean_pitch),
                "pitch_std_deg": float(std_pitch),
                "mean_wheel_tau_Nm": float(mean_wheel_tau),
                "max_wheel_tau_Nm": float(max_wheel_tau),
                "mean_trim_mps": float(mean_trim),
            }

            print(f"[{mode}] H = {h_user:.3f}m | 20s位移ΔY: {mean_disp_y:+.4f}m | 稳态vy: {ss_vy:+.4f} m/s (std: {ss_vy_std:.4f}) | 俯仰角: {mean_pitch:+.2f}° (std: {std_pitch:.2f}°) | 补偿trim: {mean_trim:+.4f} m/s")

        if enable_pi:
            results_pi = mode_records
        else:
            results_baseline = mode_records

    print(f"\n================ DYNAMIC TRANSITION TEST (Start-Stop-Restart) ================")
    print(f"Scenario at H = 0.34m: 5s Stop (PI ON) -> 4s Cruise @ +0.30 m/s (PI OFF) -> 5s Stop (PI ON)")

    torch.manual_seed(42)
    np.random.seed(42)
    env.reset()
    pi_controller.reset_env()

    h_target = 0.340
    delta_h = float(np.clip(0.08 * (h_target - 0.39774), -0.010, 0.010))
    h_ctrl_val = round(h_target + delta_h, 4)

    # Warmup
    h_cur = 0.39774
    for _ in range(warmup_steps):
        h_cur = max(h_cur - 0.04 * dt, h_ctrl_val)
        env.commands[:, :3] = 0.0
        env.height_cmd_buf[:] = h_cur
        env.compute_observations()
        obs = env.get_observations()
        with torch.no_grad():
            actions = pol(obs)
        env.step(actions)

    phase_transitions = [
        ("Phase 1: 5s Initial Stop (PI Active)", 250, 0.0),
        ("Phase 2: 4s Forward Cruise (+0.30 m/s, PI Inactive)", 200, 0.30),
        ("Phase 3: 5s Final Stop & Re-lock (PI Active)", 250, 0.0),
    ]

    transition_log = {}

    for phase_name, steps, target_vy in phase_transitions:
        v_list = []
        trim_list = []
        tau_list = []
        pitch_list = []

        for _ in range(steps):
            actual_vy_tensor = env.base_lin_vel[:, 1]
            fn_L = env.contact_forces[:, env.feet_indices[0], 2]
            fn_R = env.contact_forces[:, env.feet_indices[1], 2]
            is_grounded = (fn_L > 10.0) & (fn_R > 10.0)

            user_cmd_vy = torch.full((args.num_envs,), target_vy, device=env.device)
            user_cmd_yaw = torch.zeros(args.num_envs, device=env.device)

            v_trim, is_act = pi_controller.compute_trim(
                actual_vy=actual_vy_tensor,
                user_cmd_vy=user_cmd_vy,
                user_cmd_yaw=user_cmd_yaw,
                is_elevating=False,
                is_grounded=is_grounded
            )

            env.commands[:, 0] = 0.0
            env.commands[:, 1] = user_cmd_vy + v_trim
            env.commands[:, 2] = user_cmd_yaw
            env.height_cmd_buf[:] = h_ctrl_val

            env.compute_observations()
            obs = env.get_observations()
            with torch.no_grad():
                actions = pol(obs)
            env.step(actions)

            v_list.append(env.base_lin_vel[:, 1].cpu().numpy())
            trim_list.append(v_trim.cpu().numpy())
            torques = env.torques.cpu().numpy()
            tau_list.append(0.5 * (np.abs(torques[:, env.left_wheel_idx]) + np.abs(torques[:, env.right_wheel_idx])))
            proj_gy = env.projected_gravity[:, 1].cpu().numpy()
            pitch_list.append(np.arcsin(np.clip(proj_gy, -1.0, 1.0)) * rad_to_deg)

        mean_v = float(np.mean(v_list[int(len(v_list)*0.5):]))
        mean_trim = float(np.mean(trim_list[int(len(trim_list)*0.5):]))
        max_tau = float(np.max(tau_list))
        pitch_rms = float(np.std(pitch_list))

        transition_log[phase_name] = {
            "target_vy": target_vy,
            "actual_steady_vy": mean_v,
            "mean_trim": mean_trim,
            "max_wheel_tau": max_tau,
            "pitch_rms_deg": pitch_rms,
        }
        print(f"{phase_name}: 稳态线速: {mean_v:+.4f} m/s | 补偿trim: {mean_trim:+.4f} m/s | 最大轮扭: {max_tau:.2f} Nm | 俯仰RMS: {pitch_rms:.2f}°")

    out_file = "evaluation_assets/phase15/parking_pi_ab_validation.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "results_baseline_off": results_baseline,
            "results_pi_on": results_pi,
            "transition_test": transition_log,
        }, f, indent=2)
    print(f"\nA/B Validation complete! Results saved to: {out_file}")


if __name__ == '__main__':
    run_ab_evaluation()
