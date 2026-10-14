"""
Phase 15 Comprehensive Verification:
1. 5-Seed Steady Parking Verification (20s hold each, seeds 42~46) at H = [0.34, 0.40, 0.52] m.
   - Compares [OFF] vs [ON].
   - Reports True Single-Motor Peak Torque (Left & Right separate) and Torque Saturation Rate.
2. 60-Second Long-Duration Continuous Multi-Task Mission (5 seeds, 42~46):
   - Sequence: Squat to 0.34m -> Cruise +0.35m/s -> Brake & Rise to 0.52m -> Reverse -0.20m/s -> Brake to 0.40m -> Turn +0.25rad/s -> Final Park.
   - Evaluates survival rate, dynamic transition stability, parking re-locking accuracy, single-motor peaks and saturation rate.
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
    驻车速度闭环 PI 补偿器
    """
    def __init__(self, dt=0.02, Kp=0.35, Ki=0.60, trim_max=0.045, deadband=0.003):
        self.dt = dt
        self.Kp = Kp
        self.Ki = Ki
        self.trim_max = trim_max
        self.deadband = deadband
        self.integral = None
        self.filtered_vy = None
        self.park_settle_counter = None
        self.alpha_lpf = 0.15
        self.i_max = self.trim_max / max(1e-4, self.Ki)

    def init_buffers(self, num_envs, device):
        self.integral = torch.zeros(num_envs, device=device, dtype=torch.float)
        self.filtered_vy = torch.zeros(num_envs, device=device, dtype=torch.float)
        self.park_settle_counter = torch.zeros(num_envs, device=device, dtype=torch.long)

    def reset_env(self, env_ids=None):
        if self.integral is None:
            return
        if env_ids is None:
            self.integral.zero_()
            self.filtered_vy.zero_()
            self.park_settle_counter.zero_()
        else:
            self.integral[env_ids] = 0.0
            self.filtered_vy[env_ids] = 0.0
            self.park_settle_counter[env_ids] = 0

    def compute_trim(self, actual_vy, user_cmd_vy, user_cmd_yaw, is_elevating=False, is_grounded=True):
        self.filtered_vy = (1.0 - self.alpha_lpf) * self.filtered_vy + self.alpha_lpf * actual_vy
        is_zero_cmd = (torch.abs(user_cmd_vy) < 0.005) & (torch.abs(user_cmd_yaw) < 0.010)
        is_slow = torch.abs(self.filtered_vy) < 0.12

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

        self.park_settle_counter = torch.where(
            can_engage,
            self.park_settle_counter + 1,
            torch.zeros_like(self.park_settle_counter)
        )
        is_active = (self.park_settle_counter >= 15).bool()

        err_v = -self.filtered_vy
        in_deadband = torch.abs(err_v) < self.deadband
        err_int = torch.where(in_deadband, torch.zeros_like(err_v), err_v)

        new_integral = torch.clamp(self.integral + err_int * self.dt, -self.i_max, self.i_max)
        self.integral = torch.where(is_active, new_integral, torch.zeros_like(self.integral))

        p_term = self.Kp * err_v
        i_term = self.Ki * self.integral
        trim_raw = p_term + i_term
        trim_clamped = torch.clamp(trim_raw, -self.trim_max, self.trim_max)

        delta_v_trim = torch.where(is_active, trim_clamped, torch.zeros_like(trim_clamped))
        return delta_v_trim, is_active


def run_5seeds_evaluation():
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
    env.max_episode_length = 4000

    loaded = torch.load(model_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    pol = ac.act_inference

    rad_to_deg = 180.0 / np.pi
    dt = env.dt
    seeds = [42, 43, 44, 45, 46]
    test_heights = [0.340, 0.400, 0.520]
    hold_steps = int(20.0 / dt) # 1000 steps
    warmup_steps = 150

    pi_controller = ParkingPIController(dt=dt, Kp=0.35, Ki=0.60, trim_max=0.045, deadband=0.003)
    pi_controller.init_buffers(args.num_envs, env.device)

    print(f"\n================ PART 1: 5-SEED STEADY PARKING EVALUATION (20s) ================")
    print(f"Seeds: {seeds} | Heights: {test_heights} | Model: {model_path}\n")

    part1_results = {"OFF": {}, "ON": {}}

    for mode in ["OFF", "ON"]:
        enable_pi = (mode == "ON")
        part1_results[mode] = {}
        print(f"--- Testing Mode: [{mode}] ---")

        for h_user in test_heights:
            seed_disp = []
            seed_steady_vy = []
            seed_vy_jitter = []
            seed_pitch_mean = []
            seed_pitch_std = []
            seed_peak_tau_L = []
            seed_peak_tau_R = []
            seed_peak_single_motor = []
            seed_sat_steps = 0
            seed_total_steps = 0
            seed_survival = []

            delta_h = float(np.clip(0.08 * (h_user - 0.39774), -0.010, 0.010))
            h_ctrl_val = round(h_user + delta_h, 4)

            for s in seeds:
                torch.manual_seed(s)
                np.random.seed(s)
                env.reset()
                pi_controller.reset_env()

                # Warmup
                h_cur = 0.39774
                for _ in range(warmup_steps):
                    if h_cur < h_ctrl_val:
                        h_cur = min(h_cur + 0.04 * dt, h_ctrl_val)
                    elif h_cur > h_ctrl_val:
                        h_cur = max(h_cur - 0.04 * dt, h_ctrl_val)
                    env.commands[:, :3] = 0.0
                    env.height_cmd_buf[:] = h_cur
                    env.compute_observations()
                    obs = env.get_observations()
                    with torch.no_grad():
                        actions = pol(obs)
                    env.step(actions)

                start_y = env.root_states[:, 1].clone().cpu().numpy()
                vy_list = []
                pitch_list = []
                tau_L_list = []
                tau_R_list = []
                has_reset = False

                for _ in range(hold_steps):
                    fn_L = env.contact_forces[:, env.feet_indices[0], 2]
                    fn_R = env.contact_forces[:, env.feet_indices[1], 2]
                    is_grounded = (fn_L > 10.0) & (fn_R > 10.0)

                    user_cmd_vy = torch.zeros(args.num_envs, device=env.device)
                    user_cmd_yaw = torch.zeros(args.num_envs, device=env.device)

                    if enable_pi:
                        v_trim, is_act = pi_controller.compute_trim(
                            actual_vy=env.base_lin_vel[:, 1],
                            user_cmd_vy=user_cmd_vy,
                            user_cmd_yaw=user_cmd_yaw,
                            is_elevating=False,
                            is_grounded=is_grounded
                        )
                    else:
                        v_trim = torch.zeros_like(user_cmd_vy)

                    env.commands[:, 0] = 0.0
                    env.commands[:, 1] = user_cmd_vy + v_trim
                    env.commands[:, 2] = user_cmd_yaw
                    env.height_cmd_buf[:] = h_ctrl_val

                    env.compute_observations()
                    obs = env.get_observations()
                    with torch.no_grad():
                        actions = pol(obs)
                    obs, _, _, dones, _ = env.step(actions)

                    # Auto-reset cleanup
                    reset_mask = dones.bool()
                    if torch.any(reset_mask):
                        has_reset = True
                        pi_controller.reset_env(reset_mask.nonzero(as_tuple=False).squeeze(-1))

                    vy = env.base_lin_vel[:, 1].cpu().numpy()
                    proj_gy = env.projected_gravity[:, 1].cpu().numpy()
                    pitch_deg = np.arcsin(np.clip(proj_gy, -1.0, 1.0)) * rad_to_deg
                    torques = env.torques.cpu().numpy()

                    tau_L = np.abs(torques[:, env.left_wheel_idx])
                    tau_R = np.abs(torques[:, env.right_wheel_idx])

                    vy_list.append(vy)
                    pitch_list.append(pitch_deg)
                    tau_L_list.append(tau_L)
                    tau_R_list.append(tau_R)

                    # Saturation check (threshold: 6.99 Nm on 7.0 Nm limit)
                    sat = (tau_L >= 6.99) | (tau_R >= 6.99)
                    seed_sat_steps += int(np.sum(sat))
                    seed_total_steps += args.num_envs

                end_y = env.root_states[:, 1].clone().cpu().numpy()
                disp_y = np.mean(end_y - start_y)

                # Steady window: last 10 seconds
                ss_vy = np.mean(vy_list[500:])
                ss_vy_jitter = np.std(vy_list[500:])
                p_mean = np.mean(pitch_list[500:])
                p_std = np.std(pitch_list[500:])

                peak_L = float(np.max(tau_L_list))
                peak_R = float(np.max(tau_R_list))
                peak_single = max(peak_L, peak_R)

                seed_disp.append(disp_y)
                seed_steady_vy.append(ss_vy)
                seed_vy_jitter.append(ss_vy_jitter)
                seed_pitch_mean.append(p_mean)
                seed_pitch_std.append(p_std)
                seed_peak_tau_L.append(peak_L)
                seed_peak_tau_R.append(peak_R)
                seed_peak_single_motor.append(peak_single)
                seed_survival.append(0.0 if has_reset else 1.0)

            sat_rate = float(seed_sat_steps / max(1, seed_total_steps))
            summary_h = {
                "height": h_user,
                "disp_20s_mean_m": float(np.mean(seed_disp)),
                "disp_20s_std_m": float(np.std(seed_disp)),
                "steady_vy_mean_mps": float(np.mean(seed_steady_vy)),
                "steady_vy_std_mps": float(np.std(seed_steady_vy)),
                "instantaneous_vy_jitter_mps": float(np.mean(seed_vy_jitter)),
                "pitch_mean_deg": float(np.mean(seed_pitch_mean)),
                "pitch_std_deg": float(np.mean(seed_pitch_std)),
                "single_motor_peak_L_Nm": float(np.max(seed_peak_tau_L)),
                "single_motor_peak_R_Nm": float(np.max(seed_peak_tau_R)),
                "overall_single_motor_peak_Nm": float(np.max(seed_peak_single_motor)),
                "wheel_torque_sat_rate": sat_rate,
                "survival_rate": float(np.mean(seed_survival)),
            }
            part1_results[mode][f"H_{h_user:.3f}m"] = summary_h

            print(f"[{mode}] H = {h_user:.3f}m (5 seeds): 20s位移={summary_h['disp_20s_mean_m']:+.4f}±{summary_h['disp_20s_std_m']:.4f}m | "
                  f"稳态vy={summary_h['steady_vy_mean_mps']:+.4f} m/s (瞬时波动std: {summary_h['instantaneous_vy_jitter_mps']:.4f}) | "
                  f"单电机峰值={summary_h['overall_single_motor_peak_Nm']:.2f} Nm (L:{summary_h['single_motor_peak_L_Nm']:.2f}, R:{summary_h['single_motor_peak_R_Nm']:.2f}) | "
                  f"饱和率={sat_rate*100:.2f}% | 存活率={summary_h['survival_rate']*100:.0f}%")

    print(f"\n================ PART 2: 60-SECOND CONTINUOUS MISSION (5 SEEDS) ================")
    print(f"Sequence: Squat(0.34m) -> Cruise(+0.35m/s) -> Rise(0.52m) & Park -> Reverse(-0.20m/s) -> Park(0.40m) -> Turn(+0.25rad/s) -> Park")

    # 60-second mission schedule (total 3000 steps at dt=0.02s)
    # 6 phases of 500 steps (10.0s each)
    mission_phases = [
        {"name": "Phase 1: Squat to 0.34m & Park Hold", "target_h": 0.340, "vy": 0.0, "yaw": 0.0, "is_park": True},
        {"name": "Phase 2: Forward Cruise @ +0.35 m/s", "target_h": 0.340, "vy": 0.35, "yaw": 0.0, "is_park": False},
        {"name": "Phase 3: Brake & Elevate to 0.52m & Park", "target_h": 0.520, "vy": 0.0, "yaw": 0.0, "is_park": True},
        {"name": "Phase 4: Backward Reverse @ -0.20 m/s", "target_h": 0.520, "vy": -0.20, "yaw": 0.0, "is_park": False},
        {"name": "Phase 5: Brake to 0.40m Nominal & Park", "target_h": 0.400, "vy": 0.0, "yaw": 0.0, "is_park": True},
        {"name": "Phase 6: Turn @ +0.25 rad/s & Final Park", "target_h": 0.400, "vy": 0.0, "yaw": 0.25, "is_park": False}, # will park last 5s
    ]

    mission_records = {}

    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        env.reset()
        pi_controller.reset_env()

        h_cmd_cur = 0.39774
        vy_cmd_cur = 0.0
        ang_cmd_cur = 0.0

        seed_mission_log = {"resets": 0, "phases": {}}

        for p_idx, phase in enumerate(mission_phases):
            p_name = phase["name"]
            tgt_h = phase["target_h"]
            tgt_vy = phase["vy"]
            tgt_yaw = phase["yaw"]

            delta_h = float(np.clip(0.08 * (tgt_h - 0.39774), -0.010, 0.010))
            h_ctrl_goal = round(tgt_h + delta_h, 4)

            p_vy_history = []
            p_pitch_history = []
            p_tau_L = []
            p_tau_R = []

            for step_in_phase in range(500):
                # For Phase 6: first 250 steps turn, last 250 steps park
                if p_idx == 5:
                    if step_in_phase < 250:
                        cur_target_yaw = 0.25
                        cur_target_vy = 0.0
                    else:
                        cur_target_yaw = 0.0
                        cur_target_vy = 0.0
                else:
                    cur_target_yaw = tgt_yaw
                    cur_target_vy = tgt_vy

                # Smooth velocity ramping
                d_vy = cur_target_vy - vy_cmd_cur
                max_step_vy = 0.8 * dt
                if abs(d_vy) <= max_step_vy:
                    vy_cmd_cur = cur_target_vy
                else:
                    vy_cmd_cur += math.copysign(max_step_vy, d_vy)

                # Smooth yaw ramping
                d_yaw = cur_target_yaw - ang_cmd_cur
                max_step_yaw = 2.0 * dt
                if abs(d_yaw) <= max_step_yaw:
                    ang_cmd_cur = cur_target_yaw
                else:
                    ang_cmd_cur += math.copysign(max_step_yaw, d_yaw)

                # Smooth height ramping
                d_h = h_ctrl_goal - h_cmd_cur
                max_step_h = 0.04 * dt
                if abs(d_h) <= max_step_h:
                    h_cmd_cur = h_ctrl_goal
                else:
                    h_cmd_cur += math.copysign(max_step_h, d_h)

                is_elevating = abs(h_ctrl_goal - h_cmd_cur) > 0.003
                fn_L = env.contact_forces[:, env.feet_indices[0], 2]
                fn_R = env.contact_forces[:, env.feet_indices[1], 2]
                is_grounded = (fn_L > 10.0) & (fn_R > 10.0)

                u_vy = torch.full((args.num_envs,), vy_cmd_cur, device=env.device)
                u_yaw = torch.full((args.num_envs,), ang_cmd_cur, device=env.device)

                v_trim, is_act = pi_controller.compute_trim(
                    actual_vy=env.base_lin_vel[:, 1],
                    user_cmd_vy=u_vy,
                    user_cmd_yaw=u_yaw,
                    is_elevating=is_elevating,
                    is_grounded=is_grounded
                )

                env.commands[:, 0] = 0.0
                env.commands[:, 1] = u_vy + v_trim
                env.commands[:, 2] = u_yaw
                env.height_cmd_buf[:] = h_cmd_cur

                env.compute_observations()
                obs = env.get_observations()
                with torch.no_grad():
                    actions = pol(obs)
                obs, _, _, dones, _ = env.step(actions)

                if torch.any(dones):
                    seed_mission_log["resets"] += int(torch.sum(dones.long()).item())
                    pi_controller.reset_env(dones.bool().nonzero(as_tuple=False).squeeze(-1))
                    vy_cmd_cur = 0.0
                    ang_cmd_cur = 0.0
                    h_cmd_cur = 0.39774

                vy = env.base_lin_vel[:, 1].cpu().numpy()
                proj_gy = env.projected_gravity[:, 1].cpu().numpy()
                pitch_deg = np.arcsin(np.clip(proj_gy, -1.0, 1.0)) * rad_to_deg
                torques = env.torques.cpu().numpy()

                p_vy_history.append(vy)
                p_pitch_history.append(pitch_deg)
                p_tau_L.append(np.abs(torques[:, env.left_wheel_idx]))
                p_tau_R.append(np.abs(torques[:, env.right_wheel_idx]))

            # Steady statistics for last 4 seconds of each phase (steps 300~500)
            ss_vy = float(np.mean(p_vy_history[300:]))
            ss_vy_jitter = float(np.std(p_vy_history[300:]))
            p_mean = float(np.mean(p_pitch_history[300:]))
            p_std = float(np.std(p_pitch_history[300:]))
            peak_L = float(np.max(p_tau_L))
            peak_R = float(np.max(p_tau_R))
            sat_count = int(np.sum((np.array(p_tau_L) >= 6.99) | (np.array(p_tau_R) >= 6.99)))

            seed_mission_log["phases"][p_name] = {
                "steady_vy": ss_vy,
                "steady_vy_jitter": ss_vy_jitter,
                "pitch_mean": p_mean,
                "pitch_std": p_std,
                "peak_L_Nm": peak_L,
                "peak_R_Nm": peak_R,
                "peak_single_motor_Nm": max(peak_L, peak_R),
                "sat_count": sat_count,
            }

        mission_records[f"seed_{s}"] = seed_mission_log

    # Compute overall mission 5-seed statistics
    total_resets = sum(mission_records[f"seed_{s}"]["resets"] for s in seeds)
    mission_survival_rate = 1.0 if total_resets == 0 else max(0.0, 1.0 - total_resets / (5 * args.num_envs))

    phase_summaries = {}
    for phase in mission_phases:
        p_name = phase["name"]
        p_vys = [mission_records[f"seed_{s}"]["phases"][p_name]["steady_vy"] for s in seeds]
        p_jitters = [mission_records[f"seed_{s}"]["phases"][p_name]["steady_vy_jitter"] for s in seeds]
        p_peaks = [mission_records[f"seed_{s}"]["phases"][p_name]["peak_single_motor_Nm"] for s in seeds]
        p_peaks_L = [mission_records[f"seed_{s}"]["phases"][p_name]["peak_L_Nm"] for s in seeds]
        p_peaks_R = [mission_records[f"seed_{s}"]["phases"][p_name]["peak_R_Nm"] for s in seeds]
        p_sats = [mission_records[f"seed_{s}"]["phases"][p_name]["sat_count"] for s in seeds]

        phase_summaries[p_name] = {
            "steady_vy_mean": float(np.mean(p_vys)),
            "steady_vy_std": float(np.std(p_vys)),
            "instantaneous_jitter": float(np.mean(p_jitters)),
            "peak_single_motor_Nm": float(np.max(p_peaks)),
            "peak_motor_L_Nm": float(np.max(p_peaks_L)),
            "peak_motor_R_Nm": float(np.max(p_peaks_R)),
            "total_sat_steps": int(sum(p_sats)),
            "sat_rate": float(sum(p_sats) / (5 * 500 * args.num_envs)),
        }
        print(f"{p_name}: 稳态线速={np.mean(p_vys):+.4f} m/s | 峰值单轮力矩={np.max(p_peaks):.2f} Nm (L:{np.max(p_peaks_L):.2f}, R:{np.max(p_peaks_R):.2f}) | 饱和率={phase_summaries[p_name]['sat_rate']*100:.3f}%")

    out_file = "evaluation_assets/phase15/parking_pi_5seeds_long_validation.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "part1_5seed_steady_parking": part1_results,
            "part2_60s_continuous_mission": {
                "survival_rate": mission_survival_rate,
                "total_resets": total_resets,
                "phase_summaries": phase_summaries,
                "per_seed_records": mission_records,
            }
        }, f, indent=2)
    print(f"\nComprehensive 5-Seed & Long Mission Validation saved to: {out_file}")


if __name__ == '__main__':
    run_5seeds_evaluation()
