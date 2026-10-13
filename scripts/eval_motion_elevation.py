"""
Phase 14: In-Motion Elevation Benchmark Evaluation Script (Phase 3)
Evaluates candidate policy models across all 12 combinations of moving and turning while elevating:
Straight + Height:
  1. fwd_straight_squat (vy=+0.20, wz=0, H=0.32m)
  2. fwd_straight_rise  (vy=+0.20, wz=0, H=0.48m)
  3. bwd_straight_squat (vy=-0.20, wz=0, H=0.32m)
  4. bwd_straight_rise  (vy=-0.20, wz=0, H=0.48m)
Turning + Height:
  5. fwd_left_squat     (vy=+0.15, wz=+0.25, H=0.32m)
  6. fwd_right_squat    (vy=+0.15, wz=-0.25, H=0.32m)
  7. fwd_left_rise      (vy=+0.15, wz=+0.25, H=0.48m)
  8. fwd_right_rise     (vy=+0.15, wz=-0.25, H=0.48m)
  9. bwd_left_squat     (vy=-0.15, wz=+0.25, H=0.32m)
 10. bwd_right_squat    (vy=-0.15, wz=-0.25, H=0.32m)
 11. bwd_left_rise      (vy=-0.15, wz=+0.25, H=0.48m)
 12. bwd_right_rise     (vy=-0.15, wz=-0.25, H=0.48m)
Extreme Boundary Height Exploration:
 13. fwd_lowspeed_squat_030m (vy=+0.10, wz=0, H=0.30m)
 14. bwd_lowspeed_squat_030m (vy=-0.10, wz=0, H=0.30m)
 15. turn_lowspeed_squat_030m (vy=+0.10, wz=+0.15, H=0.30m)
 16. fwd_lowspeed_rise_050m  (vy=+0.10, wz=0, H=0.50m)
 17. bwd_lowspeed_rise_050m  (vy=-0.10, wz=0, H=0.50m)
 18. turn_lowspeed_rise_050m  (vy=+0.10, wz=-0.15, H=0.50m)

Operational Completion Definition:
- Execute elevation and movement command.
- Brake to stop in the same environment and hold target height.
- Last 5.0 seconds (250 steps): height error <= 1cm, |vy| <= 0.03 m/s, |wz| <= 0.05 rad/s for >= 90% samples.
- Zero unexpected resets (100% survival).
- Moving-phase Joint% reported as diagnostic reference.
"""

import sys
import os
import json
import math

# isaacgym must be imported before torch
import isaacgym
from isaacgym import gymtorch
import numpy as np
import torch

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def compute_h_ctrl(H_target, enable_compensation=True):
    if not enable_compensation:
        return H_target
    delta_h = np.clip(0.08 * (H_target - 0.39774), -0.010, 0.010)
    return round(H_target + float(delta_h), 4)

def parse_args():
    model_path = "evaluation_assets/champion_v6/models/champion_v6_m115.pt"
    if not os.path.exists(model_path):
        model_path = "evaluation_assets/phase14/models/phase14_exp1_m115.pt"
    output_path = "evaluation_assets/phase14/results/eval_motion_elevation_v6_refined.json"
    enable_compensation = True
    
    cleaned_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--model_path" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output_path" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--no_compensation":
            enable_compensation = False
            i += 1
        else:
            cleaned_argv.append(sys.argv[i])
            i += 1
    sys.argv = cleaned_argv
    return model_path, output_path, enable_compensation

def run_motion_elevation_eval():
    model_path, output_path, enable_compensation = parse_args()
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False
    env.max_episode_length = 3000

    rad_to_deg = 180.0 / np.pi

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    print(f"Loading policy model from: {model_path}")
    loaded = torch.load(model_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    policy = ac.act_inference

    core_12_cases = [
        # (Case Name, cmd_vy, cmd_wz, target_H)
        ("1. fwd_straight_squat", +0.20,  0.00, 0.32),
        ("2. fwd_straight_rise",  +0.20,  0.00, 0.48),
        ("3. bwd_straight_squat", -0.20,  0.00, 0.32),
        ("4. bwd_straight_rise",  -0.20,  0.00, 0.48),
        ("5. fwd_left_squat",     +0.15, +0.25, 0.32),
        ("6. fwd_right_squat",    +0.15, -0.25, 0.32),
        ("7. fwd_left_rise",      +0.15, +0.25, 0.48),
        ("8. fwd_right_rise",     +0.15, -0.25, 0.48),
        ("9. bwd_left_squat",     -0.15, +0.25, 0.32),
        ("10. bwd_right_squat",   -0.15, -0.25, 0.32),
        ("11. bwd_left_rise",     -0.15, +0.25, 0.48),
        ("12. bwd_right_rise",    -0.15, -0.25, 0.48),
    ]

    boundary_cases = [
        ("13. fwd_straight_squat_030m", +0.10,  0.00, 0.30),
        ("14. bwd_straight_squat_030m", -0.10,  0.00, 0.30),
        ("15. fwd_turn_squat_030m",     +0.10, +0.15, 0.30),
        ("16. fwd_straight_rise_050m",  +0.10,  0.00, 0.50),
        ("17. bwd_straight_rise_050m",  -0.10,  0.00, 0.50),
        ("18. fwd_turn_rise_050m",      +0.10, -0.15, 0.50),
    ]

    all_cases = core_12_cases + boundary_cases

    step_cmd_start = 50     # 1.0s settle at nominal
    step_motion_end = 350   # 7.0s: motion phase complete
    step_brake_end = 450    # 9.0s: smooth brake + transition settle (2.0s = 100 steps)
    total_steps = 700       # 14.0s: steps 450~700 is exactly 5.0s (250 steps) steady parking hold

    steady_motion_start = 180 # step 180~350 is steady motion evaluation

    tol_h = 0.010   # 1.0 cm
    tol_vy = 0.050  # 0.05 m/s
    tol_wz = 0.080  # 0.08 rad/s

    case_results = {}
    print(f"\n================ Refined In-Motion Elevation Benchmark (12 Core + 6 Boundary) ================")
    print(f"{'Case Name':<28} | {'Vy':>5} | {'Wz':>5} | {'H':>4} | {'MotJnt%':>7} | {'ParkJnt%':>8} | {'ParkH%':>6} | {'ParkVy%':>7} | {'ParkWz%':>7} | {'SignVy':>7} | {'StdVy':>5} | {'Disp':>6} | {'Surv%':>6} | {'Done':>5}")
    print("-" * 145)

    all_jnt_pass = []
    all_env_pass = []
    all_surv = []
    all_park_pass = []
    all_park_h_pass = []
    all_park_vy_pass = []
    all_park_wz_pass = []
    all_action_completed = []

    for case_name, cmd_vy, cmd_wz, target_H in all_cases:
        torch.manual_seed(42)
        np.random.seed(42)
        env.reset()

        h_ctrl_target = compute_h_ctrl(target_H, enable_compensation)
        h_current = 0.39774
        c_vy = 0.0
        c_wz = 0.0

        elevation_speed = 0.04 # 4 cm/s
        step_dh = elevation_speed * env.dt
        accel_y = 0.8 * env.dt # 0.8 m/s^2 smooth linear ramp (matching teleop)
        accel_w = 2.0 * env.dt # 2.0 rad/s^2 smooth yaw ramp (matching teleop)

        steady_actual_h = []
        steady_errors_h = []
        steady_errors_vy = []
        steady_errors_wz = []
        steady_jnt_ok = []

        parking_err_h = []
        parking_err_vy = []
        parking_err_wz = []
        parking_raw_vy = []
        parking_ok_h = []
        parking_ok_vy = []
        parking_ok_wz = []
        parking_ok_jnt = []

        done_env_flags = np.zeros(args.num_envs, dtype=bool)

        for step in range(total_steps):
            if step < step_cmd_start:
                # Stage 1: Nominal settle (0.0s ~ 1.0s)
                tgt_vy = 0.0
                tgt_wz = 0.0
                tgt_h = 0.39774
            elif step < step_motion_end:
                # Stage 2: Moving elevation (1.0s ~ 7.0s)
                tgt_vy = cmd_vy
                tgt_wz = cmd_wz
                tgt_h = h_ctrl_target
            else:
                # Stage 3 & 4: Smooth brake to stop and hold target height (7.0s ~ 14.0s)
                tgt_vy = 0.0
                tgt_wz = 0.0
                tgt_h = h_ctrl_target

            # Smooth ramps (aligned with teleop and 60s sequential test)
            if c_vy < tgt_vy:
                c_vy = min(c_vy + accel_y, tgt_vy)
            elif c_vy > tgt_vy:
                c_vy = max(c_vy - accel_y, tgt_vy)

            if c_wz < tgt_wz:
                c_wz = min(c_wz + accel_w, tgt_wz)
            elif c_wz > tgt_wz:
                c_wz = max(c_wz - accel_w, tgt_wz)

            if h_current < tgt_h:
                h_current = min(h_current + step_dh, tgt_h)
            elif h_current > tgt_h:
                h_current = max(h_current - step_dh, tgt_h)

            env.commands[:, 0] = 0.0
            env.commands[:, 1] = c_vy
            env.commands[:, 2] = c_wz
            env.height_cmd_buf[:] = h_current
            if hasattr(env, 'target_height_buf'):
                env.target_height_buf[:] = h_ctrl_target

            env.compute_observations()
            obs = env.get_observations()

            with torch.no_grad():
                actions = policy(obs)

            obs, _, _, dones, infos = env.step(actions)
            d_arr = dones.cpu().numpy()
            done_env_flags |= d_arr

            base_h = env.root_states[:, 2].cpu().numpy()
            base_vy = env.base_lin_vel[:, 1].cpu().numpy()
            base_wz = env.base_ang_vel[:, 2].cpu().numpy()

            # Record steady motion (3.6s ~ 7.0s)
            if steady_motion_start <= step < step_motion_end:
                err_h = np.abs(base_h - target_H)
                err_vy = np.abs(base_vy - cmd_vy)
                err_wz = np.abs(base_wz - cmd_wz)
                is_jnt_ok = (err_h <= tol_h) & (err_vy <= tol_vy) & (err_wz <= tol_wz)

                steady_actual_h.append(base_h)
                steady_errors_h.append(err_h)
                steady_errors_vy.append(err_vy)
                steady_errors_wz.append(err_wz)
                steady_jnt_ok.append(is_jnt_ok)

            # Record steady parking hold (last 5.0 seconds = steps 450 to 700, 250 steps)
            if step >= step_brake_end:
                p_err_h = np.abs(base_h - target_H)
                p_err_vy = np.abs(base_vy)
                p_err_wz = np.abs(base_wz)

                p_ok_h = (p_err_h <= tol_h)
                p_ok_vy = (p_err_vy <= 0.030)
                p_ok_wz = (p_err_wz <= 0.050)
                p_ok_jnt = p_ok_h & p_ok_vy & p_ok_wz

                parking_err_h.append(p_err_h)
                parking_err_vy.append(p_err_vy)
                parking_err_wz.append(p_err_wz)
                parking_raw_vy.append(base_vy)
                parking_ok_h.append(p_ok_h)
                parking_ok_vy.append(p_ok_vy)
                parking_ok_wz.append(p_ok_wz)
                parking_ok_jnt.append(p_ok_jnt)

        survival_rate = float(1.0 - np.mean(done_env_flags)) * 100.0

        # Motion metrics
        arr_h = np.array(steady_actual_h)
        arr_err_h = np.array(steady_errors_h)
        arr_err_vy = np.array(steady_errors_vy)
        arr_err_wz = np.array(steady_errors_wz)
        arr_jnt = np.array(steady_jnt_ok)

        mean_act_h = float(np.mean(arr_h))
        mean_err_h_cm = float(np.mean(arr_err_h) * 100.0)
        mean_err_vy = float(np.mean(arr_err_vy))
        mean_err_wz = float(np.mean(arr_err_wz))
        joint_pass_pct = float(np.mean(arr_jnt)) * 100.0
        env_pass_pct = float(np.mean(np.mean(arr_jnt, axis=0) >= 0.85)) * 100.0

        # Disaggregated parking retention metrics (operational completion)
        arr_park_h = np.array(parking_ok_h)     # (250, 32)
        arr_park_vy = np.array(parking_ok_vy)
        arr_park_wz = np.array(parking_ok_wz)
        arr_park_jnt = np.array(parking_ok_jnt)

        park_h_pct = float(np.mean(arr_park_h)) * 100.0
        park_vy_pct = float(np.mean(arr_park_vy)) * 100.0
        park_wz_pct = float(np.mean(arr_park_wz)) * 100.0
        park_jnt_pct = float(np.mean(arr_park_jnt)) * 100.0
        park_env_pct = float(np.mean(np.mean(arr_park_jnt, axis=0) >= 0.90)) * 100.0

        arr_p_err_h = np.array(parking_err_h) * 100.0 # cm
        arr_p_err_vy = np.array(parking_err_vy)
        arr_p_err_wz = np.array(parking_err_wz)
        arr_p_raw_vy = np.array(parking_raw_vy)

        mean_park_err_h_cm = float(np.mean(arr_p_err_h))
        mean_park_vy = float(np.mean(arr_p_err_vy))
        p95_park_vy = float(np.percentile(arr_p_err_vy, 95))
        mean_park_wz = float(np.mean(arr_p_err_wz))
        p95_park_wz = float(np.percentile(arr_p_err_wz, 95))

        # 有符号平均速度、波动及累计位移
        signed_mean_vy = float(np.mean(arr_p_raw_vy))
        std_vy = float(np.std(arr_p_raw_vy))
        net_disp_m = float(np.mean(np.sum(arr_p_raw_vy, axis=0) * env.dt))
        total_dist_m = float(np.mean(np.sum(np.abs(arr_p_raw_vy), axis=0) * env.dt))

        action_completed = bool(park_jnt_pct >= 90.0 and survival_rate >= 100.0)

        all_jnt_pass.append(joint_pass_pct)
        all_env_pass.append(env_pass_pct)
        all_surv.append(survival_rate)
        all_park_pass.append(park_jnt_pct)
        all_park_h_pass.append(park_h_pct)
        all_park_vy_pass.append(park_vy_pct)
        all_park_wz_pass.append(park_wz_pct)
        all_action_completed.append(action_completed)

        action_str = "YES" if action_completed else "NO"
        print(f"{case_name:<28} | {cmd_vy:+5.2f} | {cmd_wz:+5.2f} | {target_H:4.2f} | {joint_pass_pct:6.1f}% | {park_jnt_pct:7.1f}% | {park_h_pct:5.1f}% | {park_vy_pct:6.1f}% | {park_wz_pct:6.1f}% | {signed_mean_vy:+6.3f} | {std_vy:5.3f} | {net_disp_m:+5.2f}m | {survival_rate:5.1f}% | {action_str:>5}")

        case_results[case_name] = {
            "cmd_vy": cmd_vy,
            "cmd_wz": cmd_wz,
            "target_H": target_H,
            "mean_act_h": mean_act_h,
            "mean_err_h_cm": mean_err_h_cm,
            "mean_err_vy": mean_err_vy,
            "mean_err_wz": mean_err_wz,
            "joint_pass_pct": joint_pass_pct,
            "env_pass_pct": env_pass_pct,
            "parking_hold": {
                "joint_pass_pct": park_jnt_pct,
                "env_pass_pct": park_env_pct,
                "height_pass_pct": park_h_pct,
                "lin_vel_pass_pct": park_vy_pct,
                "yaw_rate_pass_pct": park_wz_pct,
                "mean_err_h_cm": mean_park_err_h_cm,
                "mean_abs_vy_mps": mean_park_vy,
                "signed_mean_vy_mps": signed_mean_vy,
                "std_vy_mps": std_vy,
                "net_displacement_m": net_disp_m,
                "total_distance_m": total_dist_m,
                "p95_abs_vy_mps": p95_park_vy,
                "mean_abs_wz_radps": mean_park_wz,
                "p95_abs_wz_radps": p95_park_wz
            },
            "survival_rate": survival_rate,
            "action_completed": action_completed
        }

    # Split core 12 vs 6 boundary
    core_jnt = float(np.mean(all_jnt_pass[:12]))
    core_env = float(np.mean(all_env_pass[:12]))
    core_surv = float(np.mean(all_surv[:12]))
    core_park_jnt = float(np.mean(all_park_pass[:12]))
    core_park_h = float(np.mean(all_park_h_pass[:12]))
    core_park_vy = float(np.mean(all_park_vy_pass[:12]))
    core_park_wz = float(np.mean(all_park_wz_pass[:12]))
    core_completed_count = int(np.sum(all_action_completed[:12]))

    boundary_jnt = float(np.mean(all_jnt_pass[12:]))
    boundary_surv = float(np.mean(all_surv[12:]))
    boundary_park_jnt = float(np.mean(all_park_pass[12:]))
    boundary_park_h = float(np.mean(all_park_h_pass[12:]))
    boundary_park_vy = float(np.mean(all_park_vy_pass[12:]))
    boundary_park_wz = float(np.mean(all_park_wz_pass[12:]))
    boundary_completed_count = int(np.sum(all_action_completed[12:]))

    print("-" * 125)
    print(f"Core 12 Combinations: Motion Joint%: {core_jnt:.2f}% | Env%: {core_env:.2f}% | Parking Joint%: {core_park_jnt:.2f}% (H:{core_park_h:.1f}%, Vy:{core_park_vy:.1f}%, Wz:{core_park_wz:.1f}%) | Completed: {core_completed_count}/12 | Survival: {core_surv:.1f}%")
    print(f"Boundary 6 Cases:     Motion Joint%: {boundary_jnt:.2f}% | Parking Joint%: {boundary_park_jnt:.2f}% (H:{boundary_park_h:.1f}%, Vy:{boundary_park_vy:.1f}%, Wz:{boundary_park_wz:.1f}%) | Completed: {boundary_completed_count}/6 | Survival: {boundary_surv:.1f}%")

    summary = {
        "model_path": model_path,
        "enable_compensation": enable_compensation,
        "core_12": {
            "overall_joint_pass_pct": core_jnt,
            "overall_env_pass_pct": core_env,
            "overall_parking_joint_pct": core_park_jnt,
            "overall_parking_height_pct": core_park_h,
            "overall_parking_vy_pct": core_park_vy,
            "overall_parking_wz_pct": core_park_wz,
            "actions_completed_count": core_completed_count,
            "total_actions": 12,
            "overall_survival_rate": core_surv
        },
        "boundary_6": {
            "overall_joint_pass_pct": boundary_jnt,
            "overall_parking_joint_pct": boundary_park_jnt,
            "overall_parking_height_pct": boundary_park_h,
            "overall_parking_vy_pct": boundary_park_vy,
            "overall_parking_wz_pct": boundary_park_wz,
            "actions_completed_count": boundary_completed_count,
            "total_actions": 6,
            "overall_survival_rate": boundary_surv
        },
        "case_results": case_results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRefined motion elevation report saved to: {output_path}")

if __name__ == '__main__':
    run_motion_elevation_eval()
