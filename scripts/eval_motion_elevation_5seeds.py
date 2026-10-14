"""
Phase 14: 5-Seed In-Motion Elevation & Parking Verification Script
Evaluates candidate model across 5 seeds on all 18 cases (12 Core + 6 Boundary).

Focus Areas:
1. Post-rise height hold capability (0.48m and 0.50m).
2. Directional drift and crawl analysis at 0.30m and regular heights (signed Vy, net displacement vs total distance).
3. Degradation in moving phase Joint% (quantifying mobility loss).
4. Overall parking joint pass rate consistency and survival across 5 seeds.

Usage:
    python scripts/eval_motion_elevation_5seeds.py --model_path <path> --output_path <path> --num_seeds 5
"""

import sys
import os
import json
import math

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
    model_path = "legged_gym/logs/myrobot/Oct13_15-16-03_phase14_exp3_parking_focus_resume10iter/model_125.pt"
    output_path = "evaluation_assets/phase14/results/eval_motion_elevation_exp3_m125_5seeds.json"
    enable_compensation = True
    num_seeds = 5
    
    cleaned_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--model_path" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i].startswith("--model_path="):
            model_path = sys.argv[i].split("=", 1)[1]
            i += 1
        elif sys.argv[i] == "--output_path" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i].startswith("--output_path="):
            output_path = sys.argv[i].split("=", 1)[1]
            i += 1
        elif sys.argv[i] == "--num_seeds" and i + 1 < len(sys.argv):
            num_seeds = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i].startswith("--num_seeds="):
            num_seeds = int(sys.argv[i].split("=", 1)[1])
            i += 1
        elif sys.argv[i] == "--no_compensation":
            enable_compensation = False
            i += 1
        else:
            cleaned_argv.append(sys.argv[i])
            i += 1
    sys.argv = cleaned_argv
    return model_path, output_path, enable_compensation, num_seeds

def run_5seeds_motion_elevation():
    model_path, output_path, enable_compensation, num_seeds = parse_args()
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False
    env.max_episode_length = 3000

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    print(f"Loading policy model from: {model_path}")
    loaded = torch.load(model_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    policy = ac.act_inference

    core_12_cases = [
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

    print(f"\n================ Phase 14 5-Seed In-Motion Elevation & Parking Verification ================")
    print(f"Model: {model_path}")
    print(f"Num Seeds: {num_seeds} | Envs: {args.num_envs} | Compensation: {enable_compensation}")

    per_seed_data = []

    for seed_idx in range(1, num_seeds + 1):
        seed_val = seed_idx * 100 + 42
        case_records = {}

        print(f"\n--- Running Seed {seed_idx}/{num_seeds} (seed={seed_val}) ---")

        for case_name, cmd_vy, cmd_wz, target_H in all_cases:
            torch.manual_seed(seed_val)
            np.random.seed(seed_val)
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
                    tgt_vy = 0.0
                    tgt_wz = 0.0
                    tgt_h = 0.39774
                elif step < step_motion_end:
                    tgt_vy = cmd_vy
                    tgt_wz = cmd_wz
                    tgt_h = h_ctrl_target
                else:
                    tgt_vy = 0.0
                    tgt_wz = 0.0
                    tgt_h = h_ctrl_target

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
            joint_pass_pct = float(np.mean(arr_jnt) * 100.0)
            env_pass_pct = float(np.mean(np.all(arr_jnt, axis=0)) * 100.0)

            # Parking hold metrics
            p_arr_err_h = np.array(parking_err_h)
            p_arr_raw_vy = np.array(parking_raw_vy)
            p_arr_err_wz = np.array(parking_err_wz)

            p_arr_ok_h = np.array(parking_ok_h)
            p_arr_ok_vy = np.array(parking_ok_vy)
            p_arr_ok_wz = np.array(parking_ok_wz)
            p_arr_ok_jnt = np.array(parking_ok_jnt)

            p_joint_pass = float(np.mean(p_arr_ok_jnt) * 100.0)
            p_env_pass = float(np.mean(np.all(p_arr_ok_jnt, axis=0)) * 100.0)
            p_h_pass = float(np.mean(p_arr_ok_h) * 100.0)
            p_vy_pass = float(np.mean(p_arr_ok_vy) * 100.0)
            p_wz_pass = float(np.mean(p_arr_ok_wz) * 100.0)

            mean_p_h_err_cm = float(np.mean(p_arr_err_h) * 100.0)
            mean_p_abs_vy = float(np.mean(np.abs(p_arr_raw_vy)))
            signed_mean_vy = float(np.mean(p_arr_raw_vy))
            std_vy = float(np.std(p_arr_raw_vy))
            net_displacement = float(np.mean(np.sum(p_arr_raw_vy * env.dt, axis=0)))
            total_distance = float(np.mean(np.sum(np.abs(p_arr_raw_vy) * env.dt, axis=0)))

            action_completed = (survival_rate >= 99.9) and (p_joint_pass >= 90.0)

            case_records[case_name] = {
                "cmd_vy": cmd_vy,
                "cmd_wz": cmd_wz,
                "target_H": target_H,
                "motion": {
                    "mean_act_h": mean_act_h,
                    "mean_err_h_cm": mean_err_h_cm,
                    "mean_err_vy": mean_err_vy,
                    "mean_err_wz": mean_err_wz,
                    "joint_pass_pct": joint_pass_pct,
                    "env_pass_pct": env_pass_pct,
                },
                "parking": {
                    "joint_pass_pct": p_joint_pass,
                    "env_pass_pct": p_env_pass,
                    "height_pass_pct": p_h_pass,
                    "lin_vel_pass_pct": p_vy_pass,
                    "yaw_rate_pass_pct": p_wz_pass,
                    "mean_err_h_cm": mean_p_h_err_cm,
                    "signed_mean_vy_mps": signed_mean_vy,
                    "std_vy_mps": std_vy,
                    "net_displacement_m": net_displacement,
                    "total_distance_m": total_distance,
                },
                "survival_rate": survival_rate,
                "action_completed": action_completed
            }

        # Calculate seed summary
        core_records = [case_records[c[0]] for c in core_12_cases]
        bound_records = [case_records[c[0]] for c in boundary_cases]
        squat_records = [case_records[c[0]] for c in core_12_cases if "squat" in c[0]]
        rise_records = [case_records[c[0]] for c in core_12_cases if "rise" in c[0]]

        c12_mot_jnt = float(np.mean([r["motion"]["joint_pass_pct"] for r in core_records]))
        c12_park_jnt = float(np.mean([r["parking"]["joint_pass_pct"] for r in core_records]))
        c12_park_vy = float(np.mean([r["parking"]["lin_vel_pass_pct"] for r in core_records]))
        c12_park_h = float(np.mean([r["parking"]["height_pass_pct"] for r in core_records]))
        c12_done_count = sum(1 for r in core_records if r["action_completed"])
        c12_signed_vy = float(np.mean([r["parking"]["signed_mean_vy_mps"] for r in core_records]))
        c12_disp = float(np.mean([r["parking"]["net_displacement_m"] for r in core_records]))

        squat_park_jnt = float(np.mean([r["parking"]["joint_pass_pct"] for r in squat_records]))
        rise_park_jnt = float(np.mean([r["parking"]["joint_pass_pct"] for r in rise_records]))
        rise_park_h = float(np.mean([r["parking"]["height_pass_pct"] for r in rise_records]))
        rise_mot_jnt = float(np.mean([r["motion"]["joint_pass_pct"] for r in rise_records]))

        b6_park_jnt = float(np.mean([r["parking"]["joint_pass_pct"] for r in bound_records]))
        b6_mot_jnt = float(np.mean([r["motion"]["joint_pass_pct"] for r in bound_records]))

        # 0.30m squat cases (13, 14, 15)
        squat_030_records = [case_records[c[0]] for c in boundary_cases if "030m" in c[0]]
        squat_030_signed_vy = float(np.mean([r["parking"]["signed_mean_vy_mps"] for r in squat_030_records]))
        squat_030_disp = float(np.mean([r["parking"]["net_displacement_m"] for r in squat_030_records]))

        # 0.50m rise cases (16, 17, 18)
        rise_050_records = [case_records[c[0]] for c in boundary_cases if "050m" in c[0]]
        rise_050_signed_vy = float(np.mean([r["parking"]["signed_mean_vy_mps"] for r in rise_050_records]))
        rise_050_disp = float(np.mean([r["parking"]["net_displacement_m"] for r in rise_050_records]))
        rise_050_park_jnt = float(np.mean([r["parking"]["joint_pass_pct"] for r in rise_050_records]))

        seed_summary = {
            "seed_idx": seed_idx,
            "seed_val": seed_val,
            "core_12": {
                "motion_joint_pct": c12_mot_jnt,
                "parking_joint_pct": c12_park_jnt,
                "parking_vy_pct": c12_park_vy,
                "parking_h_pct": c12_park_h,
                "actions_completed": c12_done_count,
                "signed_vy_mps": c12_signed_vy,
                "net_displacement_m": c12_disp,
                "squat_parking_joint_pct": squat_park_jnt,
                "rise_parking_joint_pct": rise_park_jnt,
                "rise_parking_h_pct": rise_park_h,
                "rise_motion_joint_pct": rise_mot_jnt,
            },
            "boundary_6": {
                "motion_joint_pct": b6_mot_jnt,
                "parking_joint_pct": b6_park_jnt,
                "squat_030_signed_vy": squat_030_signed_vy,
                "squat_030_disp": squat_030_disp,
                "rise_050_signed_vy": rise_050_signed_vy,
                "rise_050_disp": rise_050_disp,
                "rise_050_parking_joint_pct": rise_050_park_jnt,
            },
            "case_results": case_records
        }
        per_seed_data.append(seed_summary)

        print(f"Seed {seed_idx} Result: Core12 ParkJoint={c12_park_jnt:.2f}% | Done={c12_done_count}/12 | MotJoint={c12_mot_jnt:.2f}% | Squat={squat_park_jnt:.2f}% | Rise={rise_park_jnt:.2f}% (RiseH={rise_park_h:.2f}%) | 0.30m Disp={squat_030_disp:.3f}m")

    # Overall 5-seed statistics
    c12_park_jnts = [s["core_12"]["parking_joint_pct"] for s in per_seed_data]
    c12_mot_jnts = [s["core_12"]["motion_joint_pct"] for s in per_seed_data]
    c12_dones = [s["core_12"]["actions_completed"] for s in per_seed_data]
    squat_park_jnts = [s["core_12"]["squat_parking_joint_pct"] for s in per_seed_data]
    rise_park_jnts = [s["core_12"]["rise_parking_joint_pct"] for s in per_seed_data]
    rise_park_hs = [s["core_12"]["rise_parking_h_pct"] for s in per_seed_data]
    c12_signed_vys = [s["core_12"]["signed_vy_mps"] for s in per_seed_data]
    c12_disps = [s["core_12"]["net_displacement_m"] for s in per_seed_data]

    squat_030_disps = [s["boundary_6"]["squat_030_disp"] for s in per_seed_data]
    squat_030_vys = [s["boundary_6"]["squat_030_signed_vy"] for s in per_seed_data]
    rise_050_jnts = [s["boundary_6"]["rise_050_parking_joint_pct"] for s in per_seed_data]

    overall_summary = {
        "num_seeds": num_seeds,
        "core_12": {
            "parking_joint_pct_mean": float(np.mean(c12_park_jnts)),
            "parking_joint_pct_std": float(np.std(c12_park_jnts)),
            "motion_joint_pct_mean": float(np.mean(c12_mot_jnts)),
            "motion_joint_pct_std": float(np.std(c12_mot_jnts)),
            "actions_completed_mean": float(np.mean(c12_dones)),
            "squat_parking_joint_pct_mean": float(np.mean(squat_park_jnts)),
            "rise_parking_joint_pct_mean": float(np.mean(rise_park_jnts)),
            "rise_parking_h_pct_mean": float(np.mean(rise_park_hs)),
            "signed_vy_mps_mean": float(np.mean(c12_signed_vys)),
            "net_displacement_m_mean": float(np.mean(c12_disps)),
        },
        "boundary_analysis": {
            "squat_030_disp_mean": float(np.mean(squat_030_disps)),
            "squat_030_signed_vy_mean": float(np.mean(squat_030_vys)),
            "rise_050_parking_joint_pct_mean": float(np.mean(rise_050_jnts)),
        }
    }

    full_output = {
        "model_path": model_path,
        "overall_summary": overall_summary,
        "per_seed_data": per_seed_data
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(full_output, f, indent=2)

    print(f"\n================ 5-Seed Cross-Validation Summary ================")
    print(f"Core 12 Parking Joint%:  {overall_summary['core_12']['parking_joint_pct_mean']:.2f}% ± {overall_summary['core_12']['parking_joint_pct_std']:.2f}%")
    print(f"Core 12 Motion Joint%:   {overall_summary['core_12']['motion_joint_pct_mean']:.2f}% ± {overall_summary['core_12']['motion_joint_pct_std']:.2f}%")
    print(f"Actions Completed:       {overall_summary['core_12']['actions_completed_mean']:.1f} / 12")
    print(f"Squat Parking Joint%:    {overall_summary['core_12']['squat_parking_joint_pct_mean']:.2f}%")
    print(f"Rise Parking Joint%:     {overall_summary['core_12']['rise_parking_joint_pct_mean']:.2f}% (Height Pass: {overall_summary['core_12']['rise_parking_h_pct_mean']:.2f}%)")
    print(f"Regular 5s Net Disp:     {overall_summary['core_12']['net_displacement_m_mean']:.3f} m (Vy={overall_summary['core_12']['signed_vy_mps_mean']:.3f} m/s)")
    print(f"0.30m 5s Net Disp:       {overall_summary['boundary_analysis']['squat_030_disp_mean']:.3f} m (Vy={overall_summary['boundary_analysis']['squat_030_signed_vy_mean']:.3f} m/s)")
    print(f"0.50m Rise Parking%:     {overall_summary['boundary_analysis']['rise_050_parking_joint_pct_mean']:.2f}%")
    print(f"Full report saved to: {output_path}")

if __name__ == "__main__":
    run_5seeds_motion_elevation()
