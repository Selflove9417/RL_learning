"""
Step 1: Scan Elevation Height Compensation for model_125.pt
Scans feedforward height compensation offsets on rise cases (H=0.48m and H=0.50m).
Records signed height error (base_h - target_H) and checks if parking velocity deteriorates.

Usage:
    python scripts/scan_rise_height_compensation.py --model_path <path> --output_path <path>
"""

import sys
import os
import json
import isaacgym
from isaacgym import gymtorch
import numpy as np
import torch

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def parse_args():
    model_path = "legged_gym/logs/myrobot/Oct13_15-16-03_phase14_exp3_parking_focus_resume10iter/model_125.pt"
    output_path = "evaluation_assets/phase14/results/scan_rise_compensation_m125.json"
    
    cleaned_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--model_path" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output_path" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        else:
            cleaned_argv.append(sys.argv[i])
            i += 1
    sys.argv = cleaned_argv
    return model_path, output_path

def run_compensation_scan():
    model_path, output_path = parse_args()
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

    # 6 Core Rise Cases (target_H = 0.48m)
    rise_cases_048 = [
        ("2. fwd_straight_rise",  +0.20,  0.00, 0.48),
        ("4. bwd_straight_rise",  -0.20,  0.00, 0.48),
        ("7. fwd_left_rise",      +0.15, +0.25, 0.48),
        ("8. fwd_right_rise",     +0.15, -0.25, 0.48),
        ("11. bwd_left_rise",     -0.15, +0.25, 0.48),
        ("12. bwd_right_rise",    -0.15, -0.25, 0.48),
    ]

    # 3 Boundary Rise Cases (target_H = 0.50m)
    rise_cases_050 = [
        ("16. fwd_straight_rise_050m", +0.10,  0.00, 0.50),
        ("17. bwd_straight_rise_050m", -0.10,  0.00, 0.50),
        ("18. fwd_turn_rise_050m",      +0.10, -0.15, 0.50),
    ]

    # Test candidate delta_H offsets for 0.48m: [0.0mm, +4.0mm, +6.6mm (current), +10.0mm, +12.0mm, +14.0mm]
    delta_h_candidates_048 = [0.000, 0.004, 0.0066, 0.010, 0.012, 0.014]

    step_cmd_start = 50     # 1.0s settle
    step_motion_end = 350   # 7.0s motion end
    step_brake_end = 450    # 9.0s brake end
    total_steps = 700       # 14.0s (steps 450~700: 5.0s steady parking)

    tol_h = 0.010   # 1.0 cm
    tol_vy = 0.030  # 0.03 m/s
    tol_wz = 0.050  # 0.05 rad/s

    print(f"\n================ Step 1: Scan Elevation Height Compensation (model_125.pt) ================")
    print(f"Scanning delta_H for H=0.48m: {delta_h_candidates_048}")

    results_scan = []

    for delta_h in delta_h_candidates_048:
        h_ctrl_val = round(0.48 + delta_h, 4)
        case_summaries = []

        for case_name, cmd_vy, cmd_wz, target_H in rise_cases_048:
            torch.manual_seed(42)
            np.random.seed(42)
            env.reset()

            h_current = 0.39774
            c_vy = 0.0
            c_wz = 0.0

            elevation_speed = 0.04
            step_dh = elevation_speed * env.dt
            accel_y = 0.8 * env.dt
            accel_w = 2.0 * env.dt

            p_h_errors_signed = []
            p_vy_raw = []
            p_ok_jnt = []
            p_ok_h = []
            p_ok_vy = []

            done_flags = np.zeros(args.num_envs, dtype=bool)

            for step in range(total_steps):
                if step < step_cmd_start:
                    tgt_vy, tgt_wz, tgt_h = 0.0, 0.0, 0.39774
                elif step < step_motion_end:
                    tgt_vy, tgt_wz, tgt_h = cmd_vy, cmd_wz, h_ctrl_val
                else:
                    tgt_vy, tgt_wz, tgt_h = 0.0, 0.0, h_ctrl_val

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
                    env.target_height_buf[:] = h_ctrl_val

                env.compute_observations()
                obs = env.get_observations()

                with torch.no_grad():
                    actions = policy(obs)

                obs, _, _, dones, infos = env.step(actions)
                done_flags |= dones.cpu().numpy()

                base_h = env.root_states[:, 2].cpu().numpy()
                base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                base_wz = env.base_ang_vel[:, 2].cpu().numpy()

                if step >= step_brake_end:
                    # Signed height error: base_h - target_H (0.48m)
                    err_h_signed = base_h - target_H
                    ok_h = (np.abs(err_h_signed) <= tol_h)
                    ok_vy = (np.abs(base_vy) <= tol_vy)
                    ok_wz = (np.abs(base_wz) <= tol_wz)
                    ok_jnt = ok_h & ok_vy & ok_wz

                    p_h_errors_signed.append(err_h_signed)
                    p_vy_raw.append(base_vy)
                    p_ok_jnt.append(ok_jnt)
                    p_ok_h.append(ok_h)
                    p_ok_vy.append(ok_vy)

            mean_signed_h_cm = float(np.mean(p_h_errors_signed) * 100.0)
            mean_signed_vy = float(np.mean(p_vy_raw))
            mean_h_pass = float(np.mean(p_ok_h) * 100.0)
            mean_vy_pass = float(np.mean(p_ok_vy) * 100.0)
            mean_jnt_pass = float(np.mean(p_ok_jnt) * 100.0)

            case_summaries.append({
                "case_name": case_name,
                "mean_signed_h_cm": mean_signed_h_cm,
                "mean_signed_vy_mps": mean_signed_vy,
                "height_pass_pct": mean_h_pass,
                "vy_pass_pct": mean_vy_pass,
                "joint_pass_pct": mean_jnt_pass
            })

        avg_signed_h = float(np.mean([c["mean_signed_h_cm"] for c in case_summaries]))
        avg_h_pass = float(np.mean([c["height_pass_pct"] for c in case_summaries]))
        avg_vy_pass = float(np.mean([c["vy_pass_pct"] for c in case_summaries]))
        avg_jnt_pass = float(np.mean([c["joint_pass_pct"] for c in case_summaries]))
        avg_signed_vy = float(np.mean([c["mean_signed_vy_mps"] for c in case_summaries]))
        completed_count = sum(1 for c in case_summaries if c["joint_pass_pct"] >= 90.0)

        results_scan.append({
            "delta_h_offset_m": delta_h,
            "h_ctrl_sent": h_ctrl_val,
            "mean_signed_h_err_cm": avg_signed_h,
            "mean_signed_vy_mps": avg_signed_vy,
            "height_pass_pct": avg_h_pass,
            "vy_pass_pct": avg_vy_pass,
            "joint_pass_pct": avg_jnt_pass,
            "completed_count": completed_count,
            "cases": case_summaries
        })

        print(f"delta_H = {delta_h*1000:+5.1f}mm (H_ctrl={h_ctrl_val:.4f}m): Signed_H_Err={avg_signed_h:+5.2f}cm | H_Pass={avg_h_pass:5.1f}% | Vy_Pass={avg_vy_pass:5.1f}% | Signed_Vy={avg_signed_vy:+6.3f}m/s | ParkJoint={avg_jnt_pass:5.1f}% | Completed={completed_count}/6")

    output_data = {
        "model_path": model_path,
        "target_H": 0.48,
        "scan_results": results_scan
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nScan results saved to: {output_path}")

if __name__ == "__main__":
    run_compensation_scan()
