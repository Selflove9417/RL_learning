"""
Fine-Grained 8~10mm Rise Height Compensation Scan & Validation Script for model_125.pt
Covers both H=0.48m (6 Core cases) and H=0.50m (3 Boundary cases).

Candidates: [7.5mm, 8.0mm, 8.5mm, 9.0mm, 9.5mm, 10.0mm]

Usage:
    python scripts/scan_rise_fine_8_10mm.py --model_path <path> --output_path <path>
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
    output_path = "evaluation_assets/phase14/results/scan_rise_fine_8_10mm_m125.json"
    
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

def run_fine_scan():
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

    rise_cases_048 = [
        ("2. fwd_straight_rise",  +0.20,  0.00, 0.48),
        ("4. bwd_straight_rise",  -0.20,  0.00, 0.48),
        ("7. fwd_left_rise",      +0.15, +0.25, 0.48),
        ("8. fwd_right_rise",     +0.15, -0.25, 0.48),
        ("11. bwd_left_rise",     -0.15, +0.25, 0.48),
        ("12. bwd_right_rise",    -0.15, -0.25, 0.48),
    ]

    rise_cases_050 = [
        ("16. fwd_straight_rise_050m", +0.10,  0.00, 0.50),
        ("17. bwd_straight_rise_050m", -0.10,  0.00, 0.50),
        ("18. fwd_turn_rise_050m",      +0.10, -0.15, 0.50),
    ]

    delta_h_candidates = [0.0075, 0.0080, 0.0085, 0.0090, 0.0095, 0.0100]

    step_cmd_start = 50
    step_motion_end = 350
    step_brake_end = 450
    total_steps = 700

    tol_h = 0.010
    tol_vy = 0.030
    tol_wz = 0.050

    print(f"\n================ Fine-Grained 8~10mm Rise Height Compensation Scan ================")
    print(f"Model: {model_path}")
    print(f"Candidates (mm): {[round(d*1000, 1) for d in delta_h_candidates]}")

    scan_048_results = []
    scan_050_results = []

    for delta_h in delta_h_candidates:
        # 1. Evaluate 0.48m
        h_ctrl_048 = round(0.48 + delta_h, 4)
        c048_list = []

        for case_name, cmd_vy, cmd_wz, target_H in rise_cases_048:
            torch.manual_seed(42)
            np.random.seed(42)
            env.reset()

            h_current, c_vy, c_wz = 0.39774, 0.0, 0.0
            accel_y = 0.8 * env.dt
            accel_w = 2.0 * env.dt
            step_dh = 0.04 * env.dt

            p_h_errors_signed, p_vy_raw = [], []
            p_ok_jnt, p_ok_h, p_ok_vy = [], [], []

            for step in range(total_steps):
                if step < step_cmd_start:
                    tgt_vy, tgt_wz, tgt_h = 0.0, 0.0, 0.39774
                elif step < step_motion_end:
                    tgt_vy, tgt_wz, tgt_h = cmd_vy, cmd_wz, h_ctrl_048
                else:
                    tgt_vy, tgt_wz, tgt_h = 0.0, 0.0, h_ctrl_048

                if c_vy < tgt_vy: c_vy = min(c_vy + accel_y, tgt_vy)
                elif c_vy > tgt_vy: c_vy = max(c_vy - accel_y, tgt_vy)

                if c_wz < tgt_wz: c_wz = min(c_wz + accel_w, tgt_wz)
                elif c_wz > tgt_wz: c_wz = max(c_wz - accel_w, tgt_wz)

                if h_current < tgt_h: h_current = min(h_current + step_dh, tgt_h)
                elif h_current > tgt_h: h_current = max(h_current - step_dh, tgt_h)

                env.commands[:, 0] = 0.0
                env.commands[:, 1] = c_vy
                env.commands[:, 2] = c_wz
                env.height_cmd_buf[:] = h_current
                if hasattr(env, 'target_height_buf'):
                    env.target_height_buf[:] = h_ctrl_048

                env.compute_observations()
                obs = env.get_observations()

                with torch.no_grad():
                    actions = policy(obs)

                obs, _, _, dones, _ = env.step(actions)
                base_h = env.root_states[:, 2].cpu().numpy()
                base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                base_wz = env.base_ang_vel[:, 2].cpu().numpy()

                if step >= step_brake_end:
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

            c048_list.append({
                "case_name": case_name,
                "signed_h_err_cm": float(np.mean(p_h_errors_signed) * 100.0),
                "signed_vy_mps": float(np.mean(p_vy_raw)),
                "height_pass_pct": float(np.mean(p_ok_h) * 100.0),
                "vy_pass_pct": float(np.mean(p_ok_vy) * 100.0),
                "joint_pass_pct": float(np.mean(p_ok_jnt) * 100.0)
            })

        avg_048_h = float(np.mean([c["signed_h_err_cm"] for c in c048_list]))
        avg_048_vy = float(np.mean([c["signed_vy_mps"] for c in c048_list]))
        avg_048_h_pass = float(np.mean([c["height_pass_pct"] for c in c048_list]))
        avg_048_vy_pass = float(np.mean([c["vy_pass_pct"] for c in c048_list]))
        avg_048_jnt_pass = float(np.mean([c["joint_pass_pct"] for c in c048_list]))
        done_048 = sum(1 for c in c048_list if c["joint_pass_pct"] >= 90.0)

        scan_048_results.append({
            "delta_h_mm": delta_h * 1000.0,
            "h_ctrl_sent": h_ctrl_048,
            "signed_h_err_cm": avg_048_h,
            "signed_vy_mps": avg_048_vy,
            "height_pass_pct": avg_048_h_pass,
            "vy_pass_pct": avg_048_vy_pass,
            "joint_pass_pct": avg_048_jnt_pass,
            "completed_count": done_048,
            "cases": c048_list
        })

        # 2. Evaluate 0.50m
        h_ctrl_050 = round(0.50 + delta_h, 4)
        c050_list = []

        for case_name, cmd_vy, cmd_wz, target_H in rise_cases_050:
            torch.manual_seed(42)
            np.random.seed(42)
            env.reset()

            h_current, c_vy, c_wz = 0.39774, 0.0, 0.0
            accel_y = 0.8 * env.dt
            accel_w = 2.0 * env.dt
            step_dh = 0.04 * env.dt

            p_h_errors_signed, p_vy_raw = [], []
            p_ok_jnt, p_ok_h, p_ok_vy = [], [], []

            for step in range(total_steps):
                if step < step_cmd_start:
                    tgt_vy, tgt_wz, tgt_h = 0.0, 0.0, 0.39774
                elif step < step_motion_end:
                    tgt_vy, tgt_wz, tgt_h = cmd_vy, cmd_wz, h_ctrl_050
                else:
                    tgt_vy, tgt_wz, tgt_h = 0.0, 0.0, h_ctrl_050

                if c_vy < tgt_vy: c_vy = min(c_vy + accel_y, tgt_vy)
                elif c_vy > tgt_vy: c_vy = max(c_vy - accel_y, tgt_vy)

                if c_wz < tgt_wz: c_wz = min(c_wz + accel_w, tgt_wz)
                elif c_wz > tgt_wz: c_wz = max(c_wz - accel_w, tgt_wz)

                if h_current < tgt_h: h_current = min(h_current + step_dh, tgt_h)
                elif h_current > tgt_h: h_current = max(h_current - step_dh, tgt_h)

                env.commands[:, 0] = 0.0
                env.commands[:, 1] = c_vy
                env.commands[:, 2] = c_wz
                env.height_cmd_buf[:] = h_current
                if hasattr(env, 'target_height_buf'):
                    env.target_height_buf[:] = h_ctrl_050

                env.compute_observations()
                obs = env.get_observations()

                with torch.no_grad():
                    actions = policy(obs)

                obs, _, _, dones, _ = env.step(actions)
                base_h = env.root_states[:, 2].cpu().numpy()
                base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                base_wz = env.base_ang_vel[:, 2].cpu().numpy()

                if step >= step_brake_end:
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

            c050_list.append({
                "case_name": case_name,
                "signed_h_err_cm": float(np.mean(p_h_errors_signed) * 100.0),
                "signed_vy_mps": float(np.mean(p_vy_raw)),
                "height_pass_pct": float(np.mean(p_ok_h) * 100.0),
                "vy_pass_pct": float(np.mean(p_ok_vy) * 100.0),
                "joint_pass_pct": float(np.mean(p_ok_jnt) * 100.0)
            })

        avg_050_h = float(np.mean([c["signed_h_err_cm"] for c in c050_list]))
        avg_050_vy = float(np.mean([c["signed_vy_mps"] for c in c050_list]))
        avg_050_h_pass = float(np.mean([c["height_pass_pct"] for c in c050_list]))
        avg_050_vy_pass = float(np.mean([c["vy_pass_pct"] for c in c050_list]))
        avg_050_jnt_pass = float(np.mean([c["joint_pass_pct"] for c in c050_list]))
        done_050 = sum(1 for c in c050_list if c["joint_pass_pct"] >= 90.0)

        scan_050_results.append({
            "delta_h_mm": delta_h * 1000.0,
            "h_ctrl_sent": h_ctrl_050,
            "signed_h_err_cm": avg_050_h,
            "signed_vy_mps": avg_050_vy,
            "height_pass_pct": avg_050_h_pass,
            "vy_pass_pct": avg_050_vy_pass,
            "joint_pass_pct": avg_050_jnt_pass,
            "completed_count": done_050,
            "cases": c050_list
        })

        print(f"delta_H={delta_h*1000:4.1f}mm | 0.48m: H_err={avg_048_h:+5.2f}cm, H_Pass={avg_048_h_pass:5.1f}%, Vy_Pass={avg_048_vy_pass:5.1f}%, Vy={avg_048_vy:+6.3f}m/s, Joint%={avg_048_jnt_pass:5.1f}% (Done={done_048}/6) | 0.50m: H_err={avg_050_h:+5.2f}cm, Joint%={avg_050_jnt_pass:5.1f}%")

    output_data = {
        "model_path": model_path,
        "delta_h_candidates_mm": [round(d*1000, 1) for d in delta_h_candidates],
        "scan_048m": scan_048_results,
        "scan_050m": scan_050_results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nFine scan output saved to: {output_path}")

if __name__ == "__main__":
    run_fine_scan()
