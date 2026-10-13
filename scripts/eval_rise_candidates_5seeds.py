"""
5-Seed Cross-Validation for Rise Height Compensation Candidates on model_125.pt
Evaluates Delta_H = 8.0mm against baseline Delta_H = 6.6mm across 5 seeds.
Checks:
- 0.48m (6 Core cases): Joint%, Done count, Signed H_err, Signed Vy, Net Disp, Motion Joint%
- 0.50m (3 Boundary cases): Joint%, Signed H_err, Signed Vy, Net Disp, Motion Joint%

Usage:
    python scripts/eval_rise_candidates_5seeds.py --model_path <path> --output_path <path>
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
    output_path = "evaluation_assets/phase14/results/eval_rise_candidate_8mm_5seeds.json"
    num_seeds = 5
    return model_path, output_path, num_seeds

def run_5seeds_candidate_eval():
    model_path, output_path, num_seeds = parse_args()
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

    all_cases = rise_cases_048 + rise_cases_050

    # Test candidate: delta_h = 0.0080 (8.0mm) vs baseline 0.0066 (6.6mm)
    candidate_delta_h = 0.0080

    step_cmd_start = 50
    step_motion_end = 350
    step_brake_end = 450
    total_steps = 700
    steady_motion_start = 180

    tol_h = 0.010
    tol_vy = 0.030
    tol_wz = 0.050
    tol_motion_vy = 0.050
    tol_motion_wz = 0.080

    print(f"\n================ 5-Seed Cross-Validation for Candidate Delta_H = 8.0mm ================")

    per_seed_results = []

    for seed_idx in range(1, num_seeds + 1):
        seed_val = seed_idx * 100 + 42
        case_records = {}

        for case_name, cmd_vy, cmd_wz, target_H in all_cases:
            torch.manual_seed(seed_val)
            np.random.seed(seed_val)
            env.reset()

            h_ctrl = round(target_H + candidate_delta_h, 4)
            h_current, c_vy, c_wz = 0.39774, 0.0, 0.0
            accel_y = 0.8 * env.dt
            accel_w = 2.0 * env.dt
            step_dh = 0.04 * env.dt

            mot_jnt_oks = []
            p_h_errors_signed, p_vy_raw = [], []
            p_ok_jnt, p_ok_h, p_ok_vy = [], [], []

            for step in range(total_steps):
                if step < step_cmd_start:
                    tgt_vy, tgt_wz, tgt_h = 0.0, 0.0, 0.39774
                elif step < step_motion_end:
                    tgt_vy, tgt_wz, tgt_h = cmd_vy, cmd_wz, h_ctrl
                else:
                    tgt_vy, tgt_wz, tgt_h = 0.0, 0.0, h_ctrl

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
                    env.target_height_buf[:] = h_ctrl

                env.compute_observations()
                obs = env.get_observations()

                with torch.no_grad():
                    actions = policy(obs)

                obs, _, _, dones, _ = env.step(actions)
                base_h = env.root_states[:, 2].cpu().numpy()
                base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                base_wz = env.base_ang_vel[:, 2].cpu().numpy()

                # Steady motion (step 180~350)
                if steady_motion_start <= step < step_motion_end:
                    m_ok_h = (np.abs(base_h - target_H) <= tol_h)
                    m_ok_vy = (np.abs(base_vy - cmd_vy) <= tol_motion_vy)
                    m_ok_wz = (np.abs(base_wz - cmd_wz) <= tol_motion_wz)
                    mot_jnt_oks.append(m_ok_h & m_ok_vy & m_ok_wz)

                # Steady parking (step 450~700)
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

            p_raw_vy_arr = np.array(p_vy_raw)
            net_disp = float(np.mean(np.sum(p_raw_vy_arr * env.dt, axis=0)))

            case_records[case_name] = {
                "target_H": target_H,
                "motion_joint_pct": float(np.mean(mot_jnt_oks) * 100.0),
                "parking_joint_pct": float(np.mean(p_ok_jnt) * 100.0),
                "height_pass_pct": float(np.mean(p_ok_h) * 100.0),
                "vy_pass_pct": float(np.mean(p_ok_vy) * 100.0),
                "signed_h_err_cm": float(np.mean(p_h_errors_signed) * 100.0),
                "signed_vy_mps": float(np.mean(p_vy_raw)),
                "net_displacement_m": net_disp
            }

        c048_records = [case_records[c[0]] for c in rise_cases_048]
        c050_records = [case_records[c[0]] for c in rise_cases_050]

        s_summary = {
            "seed_idx": seed_idx,
            "seed_val": seed_val,
            "c048": {
                "parking_joint_pct": float(np.mean([r["parking_joint_pct"] for r in c048_records])),
                "motion_joint_pct": float(np.mean([r["motion_joint_pct"] for r in c048_records])),
                "height_pass_pct": float(np.mean([r["height_pass_pct"] for r in c048_records])),
                "vy_pass_pct": float(np.mean([r["vy_pass_pct"] for r in c048_records])),
                "signed_h_err_cm": float(np.mean([r["signed_h_err_cm"] for r in c048_records])),
                "signed_vy_mps": float(np.mean([r["signed_vy_mps"] for r in c048_records])),
                "net_displacement_m": float(np.mean([r["net_displacement_m"] for r in c048_records])),
                "completed_count": sum(1 for r in c048_records if r["parking_joint_pct"] >= 90.0)
            },
            "c050": {
                "parking_joint_pct": float(np.mean([r["parking_joint_pct"] for r in c050_records])),
                "motion_joint_pct": float(np.mean([r["motion_joint_pct"] for r in c050_records])),
                "height_pass_pct": float(np.mean([r["height_pass_pct"] for r in c050_records])),
                "vy_pass_pct": float(np.mean([r["vy_pass_pct"] for r in c050_records])),
                "signed_h_err_cm": float(np.mean([r["signed_h_err_cm"] for r in c050_records])),
                "signed_vy_mps": float(np.mean([r["signed_vy_mps"] for r in c050_records])),
                "net_displacement_m": float(np.mean([r["net_displacement_m"] for r in c050_records])),
                "completed_count": sum(1 for r in c050_records if r["parking_joint_pct"] >= 90.0)
            },
            "case_records": case_records
        }
        per_seed_results.append(s_summary)
        print(f"Seed {seed_idx} (seed={seed_val}): 0.48m ParkJoint={s_summary['c048']['parking_joint_pct']:.2f}% (Done={s_summary['c048']['completed_count']}/6, H_err={s_summary['c048']['signed_h_err_cm']:+.2f}cm, Vy={s_summary['c048']['signed_vy_mps']:+.3f}m/s, Disp={s_summary['c048']['net_displacement_m']:+.3f}m) | 0.50m ParkJoint={s_summary['c050']['parking_joint_pct']:.2f}%")

    # Aggregate
    c048_parks = [s["c048"]["parking_joint_pct"] for s in per_seed_results]
    c048_dones = [s["c048"]["completed_count"] for s in per_seed_results]
    c048_hs = [s["c048"]["signed_h_err_cm"] for s in per_seed_results]
    c048_vys = [s["c048"]["signed_vy_mps"] for s in per_seed_results]
    c048_disps = [s["c048"]["net_displacement_m"] for s in per_seed_results]
    c048_mots = [s["c048"]["motion_joint_pct"] for s in per_seed_results]

    c050_parks = [s["c050"]["parking_joint_pct"] for s in per_seed_results]
    c050_hs = [s["c050"]["signed_h_err_cm"] for s in per_seed_results]
    c050_vys = [s["c050"]["signed_vy_mps"] for s in per_seed_results]

    overall_stats = {
        "candidate_delta_h_mm": candidate_delta_h * 1000.0,
        "c048_mean": {
            "parking_joint_pct": float(np.mean(c048_parks)),
            "parking_joint_pct_std": float(np.std(c048_parks)),
            "completed_count_mean": float(np.mean(c048_dones)),
            "signed_h_err_cm": float(np.mean(c048_hs)),
            "signed_vy_mps": float(np.mean(c048_vys)),
            "net_displacement_m": float(np.mean(c048_disps)),
            "motion_joint_pct": float(np.mean(c048_mots))
        },
        "c050_mean": {
            "parking_joint_pct": float(np.mean(c050_parks)),
            "parking_joint_pct_std": float(np.std(c050_parks)),
            "signed_h_err_cm": float(np.mean(c050_hs)),
            "signed_vy_mps": float(np.mean(c050_vys))
        },
        "per_seed_results": per_seed_results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(overall_stats, f, indent=2)

    print(f"\n================ Overall 5-Seed Stats (Delta_H = 8.0mm) ================")
    print(f"0.48m Parking Joint%: {overall_stats['c048_mean']['parking_joint_pct']:.2f}% ± {overall_stats['c048_mean']['parking_joint_pct_std']:.2f}%")
    print(f"0.48m Actions Done:   {overall_stats['c048_mean']['completed_count_mean']:.1f} / 6")
    print(f"0.48m Signed H_err:   {overall_stats['c048_mean']['signed_h_err_cm']:+.2f} cm")
    print(f"0.48m Signed Vy:      {overall_stats['c048_mean']['signed_vy_mps']:+.3f} m/s (Net Disp: {overall_stats['c048_mean']['net_displacement_m']:+.3f}m)")
    print(f"0.48m Motion Joint%:  {overall_stats['c048_mean']['motion_joint_pct']:.2f}%")
    print(f"0.50m Parking Joint%: {overall_stats['c050_mean']['parking_joint_pct']:.2f}% ± {overall_stats['c050_mean']['parking_joint_pct_std']:.2f}%")
    print(f"0.50m Signed H_err:   {overall_stats['c050_mean']['signed_h_err_cm']:+.2f} cm (Vy: {overall_stats['c050_mean']['signed_vy_mps']:+.3f}m/s)")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    run_5seeds_candidate_eval()
