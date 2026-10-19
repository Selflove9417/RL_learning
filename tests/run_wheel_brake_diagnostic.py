#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Phase 3.1 Wheel Brake Diagnostic Experiment Batch Runner
# Executes 3 diagnostic modes x 20 deterministic seeds = 60 total runs.
# Modes:
#   - Mode A: Control (k_brake = 0.00)
#   - Mode B: Weak Brake (k_brake = 0.02)
#   - Mode C: Strong Brake (k_brake = 0.05)
# Window: Strictly step <= 5 (t <= 0.10s)
# ==============================================================================

import os
import sys
import csv
import json
import subprocess
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
python_bin = "/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/bin/python"
single_eval_script = os.path.join(workspace_root, "tests", "eval_single_diagnostic.py")

output_base_dir = os.path.join(workspace_root, "evaluation_assets", "phase3_1_forward_distance", "wheel_brake_diagnostic")
os.makedirs(output_base_dir, exist_ok=True)

SEEDS = [
    42, 100, 200, 300, 400,            # Original 5 seeds
    1, 2, 3, 5, 7, 11, 13, 17, 19, 23, # 10 prime seeds
    50, 150, 250, 350, 450             # 5 round seeds
]

MODES = [
    {"name": "Mode_A_control", "k": 0.00, "label": "原始 Champion (Control)"},
    {"name": "Mode_B_weak_brake", "k": 0.02, "label": "弱刹车 (k=0.02)"},
    {"name": "Mode_C_strong_brake", "k": 0.05, "label": "强刹车 (k=0.05)"}
]

def run_all_modes():
    print("==============================================================================")
    print(" Phase 3.1 Wheel Brake Diagnostic Experiment Runner")
    print(f" Output Base: {output_base_dir}")
    print(f" Seeds ({len(SEEDS)}): {SEEDS}")
    print(f" Modes ({len(MODES)}): {[m['name'] for m in MODES]}")
    print("==============================================================================\n")

    env = os.environ.copy()
    bin_dir = "/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/bin"
    lib_dir = "/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/lib"
    torch_ext_dir = "/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/torch-extensions"

    env["PATH"] = f"{bin_dir}:" + env.get("PATH", "")
    env["PYTHONPATH"] = f"{workspace_root}/legged_gym:{workspace_root}/rsl_rl:" + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:" + env.get("LD_LIBRARY_PATH", "")
    env["TORCH_EXTENSIONS_DIR"] = torch_ext_dir
    env["MAX_JOBS"] = "2"
    env["PYTHONNOUSERSITE"] = "1"

    all_mode_results = {}

    for mode_info in MODES:
        mode_name = mode_info["name"]
        brake_k = mode_info["k"]
        mode_label = mode_info["label"]

        mode_dir = os.path.join(output_base_dir, mode_name)
        os.makedirs(mode_dir, exist_ok=True)

        print(f"\n>>> Running Mode: {mode_name} (k={brake_k:.2f}) - {mode_label} <<<")
        mode_runs = []

        for idx, seed in enumerate(SEEDS, 1):
            out_json = os.path.join(mode_dir, f"seed_{seed}.json")
            cmd = [
                python_bin, single_eval_script,
                "--seed", str(seed),
                "--mode", mode_name,
                "--brake_k", str(brake_k),
                "--out", out_json
            ]

            print(f"  [{idx:2d}/{len(SEEDS)}] Seed {seed:3d}...", end=" ", flush=True)
            proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if proc.returncode != 0:
                print(f"FAILED!\nError: {proc.stderr}")
                continue

            with open(out_json, "r") as f:
                res = json.load(f)

            status = "✅ PASS" if res["joint_success"] else "❌ FAIL"
            print(f"{status} | Clr={res['max_clearance_cm']:5.2f}cm | FwdDist={res['com_forward_dist_cm']:+6.2f}cm | S5_Vy={res['step5_vy_mps']:+6.3f}m/s | TO_Vy={res['takeoff_vy_mps']:+6.3f}m/s | TO_Vz={res['takeoff_vz_mps']:+5.3f}m/s | TDRoll={res['touchdown_roll_deg']:+5.2f}°")
            mode_runs.append(res)

        all_mode_results[mode_name] = mode_runs

        # Save mode CSV summary
        mode_csv = os.path.join(output_base_dir, f"{mode_name}_telemetry.csv")
        save_mode_csv(mode_csv, mode_runs)

    # Generate comparative analysis
    generate_comparison_summary(all_mode_results)
    extract_key_trajectories(all_mode_results)

def save_mode_csv(csv_path, runs):
    if not runs:
        return
    fieldnames = [
        "seed", "mode", "brake_k", "joint_success", "valid_jump", "abnormal_resets",
        "is_normal_jump", "is_short_jump", "is_backward_jump", "is_low_clearance", "is_prep_slip",
        "step0_vy_mps", "step3_vy_mps", "step5_vy_mps", "max_prep_base_vel_mps", "prep_ground_slip_cm", "max_prep_wheel_omega_radps",
        "step0_pitch_deg", "step0_roll_deg", "step5_pitch_deg", "step5_roll_deg",
        "step5_w_L_radps", "step5_w_R_radps", "step5_tau_w_L_nm", "step5_tau_w_R_nm",
        "step7_vy_mps", "step8_vy_mps", "takeoff_vy_mps", "takeoff_vz_mps", "delta_push_vy_mps",
        "step7_tau_w_L_nm", "step7_tau_w_R_nm", "step7_tau_k_L_nm", "step7_tau_k_R_nm", "step7_tau_h_L_nm", "step7_tau_h_R_nm",
        "step8_tau_w_L_nm", "step8_tau_w_R_nm", "step8_tau_k_L_nm", "step8_tau_k_R_nm", "step8_tau_h_L_nm", "step8_tau_h_R_nm",
        "max_clearance_cm", "max_clearance_L_cm", "max_clearance_R_cm", "com_forward_dist_cm", "flight_duration_s",
        "flight_pitch_max_deg", "flight_roll_max_deg",
        "touchdown_pitch_deg", "touchdown_roll_deg", "touchdown_vy_mps", "touchdown_plus_20s_vy_mps", "recovery_vy_mps",
        "takeoff_step", "touchdown_step"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in runs:
            writer.writerow(r)

def generate_comparison_summary(all_mode_results):
    comparison = {}

    for mode_name, runs in all_mode_results.items():
        if not runs:
            continue
        n = len(runs)
        prep_max_vy = [r["max_prep_base_vel_mps"] for r in runs]
        step5_vy = [r["step5_vy_mps"] for r in runs]
        takeoff_vy = [r["takeoff_vy_mps"] for r in runs]
        takeoff_vz = [r["takeoff_vz_mps"] for r in runs]
        delta_push_vy = [r["delta_push_vy_mps"] for r in runs]
        fwd_dist = [r["com_forward_dist_cm"] for r in runs]
        clearance = [r["max_clearance_cm"] for r in runs]
        td_roll = [abs(r["touchdown_roll_deg"]) for r in runs]
        td_pitch = [abs(r["touchdown_pitch_deg"]) for r in runs]
        rec_vy = [r["recovery_vy_mps"] for r in runs]
        prep_slip = [r["prep_ground_slip_cm"] for r in runs]
        prep_omega = [r["max_prep_wheel_omega_radps"] for r in runs]

        normal_count = sum(1 for r in runs if r["is_normal_jump"])
        short_count = sum(1 for r in runs if r["is_short_jump"])
        backward_count = sum(1 for r in runs if r["is_backward_jump"])
        low_clearance_count = sum(1 for r in runs if r["is_low_clearance"])
        prep_slip_count = sum(1 for r in runs if r["is_prep_slip"])
        joint_pass_count = sum(1 for r in runs if r["joint_success"])
        abnormal_resets_total = sum(r["abnormal_resets"] for r in runs)

        comparison[mode_name] = {
            "num_runs": n,
            "pass_rate_pct": round(joint_pass_count / n * 100.0, 1),
            "counts": {
                "normal_jump": normal_count,
                "short_jump": short_count,
                "backward_jump": backward_count,
                "low_clearance": low_clearance_count,
                "prep_slip": prep_slip_count,
                "joint_pass": joint_pass_count,
                "abnormal_resets": abnormal_resets_total
            },
            "stats": {
                "prep_max_abs_vy": {
                    "mean": round(float(np.mean(prep_max_vy)), 4),
                    "std": round(float(np.std(prep_max_vy)), 4),
                    "min": round(float(np.min(prep_max_vy)), 4),
                    "max": round(float(np.max(prep_max_vy)), 4)
                },
                "step5_vy": {
                    "mean": round(float(np.mean(step5_vy)), 4),
                    "std": round(float(np.std(step5_vy)), 4),
                    "min": round(float(np.min(step5_vy)), 4),
                    "max": round(float(np.max(step5_vy)), 4)
                },
                "takeoff_vy": {
                    "mean": round(float(np.mean(takeoff_vy)), 4),
                    "std": round(float(np.std(takeoff_vy)), 4),
                    "min": round(float(np.min(takeoff_vy)), 4),
                    "max": round(float(np.max(takeoff_vy)), 4)
                },
                "takeoff_vz": {
                    "mean": round(float(np.mean(takeoff_vz)), 4),
                    "std": round(float(np.std(takeoff_vz)), 4),
                    "min": round(float(np.min(takeoff_vz)), 4),
                    "max": round(float(np.max(takeoff_vz)), 4)
                },
                "delta_push_vy": {
                    "mean": round(float(np.mean(delta_push_vy)), 4),
                    "std": round(float(np.std(delta_push_vy)), 4),
                    "min": round(float(np.min(delta_push_vy)), 4),
                    "max": round(float(np.max(delta_push_vy)), 4)
                },
                "com_forward_dist_cm": {
                    "mean": round(float(np.mean(fwd_dist)), 2),
                    "std": round(float(np.std(fwd_dist)), 2),
                    "min": round(float(np.min(fwd_dist)), 2),
                    "max": round(float(np.max(fwd_dist)), 2)
                },
                "max_clearance_cm": {
                    "mean": round(float(np.mean(clearance)), 2),
                    "std": round(float(np.std(clearance)), 2),
                    "min": round(float(np.min(clearance)), 2),
                    "max": round(float(np.max(clearance)), 2)
                },
                "touchdown_abs_roll_deg": {
                    "mean": round(float(np.mean(td_roll)), 2),
                    "std": round(float(np.std(td_roll)), 2),
                    "min": round(float(np.min(td_roll)), 2),
                    "max": round(float(np.max(td_roll)), 2)
                },
                "touchdown_abs_pitch_deg": {
                    "mean": round(float(np.mean(td_pitch)), 2),
                    "std": round(float(np.std(td_pitch)), 2),
                    "min": round(float(np.min(td_pitch)), 2),
                    "max": round(float(np.max(td_pitch)), 2)
                },
                "recovery_vy": {
                    "mean": round(float(np.mean(rec_vy)), 4),
                    "std": round(float(np.std(rec_vy)), 4),
                    "min": round(float(np.min(rec_vy)), 4),
                    "max": round(float(np.max(rec_vy)), 4)
                },
                "prep_slip_cm": {
                    "mean": round(float(np.mean(prep_slip)), 2),
                    "std": round(float(np.std(prep_slip)), 2),
                    "min": round(float(np.min(prep_slip)), 2),
                    "max": round(float(np.max(prep_slip)), 2)
                },
                "prep_wheel_omega_radps": {
                    "mean": round(float(np.mean(prep_omega)), 2),
                    "std": round(float(np.std(prep_omega)), 2),
                    "min": round(float(np.min(prep_omega)), 2),
                    "max": round(float(np.max(prep_omega)), 2)
                }
            }
        }

    summary_file = os.path.join(output_base_dir, "wheel_brake_diagnostic_summary.json")
    with open(summary_file, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[Saved Summary] {summary_file}")

    # Print the requested Section XI Comparison Table to stdout
    print_markdown_table(comparison)

def print_markdown_table(comparison):
    print("\n" + "="*80)
    print(" SECTION XI: COMPARISON TABLE (Across 20 Deterministic Seeds)")
    print("="*80)
    mA = comparison.get("Mode_A_control", {})
    mB = comparison.get("Mode_B_weak_brake", {})
    mC = comparison.get("Mode_C_strong_brake", {})

    def fmt_stat(m_dict, key, unit=""):
        if not m_dict or key not in m_dict.get("stats", {}):
            return "N/A"
        s = m_dict["stats"][key]
        return f"{s['mean']:+.3f} ± {s['std']:.3f}{unit} [{s['min']:+.2f}, {s['max']:+.2f}]"

    def fmt_dist(m_dict, key, unit="cm"):
        if not m_dict or key not in m_dict.get("stats", {}):
            return "N/A"
        s = m_dict["stats"][key]
        return f"{s['mean']:+.2f} ± {s['std']:.2f} {unit} [{s['min']:+.1f}, {s['max']:+.1f}]"

    def fmt_deg(m_dict, key):
        if not m_dict or key not in m_dict.get("stats", {}):
            return "N/A"
        s = m_dict["stats"][key]
        return f"{s['mean']:.2f} ± {s['std']:.2f}° [{s['min']:.1f}, {s['max']:.1f}]"

    def fmt_cnt(m_dict, key):
        if not m_dict or key not in m_dict.get("counts", {}):
            return "N/A"
        return f"{m_dict['counts'][key]}/{m_dict['num_runs']}"

    rows = [
        ("准备期最大 |Vy|", fmt_stat(mA, "prep_max_abs_vy", " m/s"), fmt_stat(mB, "prep_max_abs_vy", " m/s"), fmt_stat(mC, "prep_max_abs_vy", " m/s")),
        ("Step 5 Vy", fmt_stat(mA, "step5_vy", " m/s"), fmt_stat(mB, "step5_vy", " m/s"), fmt_stat(mC, "step5_vy", " m/s")),
        ("Takeoff Vy", fmt_stat(mA, "takeoff_vy", " m/s"), fmt_stat(mB, "takeoff_vy", " m/s"), fmt_stat(mC, "takeoff_vy", " m/s")),
        ("Takeoff Vz", fmt_stat(mA, "takeoff_vz", " m/s"), fmt_stat(mB, "takeoff_vz", " m/s"), fmt_stat(mC, "takeoff_vz", " m/s")),
        ("推蹬 ΔVy", fmt_stat(mA, "delta_push_vy", " m/s"), fmt_stat(mB, "delta_push_vy", " m/s"), fmt_stat(mC, "delta_push_vy", " m/s")),
        ("前向腾空位移", fmt_dist(mA, "com_forward_dist_cm", "cm"), fmt_dist(mB, "com_forward_dist_cm", "cm"), fmt_dist(mC, "com_forward_dist_cm", "cm")),
        ("最低共同净空", fmt_dist(mA, "max_clearance_cm", "cm"), fmt_dist(mB, "max_clearance_cm", "cm"), fmt_dist(mC, "max_clearance_cm", "cm")),
        ("Touchdown |Roll|", fmt_deg(mA, "touchdown_abs_roll_deg"), fmt_deg(mB, "touchdown_abs_roll_deg"), fmt_deg(mC, "touchdown_abs_roll_deg")),
        ("Touchdown |Pitch|", fmt_deg(mA, "touchdown_abs_pitch_deg"), fmt_deg(mB, "touchdown_abs_pitch_deg"), fmt_deg(mC, "touchdown_abs_pitch_deg")),
        ("恢复后 Vy", fmt_stat(mA, "recovery_vy", " m/s"), fmt_stat(mB, "recovery_vy", " m/s"), fmt_stat(mC, "recovery_vy", " m/s")),
        ("综合通过率", f"{mA.get('pass_rate_pct', 0.0)}% ({fmt_cnt(mA, 'joint_pass')})", f"{mB.get('pass_rate_pct', 0.0)}% ({fmt_cnt(mB, 'joint_pass')})", f"{mC.get('pass_rate_pct', 0.0)}% ({fmt_cnt(mC, 'joint_pass')})"),
        ("异常 reset", fmt_cnt(mA, "abnormal_resets"), fmt_cnt(mB, "abnormal_resets"), fmt_cnt(mC, "abnormal_resets")),
        ("正常前跃数量 (≥8cm & ≥20cm)", fmt_cnt(mA, "normal_jump"), fmt_cnt(mB, "normal_jump"), fmt_cnt(mC, "normal_jump")),
        ("短跳数量 (0~8cm)", fmt_cnt(mA, "short_jump"), fmt_cnt(mB, "short_jump"), fmt_cnt(mC, "short_jump")),
        ("倒跳数量 (<0cm)", fmt_cnt(mA, "backward_jump"), fmt_cnt(mB, "backward_jump"), fmt_cnt(mC, "backward_jump")),
        ("净空不足数量 (<20cm)", fmt_cnt(mA, "low_clearance"), fmt_cnt(mB, "low_clearance"), fmt_cnt(mC, "low_clearance")),
        ("准备期滑移数量 (slip≥2cm or ω≥15rad/s)", fmt_cnt(mA, "prep_slip"), fmt_cnt(mB, "prep_slip"), fmt_cnt(mC, "prep_slip"))
    ]

    print(f"| {'指标':<35} | {'原始 Champion (Mode A)':<30} | {'弱刹车 (Mode B)':<30} | {'强刹车 (Mode C)':<30} |")
    print(f"|:{'-'*35}-|-{'-'*30}:|-{'-'*30}:|-{'-'*30}:|")
    for r in rows:
        print(f"| {r[0]:<35} | {r[1]:>30} | {r[2]:>30} | {r[3]:>30} |")
    print("="*80 + "\n")

def extract_key_trajectories(all_mode_results):
    traj_dir = os.path.join(output_base_dir, "key_trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    # Key representative seeds:
    # Seed 100: Normal jump baseline
    # Seed 2: Class B Short jump archetype
    # Seed 17: Backward jump archetype
    # Seed 23: Backward jump archetype
    target_seeds = [100, 2, 17, 23]

    for seed in target_seeds:
        csv_file = os.path.join(traj_dir, f"trajectory_comparison_seed_{seed}.csv")
        fieldnames = [
            "step", "time_s",
            "A_com_vy", "A_com_vz", "A_clr_cm", "A_pitch", "A_w_L", "A_w_R", "A_tau_w_L", "A_is_air",
            "B_com_vy", "B_com_vz", "B_clr_cm", "B_pitch", "B_w_L", "B_w_R", "B_tau_w_L", "B_is_air",
            "C_com_vy", "C_com_vz", "C_clr_cm", "C_pitch", "C_w_L", "C_w_R", "C_tau_w_L", "C_is_air"
        ]

        runs_by_mode = {}
        for mode_info in MODES:
            m_name = mode_info["name"]
            runs = all_mode_results.get(m_name, [])
            for r in runs:
                if r["seed"] == seed:
                    runs_by_mode[m_name] = r
                    break

        if len(runs_by_mode) < 3:
            continue

        traj_A = runs_by_mode["Mode_A_control"].get("trajectory", [])
        traj_B = runs_by_mode["Mode_B_weak_brake"].get("trajectory", [])
        traj_C = runs_by_mode["Mode_C_strong_brake"].get("trajectory", [])

        min_steps = min(len(traj_A), len(traj_B), len(traj_C), 40) # focus on steps 0-40 (prep, push, flight, td)

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in range(min_steps):
                row = {
                    "step": s,
                    "time_s": traj_A[s]["time_s"],
                    "A_com_vy": traj_A[s]["com_vy"],
                    "A_com_vz": traj_A[s]["com_vz"],
                    "A_clr_cm": traj_A[s]["clr_min_cm"],
                    "A_pitch": traj_A[s]["pitch_deg"],
                    "A_w_L": traj_A[s]["w_L_radps"],
                    "A_w_R": traj_A[s]["w_R_radps"],
                    "A_tau_w_L": traj_A[s]["tau_w_L_nm"],
                    "A_is_air": int(traj_A[s]["is_airborne"]),

                    "B_com_vy": traj_B[s]["com_vy"],
                    "B_com_vz": traj_B[s]["com_vz"],
                    "B_clr_cm": traj_B[s]["clr_min_cm"],
                    "B_pitch": traj_B[s]["pitch_deg"],
                    "B_w_L": traj_B[s]["w_L_radps"],
                    "B_w_R": traj_B[s]["w_R_radps"],
                    "B_tau_w_L": traj_B[s]["tau_w_L_nm"],
                    "B_is_air": int(traj_B[s]["is_airborne"]),

                    "C_com_vy": traj_C[s]["com_vy"],
                    "C_com_vz": traj_C[s]["com_vz"],
                    "C_clr_cm": traj_C[s]["clr_min_cm"],
                    "C_pitch": traj_C[s]["pitch_deg"],
                    "C_w_L": traj_C[s]["w_L_radps"],
                    "C_w_R": traj_C[s]["w_R_radps"],
                    "C_tau_w_L": traj_C[s]["tau_w_L_nm"],
                    "C_is_air": int(traj_C[s]["is_airborne"])
                }
                writer.writerow(row)
        print(f"[Saved Key Trajectory] {csv_file}")

if __name__ == "__main__":
    run_all_modes()
