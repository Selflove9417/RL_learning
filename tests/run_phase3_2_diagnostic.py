#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Phase 3.2 Static Start Purity Audit Batch Runner
# Executes:
#   - Mode A: Control (Original Champion + Current Reset)
#   - Mode D: Clean-Start Diagnostic (Counterfactual True Zero-Velocity Start)
# Over 20 Deterministic Seeds:
#   [42, 100, 200, 300, 400, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 50, 150, 250, 350, 450]
# Total 40 runs.
# ==============================================================================

import os
import sys
import csv
import json
import subprocess
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
python_bin = "/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/bin/python"
single_eval_script = os.path.join(workspace_root, "tests", "eval_phase3_2_single_seed.py")

output_base_dir = os.path.join(workspace_root, "evaluation_assets", "phase3_2_static_start_purity")
os.makedirs(output_base_dir, exist_ok=True)

SEEDS = [
    42, 100, 200, 300, 400,            # Original 5 seeds
    1, 2, 3, 5, 7, 11, 13, 17, 19, 23, # 10 prime seeds
    50, 150, 250, 350, 450             # 5 round seeds
]

MODES = [
    {"name": "Mode_A_control", "label": "原始 Champion (Control)"},
    {"name": "Mode_D_clean_start", "label": "Clean-Start Diagnostic (Mode D)"}
]


def save_csv(filepath, rows, headers):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def run_all():
    print("==============================================================================")
    print(" Phase 3.2 Static Start Purity Audit Batch Runner")
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
        mode_label = mode_info["label"]

        mode_dir = os.path.join(output_base_dir, mode_name)
        os.makedirs(mode_dir, exist_ok=True)

        print(f"\n>>> Running Mode: {mode_name} - {mode_label} <<<")
        mode_runs = []

        for idx, seed in enumerate(SEEDS, 1):
            out_json = os.path.join(mode_dir, f"seed_{seed}.json")
            cmd = [
                python_bin, single_eval_script,
                "--seed", str(seed),
                "--mode", mode_name,
                "--out", out_json
            ]

            print(f"  [{idx:2d}/{len(SEEDS)}] Seed {seed:3d}...", end=" ", flush=True)
            proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if proc.returncode != 0:
                print(f"FAILED!\nError: {proc.stderr}")
                continue

            with open(out_json, "r") as f:
                res = json.load(f)

            status = "✅ PASS" if res["joint_pass"] else "❌ FAIL"
            print(f"{status} | Clr={res['max_clearance_cm']:5.2f}cm | FwdDist={res['com_forward_dist_cm']:+6.2f}cm | S0_Vy={res['step0_base_vy_mps']:+6.3f}m/s | S5_Vy={res['step5_base_vy_mps']:+6.3f}m/s | TO_Vy={res['takeoff_com_vy_mps']:+6.3f}m/s | TO_Vz={res['takeoff_com_vz_mps']:+5.3f}m/s | Class={res['jump_class']}")
            mode_runs.append(res)

        all_mode_results[mode_name] = mode_runs

    # --------------------------------------------------------------------------
    # Generate Phase 3.2 Deliverables
    # --------------------------------------------------------------------------
    print("\n==============================================================================")
    print(" Compiling Phase 3.2 Audit Deliverables")
    print("==============================================================================")

    # 1. Reset Timeline Analysis across 20 seeds (from Mode A)
    mode_A_runs = all_mode_results["Mode_A_control"]
    mode_D_runs = all_mode_results["Mode_D_clean_start"]

    timeline_rows = []
    timeline_headers = [
        "seed",
        "state_A_base_vy", "state_A_base_vz", "state_A_pitch", "state_A_roll", "state_A_yaw",
        "state_A_w_L", "state_A_w_R",
        "state_B_base_vy", "state_B_base_vz", "state_B_pitch", "state_B_roll", "state_B_yaw",
        "state_B_w_L", "state_B_w_R",
        "delta_vy_0_02s", "fn_L_0_02s", "fn_R_0_02s", "friction_L_0_02s", "friction_R_0_02s",
        "state_C_base_vy", "step5_base_vy"
    ]

    for run in mode_A_runs:
        s = run["seed"]
        sA = run["state_A"]
        sB = run["state_B"]
        cp = run["contact_process_0_02s"]
        sC = run["state_C"]
        s5 = run["step5_base_vy_mps"]

        timeline_rows.append([
            s,
            round(sA["base_vy"], 4), round(sA["base_vz"], 4), round(sA["base_pitch_deg"], 2), round(sA["base_roll_deg"], 2), round(sA["base_yaw_deg"], 2),
            round(sA["wheel_w_L_radps"], 4), round(sA["wheel_w_R_radps"], 4),
            round(sB["base_vy"], 4), round(sB["base_vz"], 4), round(sB["base_pitch_deg"], 2), round(sB["base_roll_deg"], 2), round(sB["base_yaw_deg"], 2),
            round(sB["wheel_w_L_radps"], 4), round(sB["wheel_w_R_radps"], 4),
            round(cp["delta_base_vy"], 4), round(cp["fn_L_N"], 2), round(cp["fn_R_N"], 2), round(cp["friction_L_N"], 2), round(cp["friction_R_N"], 2),
            round(sC["base_vy"], 4), round(s5, 4)
        ])

    csv_timeline = os.path.join(output_base_dir, "reset_timeline_20seed.csv")
    save_csv(csv_timeline, timeline_rows, timeline_headers)

    # 2. Section V Table: Per-Seed Vy Progression
    # | Seed | reset前 Vy | reset后 Vy | 正式控制开始 Vy | Step 5 Vy |
    vy_progression_rows = []
    vy_progression_headers = ["Seed", "reset前 Vy (m/s)", "reset后 Vy (m/s)", "正式控制开始 Vy (m/s)", "Step 5 Vy (m/s)"]
    vys_after = []

    for run in mode_A_runs:
        s = run["seed"]
        v_pre = run["state_A"]["base_vy"]
        v_post = run["state_B"]["base_vy"]
        v_start = run["state_C"]["base_vy"]
        v_s5 = run["step5_base_vy_mps"]
        vys_after.append(v_post)

        vy_progression_rows.append([s, f"{v_pre:+.4f}", f"{v_post:+.4f}", f"{v_start:+.4f}", f"{v_s5:+.4f}"])

    csv_progression = os.path.join(output_base_dir, "vy_progression_20seed.csv")
    save_csv(csv_progression, vy_progression_rows, vy_progression_headers)

    # Stats for Vy_reset_after
    vys_after_arr = np.array(vys_after)
    vy_after_mean = float(np.mean(vys_after_arr))
    vy_after_std = float(np.std(vys_after_arr))
    vy_after_min = float(np.min(vys_after_arr))
    vy_after_max = float(np.max(vys_after_arr))
    count_abs_gt_005 = int(np.sum(np.abs(vys_after_arr) > 0.05))
    count_abs_gt_010 = int(np.sum(np.abs(vys_after_arr) > 0.10))

    reset_audit_stats = {
        "vy_reset_after_mean_mps": vy_after_mean,
        "vy_reset_after_std_mps": vy_after_std,
        "vy_reset_after_min_mps": vy_after_min,
        "vy_reset_after_max_mps": vy_after_max,
        "count_abs_gt_005_mps": count_abs_gt_005,
        "count_abs_gt_010_mps": count_abs_gt_010
    }
    with open(os.path.join(output_base_dir, "reset_audit_stats.json"), "w") as f:
        json.dump(reset_audit_stats, f, indent=2)

    # 3. Section VII: Mode A vs Mode D Comparative Analysis
    def compute_mode_metrics(runs):
        n = len(runs)
        s0_vys = [r["step0_base_vy_mps"] for r in runs]
        s5_vys = [r["step5_base_vy_mps"] for r in runs]
        to_vys = [r["takeoff_com_vy_mps"] for r in runs]
        to_vzs = [r["takeoff_com_vz_mps"] for r in runs]
        dists = [r["com_forward_dist_cm"] for r in runs]
        clrs = [r["max_clearance_cm"] for r in runs]
        pitches = [abs(r["touchdown_pitch_deg"]) for r in runs]
        rolls = [abs(r["touchdown_roll_deg"]) for r in runs]
        rec_vys = [abs(r["post_recovery_vy_mps"]) for r in runs]
        abnormals = [r["abnormal_resets"] for r in runs]

        n_norm = sum(1 for r in runs if r["jump_class"] == "Class_A_normal_forward")
        n_short = sum(1 for r in runs if r["jump_class"] == "Class_B_short_jump")
        n_back = sum(1 for r in runs if r["jump_class"] == "Class_C_backward_jump")
        n_pass = sum(1 for r in runs if r["joint_pass"])

        return {
            "num_runs": n,
            "step0_vy": {"mean": float(np.mean(s0_vys)), "std": float(np.std(s0_vys)), "min": float(np.min(s0_vys)), "max": float(np.max(s0_vys))},
            "step5_vy": {"mean": float(np.mean(s5_vys)), "std": float(np.std(s5_vys)), "min": float(np.min(s5_vys)), "max": float(np.max(s5_vys))},
            "takeoff_vy": {"mean": float(np.mean(to_vys)), "std": float(np.std(to_vys)), "min": float(np.min(to_vys)), "max": float(np.max(to_vys))},
            "takeoff_vz": {"mean": float(np.mean(to_vzs)), "std": float(np.std(to_vzs)), "min": float(np.min(to_vzs)), "max": float(np.max(to_vzs))},
            "forward_displacement_cm": {"mean": float(np.mean(dists)), "std": float(np.std(dists)), "min": float(np.min(dists)), "max": float(np.max(dists))},
            "min_clearance_cm": {"mean": float(np.mean(clrs)), "std": float(np.std(clrs)), "min": float(np.min(clrs)), "max": float(np.max(clrs))},
            "touchdown_pitch_abs_deg": {"mean": float(np.mean(pitches)), "std": float(np.std(pitches)), "min": float(np.min(pitches)), "max": float(np.max(pitches))},
            "touchdown_roll_abs_deg": {"mean": float(np.mean(rolls)), "std": float(np.std(rolls)), "min": float(np.min(rolls)), "max": float(np.max(rolls))},
            "post_recovery_vy": {"mean": float(np.mean(rec_vys)), "std": float(np.std(rec_vys)), "min": float(np.min(rec_vys)), "max": float(np.max(rec_vys))},
            "abnormal_resets_total": int(np.sum(abnormals)),
            "normal_forward_count": n_norm,
            "short_jump_count": n_short,
            "backward_jump_count": n_back,
            "joint_pass_count": n_pass,
            "joint_pass_rate_pct": float(n_pass / n * 100.0)
        }

    summary_A = compute_mode_metrics(mode_A_runs)
    summary_D = compute_mode_metrics(mode_D_runs)

    comparative_summary = {
        "Mode_A_control": summary_A,
        "Mode_D_clean_start": summary_D,
        "reset_audit_stats": reset_audit_stats
    }

    with open(os.path.join(output_base_dir, "phase3_2_comparison_summary.json"), "w") as f:
        json.dump(comparative_summary, f, indent=2)

    # 4. Save mode telemetries
    def make_mode_telemetry(runs, filepath):
        headers = [
            "seed", "step0_vy_mps", "step5_vy_mps", "takeoff_vy_mps", "takeoff_vz_mps",
            "pushoff_delta_vy_mps", "com_forward_dist_cm", "max_clearance_cm",
            "touchdown_pitch_deg", "touchdown_roll_deg", "post_recovery_vy_mps",
            "abnormal_resets", "jump_class", "joint_pass"
        ]
        rows = []
        for r in runs:
            rows.append([
                r["seed"],
                round(r["step0_base_vy_mps"], 4),
                round(r["step5_base_vy_mps"], 4),
                round(r["takeoff_com_vy_mps"], 4),
                round(r["takeoff_com_vz_mps"], 4),
                round(r["pushoff_delta_com_vy_mps"], 4),
                round(r["com_forward_dist_cm"], 2),
                round(r["max_clearance_cm"], 2),
                round(r["touchdown_pitch_deg"], 2),
                round(r["touchdown_roll_deg"], 2),
                round(r["post_recovery_vy_mps"], 4),
                r["abnormal_resets"],
                r["jump_class"],
                r["joint_pass"]
            ])
        save_csv(filepath, rows, headers)

    make_mode_telemetry(mode_A_runs, os.path.join(output_base_dir, "Mode_A_control_telemetry.csv"))
    make_mode_telemetry(mode_D_runs, os.path.join(output_base_dir, "Mode_D_clean_start_telemetry.csv"))

    # 5. Save Key Trajectories (Seed 42, 2, 7, 11, 13, 17, 19, 23)
    key_dir = os.path.join(output_base_dir, "key_trajectories")
    os.makedirs(key_dir, exist_ok=True)
    key_seeds = [42, 2, 7, 11, 13, 17, 19, 23]
    for ks in key_seeds:
        run_A = next((r for r in mode_A_runs if r["seed"] == ks), None)
        run_D = next((r for r in mode_D_runs if r["seed"] == ks), None)
        if run_A and run_D:
            comp_traj = {
                "seed": ks,
                "Mode_A_control": {
                    "com_forward_dist_cm": run_A["com_forward_dist_cm"],
                    "max_clearance_cm": run_A["max_clearance_cm"],
                    "step0_vy": run_A["step0_base_vy_mps"],
                    "step5_vy": run_A["step5_base_vy_mps"],
                    "takeoff_vy": run_A["takeoff_com_vy_mps"],
                    "jump_class": run_A["jump_class"],
                    "trajectory": run_A["trajectory"]
                },
                "Mode_D_clean_start": {
                    "com_forward_dist_cm": run_D["com_forward_dist_cm"],
                    "max_clearance_cm": run_D["max_clearance_cm"],
                    "step0_vy": run_D["step0_base_vy_mps"],
                    "step5_vy": run_D["step5_base_vy_mps"],
                    "takeoff_vy": run_D["takeoff_com_vy_mps"],
                    "jump_class": run_D["jump_class"],
                    "trajectory": run_D["trajectory"]
                }
            }
            with open(os.path.join(key_dir, f"trajectory_seed_{ks}.json"), "w") as f:
                json.dump(comp_traj, f, indent=2)

    print("\n>>> Phase 3.2 Audit Batch Run Completed Successfully! <<<")


if __name__ == "__main__":
    run_all()
