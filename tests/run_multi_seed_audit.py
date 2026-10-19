#!/usr/bin/env python3
# ==============================================================================
# Runner: Multi-Seed Benchmark for Stage 8 Baseline vs. Phase 1 Trial 2
# ==============================================================================

import os
import sys
import json
import subprocess
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEEDS = [42, 100, 200, 300, 400]
DURATION_S = 2.0  # Exactly 1 full single jump cycle

baseline_model = os.path.join(workspace_root, "evaluation_assets", "frozen_stage8", "model_40.pt")
trial2_model = os.path.join(workspace_root, "evaluation_assets", "phase1_forward_jump", "models", "model_30_yaw_aligned.pt")

output_dir = os.path.join(workspace_root, "evaluation_assets", "phase1_forward_jump", "multi_seed_benchmark")
os.makedirs(output_dir, exist_ok=True)

models = [
    {"name": "Stage 8 Baseline (In-Place)", "path": baseline_model, "is_forward": False, "tag": "stage8"},
    {"name": "Trial 2 (Forward Jump)", "path": trial2_model, "is_forward": True, "tag": "trial2"}
]

python_bin = sys.executable
worker_script = os.path.join(workspace_root, "tests", "eval_single_seed_fwd.py")

all_summaries = {}

for m in models:
    m_name = m["name"]
    m_path = m["path"]
    is_fwd = m["is_forward"]
    tag = m["tag"]

    print("=" * 80)
    print(f"🚀 Running Multi-Seed Evaluation for: {m_name}")
    print("=" * 80)

    records = []
    for seed in SEEDS:
        out_json = os.path.join(output_dir, f"{tag}_seed_{seed}.json")
        cmd = [
            python_bin, worker_script,
            "--seed", str(seed),
            "--model_path", m_path,
            "--duration", str(DURATION_S),
            "--out_json", out_json
        ]
        if is_fwd:
            cmd.append("--is_forward")

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{workspace_root}/legged_gym:{workspace_root}/rsl_rl:" + env.get("PYTHONPATH", "")

        print(f"  --> Seed {seed:3d} executing...")
        res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if res.returncode != 0:
            print(f"  [ERROR] Seed {seed} failed with code {res.returncode}")
            print(res.stderr[:500])
        else:
            with open(out_json) as f:
                data = json.load(f)
            records.append(data)
            print(f"  Seed {seed:3d}: Valid={data['valid_jump']}, Clr={data['max_clearance_cm']:5.2f}cm, "
                  f"FwdDist(Root)={data['root_forward_dist_cm']:+5.2f}cm, FwdDist(CoM)={data['com_forward_dist_cm']:+5.2f}cm, "
                  f"T_flight={data['flight_duration_s']:.3f}s, YawDrift={data['yaw_drift_deg']:4.2f}deg, "
                  f"RecVy={data['recovery_vy_mps']:+.3f}m/s, PeakKnee={data['peak_knee_tau_nm']:.1f}Nm")

    def agg(key):
        vals = [r[key] for r in records]
        return {
            "mean": round(float(np.mean(vals)), 3),
            "std": round(float(np.std(vals)), 3),
            "min": round(float(np.min(vals)), 3),
            "max": round(float(np.max(vals)), 3)
        }

    summary = {
        "model_path": m_path,
        "seeds": SEEDS,
        "total_trials": len(SEEDS),
        "valid_jumps": sum(1 for r in records if r["valid_jump"]),
        "abnormal_resets": sum(r["abnormal_resets"] for r in records),
        "max_clearance_cm": agg("max_clearance_cm"),
        "flight_duration_s": agg("flight_duration_s"),
        "root_forward_dist_cm": agg("root_forward_dist_cm"),
        "com_forward_dist_cm": agg("com_forward_dist_cm"),
        "takeoff_root_vy_mps": agg("takeoff_root_vy_mps"),
        "takeoff_root_vz_mps": agg("takeoff_root_vz_mps"),
        "takeoff_com_vy_mps": agg("takeoff_com_vy_mps"),
        "takeoff_com_vz_mps": agg("takeoff_com_vz_mps"),
        "touchdown_pitch_deg": agg("touchdown_pitch_deg"),
        "touchdown_roll_deg": agg("touchdown_roll_deg"),
        "yaw_drift_deg": agg("yaw_drift_deg"),
        "recovery_vy_mps": agg("recovery_vy_mps"),
        "recovery_vx_mps": agg("recovery_vx_mps"),
        "peak_hip_tau_nm": agg("peak_hip_tau_nm"),
        "peak_knee_tau_nm": agg("peak_knee_tau_nm"),
        "peak_wheel_tau_nm": agg("peak_wheel_tau_nm"),
        "knee_saturation_steps": agg("knee_saturation_steps"),
        "records": records
    }
    all_summaries[m_name] = summary

summary_all_path = os.path.join(output_dir, "multi_seed_comparison_summary.json")
with open(summary_all_path, "w", encoding="utf-8") as f:
    json.dump(all_summaries, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print(f"✅ Multi-Seed Benchmark Complete! Consolidated summary saved to:\n   {summary_all_path}")
print("=" * 80)
