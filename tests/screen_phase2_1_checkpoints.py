#!/usr/bin/env python3
import os
import sys
import json
import subprocess

workspace_root = "/home/xy/IsaacTu_jump"
python_bin = "/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/bin/python"
eval_script = os.path.join(workspace_root, "tests", "eval_joint_success.py")

run_dir = "/home/xy/IsaacTu_jump/legged_gym/logs/myrobot_forward_jump_static/Oct16_15-26-49_phase2_1_clearance_priority"
checkpoints = ["model_5.pt", "model_10.pt", "model_15.pt", "model_20.pt", "model_25.pt"]
seeds = [42, 100, 200, 300, 400]

summary = {}

for ckpt in checkpoints:
    ckpt_path = os.path.join(run_dir, ckpt)
    print(f"\n=================== Evaluating {ckpt} ===================")
    ckpt_results = []
    for seed in seeds:
        out_json = f"/tmp/p21_{ckpt}_seed_{seed}.json"
        cmd = [python_bin, eval_script, "--model", ckpt_path, "--seed", str(seed), "--clip", "100.0", "--out", out_json]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{workspace_root}/legged_gym:{workspace_root}/rsl_rl:" + env.get("PYTHONPATH", "")
        env["LD_LIBRARY_PATH"] = f"/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/lib:" + env.get("LD_LIBRARY_PATH", "")

        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if proc.returncode != 0:
            print(f"Error on Seed {seed}: {proc.stderr[:200]}")
            continue
        
        with open(out_json) as f:
            res = json.load(f)

        status = "✅ PASS" if res["joint_success"] else f"❌ FAIL ({', '.join(res['failure_reasons'])})"
        print(f"Seed {seed:3d}: {status} | Clr={res['max_clearance_cm']:5.2f}cm | FwdDist={res['com_forward_dist_cm']:5.2f}cm | Vz={res['takeoff_com_vz_mps']:+.3f}m/s | Vy={res['takeoff_com_vy_mps']:+.3f}m/s | Slip={res['prep_ground_slip_cm']:.2f}cm | RecVy={res['recovery_vy_mps']:+.3f}m/s | YawDrift={res['yaw_drift_deg']:.1f}°")
        ckpt_results.append(res)
    
    n = len(ckpt_results)
    if n > 0:
        valid_rate = sum(1 for r in ckpt_results if r["valid_jump"]) / n * 100.0
        clr_rate = sum(1 for r in ckpt_results if r["c2_clearance_ok"]) / n * 100.0
        fwd_rate = sum(1 for r in ckpt_results if r["c1_fwd_dist_ok"]) / n * 100.0
        joint_rate = sum(1 for r in ckpt_results if r["joint_success"]) / n * 100.0
        
        mean_clr = sum(r["max_clearance_cm"] for r in ckpt_results) / n
        mean_dist = sum(r["com_forward_dist_cm"] for r in ckpt_results) / n
        mean_vz = sum(r["takeoff_com_vz_mps"] for r in ckpt_results) / n
        mean_vy = sum(r["takeoff_com_vy_mps"] for r in ckpt_results) / n
        
        print(f"--- {ckpt} Summary: JointPass={joint_rate:.1f}% (ClrPass={clr_rate:.1f}%, FwdPass={fwd_rate:.1f}%) | MeanClr={mean_clr:.2f}cm, MeanDist={mean_dist:.2f}cm, Vz={mean_vz:.3f}m/s, Vy={mean_vy:.3f}m/s ---")
        summary[ckpt] = {
            "joint_success_rate": joint_rate,
            "clearance_pass_rate": clr_rate,
            "forward_pass_rate": fwd_rate,
            "valid_jump_rate": valid_rate,
            "mean_clearance_cm": round(mean_clr, 2),
            "mean_forward_dist_cm": round(mean_dist, 2),
            "mean_takeoff_vz": round(mean_vz, 3),
            "mean_takeoff_vy": round(mean_vy, 3),
            "seeds": ckpt_results
        }

with open("/tmp/phase2_1_screening_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nScreening complete! Saved to /tmp/phase2_1_screening_summary.json")
