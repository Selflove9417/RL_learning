#!/usr/bin/env python3
# ==============================================================================
# Phase 4.4 Pilot Evaluation Runner
# Evaluates Baseline Champion, Iter 1, Iter 3, Iter 5 across Dev Seeds [3001..3005]
# ==============================================================================

import os
import sys
import json
import subprocess
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
eval_script = os.path.join(workspace_root, "tests", "eval_phase4_seed.py")
output_dir = os.path.join(workspace_root, "evaluation_assets", "phase4_4_pilot", "eval_results")
os.makedirs(output_dir, exist_ok=True)

models = {
    "Baseline_Champion": "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt",
    "PPO_Pilot_Iter1": "/home/xy/IsaacTu_jump/legged_gym/logs/myrobot_forward_jump_phase4/Oct17_10-50-44_phase4_4_pilot_sigmoid_5iter/model_1.pt",
    "PPO_Pilot_Iter3": "/home/xy/IsaacTu_jump/legged_gym/logs/myrobot_forward_jump_phase4/Oct17_10-50-44_phase4_4_pilot_sigmoid_5iter/model_3.pt",
    "PPO_Pilot_Iter5": "/home/xy/IsaacTu_jump/legged_gym/logs/myrobot_forward_jump_phase4/Oct17_10-50-44_phase4_4_pilot_sigmoid_5iter/model_5.pt",
}

dev_seeds = [3001, 3002, 3003, 3004, 3005]
summary_results = {}

for model_name, model_path in models.items():
    summary_results[model_name] = []
    print(f"\n=======================================================")
    print(f"Evaluating Model: {model_name}")
    print(f"Path: {model_path}")
    print(f"=======================================================")
    
    for seed in dev_seeds:
        out_json = os.path.join(output_dir, f"{model_name}_seed_{seed}.json")
        cmd = [
            "/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/bin/python",
            eval_script,
            "--seed", str(seed),
            "--jump_model", model_path,
            "--hip_bias", "0.10",
            "--phase_threshold", "0.22",
            "--out", out_json
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{workspace_root}/legged_gym:{workspace_root}/rsl_rl:" + env.get("PYTHONPATH", "")
        
        print(f"Running seed {seed}...", end=" ", flush=True)
        ret = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if ret.returncode != 0:
            print(f"FAILED (returncode {ret.returncode})")
            print(ret.stderr[-500:])
            continue
            
        with open(out_json, "r") as f:
            data = json.load(f)
        summary_results[model_name].append(data)
        print(f"Done. fwd={data['fwd_dist_cm']:.2f}cm, clr={data['max_clearance_cm']:.2f}cm, pass_12={data['passed_fwd_12cm']}, joint={data['joint_passed_phase4']}")

# Save aggregated summary
summary_file = os.path.join(output_dir, "phase4_4_pilot_summary.json")
with open(summary_file, "w") as f:
    json.dump(summary_results, f, indent=2)

print("\n\n" + "="*80)
print("PHASE 4.4 PILOT EVALUATION SUMMARY (Dev Seeds 3001-3005)")
print("="*80)

# Print Markdown Table
header = "| Model | Seed | Fwd Dist (cm) | Max Clr (cm) | Pitch (deg) | Roll (deg) | Rec Vy (m/s) | Pass >=12cm | Strict Joint Pass |"
sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
print(header)
print(sep)

for model_name, res_list in summary_results.items():
    for r in res_list:
        print(f"| {model_name} | {r['seed']} | {r['fwd_dist_cm']:.2f} | {r['max_clearance_cm']:.2f} | {r['touchdown_pitch_deg']:.2f} | {r['touchdown_roll_deg']:.2f} | {r['post_rec_vy_last10']:.4f} | {r['passed_fwd_12cm']} | {r['joint_passed_phase4']} |")

print("\n" + "="*80)
print("AGGREGATE METRICS COMPARISON")
print("="*80)
print(f"| Model | Mean Fwd (cm) | Min Fwd (cm) | Max Fwd (cm) | Mean Clr (cm) | Mean Pitch (deg) | Mean Roll (deg) | Fwd >=12cm Rate | Strict Joint Pass Rate |")
print(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

for model_name, res_list in summary_results.items():
    if not res_list:
        continue
    fwds = [r['fwd_dist_cm'] for r in res_list]
    clrs = [r['max_clearance_cm'] for r in res_list]
    pitches = [r['touchdown_pitch_deg'] for r in res_list]
    rolls = [r['touchdown_roll_deg'] for r in res_list]
    pass12_cnt = sum(1 for r in res_list if r['passed_fwd_12cm'])
    joint_cnt = sum(1 for r in res_list if r['joint_passed_phase4'])
    n = len(res_list)
    print(f"| {model_name} | {np.mean(fwds):.2f} ± {np.std(fwds):.2f} | {np.min(fwds):.2f} | {np.max(fwds):.2f} | {np.mean(clrs):.2f} | {np.mean(pitches):.2f} | {np.mean(rolls):.2f} | {pass12_cnt}/{n} ({pass12_cnt/n*100:.0f}%) | {joint_cnt}/{n} ({joint_cnt/n*100:.0f}%) |")

