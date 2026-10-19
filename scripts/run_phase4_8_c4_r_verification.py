#!/usr/bin/env python3
import os
import sys
import subprocess
import json
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
eval_script = os.path.join(workspace_root, "tests/eval_phase4_8_c4_r_zerocmd.py")
activate_script = os.path.join(workspace_root, "activate_jump.sh")
model_path = os.path.join(workspace_root, "evaluation_assets/phase4_8_pilot_b/model_pilot_b_step1.pt")

out_dir = os.path.join(workspace_root, "evaluation_assets/phase4_8_pilot_b/zerocmd_verification")
os.makedirs(out_dir, exist_ok=True)

seeds = [3001, 3002, 3003, 3004, 3005]

results = {
    "group_a_baseline": {},
    "group_b_zerocmd": {}
}

print("=" * 80)
print("PHASE 4.8-C4-R: ZEROCMD RIGOROUS PAIRED 5-SEED VERIFICATION")
print(f"Model Path: {model_path}")
print("=" * 80)

# 1. Group A: Baseline (zero_cmd = False)
print("\n[GROUP A: BASELINE PILOT B (zero_cmd = False)]")
for seed in seeds:
    out_file = os.path.join(out_dir, f"group_a_baseline_seed{seed}.json")
    cmd = f"source {activate_script} && python {eval_script} --seed {seed} --model \"{model_path}\" --out \"{out_file}\""
    print(f"  Running Seed {seed}...")
    res = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"Error on seed {seed}:\n{res.stderr[-300:]}")
        sys.exit(1)
    with open(out_file) as f:
        d = json.load(f)
    results["group_a_baseline"][seed] = d
    print(f"    Seed {seed}: Fwd={d['fwd_dist_cm']:.2f}cm, Clr={d['max_clearance_cm']:.2f}cm, C4={d['post_rec_vy_last10']:.4f} m/s (Pass C4: {d['passed_rec_vy_003']}, Joint Pass: {d['joint_passed_phase4']})")

# 2. Group B: ZeroCmd (zero_cmd = True)
print("\n[GROUP B: ZEROCMD PILOT B (zero_cmd = True)]")
for seed in seeds:
    out_file = os.path.join(out_dir, f"group_b_zerocmd_seed{seed}.json")
    cmd = f"source {activate_script} && python {eval_script} --seed {seed} --model \"{model_path}\" --zero_cmd --out \"{out_file}\""
    print(f"  Running Seed {seed}...")
    res = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"Error on seed {seed}:\n{res.stderr[-300:]}")
        sys.exit(1)
    with open(out_file) as f:
        d = json.load(f)
    results["group_b_zerocmd"][seed] = d
    print(f"    Seed {seed}: Fwd={d['fwd_dist_cm']:.2f}cm, Clr={d['max_clearance_cm']:.2f}cm, C4={d['post_rec_vy_last10']:.4f} m/s (Pass C4: {d['passed_rec_vy_003']}, Joint Pass: {d['joint_passed_phase4']})")

# Summary saving
summary_file = os.path.join(out_dir, "zerocmd_paired_verification_summary.json")
with open(summary_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved full paired verification summary to: {summary_file}")
