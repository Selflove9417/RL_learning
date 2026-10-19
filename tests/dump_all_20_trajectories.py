#!/usr/bin/env python3
import os
import sys
import subprocess

workspace_root = "/home/xy/IsaacTu_jump"
python_bin = "/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/bin/python"
script = os.path.join(workspace_root, "tests", "diagnose_failure_root_cause.py")
out_dir = "/tmp/audit_trajectories"
os.makedirs(out_dir, exist_ok=True)

SEEDS = [
    42, 100, 200, 300, 400,
    1, 2, 3, 5, 7, 11, 13, 17, 19, 23,
    50, 150, 250, 350, 450
]

for seed in SEEDS:
    out_file = os.path.join(out_dir, f"traj_seed_{seed}.json")
    if os.path.exists(out_file):
        print(f"Seed {seed} already dumped.")
        continue
    print(f"Dumping trajectory for Seed {seed}...", end=" ", flush=True)
    cmd = [python_bin, script, "--seed", str(seed), "--out", out_file]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{workspace_root}/legged_gym:{workspace_root}/rsl_rl:" + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = f"/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/lib:" + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if proc.returncode == 0:
        print("Done.")
    else:
        print(f"FAILED: {proc.stderr[:100]}")
