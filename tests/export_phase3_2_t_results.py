#!/usr/bin/env python3
import os
import sys
import json
import subprocess

workspace_root = "/home/xy/IsaacTu_jump"
seeds = [100, 2, 17, 23]
results = {}

for s in seeds:
    cmd = [
        "bash", "-c",
        f"source /home/xy/IsaacTu_jump/activate_jump.sh && python tests/test_phase3_2_t_stable_entry.py --seed {s}"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = proc.stdout
    if "===JSON_START===" in out and "===JSON_END===" in out:
        json_str = out.split("===JSON_START===")[1].split("===JSON_END===")[0].strip()
        results[str(s)] = json.loads(json_str)
    else:
        print(f"Error on seed {s}:")
        print(out)
        print(proc.stderr)

out_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_2_t_ground_reset/phase3_2_t_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Exported Phase 3.2-T results to {out_path}")
