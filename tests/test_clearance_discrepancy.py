import os
import sys
import json
import subprocess

workspace_root = "/home/xy/IsaacTu_jump"
model_path = os.path.join(workspace_root, "evaluation_assets", "phase1_forward_jump", "models", "model_30_yaw_aligned.pt")

# 1. Run eval_single_seed_fwd (Phase 1.1) on seed 42
cmd1 = [sys.executable, "tests/eval_single_seed_fwd.py", "--seed", "42", "--model_path", model_path, "--duration", "2.0", "--out_json", "/tmp/test_fwd_s42.json", "--is_forward"]
subprocess.run(cmd1)
with open("/tmp/test_fwd_s42.json") as f:
    d1 = json.load(f)

# 2. Run eval_single_seed_fwd_static (Phase 2) on seed 42
cmd2 = [sys.executable, "tests/eval_single_seed_fwd_static.py", "--seed", "42", "--model_path", model_path, "--out_json", "/tmp/test_static_s42.json"]
subprocess.run(cmd2)
with open("/tmp/test_static_s42.json") as f:
    d2 = json.load(f)

print("=== COMPARISON SEED 42 ===")
print("Phase 1.1 eval_single_seed_fwd:")
print("  Clearance:", d1['max_clearance_cm'], "cm")
print("  Takeoff Vz:", d1['takeoff_com_vz_mps'], "m/s, Vy:", d1['takeoff_com_vy_mps'], "m/s")
print("  Flight steps:", d1['flight_steps'], "Flight dur:", d1['flight_duration_s'], "s")
print("Phase 2 eval_single_seed_fwd_static:")
print("  Clearance:", d2['max_clearance_cm'], "cm")
print("  Takeoff Vz:", d2['takeoff_com_vz_mps'], "m/s, Vy:", d2['takeoff_com_vy_mps'], "m/s")
print("  Flight steps:", d2['flight_steps'], "Flight dur:", d2['flight_duration_s'], "s")
