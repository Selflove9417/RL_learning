import subprocess
import json
import numpy as np

seeds = [3001, 3002, 3003, 3004, 3005]
results = []

for s in seeds:
    out_file = f"/tmp/dyn_{s}.json"
    cmd = f"source activate_jump.sh && python tests/diagnose_paired_dynamics.py --seed {s} --out {out_file}"
    print(f"Running paired dynamics for seed {s}...")
    res = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error on seed {s}:\n{res.stderr}")
        continue
    with open(out_file, "r") as f:
        data = json.load(f)
        results.append(data)

combined_out = "/home/xy/IsaacTu_jump/tests/paired_dynamics_results.json"
with open(combined_out, "w") as f:
    json.dump(results, f, indent=2)

print("\n========== Dev Set (5 Seeds) Dynamics Summary ==========")
print(f"{'Seed':<6} | {'Vy_c (m/s)':<12} {'Vy_5 (m/s)':<12} | {'Vz_c (m/s)':<12} {'Vz_5 (m/s)':<12} | {'Pitch_c (deg)':<14} {'Pitch_5 (deg)':<14} | {'RewVy_c':<10} {'RewVy_5':<10}")
print("-" * 105)

for r in results:
    s = r["seed"]
    vy_c, vy_5 = r["com_vy_lift_c"], r["com_vy_lift_5"]
    vz_c, vz_5 = r["com_vz_lift_c"], r["com_vz_lift_5"]
    p_c, p_5 = r["pitch_lift_c"], r["pitch_lift_5"]
    rw_c, rw_5 = r["sum_rew_vy_c"], r["sum_rew_vy_5"]
    print(f"{s:<6} | {vy_c:<12.4f} {vy_5:<12.4f} | {vz_c:<12.4f} {vz_5:<12.4f} | {p_c:<14.2f} {p_5:<14.2f} | {rw_c:<10.3f} {rw_5:<10.3f}")

print("-" * 105)
mean_vy_c = np.mean([r["com_vy_lift_c"] for r in results])
mean_vy_5 = np.mean([r["com_vy_lift_5"] for r in results])
mean_vz_c = np.mean([r["com_vz_lift_c"] for r in results])
mean_vz_5 = np.mean([r["com_vz_lift_5"] for r in results])
mean_p_c = np.mean([r["pitch_lift_c"] for r in results])
mean_p_5 = np.mean([r["pitch_lift_5"] for r in results])
mean_rw_c = np.mean([r["sum_rew_vy_c"] for r in results])
mean_rw_5 = np.mean([r["sum_rew_vy_5"] for r in results])

print(f"{'MEAN':<6} | {mean_vy_c:<12.4f} {mean_vy_5:<12.4f} | {mean_vz_c:<12.4f} {mean_vz_5:<12.4f} | {mean_p_c:<14.2f} {mean_p_5:<14.2f} | {mean_rw_c:<10.3f} {mean_rw_5:<10.3f}")
