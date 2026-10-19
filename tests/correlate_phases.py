#!/usr/bin/env python3
import json

with open("/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/20seed_robustness/phase3_1_champion_20seed_summary.json") as f:
    summary = json.load(f)

SEEDS = summary["metadata"]["seeds"]
res_map = {r["seed"]: r for r in summary["all_seeds_results"]}

print(f"{'Seed':<5} | {'Status':<6} | {'FwdDist':<8} | {'ClrMin':<7} | {'Step0_CoM_Vy':<13} | {'Step5_CoM_Vy':<13} | {'Takeoff_Vy':<11} | {'Delta_Push_Vy':<14} | {'Step0_Pitch':<12} | {'Step5_Pitch':<12} | {'Friction':<9}")
print("-" * 135)

for s in SEEDS:
    with open(f"/tmp/audit_trajectories/traj_seed_{s}.json") as f:
        t = json.load(f)
    r = res_map[s]
    status = "PASS" if r["joint_success"] else "FAIL"
    f0 = t["frames"][0]
    f5 = t["frames"][5]
    f_to = t["frames"][t["takeoff_step"]]
    to_vy = f_to["com_vel_y_mps"]
    s0_vy = f0["com_vel_y_mps"]
    s5_vy = f5["com_vel_y_mps"]
    delta_vy = to_vy - s5_vy
    fric = t["friction_coefficient"]

    print(f"{s:<5d} | {status:<6} | {r['com_forward_dist_cm']:<+8.2f} | {r['max_clearance_cm']:<7.2f} | {s0_vy:<+13.3f} | {s5_vy:<+13.3f} | {to_vy:<+11.3f} | {delta_vy:<+14.3f} | {f0['pitch_deg']:<+12.2f} | {f5['pitch_deg']:<+12.2f} | {fric:<9.4f}")
