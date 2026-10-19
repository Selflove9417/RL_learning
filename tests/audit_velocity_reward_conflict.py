#!/usr/bin/env python3
# ==============================================================================
# Phase 4.6: Takeoff Velocity Reward Conflict Residual Impact Audit (Section VII)
# ==============================================================================

import os
import json
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
out_dir = os.path.join(workspace_root, "evaluation_assets", "phase4_6_critic_calibration")
os.makedirs(out_dir, exist_ok=True)

def run_velocity_conflict_audit():
    target_vy = 0.22
    sigma = 0.06
    scale_takeoff = 3.0
    scale_air = 3.5
    scale_td = 2.0
    sig_center = 0.110
    sig_scale = 0.025
    t_flight = 0.407  # ~0.407s flight time (vz ~ 2.0 m/s -> 2 * vz / g)

    vy_points = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.32, 0.35, 0.38, 0.42]

    table = []
    for vy in vy_points:
        d_fwd = vy * t_flight # approximate forward flight distance (m)

        # 1. Gaussian Takeoff Velocity Reward
        diff = vy - target_vy
        gauss_val = scale_takeoff * np.exp(-(diff**2) / (2.0 * sigma**2))
        gauss_grad = -scale_takeoff * (diff / (sigma**2)) * np.exp(-(diff**2) / (2.0 * sigma**2))

        # 2. Sigmoid Distance Rewards (approximate cumulative in flight: ~20 flight steps + 1 TD step)
        z = (d_fwd - sig_center) / sig_scale
        sig_score = 1.0 / (1.0 + np.exp(-z))
        sig_grad = (np.exp(-z) / ((1.0 + np.exp(-z))**2)) / sig_scale

        # Flight reward over ~20 steps (with clearance gate = 1.0)
        cum_air_rew = 20 * scale_air * (0.05 * sig_score) # dt * scale * score
        cum_td_rew = scale_td * sig_score

        # Takeoff reward over ~3 push-off steps
        cum_takeoff_rew = 3 * gauss_val

        cum_fwd_incentive = cum_air_rew + cum_td_rew
        cum_total = cum_takeoff_rew + cum_fwd_incentive

        table.append({
            "vy_m_s": vy,
            "est_fwd_dist_cm": round(d_fwd * 100.0, 2),
            "takeoff_rew_step": round(gauss_val, 4),
            "takeoff_rew_grad": round(gauss_grad, 4),
            "sigmoid_score": round(sig_score, 4),
            "cum_takeoff_rew_3steps": round(cum_takeoff_rew, 4),
            "cum_distance_rew": round(cum_fwd_incentive, 4),
            "cum_combined_rew": round(cum_total, 4),
        })

    # Net delta between baseline (0.22 m/s) and target (0.35 m/s)
    base_entry = next(e for e in table if e["vy_m_s"] == 0.22)
    target_entry = next(e for e in table if e["vy_m_s"] == 0.35)

    delta_takeoff = target_entry["cum_takeoff_rew_3steps"] - base_entry["cum_takeoff_rew_3steps"]
    delta_distance = target_entry["cum_distance_rew"] - base_entry["cum_distance_rew"]
    net_reward_delta = delta_takeoff + delta_distance

    report = {
        "parameters": {
            "target_vy": target_vy,
            "sigma": sigma,
            "scale_takeoff": scale_takeoff,
            "scale_air": scale_air,
            "scale_td": scale_td,
            "sigmoid_center_m": sig_center,
            "sigmoid_scale_m": sig_scale,
            "t_flight_s": t_flight,
        },
        "velocity_progression_table": table,
        "conflict_analysis": {
            "baseline_vy": 0.22,
            "target_15cm_vy": 0.35,
            "takeoff_reward_delta": round(delta_takeoff, 4),
            "distance_reward_delta": round(delta_distance, 4),
            "net_reward_delta": round(net_reward_delta, 4),
            "gradient_at_035ms": target_entry["takeoff_rew_grad"],
            "conclusion": (
                "Even with perfect Critic value calibration, the Gaussian takeoff velocity reward strongly opposes forward acceleration. "
                f"Increasing takeoff velocity from 0.22 m/s to 0.35 m/s incurs a penalty of {delta_takeoff:.2f} points from the velocity reward, "
                f"which completely swamps the +{delta_distance:.2f} points gain from the Sigmoid distance reward, resulting in a net negative delta of {net_reward_delta:.2f} points. "
                "The velocity reward peak acts as an artificial energy barrier preventing policy progression toward 15cm jumps."
            )
        }
    }

    out_file = os.path.join(out_dir, "velocity_reward_conflict_audit.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("PHASE 4.6 SECTION VII: VELOCITY REWARD CONFLICT AUDIT")
    print("=" * 80)
    print(f"{'Vy (m/s)':<10} | {'Est Fwd (cm)':<14} | {'Takeoff Rew':<12} {'dRew/dVy':<12} | {'Cum Takeoff':<12} {'Cum Dist':<12} {'Combined':<12}")
    print("-" * 88)
    for row in table:
        print(f"{row['vy_m_s']:<10.2f} | {row['est_fwd_dist_cm']:<14.2f} | {row['takeoff_rew_step']:<12.4f} {row['takeoff_rew_grad']:<12.4f} | {row['cum_takeoff_rew_3steps']:<12.4f} {row['cum_distance_rew']:<12.4f} {row['cum_combined_rew']:<12.4f}")

    print("\n" + "-" * 80)
    print(f"Baseline (0.22 m/s) -> Target (0.35 m/s):")
    print(f"  Takeoff Velocity Reward Delta: {delta_takeoff:+.4f}")
    print(f"  Distance Reward Delta        : {delta_distance:+.4f}")
    print(f"  Net Combined Reward Delta    : {net_reward_delta:+.4f} (NEGATIVE INCENTIVE!)")
    print(f"  Gradient at 0.35 m/s         : {target_entry['takeoff_rew_grad']:.4f} (HEAVILY OPPOSING)")
    print("-" * 80)
    print(f"Report saved to: {out_file}")

if __name__ == "__main__":
    run_velocity_conflict_audit()
