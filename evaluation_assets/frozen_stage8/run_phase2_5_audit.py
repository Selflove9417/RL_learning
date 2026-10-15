"""
Phase 2.5 5-Seed Comparative Audit Script
-----------------------------------------
Runs Seed 1 to 5 for:
1. Baseline (Stage 8 as-is with randomized velocity reset)
2. Optimized (Stage 8 with zero initial velocity reset)
Saves per-seed JSONs and consolidated summary.
"""

import os
import sys
import json
import numpy as np

# We invoke tests/eval_per_jump.py via subprocess to cleanly isolate Isaac Gym memory between seeds
import subprocess

MODEL_PATH = '/home/xy/IsaacTu_jump/legged_gym/logs/myrobot_jump/Oct14_17-04-58_jump_stage8_earlyzero/model_40.pt'
OUT_DIR = '/home/xy/IsaacTu_jump/evaluation_assets/jump_20cm_audit/phase2_5'
SEEDS = [1, 2, 3, 4, 5]

def run_seed_eval(seed, zero_vel, prefix):
    out_file = os.path.join(OUT_DIR, f"{prefix}_seed{seed}.json")
    cmd = [
        "/home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/rl-runtime/env/bin/python",
        "/home/xy/IsaacTu_jump/tests/eval_per_jump.py",
        MODEL_PATH,
        out_file,
        str(seed),
        str(zero_vel).lower()
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "/home/xy/IsaacTu_jump/legged_gym:/home/xy/IsaacTu_jump/rsl_rl:" + env.get("PYTHONPATH", "")
    print(f"--> Running {prefix} Seed {seed} (zero_vel={zero_vel})...")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error on Seed {seed}:\n{res.stderr}")
        raise RuntimeError(f"Eval failed on Seed {seed}")
    with open(out_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"    Done {prefix} Seed {seed}: Exact 20cm={data['total_reach_20cm_count_exact']}/{data['total_jumps']} ({data['per_jump_20cm_rate_exact_pct']:.1f}%), Cycle 1={data['cycles'][0]['reach_20cm_count_exact']}/16")
    return data

def aggregate_results(seed_results):
    total_jumps = sum(r['total_jumps'] for r in seed_results)
    tot_exact = sum(r['total_reach_20cm_count_exact'] for r in seed_results)
    tot_simple = sum(r['total_reach_20cm_count_simple'] for r in seed_results)
    
    c1_exact = sum(r['cycles'][0]['reach_20cm_count_exact'] for r in seed_results)
    c2_exact = sum(r['cycles'][1]['reach_20cm_count_exact'] for r in seed_results)
    c3_exact = sum(r['cycles'][2]['reach_20cm_count_exact'] for r in seed_results)
    c_tot = sum(r['cycles'][0]['total_jumps'] for r in seed_results)
    
    mean_clr_exact = np.mean([r['per_jump_clr_exact_mean_cm'] for r in seed_results])
    mean_clr_simple = np.mean([r['per_jump_clr_simple_mean_cm'] for r in seed_results])
    mean_formula_diff = np.mean([r['mean_formula_diff_cm'] for r in seed_results])
    
    mean_td_disp = np.mean([r['touchdown_disp_mean_m'] for r in seed_results])
    mean_post_td_roll = np.mean([r['post_td_roll_disp_mean_m'] for r in seed_results])
    mean_settle_t = np.mean([r['time_touchdown_to_settled_s'] for r in seed_results])
    mean_net_6s = np.mean([r['net_displacement_6s_mean_m'] for r in seed_results])
    mean_path_6s = np.mean([r['cumul_path_dist_6s_mean_m'] for r in seed_results])
    
    max_hip_t = max(r['max_hip_torque_Nm'] for r in seed_results)
    max_knee_t = max(r['max_knee_torque_Nm'] for r in seed_results)
    max_knee_v = max(r['max_knee_vel_rads'] for r in seed_results)
    crashes = sum(r['crashed_envs'] for r in seed_results)
    
    return {
        'total_jumps': total_jumps,
        'reach_20cm_exact_count': tot_exact,
        'reach_20cm_exact_rate_pct': float(tot_exact / total_jumps * 100.0),
        'reach_20cm_simple_count': tot_simple,
        'reach_20cm_simple_rate_pct': float(tot_simple / total_jumps * 100.0),
        'cycle1_exact_count': c1_exact,
        'cycle1_exact_rate_pct': float(c1_exact / c_tot * 100.0),
        'cycle2_exact_count': c2_exact,
        'cycle2_exact_rate_pct': float(c2_exact / c_tot * 100.0),
        'cycle3_exact_count': c3_exact,
        'cycle3_exact_rate_pct': float(c3_exact / c_tot * 100.0),
        'mean_clr_exact_cm': float(mean_clr_exact),
        'mean_clr_simple_cm': float(mean_clr_simple),
        'mean_formula_diff_cm': float(mean_formula_diff),
        'touchdown_disp_mean_m': float(mean_td_disp),
        'post_td_roll_disp_mean_m': float(mean_post_td_roll),
        'time_touchdown_to_settled_s': float(mean_settle_t),
        'net_displacement_6s_mean_m': float(mean_net_6s),
        'cumul_path_dist_6s_mean_m': float(mean_path_6s),
        'max_hip_torque_Nm': float(max_hip_t),
        'max_knee_torque_Nm': float(max_knee_t),
        'max_knee_vel_rads': float(max_knee_v),
        'total_crashed_envs': crashes,
        'per_seed': seed_results
    }

def main():
    print("==================================================")
    print("  Phase 2.5: 5-Seed Comparative Audit (Seeds 1-5)")
    print("==================================================")
    
    # 1. Baseline runs (zero_vel=False)
    print("\n--- [PART 1] Running Baseline Stage 8 (zero_vel=False) ---")
    baseline_results = []
    for s in SEEDS:
        res = run_seed_eval(s, zero_vel=False, prefix="baseline")
        baseline_results.append(res)
    baseline_agg = aggregate_results(baseline_results)
    
    # 2. Optimized runs (zero_vel=True)
    print("\n--- [PART 2] Running Optimized Stage 8 (zero_vel=True) ---")
    optimized_results = []
    for s in SEEDS:
        res = run_seed_eval(s, zero_vel=True, prefix="optimized")
        optimized_results.append(res)
    optimized_agg = aggregate_results(optimized_results)
    
    summary = {
        'baseline_random_reset': baseline_agg,
        'optimized_zero_vel_reset': optimized_agg,
    }
    
    out_summary_file = os.path.join(OUT_DIR, "audit_summary.json")
    with open(out_summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nConsolidated audit summary saved to {out_summary_file}")
    
    print("\n=================== COMPARATIVE RESULTS ===================")
    print(f"Metric                      | Baseline (Random Vel) | Optimized (Zero Vel)")
    print("----------------------------+-----------------------+---------------------")
    print(f"Cycle 1 >= 20cm (80 jumps)   | {baseline_agg['cycle1_exact_count']:2d}/80 ({baseline_agg['cycle1_exact_rate_pct']:5.2f}%)       | {optimized_agg['cycle1_exact_count']:2d}/80 ({optimized_agg['cycle1_exact_rate_pct']:5.2f}%)")
    print(f"Cycle 2 >= 20cm (80 jumps)   | {baseline_agg['cycle2_exact_count']:2d}/80 ({baseline_agg['cycle2_exact_rate_pct']:5.2f}%)       | {optimized_agg['cycle2_exact_count']:2d}/80 ({optimized_agg['cycle2_exact_rate_pct']:5.2f}%)")
    print(f"Cycle 3 >= 20cm (80 jumps)   | {baseline_agg['cycle3_exact_count']:2d}/80 ({baseline_agg['cycle3_exact_rate_pct']:5.2f}%)      | {optimized_agg['cycle3_exact_count']:2d}/80 ({optimized_agg['cycle3_exact_rate_pct']:5.2f}%)")
    print(f"Overall Exact (240 jumps)   | {baseline_agg['reach_20cm_exact_count']:3d}/240 ({baseline_agg['reach_20cm_exact_rate_pct']:5.2f}%)     | {optimized_agg['reach_20cm_exact_count']:3d}/240 ({optimized_agg['reach_20cm_exact_rate_pct']:5.2f}%)")
    print(f"Overall Simple (240 jumps)  | {baseline_agg['reach_20cm_simple_count']:3d}/240 ({baseline_agg['reach_20cm_simple_rate_pct']:5.2f}%)     | {optimized_agg['reach_20cm_simple_count']:3d}/240 ({optimized_agg['reach_20cm_simple_rate_pct']:5.2f}%)")
    print(f"Mean Exact Clearance        | {baseline_agg['mean_clr_exact_cm']:5.2f} cm              | {optimized_agg['mean_clr_exact_cm']:5.2f} cm")
    print(f"Mean Simple Clearance       | {baseline_agg['mean_clr_simple_cm']:5.2f} cm              | {optimized_agg['mean_clr_simple_cm']:5.2f} cm")
    print(f"Formula Discrepancy (mm)    | {baseline_agg['mean_formula_diff_cm']*10:5.2f} mm              | {optimized_agg['mean_formula_diff_cm']*10:5.2f} mm")
    print(f"Touchdown Disp Mean         | {baseline_agg['touchdown_disp_mean_m']:5.3f} m              | {optimized_agg['touchdown_disp_mean_m']:5.3f} m")
    print(f"Post-TD Roll Disp Mean      | {baseline_agg['post_td_roll_disp_mean_m']:5.3f} m              | {optimized_agg['post_td_roll_disp_mean_m']:5.3f} m")
    print(f"Settling Time Mean          | {baseline_agg['time_touchdown_to_settled_s']:5.3f} s              | {optimized_agg['time_touchdown_to_settled_s']:5.3f} s")
    print(f"6s Net Displacement Mean    | {baseline_agg['net_displacement_6s_mean_m']:5.3f} m              | {optimized_agg['net_displacement_6s_mean_m']:5.3f} m")
    print(f"6s Cumulative Path Dist     | {baseline_agg['cumul_path_dist_6s_mean_m']:5.3f} m              | {optimized_agg['cumul_path_dist_6s_mean_m']:5.3f} m")
    print(f"Peak Hip Torque (<=140Nm)   | {baseline_agg['max_hip_torque_Nm']:5.1f} Nm              | {optimized_agg['max_hip_torque_Nm']:5.1f} Nm")
    print(f"Peak Knee Torque (<=120Nm)  | {baseline_agg['max_knee_torque_Nm']:5.1f} Nm              | {optimized_agg['max_knee_torque_Nm']:5.1f} Nm")
    print(f"Peak Knee Vel (<=14rad/s)   | {baseline_agg['max_knee_vel_rads']:5.2f} rad/s           | {optimized_agg['max_knee_vel_rads']:5.2f} rad/s")
    print(f"Total Collisions / Resets   | {baseline_agg['total_crashed_envs']:2d}                    | {optimized_agg['total_crashed_envs']:2d}")
    print("===========================================================")

if __name__ == '__main__':
    main()
