#!/usr/bin/env python3
# ==============================================================================
# Phase 4.6-R: Runner for All 5 Dev Seeds (3001..3005)
# Executes each seed in a separate OS process to comply with Isaac Gym single-context rule.
# Aggregates results across Target A, Target B, and Target C.
# ==============================================================================

import os
import sys
import subprocess
import json
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
out_dir = os.path.join(workspace_root, "evaluation_assets", "phase4_6_critic_calibration")
os.makedirs(out_dir, exist_ok=True)

DEV_SEEDS = [3001, 3002, 3003, 3004, 3005]


def main():
    print("=" * 80)
    print("PHASE 4.6-R: AUDITING STRICTLY PAIRED DEV SEEDS (3001..3005)")
    print("=" * 80)

    per_seed_results = []

    for seed in DEV_SEEDS:
        seed_out = os.path.join(out_dir, f"audit_seed{seed}.json")
        cmd = (
            f"source /home/xy/IsaacTu_jump/activate_jump.sh && "
            f"python /home/xy/IsaacTu_jump/tests/audit_phase4_6_r_paired_seeds.py "
            f"--seed {seed} --out {seed_out}"
        )
        print(f"\n---> Executing Seed {seed}...")
        res = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(f"[ERROR] Seed {seed} failed!")
            print(res.stderr[-500:])
            continue

        with open(seed_out, "r", encoding="utf-8") as f:
            data = json.load(f)
        per_seed_results.append(data)

    print("\n" + "=" * 90)
    print("ALL SEEDS AUDIT COMPLETED. GENERATING COMPREHENSIVE COMPARISON TABLE...")
    print("=" * 90)

    # Compile Table 1: Trajectory Physical Fidelity & Actor Invariance
    print(f"\n1. PHYSICAL TRAJECTORY FIDELITY & STRICT ACTOR INVARIANCE:")
    print(f"{'Seed':<6} | {'Fwd Dist (cm)':<14} {'Clearance (cm)':<15} {'Touchdown Step':<15} | {'Actor Max Diff':<15}")
    print("-" * 75)
    for r in per_seed_results:
        print(f"{r['seed']:<6} | {r['fwd_dist_cm']:<14.2f} {r['max_clearance_cm']:<15.2f} {r['touchdown_step']:<15} | {r['actor_max_diff']:<15.2e}")

    # Compile Table 2: Terminal State Values (V(s_140_pre) vs V(s_0_post))
    print(f"\n2. TERMINAL VALUES COMPARISON: PRE-RESET s_140 vs POST-RESET s_0':")
    print(f"{'Seed':<6} | {'V_A(s140_pre)':<14} {'V_A(s0_post)':<14} {'Delta A':<10} | {'V_B(s140_pre)':<14} {'V_B(s0_post)':<14} {'Delta B':<10}")
    print("-" * 85)
    for r in per_seed_results:
        t = r['terminal_values']
        dA = t['va_post_reset'] - t['va_pre_reset']
        dB = t['vb_post_reset'] - t['vb_pre_reset']
        print(f"{r['seed']:<6} | {t['va_pre_reset']:<14.2f} {t['va_post_reset']:<14.2f} {dA:<+10.2f} | {t['vb_pre_reset']:<14.2f} {t['vb_post_reset']:<14.2f} {dB:<+10.2f}")

    # Compile Table 3: Target A (Finite Horizon) vs Target B (Pre-Reset Bootstrap) vs Target C (Post-Reset Bootstrap)
    print(f"\n3. VALUE PREDICTION MSE & EXPLAINED VARIANCE ACROSS VALUE TARGET FORMULATIONS:")
    print(f"{'Seed':<6} | {'--- Target B (Pre Bootstrap) ---':<32} | {'--- Target C (Post Bootstrap) ---':<33} | {'--- Target A (Finite Horiz) ---':<31}")
    print(f"{'':<6} | {'Model A':<9} {'Model B':<9} {'B Reduc %':<11} | {'Model A':<9} {'Model B':<9} {'B Reduc %':<11} | {'Model A':<9} {'Model B':<9} {'B Delta':<10}")
    print("-" * 105)
    for r in per_seed_results:
        st = r['stats']
        mse_b_a = st['Target_B_pre_reset_bootstrap']['model_a']['mse']
        mse_b_b = st['Target_B_pre_reset_bootstrap']['model_b']['mse']
        red_b = (mse_b_a - mse_b_b) / mse_b_a * 100.0

        mse_c_a = st['Target_C_post_reset_bootstrap']['model_a']['mse']
        mse_c_b = st['Target_C_post_reset_bootstrap']['model_b']['mse']
        red_c = (mse_c_a - mse_c_b) / mse_c_a * 100.0

        mse_a_a = st['Target_A_finite']['model_a']['mse']
        mse_a_b = st['Target_A_finite']['model_b']['mse']
        del_a = mse_a_b - mse_a_a

        print(f"{r['seed']:<6} | {mse_b_a:<9.4f} {mse_b_b:<9.4f} {red_b:<11.1f}% | {mse_c_a:<9.4f} {mse_c_b:<9.4f} {red_c:<11.1f}% | {mse_a_a:<9.4f} {mse_a_b:<9.4f} {del_a:<+10.2f}")

    # Compile Table 4: Phase-by-Phase GAE Advantages (Model A vs Model B under Target B)
    print(f"\n4. PHASE-BY-PHASE ADVANTAGES UNDER TARGET B (Pre-Reset Bootstrap):")
    print(f"{'Seed':<6} | {'Launch Adv (A / B)':<22} | {'Flight Adv (A / B)':<22} | {'Touchdown Adv (A / B)':<24} | {'Recovery Adv (A / B)':<22}")
    print("-" * 100)
    for r in per_seed_results:
        st = r['stats']['Target_B_pre_reset_bootstrap']
        print(f"{r['seed']:<6} | {st['adv_launch_a']:<9.4f} / {st['adv_launch_b']:<9.4f} | {st['adv_flight_a']:<9.4f} / {st['adv_flight_b']:<9.4f} | {st['adv_touchdown_a']:<10.4f} / {st['adv_touchdown_b']:<10.4f} | {st['adv_recovery_a']:<9.4f} / {st['adv_recovery_b']:<9.4f}")

    # Compile Table 5: Terminal Step 139 TD Residuals
    print(f"\n5. STEP 139 TD RESIDUAL UNDER TARGET A vs TARGET B:")
    print(f"{'Seed':<6} | {'Target A TD_139 (A / B)':<26} | {'Target B TD_139 (A / B)':<26} | {'Mean TD_Res B (A / B)':<22}")
    print("-" * 85)
    for r in per_seed_results:
        st_a = r['stats']['Target_A_finite']
        st_b = r['stats']['Target_B_pre_reset_bootstrap']
        print(f"{r['seed']:<6} | {st_a['td_step139_a']:<10.4f} / {st_a['td_step139_b']:<12.4f} | {st_b['td_step139_a']:<10.4f} / {st_b['td_step139_b']:<12.4f} | {st_b['td_mean_a']:<9.4f} / {st_b['td_mean_b']:<9.4f}")

    # Distance vs Advantage Correlation Analysis under Target B
    fwds = [r['fwd_dist_cm'] for r in per_seed_results]
    launch_as = [r['stats']['Target_B_pre_reset_bootstrap']['adv_launch_a'] for r in per_seed_results]
    launch_bs = [r['stats']['Target_B_pre_reset_bootstrap']['adv_launch_b'] for r in per_seed_results]
    flight_as = [r['stats']['Target_B_pre_reset_bootstrap']['adv_flight_a'] for r in per_seed_results]
    flight_bs = [r['stats']['Target_B_pre_reset_bootstrap']['adv_flight_b'] for r in per_seed_results]

    corr_launch_a = float(np.corrcoef(fwds, launch_as)[0, 1])
    corr_launch_b = float(np.corrcoef(fwds, launch_bs)[0, 1])
    corr_flight_a = float(np.corrcoef(fwds, flight_as)[0, 1])
    corr_flight_b = float(np.corrcoef(fwds, flight_bs)[0, 1])

    print(f"\nForward Distance Correlation (Target B):")
    print(f"  Launch Adv  : Model A = {corr_launch_a:+.4f} | Model B = {corr_launch_b:+.4f}")
    print(f"  Flight Adv  : Model A = {corr_flight_a:+.4f} | Model B = {corr_flight_b:+.4f}")

    # Reward totals breakdown (mean across 5 seeds)
    all_keys = list(per_seed_results[0]['reward_totals'].keys())
    mean_rewards = {k: float(np.mean([r['reward_totals'][k] for r in per_seed_results])) for k in all_keys}

    print(f"\nMean Cumulative Reward Contributions across 5 Dev Seeds:")
    for k, v in sorted(mean_rewards.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {k:<32}: {v:+.4f}")

    # Save aggregated audit
    out_file = os.path.join(out_dir, "dev_seeds_phase4_6_r_audit.json")
    save_obj = {
        "per_seed_results": per_seed_results,
        "correlations": {
            "launch_adv_model_a": corr_launch_a,
            "launch_adv_model_b": corr_launch_b,
            "flight_adv_model_a": corr_flight_a,
            "flight_adv_model_b": corr_flight_b,
        },
        "mean_rewards": mean_rewards,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, indent=2)
    print(f"\nSaved full Phase 4.6-R audit to: {out_file}")


if __name__ == "__main__":
    main()
