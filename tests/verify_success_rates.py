import os
import glob
import json
import numpy as np

def verify_success_rates():
    eval_dir = "/home/xy/IsaacTu_jump/evaluation_assets/phase4_4_pilot/eval_results"
    models = ["Baseline_Champion", "PPO_Pilot_Iter1", "PPO_Pilot_Iter3", "PPO_Pilot_Iter5"]
    seeds = [3001, 3002, 3003, 3004, 3005]

    print("=== Task 4: Success Rate Dual-Standard Audit ===")

    for m in models:
        print(f"\n==========================================")
        print(f"Model: {m}")
        print(f"==========================================")
        print(f"{'Seed':<6} | {'Fwd(cm)':<8} {'Clr(cm)':<8} {'Pitch':<8} {'Roll':<8} {'RecVy':<10} {'RunUp':<8} {'Reset':<6} | {'Orig(>=8cm)':<12} {'Ph4(>=12cm)':<12}")
        print("-" * 95)

        orig_passes = 0
        ph4_passes = 0

        for s in seeds:
            path = os.path.join(eval_dir, f"{m}_seed_{s}.json")
            with open(path) as f:
                d = json.load(f)

            fwd = d["fwd_dist_cm"]
            clr = d["max_clearance_cm"]
            pitch = abs(d["touchdown_pitch_deg"])
            roll = abs(d["touchdown_roll_deg"])
            rec_vy = abs(d["post_rec_vy_last10"])
            prep_runup = d["passed_pre_runup"]
            abnormal_resets = d["abnormal_resets"]

            # Criteria:
            c_fwd_8 = fwd >= 8.0
            c_fwd_12 = fwd >= 12.0
            c_clr = clr >= 20.0
            c_pitch = pitch <= 10.0
            c_roll = roll <= 10.0
            c_rec = rec_vy <= 0.03
            c_runup = prep_runup
            c_reset = abnormal_resets == 0

            orig_pass = c_fwd_8 and c_clr and c_pitch and c_roll and c_rec and c_runup and c_reset
            ph4_pass = c_fwd_12 and c_clr and c_pitch and c_roll and c_rec and c_runup and c_reset

            if orig_pass:
                orig_passes += 1
            if ph4_pass:
                ph4_passes += 1

            o_str = "PASS" if orig_pass else "FAIL"
            p_str = "PASS" if ph4_pass else "FAIL"

            print(f"{s:<6} | {fwd:<8.2f} {clr:<8.2f} {pitch:<8.2f} {roll:<8.2f} {rec_vy:<10.4f} {str(prep_runup):<8} {abnormal_resets:<6} | {o_str:<12} {p_str:<12}")

        print("-" * 95)
        print(f"Summary for {m}:")
        print(f"  Original Joint Pass Rate (d >= 8cm) : {orig_passes}/{len(seeds)} ({orig_passes/len(seeds)*100:.1f}%)")
        print(f"  Phase 4 Joint Pass Rate  (d >= 12cm): {ph4_passes}/{len(seeds)} ({ph4_passes/len(seeds)*100:.1f}%)")

if __name__ == "__main__":
    verify_success_rates()
