"""
Phase 11 vs Champion v4: Formal Comparison Script.
"""

import json
import numpy as np

def main():
    with open("evaluation_assets/phase8b/results/eval_phase8b_m10_formal_results.json") as f:
        d_v4 = json.load(f)["phase8b_m10_formal"]

    with open("evaluation_assets/phase11/results/eval_phase11_formal_results.json") as f:
        d_m20 = json.load(f)["phase11_m20"]

    groups = ["group1_squat", "group2_rise", "group3_regression", "group4_stress"]

    print("=" * 115)
    print(f"{'Case':<24} | {'v4 Joint%':<10} | {'m20 Joint%':<10} | {'dJoint%':<9} | {'v4 Env%':<9} | {'m20 Env%':<9} | {'dEnv%':<8} | {'v4 Herr':<8} | {'m20 Herr':<8}")
    print("=" * 115)

    all_j4, all_j20, all_e4, all_e20 = [], [], [], []

    for g in groups:
        print(f"[{g}]")
        for case in d_v4[g]:
            c4 = d_v4[g][case]
            c20 = d_m20[g][case]
            j4, j20 = c4["joint_pass_rate"], c20["joint_pass_rate"]
            e4, e20 = c4["env_pass_rate"], c20["env_pass_rate"]
            h4, h20 = c4["h_signed_err_cm"], c20["h_signed_err_cm"]
            dj = j20 - j4
            de = e20 - e4
            all_j4.append(j4)
            all_j20.append(j20)
            all_e4.append(e4)
            all_e20.append(e20)
            print(f"  {case:<22} | {j4:>9.1f}% | {j20:>9.1f}% | {dj:>+8.1f}% | {e4:>8.1f}% | {e20:>8.1f}% | {de:>+7.1f}% | {h4:>+6.2f}cm | {h20:>+6.2f}cm")

    print("=" * 115)
    print(f"GLOBAL (21 Cases):")
    print(f"  Joint% : Champion v4 = {np.mean(all_j4):.2f}%, Phase 11 m20 = {np.mean(all_j20):.2f}% (delta: {np.mean(all_j20)-np.mean(all_j4):+.2f}%)")
    print(f"  Env%   : Champion v4 = {np.mean(all_e4):.2f}%, Phase 11 m20 = {np.mean(all_e20):.2f}% (delta: {np.mean(all_e20)-np.mean(all_e4):+.2f}%)")
    print("=" * 115)

if __name__ == '__main__':
    main()
