import json
import os
import numpy as np

with open("evaluation_assets/phase6b/results/eval_phase6b_pilot_results.json") as f:
    eval_data = json.load(f)

dirs = {
    "Group A (beta=0.00)": "legged_gym/logs/myrobot/Oct11_14-48-28_phase6b_pilot_groupA_baseline_beta0",
    "Group B (beta=0.05)": "legged_gym/logs/myrobot/Oct11_14-48-53_phase6b_pilot_groupB_ref_kl_beta005",
    "Group C (beta=0.20)": "legged_gym/logs/myrobot/Oct11_14-49-18_phase6b_pilot_groupC_ref_kl_beta020",
}

print("=" * 115)
print("1. EVALUATION PASS RATE SUMMARY (BASELINE vs m1 vs m3)")
print("=" * 115)

models = ["baseline_m25", "groupA_m1", "groupA_m3", "groupB_m1", "groupB_m3", "groupC_m1", "groupC_m3"]
categories = ["group1_squat", "group2_rise", "group3_regression", "group4_stress"]

for m in models:
    jnt_rates = []
    env_rates = []
    cat_jnt = {}
    for c in categories:
        c_jnt = [case["joint_pass_rate"] for case in eval_data[m][c].values()]
        cat_jnt[c] = np.mean(c_jnt)
        for case in eval_data[m][c].values():
            jnt_rates.append(case["joint_pass_rate"])
            env_rates.append(case["env_pass_rate"])
    tot_j = np.mean(jnt_rates)
    tot_e = np.mean(env_rates)
    inp = eval_data[m]["group3_regression"]["g3_inplace_squat"]["joint_pass_rate"]
    inp_e = eval_data[m]["group3_regression"]["g3_inplace_squat"]["env_pass_rate"]
    print("%-14s | Total: %5.2f%% (Env %5.2f%%) | G1: %5.2f%% | G2: %5.2f%% | G3: %5.2f%% | G4: %5.2f%% | g3_inplace: %4.1f%% (%4.1f%%)" % (
        m, tot_j, tot_e, cat_jnt["group1_squat"], cat_jnt["group2_rise"], cat_jnt["group3_regression"], cat_jnt["group4_stress"], inp, inp_e
    ))

print("\n" + "=" * 115)
print("2. PER-SCENARIO DELTA: m1 -> m3 ACROSS GROUPS")
print("=" * 115)

for grp_tag, g_name in [("A", "Group A (beta=0.00)"), ("B", "Group B (beta=0.05)"), ("C", "Group C (beta=0.20)")]:
    m1_k = "group" + grp_tag + "_m1"
    m3_k = "group" + grp_tag + "_m3"
    print("\n--- %s ---" % g_name)
    print("%-20s | %-24s | %-10s | %-10s | %-10s | %-10s | %-10s" % (
        "Group", "Scenario", "m1 Jnt%", "m3 Jnt%", "Delta%", "m1 H_err", "m3 H_err"
    ))
    print("-" * 105)
    for c in categories:
        for sc in eval_data[m1_k][c]:
            c1 = eval_data[m1_k][c][sc]
            c3 = eval_data[m3_k][c][sc]
            d_jnt = c3["joint_pass_rate"] - c1["joint_pass_rate"]
            print("%-20s | %-24s | %9.1f%% | %9.1f%% | %+9.1f%% | %8.2fcm | %8.2fcm" % (
                c, sc, c1["joint_pass_rate"], c3["joint_pass_rate"], d_jnt, c1["h_steady_err_cm"], c3["h_steady_err_cm"]
            ))

print("\n" + "=" * 115)
print("3. TRAINING REWARD COMPONENTS BREAKDOWN (Iter 0 -> Iter 1 -> Iter 2)")
print("=" * 115)

# Read terminal logs for reward components
log_files = {
    "Group A (beta=0.00)": "/home/xy/.gemini/antigravity/brain/fc54084a-62cf-486e-b173-3c563c8adf07/.system_generated/tasks/task-603.log",
    "Group B (beta=0.05)": "/home/xy/.gemini/antigravity/brain/fc54084a-62cf-486e-b173-3c563c8adf07/.system_generated/tasks/task-609.log",
    "Group C (beta=0.20)": "/home/xy/.gemini/antigravity/brain/fc54084a-62cf-486e-b173-3c563c8adf07/.system_generated/tasks/task-613.log",
}

for name, lpath in log_files.items():
    print("\n--- %s ---" % name)
    with open(lpath) as f:
        lines = f.readlines()
    curr_iter = None
    rew_dict = {}
    for l in lines:
        if "Learning iteration" in l:
            curr_iter = l.strip()
            rew_dict[curr_iter] = {}
        if curr_iter and "Mean episode rew_" in l:
            parts = l.strip().split(":")
            if len(parts) == 2:
                k = parts[0].replace("Mean episode rew_", "").strip()
                v = float(parts[1].strip())
                rew_dict[curr_iter][k] = v
        if curr_iter and "Mean reward:" in l:
            v = float(l.strip().split(":")[1].strip())
            rew_dict[curr_iter]["TOTAL_MEAN_REWARD"] = v
        if curr_iter and "Mean episode length:" in l:
            v = float(l.strip().split(":")[1].strip())
            rew_dict[curr_iter]["EPISODE_LENGTH"] = v
            
    for it, rmap in rew_dict.items():
        print("  [%s] Total Rew: %5.2f | Ep Len: %5.2f | tracking_lin: %6.4f | tracking_ang: %6.4f | stand_still_wheel: %7.4f | stand_still: %7.4f" % (
            it, rmap.get("TOTAL_MEAN_REWARD", 0), rmap.get("EPISODE_LENGTH", 0),
            rmap.get("tracking_lin_vel", 0), rmap.get("tracking_ang_vel", 0),
            rmap.get("stand_still_wheel", 0), rmap.get("stand_still", 0)
        ))
