import json

with open("evaluation_assets/phase10/results/phase10_12cases_eval.json") as f:
    d = json.load(f)

models = ["champ_v4_m10", "phase9b_m13", "phase10_m13", "phase10_m16", "phase10_m20"]
model_labels = ["Champ v4", "Phase9B m13", "P10 m13", "P10 m16", "P10 m20"]
scenarios = [
    "g1_fwd_left_squat", "g1_bwd_left_squat", "g4_exp_fwd_left_squat", "g4_exp_bwd_left_squat",
    "g1_fwd_right_squat", "g1_bwd_right_squat", "g4_exp_fwd_right_squat", "g4_exp_bwd_right_squat",
    "g3_inplace_squat", "g3_cruise_squat", "g3_nominal_compound", "g2_bwd_right_rise"
]

header = f"{'Scenario':<24} | {'Champ v4':<10} | {'Phase9B m13':<12} | {'P10 m13':<10} | {'P10 m16':<10} | {'P10 m20':<10}"
print(header)
print("-" * len(header))
for sc in scenarios:
    vals = [f"{d[m][sc]['joint_pass_rate']:>5.1f}%" for m in models]
    print(f"{sc:<24} | {vals[0]:<10} | {vals[1]:<12} | {vals[2]:<10} | {vals[3]:<10} | {vals[4]:<10}")

print("-" * len(header))
left_4 = [f"{d[m]['_summary']['left_4cases_joint']:>5.1f}%" for m in models]
right_4 = [f"{d[m]['_summary']['right_4cases_joint']:>5.1f}%" for m in models]
inplace = [f"{d[m]['_summary']['inplace_squat_joint']:>5.1f}%" for m in models]
mean_12 = [f"{d[m]['_summary']['mean_joint_12']:>5.1f}%" for m in models]
env_12 = [f"{d[m]['_summary']['mean_env_12']:>5.1f}%" for m in models]
gap = [f"{d[m]['_summary']['left_right_gap']:>5.1f}%" for m in models]

print(f"{'Left-4 Joint%':<24} | {left_4[0]:<10} | {left_4[1]:<12} | {left_4[2]:<10} | {left_4[3]:<10} | {left_4[4]:<10}")
print(f"{'Right-4 Joint%':<24} | {right_4[0]:<10} | {right_4[1]:<12} | {right_4[2]:<10} | {right_4[3]:<10} | {right_4[4]:<10}")
print(f"{'L-R Gap':<24} | {gap[0]:<10} | {gap[1]:<12} | {gap[2]:<10} | {gap[3]:<10} | {gap[4]:<10}")
print(f"{'Inplace Squat Joint%':<24} | {inplace[0]:<10} | {inplace[1]:<12} | {inplace[2]:<10} | {inplace[3]:<10} | {inplace[4]:<10}")
print(f"{'12-Case Joint%':<24} | {mean_12[0]:<10} | {mean_12[1]:<12} | {mean_12[2]:<10} | {mean_12[3]:<10} | {mean_12[4]:<10}")
print(f"{'12-Case Env%':<24} | {env_12[0]:<10} | {env_12[1]:<12} | {env_12[2]:<10} | {env_12[3]:<10} | {env_12[4]:<10}")
