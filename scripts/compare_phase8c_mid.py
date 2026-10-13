import json

with open("evaluation_assets/phase8c/results/phase8c_mid_checkpoints_eval.json", "r") as f:
    data = f.load() if hasattr(f, "load") else json.load(f)

cases = [
    "g4_exp_fwd_left_squat",
    "g4_exp_bwd_left_squat",
    "g2_bwd_right_rise",
    "g1_fwd_left_squat",
    "g3_inplace_squat",
    "g3_nominal_cruise",
    "g3_nominal_compound",
]

print("="*105)
print(f"{'Case Name':25s} | {'m10 J%':8s} {'m12 J%':8s} {'m15 J%':8s} | {'m10 H%':8s} {'m12 H%':8s} {'m15 H%':8s} | {'m10 Env%':8s} {'m12 Env%':8s} {'m15 Env%'}")
print("="*105)

for c in cases:
    m10_j = data['m10'][c]['joint_pass_rate']
    m12_j = data['m12'][c]['joint_pass_rate']
    m15_j = data['m15'][c]['joint_pass_rate']
    m10_h = data['m10'][c]['h_pass_rate']
    m12_h = data['m12'][c]['h_pass_rate']
    m15_h = data['m15'][c]['h_pass_rate']
    m10_e = data['m10'][c]['env_pass_rate']
    m12_e = data['m12'][c]['env_pass_rate']
    m15_e = data['m15'][c]['env_pass_rate']
    print(f"{c:25s} | {m10_j:7.1f}% {m12_j:7.1f}% {m15_j:7.1f}% | {m10_h:7.1f}% {m12_h:7.1f}% {m15_h:7.1f}% | {m10_e:7.1f}% {m12_e:7.1f}% {m15_e:7.1f}%")

print("="*105)
