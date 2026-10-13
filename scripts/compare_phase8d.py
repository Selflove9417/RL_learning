import json

with open("evaluation_assets/phase8d/results/phase8d_key_checkpoints_eval.json", "r") as f:
    data = json.load(f)

cases = [
    # Targeted Low-Height Left-Turn Cases
    ("g4_exp_fwd_left_squat", "Target: Fwd Left High-Squat"),
    ("g4_exp_bwd_left_squat", "Target: Bwd Left High-Squat"),
    ("g1_fwd_left_squat",     "Target: Fwd Left Squat"),
    ("g1_bwd_left_squat",     "Target: Bwd Left Squat"),
    # Preservation Check Cases
    ("g3_inplace_squat",      "Preserve: In-place Squat"),
    ("g3_nominal_cruise",     "Preserve: Nominal Cruise"),
    ("g3_nominal_compound",   "Preserve: Nominal Compound"),
    ("g1_fwd_right_squat",    "Balance: Fwd Right Squat"),
    ("g2_bwd_right_rise",     "Preserve: Bwd Right Rise"),
]

print("="*120)
print(f"{'Case Name':24s} | {'m10 J%':7s} {'m13 J%':7s} {'m15 J%':7s} | {'m10 H%':7s} {'m13 H%':7s} {'m15 H%':7s} | {'m10 Vy%':7s} {'m13 Vy%':7s} {'m15 Vy%':7s} | {'m10 Wz%':7s} {'m13 Wz%':7s} {'m15 Wz%'}")
print("="*120)

for c_name, desc in cases:
    m10 = data['m10'][c_name]
    m13 = data['m13'][c_name]
    m15 = data['m15'][c_name]
    print(f"{c_name:24s} | "
          f"{m10['joint_pass_rate']:6.1f}% {m13['joint_pass_rate']:6.1f}% {m15['joint_pass_rate']:6.1f}% | "
          f"{m10['h_pass_rate']:6.1f}% {m13['h_pass_rate']:6.1f}% {m15['h_pass_rate']:6.1f}% | "
          f"{m10['vy_pass_rate']:6.1f}% {m13['vy_pass_rate']:6.1f}% {m15['vy_pass_rate']:6.1f}% | "
          f"{m10['wz_pass_rate']:6.1f}% {m13['wz_pass_rate']:6.1f}% {m15['wz_pass_rate']:6.1f}%")

print("="*120)
print(f"{'Case Name':24s} | {'m10 Env%':8s} {'m13 Env%':8s} {'m15 Env%':8s} | {'m10 H_err':9s} {'m13 H_err':9s} {'m15 H_err':9s} | Description")
print("="*120)

for c_name, desc in cases:
    m10 = data['m10'][c_name]
    m13 = data['m13'][c_name]
    m15 = data['m15'][c_name]
    print(f"{c_name:24s} | "
          f"{m10['env_pass_rate']:7.1f}% {m13['env_pass_rate']:7.1f}% {m15['env_pass_rate']:7.1f}% | "
          f"{m10['h_steady_err_cm']:6.2f}cm  {m13['h_steady_err_cm']:6.2f}cm  {m15['h_steady_err_cm']:6.2f}cm  | {desc}")

print("="*120)
