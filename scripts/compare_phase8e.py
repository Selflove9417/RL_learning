import json

with open("evaluation_assets/phase8e/results/phase8e_12cases_eval.json", "r") as f:
    data = json.load(f)

cases = [
    # Left-turn targeted
    ("g4_exp_fwd_left_squat", "Left Target: Fwd Left High-Squat"),
    ("g4_exp_bwd_left_squat", "Left Target: Bwd Left High-Squat"),
    ("g1_fwd_left_squat",     "Left Target: Fwd Left Squat"),
    ("g1_bwd_left_squat",     "Left Target: Bwd Left Squat"),
    # Right-turn protection
    ("g4_exp_fwd_right_squat", "Right Protect: Fwd Right High-Squat"),
    ("g4_exp_bwd_right_squat", "Right Protect: Bwd Right High-Squat"),
    ("g1_fwd_right_squat",    "Right Protect: Fwd Right Squat"),
    ("g1_bwd_right_squat",    "Right Protect: Bwd Right Squat"),
    # Baseline / preservation
    ("g3_inplace_squat",      "Preserve: In-place Squat"),
    ("g3_cruise_squat",       "Preserve: Cruise Squat"),
    ("g3_nominal_compound",   "Preserve: Nominal Compound"),
    ("g2_bwd_right_rise",     "Preserve: Bwd Right Rise"),
]

print("="*120)
print("PHASE 8E 12-SCENARIO HEAD-TO-HEAD COMPARISON (Champ v4 m10 vs Phase 8D m13 vs Phase 8E m13)")
print("="*120)
print(f"{'Case Name':24s} | {'v4_m10 J%':9s} {'8D_m13 J%':9s} {'8E_m13 J%':9s} | {'v4_m10 H%':9s} {'8D_m13 H%':9s} {'8E_m13 H%':9s} | {'v4_m10 Env':10s} {'8D_m13 Env':10s} {'8E_m13 Env'}")
print("-" * 120)

for c_name, desc in cases:
    m10 = data['champ_v4_m10'][c_name]
    m8d = data['phase8d_m13'][c_name]
    m8e = data['phase8e_m13'][c_name]
    print(f"{c_name:24s} | "
          f"{m10['joint_pass_rate']:7.1f}%  {m8d['joint_pass_rate']:7.1f}%  {m8e['joint_pass_rate']:7.1f}%  | "
          f"{m10['h_pass_rate']:7.1f}%  {m8d['h_pass_rate']:7.1f}%  {m8e['h_pass_rate']:7.1f}%  | "
          f"{m10['env_pass_rate']:8.1f}%  {m8d['env_pass_rate']:8.1f}%  {m8e['env_pass_rate']:8.1f}%")

print("\n" + "=" * 135)
print("TORQUE & SATURATION AUDIT (Physical Limit Saturation >= 6.95 Nm vs Low Threshold >= 1.49 Nm)")
print("=" * 135)
print(f"{'Case Name':24s} | {'v4 TauMax':9s} {'8E TauMax':9s} | {'v4 Sat695':9s} {'8D Sat695':9s} {'8E Sat695':9s} | {'v4 Sat149':9s} {'8D Sat149':9s} {'8E Sat149':9s} | Description")
print("-" * 135)

for c_name, desc in cases:
    m10 = data['champ_v4_m10'][c_name]
    m8d = data['phase8d_m13'][c_name]
    m8e = data['phase8e_m13'][c_name]
    print(f"{c_name:24s} | "
          f"{m10['wheel_tau_max_Nm']:7.2f}Nm {m8e['wheel_tau_max_Nm']:7.2f}Nm | "
          f"{m10['wheel_sat_rate_695']:7.2f}%  {m8d['wheel_sat_rate_695']:7.2f}%  {m8e['wheel_sat_rate_695']:7.2f}%  | "
          f"{m10['wheel_sat_rate_149']:7.2f}%  {m8d['wheel_sat_rate_149']:7.2f}%  {m8e['wheel_sat_rate_149']:7.2f}%  | {desc}")

print("=" * 135)
