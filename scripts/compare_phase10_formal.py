import json

with open("evaluation_assets/phase8b/results/eval_phase8b_m10_formal_results.json") as f:
    d8b = json.load(f)["phase8b_m10_formal"]
with open("evaluation_assets/phase10/results/eval_phase10_formal_results.json") as f:
    d10 = json.load(f)["phase10_m20"]

groups = ["group1_squat", "group2_rise", "group3_regression", "group4_stress"]

h_sc = "Scenario"
h_j8 = "Champ v4 Joint%"
h_j10 = "P10 m20 Joint%"
h_dj = "Delta Joint%"
h_e8 = "Champ v4 Env%"
h_e10 = "P10 m20 Env%"

header = f"{h_sc:<24} | {h_j8:<16} | {h_j10:<16} | {h_dj:<12} | {h_e8:<14} | {h_e10:<14}"
print(header)
print("-" * len(header))

for g in groups:
    print(f"=== {g.upper()} ===")
    for case, m8 in d8b[g].items():
        m10 = d10[g][case]
        j8 = m8["joint_pass_rate"]
        j10 = m10["joint_pass_rate"]
        dj = j10 - j8
        e8 = m8["env_pass_rate"]
        e10 = m10["env_pass_rate"]
        print(f"{case:<24} | {j8:>6.1f}%          | {j10:>6.1f}%          | {dj:>+6.1f}%      | {e8:>6.1f}%        | {e10:>6.1f}%")

print("=" * len(header))
s8_j = 66.52
s8_e = 63.24
s10_j = d10["_summary"]["joint_pass_rate"]
s10_e = d10["_summary"]["env_pass_rate"]
dj_tot = s10_j - s8_j
de_tot = s10_e - s8_e
print(f"{'OVERALL GLOBAL':<24} | {s8_j:>6.2f}%          | {s10_j:>6.2f}%          | {dj_tot:>+6.2f}%      | {s8_e:>6.2f}%        | {s10_e:>6.2f}%")

# Also print group summaries
print("\n--- GROUP SUMMARIES ---")
for g in groups:
    g8_j = sum(m["joint_pass_rate"] for m in d8b[g].values()) / len(d8b[g])
    g8_e = sum(m["env_pass_rate"] for m in d8b[g].values()) / len(d8b[g])
    g10_j = d10["_summary"]["group_summaries"][g]["joint"]
    g10_e = d10["_summary"]["group_summaries"][g]["env"]
    print(f"{g:<18}: Joint% {g8_j:.2f}% -> {g10_j:.2f}% ({g10_j - g8_j:+.2f}%) | Env% {g8_e:.2f}% -> {g10_e:.2f}% ({g10_e - g8_e:+.2f}%)")
