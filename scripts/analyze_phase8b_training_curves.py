import json
import numpy as np

# Load Phase 8A and Phase 8B diagnostics
p8a_diag = json.load(open("evaluation_assets/phase8a/model/diagnostics.json"))
p8b_diag = json.load(open("evaluation_assets/phase8b/models/diagnostics.json"))

print("=" * 80)
print("PHASE 8B TRAINING DYNAMICS & CURVES (10 Iterations Cumulative)")
print("=" * 80)
print(f"{'Iter':<5s} | {'Rew':<6s} | {'Surr Loss':<10s} | {'Val Loss':<10s} | {'Ref KL':<9s} | {'PPO KL':<9s} | {'GradNorm':<9s} | {'Delta L2 (rel)':<15s}")
print("-" * 80)

combined = [p8a_diag[0]] + p8b_diag

for idx, d in enumerate(combined):
    it = idx + 1
    rew = d.get('mean_reward', 'N/A')
    surr = d.get('surrogate_loss', 0.0)
    val = d.get('value_loss', 0.0)
    ref_kl = d.get('ref_kl_mean', 0.0)
    ppo_kl = d.get('kl_mean', 0.0)
    grad = d.get('grad_norm_mean', 0.0)
    l2_rel = d.get('actor_param_delta_rel', 0.0)
    l2_val = d.get('actor_param_delta_l2', 0.0)
    print(f"m{it:<4d} | {rew if isinstance(rew, str) else f'{rew:6.2f}':<6s} | {surr:10.4f} | {val:10.4f} | {ref_kl:9.5f} | {ppo_kl:9.5f} | {grad:9.4f} | {l2_val:.4f} ({l2_rel:.3%})")
