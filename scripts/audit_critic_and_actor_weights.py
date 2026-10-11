"""
Phase 6A-R Audit Script: Inspect Actor and Critic parameter drift and value function outputs
across Baseline, Group A, and Group B checkpoints.
"""

import os
import torch
import numpy as np

def audit_checkpoints():
    base_path = "legged_gym/logs/myrobot/Oct10_15-48-32_turning_height_stage5_3_formal/model_25.pt"
    dir_A = "legged_gym/logs/myrobot/Oct11_13-03-20_phase6a_groupA_vanilla"
    dir_B = "legged_gym/logs/myrobot/Oct11_13-04-06_phase6a_groupB_kl_early_stop"

    base_ckpt = torch.load(base_path, map_location='cpu')['model_state_dict']
    base_actor = torch.cat([p.flatten() for k, p in base_ckpt.items() if 'actor' in k])
    base_critic = torch.cat([p.flatten() for k, p in base_ckpt.items() if 'critic' in k])
    base_std = base_ckpt['std']

    print("=" * 80)
    print("PHASE 6A-R AUDIT: CHECKPOINT WEIGHT DRIFT VS BASELINE MODEL_25")
    print("=" * 80)
    print(f"Base Actor norm : {torch.norm(base_actor).item():.4f} (params: {base_actor.numel()})")
    print(f"Base Critic norm: {torch.norm(base_critic).item():.4f} (params: {base_critic.numel()})")
    print(f"Base Std        : {base_std.numpy()}")

    checkpoints = [
        ("groupA_m1", os.path.join(dir_A, "model_1.pt")),
        ("groupA_m5", os.path.join(dir_A, "model_5.pt")),
        ("groupA_m10", os.path.join(dir_A, "model_10.pt")),
        ("groupB_m1", os.path.join(dir_B, "model_1.pt")),
        ("groupB_m5", os.path.join(dir_B, "model_5.pt")),
        ("groupB_m10", os.path.join(dir_B, "model_10.pt")),
    ]

    print("\n" + "-" * 80)
    print(f"{'Model':<12} | {'Actor Delta L2':<15} | {'Actor Rel%':<11} | {'Critic Delta L2':<15} | {'Critic Rel%':<11} | {'Std Drift L2':<12}")
    print("-" * 80)

    # Standard probe observations for forward testing
    torch.manual_seed(42)
    probe_obs = torch.randn(100, 31)

    for label, path in checkpoints:
        sd = torch.load(path, map_location='cpu')['model_state_dict']
        m_actor = torch.cat([p.flatten() for k, p in sd.items() if 'actor' in k])
        m_critic = torch.cat([p.flatten() for k, p in sd.items() if 'critic' in k])
        m_std = sd['std']

        d_act = torch.norm(m_actor - base_actor).item()
        rel_act = d_act / torch.norm(base_actor).item() * 100.0

        d_crit = torch.norm(m_critic - base_critic).item()
        rel_crit = d_crit / torch.norm(base_critic).item() * 100.0

        d_std = torch.norm(m_std - base_std).item()

        print(f"{label:<12} | {d_act:<15.4f} | {rel_act:<10.2f}% | {d_crit:<15.4f} | {rel_crit:<10.2f}% | {d_std:<12.5f}")

if __name__ == '__main__':
    audit_checkpoints()
