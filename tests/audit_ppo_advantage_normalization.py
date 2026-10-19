"""
Phase 4.7-S1-R: Task 2 - Advantage Normalization and PPO Actor Loss Exact Audit
Computes:
1. Raw GAE advantages and returns under Bootstrap A (Champion) vs Bootstrap B (Calibrated Target B)
2. Normalized GAE advantages (per RolloutStorage.compute_returns)
3. Step 0-33 sign change analysis
4. Actor loss and policy gradient difference (backward without optimizer.step())
5. Step 70 advantage change exact recalculation
"""

import os
import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic

def audit():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running audit on device: {device}")

    # 1. Load data
    train_path = "evaluation_assets/phase4_7_critic_calibration/data_train_seed6001.pt"
    champ_path = "evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    calib_path = "evaluation_assets/phase4_7_critic_calibration/model_critic_calibrated_target_b.pt"

    data = torch.load(train_path, map_location=device)
    obs = data["obs"].to(device)           # [140, 64, 35]
    actions = data["actions"].to(device)   # [140, 64, 6]
    rewards = data["rewards"].to(device)   # [140, 64]
    dones = data["dones"].to(device)       # [140, 64]
    terminal_obs = data["terminal_obs"].to(device) # [64, 35]

    # Load ActorCritic models
    # Actor architecture: 35 -> 512 -> 256 -> 128 -> 6
    # Critic architecture: 35 -> 512 -> 256 -> 128 -> 1
    ac_champ = ActorCritic(num_actor_obs=35, num_critic_obs=35, num_actions=6,
                           actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128]).to(device)
    champ_ckpt = torch.load(champ_path, map_location=device)
    ac_champ.load_state_dict(champ_ckpt["model_state_dict"])
    ac_champ.eval()

    ac_calib = ActorCritic(num_actor_obs=35, num_critic_obs=35, num_actions=6,
                           actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128]).to(device)
    calib_ckpt = torch.load(calib_path, map_location=device)
    ac_calib.load_state_dict(calib_ckpt["model_state_dict"])
    ac_calib.eval()

    # Fixed Critic values across the 140 steps: evaluated using the current Calibrated Critic
    with torch.no_grad():
        values_calib = torch.zeros(140, 64, device=device)
        for t in range(140):
            values_calib[t] = ac_calib.evaluate(obs[t]).squeeze(-1)

        # Terminal state evaluations
        v_term_champ = ac_champ.evaluate(terminal_obs).squeeze(-1)  # V_A
        v_term_calib = ac_calib.evaluate(terminal_obs).squeeze(-1)  # V_B

    gamma = 0.99
    lam = 0.95

    print(f"Terminal V_A (Champion)   : mean={v_term_champ.mean().item():.4f}, std={v_term_champ.std().item():.4f}")
    print(f"Terminal V_B (Calibrated) : mean={v_term_calib.mean().item():.4f}, std={v_term_calib.std().item():.4f}")
    print(f"Terminal diff (V_B - V_A) : mean={(v_term_calib - v_term_champ).mean().item():.4f}")

    # Compute GAE for both cases
    def compute_rollout_gae(terminal_v):
        raw_adv = torch.zeros(140, 64, device=device)
        returns = torch.zeros(140, 64, device=device)
        adv = torch.zeros(64, device=device)

        # Prepare augmented rewards for step 139
        r_step = rewards.clone()
        r_step[139] += gamma * terminal_v

        for step in reversed(range(140)):
            if step == 139:
                next_val = 0.0
                next_is_not_terminal = 0.0
            else:
                next_val = values_calib[step + 1]
                next_is_not_terminal = 1.0 - dones[step].float()
            
            delta = r_step[step] + next_is_not_terminal * gamma * next_val - values_calib[step]
            adv = delta + next_is_not_terminal * gamma * lam * adv
            raw_adv[step] = adv
            returns[step] = adv + values_calib[step]

        # Advantage normalization across entire rollout (RolloutStorage.compute_returns)
        adv_mean = raw_adv.mean()
        adv_std = raw_adv.std()
        norm_adv = (raw_adv - adv_mean) / (adv_std + 1e-8)
        return raw_adv, norm_adv, returns, adv_mean.item(), adv_std.item()

    raw_adv_A, norm_adv_A, ret_A, mean_A, std_A = compute_rollout_gae(v_term_champ)
    raw_adv_B, norm_adv_B, ret_B, mean_B, std_B = compute_rollout_gae(v_term_calib)

    # 1. Raw GAE comparison
    raw_diff = raw_adv_B - raw_adv_A
    print("\n--- 1. Raw GAE Statistics ---")
    print(f"Raw Adv A (Champion Target): mean={raw_adv_A.mean().item():.6f}, std={raw_adv_A.std().item():.6f}, min={raw_adv_A.min().item():.6f}, max={raw_adv_A.max().item():.6f}")
    print(f"Raw Adv B (Target B):       mean={raw_adv_B.mean().item():.6f}, std={raw_adv_B.std().item():.6f}, min={raw_adv_B.min().item():.6f}, max={raw_adv_B.max().item():.6f}")
    print(f"Raw Adv Diff (B - A):       mean={raw_diff.mean().item():.6f}, std={raw_diff.std().item():.6f}, min={raw_diff.min().item():.6f}, max={raw_diff.max().item():.6f}")

    # 2. Normalized GAE comparison
    norm_diff = norm_adv_B - norm_adv_A
    print("\n--- 2. Normalized GAE Statistics (RolloutStorage Level) ---")
    print(f"Norm Adv A: mean={norm_adv_A.mean().item():.6e}, std={norm_adv_A.std().item():.6f}, min={norm_adv_A.min().item():.6f}, max={norm_adv_A.max().item():.6f}")
    print(f"Norm Adv B: mean={norm_adv_B.mean().item():.6e}, std={norm_adv_B.std().item():.6f}, min={norm_adv_B.min().item():.6f}, max={norm_adv_B.max().item():.6f}")
    print(f"Norm Adv Diff (B - A): mean={norm_diff.mean().item():.6e}, std={norm_diff.std().item():.6f}, min={norm_diff.min().item():.6f}, max={norm_diff.max().item():.6f}")
    print(f"Max absolute norm diff: {norm_diff.abs().max().item():.6f}")

    # 3. Step 0-33 sign change analysis
    jump_raw_A = raw_adv_A[:34]
    jump_raw_B = raw_adv_B[:34]
    jump_norm_A = norm_adv_A[:34]
    jump_norm_B = norm_adv_B[:34]

    raw_sign_flips = (torch.sign(jump_raw_A) != torch.sign(jump_raw_B)).sum().item()
    norm_sign_flips = (torch.sign(jump_norm_A) != torch.sign(jump_norm_B)).sum().item()
    total_jump_elements = 34 * 64

    print("\n--- 3. Step 0-33 Sign Change Analysis ---")
    print(f"Total elements in Step 0-33: {total_jump_elements}")
    print(f"Raw GAE sign flips in Step 0-33       : {raw_sign_flips} / {total_jump_elements} ({raw_sign_flips/total_jump_elements*100:.2f}%)")
    print(f"Normalized GAE sign flips in Step 0-33: {norm_sign_flips} / {total_jump_elements} ({norm_sign_flips/total_jump_elements*100:.2f}%)")
    print(f"Max absolute raw diff in Step 0-33    : {(jump_raw_B - jump_raw_A).abs().max().item():.6f}")
    print(f"Max absolute norm diff in Step 0-33   : {(jump_norm_B - jump_norm_A).abs().max().item():.6f}")

    # 4. Step 70 exact recalculation check
    mu = gamma * lam # 0.9405
    diff_delta_step139 = (gamma * (v_term_calib - v_term_champ)).mean().item() # ~2.4693
    step70_theoretical_weight = mu ** 69 # 139 - 70 = 69
    step70_theoretical_adv_diff = step70_theoretical_weight * diff_delta_step139
    step70_actual_raw_diff = (raw_adv_B[70] - raw_adv_A[70]).mean().item()

    print("\n--- 4. Step 70 Recalculation Check ---")
    print(f"gamma * lambda = {mu:.4f}")
    print(f"Step 139 delta diff = {diff_delta_step139:.6f}")
    print(f"Exponent (139 - 70) = 69")
    print(f"Theoretical weight (0.9405^69) = {step70_theoretical_weight:.8f}")
    print(f"Theoretical advantage diff = {step70_theoretical_adv_diff:.8f}")
    print(f"Actual rollout advantage diff = {step70_actual_raw_diff:.8f}")
    print(f"Phase 4.7-S1 Report reported value: 0.0326 (weight 0.01320)")
    print(f"Difference between actual and reported: {abs(step70_actual_raw_diff - 0.0326):.6f}")

    # 5. Actor Loss and Policy Gradient Analysis (Zero Parameter Update)
    print("\n--- 5. Actor Loss and Policy Gradient Analysis ---")
    ac_test = ActorCritic(num_actor_obs=35, num_critic_obs=35, num_actions=6,
                          actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128]).to(device)
    ac_test.load_state_dict(calib_ckpt["model_state_dict"])
    ac_test.actor.train()

    obs_flat = obs.view(-1, 35)
    actions_flat = actions.view(-1, 6)

    # 5A. Evaluation on recorded (deterministic) actions
    ac_test.act(obs_flat)
    lp_det = ac_test.get_actions_log_prob(actions_flat)
    ratio_det = torch.exp(lp_det - lp_det.detach()) # 1.0

    adv_flat_A = norm_adv_A.view(-1, 1).detach()
    adv_flat_B = norm_adv_B.view(-1, 1).detach()

    loss_det_A = torch.max(-adv_flat_A * ratio_det, -adv_flat_A * torch.clamp(ratio_det, 0.8, 1.2)).mean()
    loss_det_B = torch.max(-adv_flat_B * ratio_det, -adv_flat_B * torch.clamp(ratio_det, 0.8, 1.2)).mean()

    ac_test.zero_grad()
    loss_det_A.backward(retain_graph=True)
    grads_det_A = torch.cat([p.grad.flatten() for p in ac_test.actor.parameters() if p.grad is not None]).clone()

    ac_test.zero_grad()
    loss_det_B.backward()
    grads_det_B = torch.cat([p.grad.flatten() for p in ac_test.actor.parameters() if p.grad is not None]).clone()

    print(f"[Deterministic Actions (Dataset Saved)]")
    print(f"Actor Surrogate Loss A: {loss_det_A.item():.8e}")
    print(f"Actor Surrogate Loss B: {loss_det_B.item():.8e}")
    print(f"Surrogate Loss Diff (B - A): {(loss_det_B - loss_det_A).item():.8e}")
    print(f"Gradient Norm A: {grads_det_A.norm().item():.8e} (analytically 0 at Gaussian mode)")
    print(f"Gradient Norm B: {grads_det_B.norm().item():.8e} (analytically 0 at Gaussian mode)")

    # 5B. Evaluation on stochastic actions (sampled from pi_old during online PPO rollout)
    torch.manual_seed(42)
    ac_test.act(obs_flat)
    stoch_actions = ac_test.distribution.sample().detach()
    old_stoch_lp = ac_test.get_actions_log_prob(stoch_actions).detach()

    # Fresh forward pass for Loss A
    ac_test.act(obs_flat)
    lp_A = ac_test.get_actions_log_prob(stoch_actions)
    ratio_A = torch.exp(lp_A - old_stoch_lp)
    loss_stoch_A = torch.max(-adv_flat_A * ratio_A, -adv_flat_A * torch.clamp(ratio_A, 0.8, 1.2)).mean()

    ac_test.zero_grad()
    loss_stoch_A.backward()
    grads_stoch_A = torch.cat([p.grad.flatten() for p in ac_test.actor.parameters() if p.grad is not None]).clone()

    # Fresh forward pass for Loss B
    ac_test.act(obs_flat)
    lp_B = ac_test.get_actions_log_prob(stoch_actions)
    ratio_B = torch.exp(lp_B - old_stoch_lp)
    loss_stoch_B = torch.max(-adv_flat_B * ratio_B, -adv_flat_B * torch.clamp(ratio_B, 0.8, 1.2)).mean()

    ac_test.zero_grad()
    loss_stoch_B.backward()
    grads_stoch_B = torch.cat([p.grad.flatten() for p in ac_test.actor.parameters() if p.grad is not None]).clone()

    cos_sim = torch.cosine_similarity(grads_stoch_A.unsqueeze(0), grads_stoch_B.unsqueeze(0)).item()
    norm_gA = grads_stoch_A.norm().item()
    norm_gB = grads_stoch_B.norm().item()
    diff_norm = (grads_stoch_B - grads_stoch_A).norm().item()
    rel_diff = diff_norm / (norm_gA + 1e-8)
    max_elem = (grads_stoch_B - grads_stoch_A).abs().max().item()

    print(f"\n[Stochastic Actions (Online PPO Regime)]")
    print(f"Stochastic Surrogate Loss A: {loss_stoch_A.item():.8e}")
    print(f"Stochastic Surrogate Loss B: {loss_stoch_B.item():.8e}")
    print(f"Surrogate Loss Diff (B - A): {(loss_stoch_B - loss_stoch_A).item():.8e}")
    print(f"Gradient Norm A: {norm_gA:.6e}")
    print(f"Gradient Norm B: {norm_gB:.6e}")
    print(f"Gradient Cosine Similarity: {cos_sim:.6f}")
    print(f"Gradient Norm Difference: {diff_norm:.6e}")
    print(f"Relative Gradient Difference: {rel_diff * 100:.2f}%")
    print(f"Max Component Difference: {max_elem:.6e}")

    # 5C. Raw (Unnormalized) Advantage Gradient Comparison
    raw_flat_A = raw_adv_A.view(-1, 1).detach()
    raw_flat_B = raw_adv_B.view(-1, 1).detach()

    ac_test.act(obs_flat)
    lp_raw_A = ac_test.get_actions_log_prob(stoch_actions)
    ratio_raw_A = torch.exp(lp_raw_A - old_stoch_lp)
    loss_raw_A = torch.max(-raw_flat_A * ratio_raw_A, -raw_flat_A * torch.clamp(ratio_raw_A, 0.8, 1.2)).mean()

    ac_test.zero_grad()
    loss_raw_A.backward()
    grads_raw_A = torch.cat([p.grad.flatten() for p in ac_test.actor.parameters() if p.grad is not None]).clone()

    ac_test.act(obs_flat)
    lp_raw_B = ac_test.get_actions_log_prob(stoch_actions)
    ratio_raw_B = torch.exp(lp_raw_B - old_stoch_lp)
    loss_raw_B = torch.max(-raw_flat_B * ratio_raw_B, -raw_flat_B * torch.clamp(ratio_raw_B, 0.8, 1.2)).mean()

    ac_test.zero_grad()
    loss_raw_B.backward()
    grads_raw_B = torch.cat([p.grad.flatten() for p in ac_test.actor.parameters() if p.grad is not None]).clone()

    raw_cos_sim = torch.cosine_similarity(grads_raw_A.unsqueeze(0), grads_raw_B.unsqueeze(0)).item()
    raw_norm_gA = grads_raw_A.norm().item()
    raw_norm_gB = grads_raw_B.norm().item()
    raw_diff_norm = (grads_raw_B - grads_raw_A).norm().item()
    raw_rel_diff = raw_diff_norm / (raw_norm_gA + 1e-8)

    print(f"\n[Raw (Unnormalized) Advantage Regime]")
    print(f"Raw Surrogate Loss A: {loss_raw_A.item():.6f}")
    print(f"Raw Surrogate Loss B: {loss_raw_B.item():.6f}")
    print(f"Raw Gradient Norm A: {raw_norm_gA:.6e}")
    print(f"Raw Gradient Norm B: {raw_norm_gB:.6e}")
    print(f"Raw Gradient Cosine Similarity: {raw_cos_sim:.6f}")
    print(f"Raw Relative Gradient Difference: {raw_rel_diff * 100:.2f}%")

    # Save complete audit dictionary
    audit_results = {
        "terminal_values": {
            "v_term_champ_mean": v_term_champ.mean().item(),
            "v_term_calib_mean": v_term_calib.mean().item(),
            "diff_mean": (v_term_calib - v_term_champ).mean().item(),
        },
        "raw_gae": {
            "mean_A": raw_adv_A.mean().item(),
            "std_A": raw_adv_A.std().item(),
            "min_A": raw_adv_A.min().item(),
            "max_A": raw_adv_A.max().item(),
            "mean_B": raw_adv_B.mean().item(),
            "std_B": raw_adv_B.std().item(),
            "min_B": raw_adv_B.min().item(),
            "max_B": raw_adv_B.max().item(),
            "diff_mean": raw_diff.mean().item(),
            "diff_std": raw_diff.std().item(),
            "diff_max": raw_diff.max().item(),
        },
        "norm_gae": {
            "mean_A": norm_adv_A.mean().item(),
            "std_A": norm_adv_A.std().item(),
            "min_A": norm_adv_A.min().item(),
            "max_A": norm_adv_A.max().item(),
            "mean_B": norm_adv_B.mean().item(),
            "std_B": norm_adv_B.std().item(),
            "min_B": norm_adv_B.min().item(),
            "max_B": norm_adv_B.max().item(),
            "diff_mean": norm_diff.mean().item(),
            "diff_std": norm_diff.std().item(),
            "diff_max": norm_diff.max().item(),
            "max_abs_diff": norm_diff.abs().max().item(),
        },
        "step_0_33_jump_segment": {
            "total_elements": total_jump_elements,
            "raw_sign_flips": raw_sign_flips,
            "norm_sign_flips": norm_sign_flips,
            "max_abs_raw_diff": (jump_raw_B - jump_raw_A).abs().max().item(),
            "max_abs_norm_diff": (jump_norm_B - jump_norm_A).abs().max().item(),
        },
        "step_70_audit": {
            "exponent": 69,
            "mu_69": step70_theoretical_weight,
            "theoretical_adv_diff": step70_theoretical_adv_diff,
            "actual_raw_diff": step70_actual_raw_diff,
            "report_reported_val": 0.0326,
            "report_weight_reported": 0.01320,
            "reconciliation_note": "Report previously cited 0.0326 using weight 0.01320; exact recalculation gives weight 0.014513 and diff 0.035837.",
        },
        "actor_loss_and_gradients": {
            "loss_det_A": loss_det_A.item(),
            "loss_det_B": loss_det_B.item(),
            "loss_det_diff": (loss_det_B - loss_det_A).item(),
            "grad_norm_det_A": grads_det_A.norm().item(),
            "grad_norm_det_B": grads_det_B.norm().item(),
            "loss_stoch_A": loss_stoch_A.item(),
            "loss_stoch_B": loss_stoch_B.item(),
            "loss_stoch_diff": (loss_stoch_B - loss_stoch_A).item(),
            "grad_norm_stoch_A": norm_gA,
            "grad_norm_stoch_B": norm_gB,
            "grad_cosine_similarity": cos_sim,
            "grad_norm_diff": diff_norm,
            "rel_grad_diff_pct": rel_diff * 100,
            "max_elem_diff": max_elem,
            "loss_raw_A": loss_raw_A.item(),
            "loss_raw_B": loss_raw_B.item(),
            "raw_grad_norm_A": raw_norm_gA,
            "raw_grad_norm_B": raw_norm_gB,
            "raw_grad_cosine_similarity": raw_cos_sim,
            "raw_rel_grad_diff_pct": raw_rel_diff * 100,
        }
    }

    import json
    out_path = "evaluation_assets/phase4_7_critic_calibration/phase4_7_s1_r_advantage_audit.json"
    with open(out_path, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nAudit results successfully saved to: {out_path}")

if __name__ == "__main__":
    audit()
