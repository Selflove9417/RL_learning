"""
Phase 9B Fixed Observation Dataset Collection and Unified Symmetry Evaluation.
1. Runs Champion v4 across the 12 representative benchmark scenarios.
2. Collects steady-state observations (steps 200-300).
3. Saves the dataset to evaluation_assets/phase9b/fixed_obs.pt.
4. Evaluates L_sym across Champion v4 m10, Phase 8C m13, and Phase 9A m13 on this exact dataset.
"""

import isaacgym
import os
import json
import torch
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

# Reflection operators
obs_indices = torch.tensor([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    15, 16, 17, 12, 13, 14,
    21, 22, 23, 18, 19, 20,
    27, 28, 29, 24, 25, 26,
    30
], dtype=torch.long)
obs_signs = torch.tensor([
    -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0
], dtype=torch.float)
act_indices = torch.tensor([3, 4, 5, 0, 1, 2], dtype=torch.long)

def S_obs(obs):
    device = obs.device
    return obs[..., obs_indices.to(device)] * obs_signs.to(device)

def S_act(act):
    device = act.device
    return act[..., act_indices.to(device)]

def main():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    ckpt_path = "/home/xy/IsaacTu/evaluation_assets/phase8b/models/phase8b_m10.pt"
    loaded = torch.load(ckpt_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    pol = ac.act_inference

    scenarios = [
        ("g1_fwd_left_squat",      0.10,  0.20, 0.36),
        ("g1_bwd_left_squat",     -0.10,  0.20, 0.36),
        ("g1_fwd_right_squat",     0.10, -0.20, 0.36),
        ("g1_bwd_right_squat",    -0.10, -0.20, 0.36),
        ("g4_exp_fwd_left_squat",  0.16,  0.30, 0.36),
        ("g4_exp_bwd_left_squat", -0.16,  0.30, 0.36),
        ("g4_exp_fwd_right_squat", 0.16, -0.30, 0.36),
        ("g4_exp_bwd_right_squat",-0.16, -0.30, 0.36),
        ("g3_inplace_squat",       0.00,  0.00, 0.36),
        ("g3_cruise_squat",        0.15,  0.00, 0.36),
        ("g3_nominal_compound",    0.10,  0.20, 0.39774),
        ("g2_bwd_right_rise",     -0.10, -0.20, 0.44),
    ]

    all_obs = []
    sc_slices = {}
    current_idx = 0

    print("=== Collecting Fixed Observation Dataset from Champion v4 ===")
    for sc_name, cmd_vy, cmd_wz, target_h in scenarios:
        h_ref = target_h
        delta_h = -0.005 * float(np.clip((0.39774 - h_ref) / (0.39774 - 0.360), 0.0, 1.0))
        h_ctrl = h_ref + delta_h

        torch.manual_seed(42)
        np.random.seed(42)
        env.reset()

        sc_obs = []
        for step in range(300):
            curr_h_ctrl = 0.39774 if step < 100 else h_ctrl
            env.commands[:, 0] = 0.0
            env.commands[:, 1] = cmd_vy
            env.commands[:, 2] = cmd_wz
            env.height_cmd_buf[:] = curr_h_ctrl

            obs = env.get_observations()
            with torch.no_grad():
                actions = pol(obs)
            obs, _, _, dones, _ = env.step(actions)

            if step >= 200:  # steady state
                sc_obs.append(obs.detach().cpu())

        sc_obs_tensor = torch.cat(sc_obs, dim=0)  # (100 * 32, 31) = 3200 x 31
        all_obs.append(sc_obs_tensor)
        count = sc_obs_tensor.shape[0]
        sc_slices[sc_name] = (current_idx, current_idx + count)
        current_idx += count
        print(f"  Collected {count} obs for {sc_name}")

    full_obs = torch.cat(all_obs, dim=0)
    print(f"\nTotal fixed observations: {full_obs.shape} (38400 samples)")
    save_data = {
        "observations": full_obs,
        "sc_slices": sc_slices,
        "scenarios": [s[0] for s in scenarios],
    }
    torch.save(save_data, "evaluation_assets/phase9b/fixed_obs.pt")
    print("Saved to evaluation_assets/phase9b/fixed_obs.pt")

    # Now evaluate L_sym on this exact tensor across models
    models = {
        "champ_v4_m10": "/home/xy/IsaacTu/evaluation_assets/phase8b/models/phase8b_m10.pt",
        "phase8c_m13":  "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct12_15-41-44_phase8c_curriculum_ext_resume_15iter/model_13.pt",
        "phase9a_m13":  "/home/xy/IsaacTu/evaluation_assets/phase9a/models/phase9a_m13.pt",
    }

    obs_gpu = full_obs.to(env.device)
    obs_mirr_gpu = S_obs(obs_gpu)

    print("\n=== Evaluating Unified Mirror Error L_sym on Fixed Dataset ===")
    results = {}
    for m_name, path in models.items():
        loaded_m = torch.load(path, map_location=env.device)
        ac_m = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
        ac_m.load_state_dict(loaded_m['model_state_dict'])
        ac_m.eval()

        with torch.no_grad():
            mu_orig = ac_m.actor(obs_gpu)
            mu_mirr = ac_m.actor(obs_mirr_gpu)
            mu_expected = S_act(mu_orig)

            # L_sym = mean((mu(S_obs) - S_act(mu))**2)
            diff_sq = torch.square(mu_mirr - mu_expected)
            global_l_sym = torch.mean(diff_sq).item()

            per_sc_l_sym = {}
            for sc, (start, end) in sc_slices.items():
                sc_diff_sq = diff_sq[start:end]
                per_sc_l_sym[sc] = torch.mean(sc_diff_sq).item()

        left_cases = ["g1_fwd_left_squat", "g1_bwd_left_squat", "g4_exp_fwd_left_squat", "g4_exp_bwd_left_squat"]
        right_cases = ["g1_fwd_right_squat", "g1_bwd_right_squat", "g4_exp_fwd_right_squat", "g4_exp_bwd_right_squat"]

        left_l_sym = np.mean([per_sc_l_sym[c] for c in left_cases])
        right_l_sym = np.mean([per_sc_l_sym[c] for c in right_cases])

        results[m_name] = {
            "global_l_sym": global_l_sym,
            "left_4cases_l_sym": float(left_l_sym),
            "right_4cases_l_sym": float(right_l_sym),
            "inplace_squat_l_sym": per_sc_l_sym["g3_inplace_squat"],
            "per_sc_l_sym": per_sc_l_sym,
        }
        print(f"Model: {m_name:<14} | Global L_sym = {global_l_sym:.6f} | Left-4 L_sym = {left_l_sym:.6f} | Right-4 L_sym = {right_l_sym:.6f} | Inplace = {per_sc_l_sym['g3_inplace_squat']:.6f}")

    with open("evaluation_assets/phase9b/results/fixed_obs_baseline_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Baseline L_sym saved to evaluation_assets/phase9b/results/fixed_obs_baseline_eval.json")

if __name__ == '__main__':
    main()
