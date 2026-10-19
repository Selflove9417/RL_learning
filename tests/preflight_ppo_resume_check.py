#!/usr/bin/env python3
# ==============================================================================
# Phase 4.7-S1 Task 3: Zero-Update Preflight Rollout Verification
# Loads model_critic_calibrated_target_b.pt into PPO Runner, executes 1 full
# rollout (140 steps x 64 envs = 8,960 transitions) with ZERO optimizer updates.
# Verifies:
#   1. Actor weights and std are identical to model_champion.pt (diff = 0.00e+00)
#   2. Returns, Advantages, and Values are 100% finite (no NaN, no Inf)
#   3. Terminal observation capture at step 139 works online
#   4. PPO transition rewards are augmented with online gamma * V_calib(s_140)
#   5. ZERO parameter changes occur during rollout (guaranteed frozen)
# ==============================================================================

import os
import sys
import json
import isaacgym
from isaacgym import gymapi, gymtorch
import torch
import numpy as np

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner
from legged_gym.envs import task_registry
from legged_gym.utils import get_args


def run_preflight_check():
    print("=" * 80)
    print("PHASE 4.7-S1 TASK 3: MINIMAL PPO RESUME PREFLIGHT CHECK (ZERO UPDATE)")
    print("=" * 80)

    task_name = "myrobot_forward_jump_phase4"
    num_envs = 64
    sys.argv = ["preflight", "--task", task_name, "--num_envs", str(num_envs), "--headless"]
    args = get_args()
    args.seed = 6001

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs
    env_cfg.seed = 6001
    env_cfg.domain_rand.push_robots = False
    env_cfg.noise.add_noise = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # File paths
    champ_path = os.path.join(workspace_root, "evaluation_assets/phase3_1_forward_distance/models/model_champion.pt")
    calib_path = os.path.join(workspace_root, "evaluation_assets/phase4_7_critic_calibration/model_critic_calibrated_target_b.pt")

    # Load reference Champion ActorCritic for strict parameter comparison
    ac_champ = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac_champ.load_state_dict(torch.load(champ_path, map_location=env.device)["model_state_dict"])
    ac_champ.eval()

    # Create OnPolicyRunner with train_cfg
    train_cfg_dict = train_cfg
    from legged_gym.utils import class_to_dict
    if not isinstance(train_cfg_dict, dict):
        train_cfg_dict = class_to_dict(train_cfg)

    # Initialize runner without log_dir (test only)
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=None, device=env.device)

    # Load calibrated checkpoint into runner
    print(f"Loading calibrated model into Runner: {calib_path}")
    runner.load(calib_path, load_optimizer=False)
    runner.current_learning_iteration = 0

    # 1. Verify Actor parameters before rollout
    ac_runner = runner.alg.actor_critic
    p_champ = torch.cat([p.flatten() for p in ac_champ.actor.parameters()])
    p_runner_before = torch.cat([p.flatten() for p in ac_runner.actor.parameters()])
    diff_actor_before = (p_champ - p_runner_before).abs().max().item()
    diff_std_before = (ac_champ.std - ac_runner.std).abs().max().item()
    print(f"\n[Pre-Rollout Parameter Check]")
    print(f"  Actor weights diff vs Champion: {diff_actor_before:.2e}")
    print(f"  Actor std diff vs Champion    : {diff_std_before:.2e}")
    assert diff_actor_before == 0.0, "Actor weights in runner do not match Champion!"
    assert diff_std_before == 0.0, "Actor std in runner does not match Champion!"

    # Snapshot all model parameters before rollout
    params_all_before = torch.cat([p.flatten() for p in ac_runner.parameters()]).clone()

    # 2. Execute 1 synchronous rollout (140 steps) with NO OPTIMIZER UPDATE
    print("\nExecuting 1 synchronous rollout (140 steps x 64 envs = 8,960 transitions)...")
    obs, _ = env.reset()
    critic_obs = obs

    step139_info = {}

    with torch.inference_mode():
        for i in range(runner.num_steps_per_env):
            actions = runner.alg.act(obs, critic_obs)
            next_obs, next_critic_obs, rewards, dones, infos = env.step(actions)
            if next_critic_obs is None:
                next_critic_obs = next_obs
            
            obs = next_obs.to(env.device)
            critic_obs = next_critic_obs.to(env.device)
            rewards = rewards.to(env.device)
            dones = dones.to(env.device)

            if i == 139:
                step139_info["has_timeouts"] = 'time_outs' in infos
                step139_info["timeouts_all"] = bool(infos['time_outs'].all().item()) if 'time_outs' in infos else False
                step139_info["has_term_obs"] = 'terminal_observation' in infos
                step139_info["raw_rew_mean"] = rewards.mean().item()
                step139_info["dones_all"] = bool(dones.all().item())
                if 'terminal_observation' in infos:
                    v_term_online = ac_runner.evaluate(infos['terminal_observation'].to(env.device)).squeeze(-1)
                    step139_info["v_term_online_mean"] = v_term_online.mean().item()

            runner.alg.process_env_step(rewards, dones, infos)

        # Compute returns on storage
        runner.alg.compute_returns(critic_obs)

    # 3. Check RolloutStorage numerical validity
    storage = runner.alg.storage
    vals = storage.values
    rets = storage.returns
    advs = storage.advantages
    rews = storage.rewards

    print("\n[Rollout Numerical Integrity Check]")
    print(f"  Values     : min={vals.min().item():.4f}, max={vals.max().item():.4f}, has_nan={bool(torch.isnan(vals).any().item())}")
    print(f"  Returns    : min={rets.min().item():.4f}, max={rets.max().item():.4f}, has_nan={bool(torch.isnan(rets).any().item())}")
    print(f"  Advantages : min={advs.min().item():.4f}, max={advs.max().item():.4f}, has_nan={bool(torch.isnan(advs).any().item())}")
    print(f"  Stored Rew : min={rews.min().item():.4f}, max={rews.max().item():.4f}, has_nan={bool(torch.isnan(rews).any().item())}")
    
    assert not torch.isnan(vals).any(), "NaN detected in values!"
    assert not torch.isinf(vals).any(), "Inf detected in values!"
    assert not torch.isnan(rets).any(), "NaN detected in returns!"
    assert not torch.isinf(rets).any(), "Inf detected in returns!"
    assert not torch.isnan(advs).any(), "NaN detected in advantages!"
    assert not torch.isinf(advs).any(), "Inf detected in advantages!"

    # 4. Step 139 Online Bootstrap Check
    print("\n[Step 139 Online Bootstrap Validation]")
    print(f"  time_outs in infos           : {step139_info['has_timeouts']} (all={step139_info['timeouts_all']})")
    print(f"  terminal_obs in infos        : {step139_info['has_term_obs']}")
    print(f"  V_calib(s_140) evaluated     : {step139_info['v_term_online_mean']:.4f}")
    print(f"  Raw reward mean at step 139  : {step139_info['raw_rew_mean']:.4f}")
    
    stored_rew_139 = storage.rewards[139].mean().item()
    expected_online_rew = step139_info['raw_rew_mean'] + runner.alg.gamma * step139_info['v_term_online_mean']
    print(f"  Stored reward at step 139    : {stored_rew_139:.4f}")
    print(f"  Expected (raw + gamma*V_calib): {expected_online_rew:.4f}")
    diff_online_boot = abs(stored_rew_139 - expected_online_rew)
    print(f"  Bootstrap discrepancy        : {diff_online_boot:.6e}")
    assert diff_online_boot < 1e-4, "Online bootstrap reward discrepancy!"

    # 5. Verify Model Parameters remain 100% FROZEN (ZERO UPDATE)
    params_all_after = torch.cat([p.flatten() for p in ac_runner.parameters()])
    total_param_diff = (params_all_before - params_all_after).abs().max().item()
    print(f"\n[Zero-Update Verification]")
    print(f"  Total model parameter change across rollout: {total_param_diff:.2e}")
    assert total_param_diff == 0.0, "Model parameters changed during rollout!"

    # Actor action distribution verification
    obs_test = storage.observations[0:10, 0] # first 10 steps of env 0
    act_champ = ac_champ.act_inference(obs_test)
    act_runner = ac_runner.act_inference(obs_test)
    act_diff = (act_champ - act_runner).abs().max().item()
    print(f"  Actor action output diff vs Champion: {act_diff:.2e}")
    assert act_diff == 0.0, "Actor actions differed from Champion!"

    print("\n" + "=" * 80)
    print(">>> PREFLIGHT CHECK 100% PASSED: Zero updates, clean numbers, full online bootstrap!")
    print("=" * 80)

    # Save summary JSON
    out_dir = os.path.join(workspace_root, "evaluation_assets/phase4_7_critic_calibration")
    out_json = os.path.join(out_dir, "phase4_7_s1_preflight_check.json")
    with open(out_json, "w") as f:
        json.dump({
            "model_paths": {
                "champion": champ_path,
                "calibrated_critic": calib_path,
            },
            "actor_invariance": {
                "weight_diff": diff_actor_before,
                "std_diff": diff_std_before,
                "action_diff": act_diff,
                "total_param_diff_after_rollout": total_param_diff,
            },
            "numerical_integrity": {
                "val_min": vals.min().item(),
                "val_max": vals.max().item(),
                "ret_min": rets.min().item(),
                "ret_max": rets.max().item(),
                "adv_min": advs.min().item(),
                "adv_max": advs.max().item(),
            },
            "step139_online_bootstrap": {
                "v_term_online_mean": step139_info['v_term_online_mean'],
                "stored_rew_139": stored_rew_139,
                "expected_online_rew": expected_online_rew,
                "diff": diff_online_boot,
            }
        }, f, indent=2)
    print(f"Preflight summary saved to: {out_json}")


if __name__ == "__main__":
    run_preflight_check()
