#!/usr/bin/env python3
# ==============================================================================
# Phase 4.7: Deterministic Unit Test for Target B Bootstrap Call Chain
# Verifies:
# 1. Isolated mathematical unit test for PPO.process_env_step & storage.compute_returns
#    Proves Target B returns (G_139 = r_139 + gamma * V(s_140_pre)) are received,
#    strictly distinct from Target C (s_0_post), Target A (no bootstrap), and fallback V(s_139).
# 2. Environment integration test with MyRobotForwardJumpPhase4 for 140 steps.
#    Verifies extras['time_outs'] and extras['terminal_observation'] at step 139.
# ==============================================================================

import os
import sys
import isaacgym
import torch
import torch.nn as nn
import numpy as np
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from legged_gym.envs import task_registry
from legged_gym.utils import get_args


class MockLinearCritic(nn.Module):
    """
    Simple deterministic linear model where V(s) = sum(s)
    """
    def __init__(self, num_obs=4, num_actions=2):
        super().__init__()
        self.actor = nn.Linear(num_obs, num_actions, bias=False)
        self.critic = nn.Linear(num_obs, 1, bias=False)
        with torch.no_grad():
            self.critic.weight.fill_(1.0) # V(s) = s_0 + s_1 + s_2 + s_3
        self.std = nn.Parameter(torch.ones(num_actions))
        self.is_recurrent = False

    def act(self, obs):
        return self.actor(obs)

    def evaluate(self, critic_obs):
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions):
        return torch.zeros(actions.shape[0], 1, device=actions.device)

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return torch.zeros(1, 2)

    @property
    def action_std(self):
        return torch.ones(1, 2)


def test_isolated_target_b_math():
    print("=" * 80)
    print("[TEST 1] ISOLATED MATHEMATICAL UNIT TEST: TARGET B VS TARGET C / A / FALLBACK")
    print("=" * 80)

    device = torch.device("cpu")
    num_envs = 1
    num_transitions = 2
    num_obs = 4
    num_actions = 2
    gamma = 0.99
    lam = 0.95

    ac = MockLinearCritic(num_obs, num_actions).to(device)
    ppo = PPO(
        actor_critic=ac,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=gamma,
        lam=lam,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=False,
        schedule="fixed",
        desired_kl=0.01,
        device=device,
    )
    ppo.init_storage(num_envs, num_transitions, [num_obs], [num_obs], [num_actions])

    # Construct observations with clearly distinct, easily verifiable values:
    # V(s) = sum(s)
    # s_0: [1.0, 1.0, 0.0, 0.0] -> V(s_0) = 2.0
    # s_1 (s_139): [5.0, 5.0, 0.0, 0.0] -> V(s_139) = 10.0
    # s_terminal (s_140_pre): [2.5, 2.5, 0.0, 0.0] -> V(s_140_pre) = 5.0
    # s_post_reset (s_0_post): [12.5, 12.5, 0.0, 0.0] -> V(s_0_post) = 25.0
    s_0 = torch.tensor([[1.0, 1.0, 0.0, 0.0]], device=device)
    s_139 = torch.tensor([[5.0, 5.0, 0.0, 0.0]], device=device)
    s_140_pre = torch.tensor([[2.5, 2.5, 0.0, 0.0]], device=device)
    s_0_post = torch.tensor([[12.5, 12.5, 0.0, 0.0]], device=device)

    v_0 = ac.evaluate(s_0).item()
    v_139 = ac.evaluate(s_139).item()
    v_140_pre = ac.evaluate(s_140_pre).item()
    v_0_post = ac.evaluate(s_0_post).item()

    print(f"Constructed States:")
    print(f"  V(s_0)          = {v_0:.2f}")
    print(f"  V(s_139)        = {v_139:.2f}")
    print(f"  V(s_140_pre)    = {v_140_pre:.2f}  <--- Target B Ground Truth")
    print(f"  V(s_0_post)     = {v_0_post:.2f}  <--- Target C Bug Value")

    # Step 0: Non-terminal transition
    ppo.act(s_0, s_0)
    r_0 = torch.tensor([0.5], device=device)
    dones_0 = torch.tensor([0.0], device=device)
    infos_0 = {}
    ppo.process_env_step(r_0, dones_0, infos_0)

    # Step 1 (representing step 139): Timeout transition with terminal_observation
    ppo.act(s_139, s_139)
    r_139 = torch.tensor([1.0], device=device)
    dones_1 = torch.tensor([1.0], device=device) # Reset triggered
    infos_1 = {
        "time_outs": torch.tensor([True], device=device),
        "terminal_observation": s_140_pre, # Pre-reset terminal observation
    }
    ppo.process_env_step(r_139, dones_1, infos_1)

    # Compute returns passing s_0_post as last_critic_obs
    ppo.compute_returns(s_0_post)

    target_b_g139 = 1.0 + gamma * v_140_pre      # 1.0 + 0.99 * 5.0 = 5.95
    target_c_g139 = 1.0 + gamma * v_0_post       # 1.0 + 0.99 * 25.0 = 25.75
    target_a_g139 = 1.0                          # 1.0 (terminal value = 0)
    fallback_g139 = 1.0 + gamma * v_139          # 1.0 + 0.99 * 10.0 = 10.90

    actual_g139 = ppo.storage.returns[1, 0].item()
    actual_adv139 = ppo.storage.advantages[1, 0].item()
    actual_rew139 = ppo.storage.rewards[1, 0].item()

    print(f"\nReturn Targets Comparison at Step 139:")
    print(f"  Target B (Expected Correct) : {target_b_g139:.4f}")
    print(f"  Target C (Phase 4.6 Bug)    : {target_c_g139:.4f}")
    print(f"  Target A (True Zero Term)   : {target_a_g139:.4f}")
    print(f"  RSL-RL Fallback (V(s_t))    : {fallback_g139:.4f}")
    print(f"  Actual Received Return      : {actual_g139:.4f}")
    print(f"  Actual Stored Reward        : {actual_rew139:.4f}")
    print(f"  Actual Step 139 Advantage   : {actual_adv139:.4f}")

    raw_adv139 = (ppo.storage.returns[1, 0] - ppo.storage.values[1, 0]).item()
    print(f"  Actual Step 139 Raw Adv     : {raw_adv139:.4f}")
    assert abs(raw_adv139 - (target_b_g139 - v_139)) < 1e-5, "Raw advantage does not match delta_139"
    assert abs(actual_g139 - target_c_g139) > 10.0, "FATAL: Return is still contaminated by Target C!"
    assert abs(actual_g139 - target_a_g139) > 2.0, "FATAL: Return is still zeroed out by Target A!"

    print("\n>>> [TEST 1 PASSED]: PPO perfectly computes Target B return and rejects Target A, C, and fallback!")


def test_phase4_env_step139_integration():
    print("\n" + "=" * 80)
    print("[TEST 2] FULL PHASE 4 ENVIRONMENT END-TO-END STEP 139 TIMEOUT INTEGRATION TEST")
    print("=" * 80)

    task_name = "myrobot_forward_jump_phase4"
    sys.argv = ["test_env", "--task", task_name, "--num_envs", "2", "--headless"]
    args = get_args()
    args.seed = 6001

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = 2
    env_cfg.seed = 6001

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    champ_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    ac = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(torch.load(champ_path, map_location=env.device)["model_state_dict"])
    ac.eval()

    obs, _ = env.reset()

    # Step exactly 140 steps (0 to 139)
    step139_infos = None
    step139_next_obs = None
    step139_dones = None

    with torch.no_grad():
        for step in range(140):
            act = ac.act_inference(obs)
            next_obs, _, rew, dones, infos = env.step(act)

            if step == 139:
                step139_infos = infos
                step139_next_obs = next_obs
                step139_dones = dones

            obs = next_obs

    # Verify Step 139 payload
    print("\nStep 139 Output Verification:")
    print(f"  dones.all() == 1: {step139_dones.all().item()}")
    assert step139_dones.all().item() == True, "Dones must be True at step 139!"

    assert "time_outs" in step139_infos, "extras must contain 'time_outs'!"
    assert step139_infos["time_outs"].all().item() == True, "'time_outs' must be True at step 139!"

    assert "terminal_observation" in step139_infos, "extras must contain 'terminal_observation'!"
    term_obs = step139_infos["terminal_observation"]
    assert term_obs.shape == (2, 35), f"terminal_observation shape must be (2, 35), got {term_obs.shape}"

    # Extract jump_phase from terminal_observation vs next_obs
    # Slot 12 in observation is jump_phase (see myrobot_forward_jump_env.py line 120)
    term_jump_phase = term_obs[0, 12].item()
    post_jump_phase = step139_next_obs[0, 12].item()

    print(f"  terminal_observation jump_phase = {term_jump_phase:.4f} (Expected: 0.85 standing recovery)")
    print(f"  post-reset next_obs jump_phase  = {post_jump_phase:.4f} (Expected: 0.00 initial squat)")

    assert abs(term_jump_phase - 0.85) < 1e-3, f"terminal_observation jump_phase should be 0.85, got {term_jump_phase}"
    assert abs(post_jump_phase - 0.00) < 1e-3, f"post-reset next_obs jump_phase should be 0.00, got {post_jump_phase}"

    # Evaluate Critic on terminal_obs vs post_reset_obs
    with torch.no_grad():
        v_terminal = ac.evaluate(term_obs).mean().item()
        v_post_reset = ac.evaluate(step139_next_obs).mean().item()

    print(f"  V(terminal_observation)         = {v_terminal:.4f}")
    print(f"  V(post_reset next_obs)          = {v_post_reset:.4f}")
    print(f"  Delta (Post - Terminal)         = {v_post_reset - v_terminal:+.4f}")

    assert abs(v_terminal - v_post_reset) > 0.5, "Terminal value and post-reset value must be distinctly separated!"

    print("\n>>> [TEST 2 PASSED]: Full Phase 4 environment correctly produces Target B terminal observations at step 139!")


if __name__ == "__main__":
    test_isolated_target_b_math()
    test_phase4_env_step139_integration()
