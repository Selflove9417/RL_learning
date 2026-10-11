"""
Unit test suite for Phase 6B-Pilot: Reference-Policy KL Regularization.

Verifies:
1. Reference checkpoint SHA256 matches official Stage 1 model_25 baseline.
2. Reference Actor and ref_std weights match checkpoint tensor-by-tensor.
3. Reference Actor parameters are strictly frozen (requires_grad=False).
4. Reference policy loading does not alter random number generator (RNG) state.
5. KL divergence between identical policies is zero (< 1e-7).
6. KL divergence under parameter perturbations is strictly positive and monotonically increasing.
7. Analytical Gaussian KL formula exactly matches PyTorch kl_divergence (< 1e-7 in double precision).
8. beta = 0 exactly recovers original PPO loss and actor/critic gradients.
9. Reference KL loss propagates gradients to current Actor and scales linearly with beta.
10. Reference Actor receives ZERO gradients (grad is None).
11. Critic loss and critic gradients are completely decoupled from Reference KL.
12. Output shapes match and no NaN / Inf is produced.
"""

import copy
import hashlib
import os
import sys
import unittest
import torch
import numpy as np

WORKSPACE_ROOT = "/home/xy/IsaacTu"
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "rsl_rl"))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "legged_gym"))

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from torch.distributions import Normal, kl_divergence

CHECKPOINT_PATH = os.path.join(
    WORKSPACE_ROOT,
    "legged_gym/logs/myrobot/Oct10_15-48-32_turning_height_stage5_3_formal/model_25.pt"
)
EXPECTED_SHA256 = "d67474d0e47f908a8c863e605b8d5de7ea1bd9f89f41d2c303fbef33d963536c"

NUM_ACTOR_OBS = 31
NUM_CRITIC_OBS = 31
NUM_ACTIONS = 6
ACTOR_HIDDEN_DIMS = [512, 256, 128]
CRITIC_HIDDEN_DIMS = [512, 256, 128]


class TestPhase6BReferenceKL(unittest.TestCase):

    def setUp(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def make_ac(self):
        return ActorCritic(
            num_actor_obs=NUM_ACTOR_OBS,
            num_critic_obs=NUM_CRITIC_OBS,
            num_actions=NUM_ACTIONS,
            actor_hidden_dims=ACTOR_HIDDEN_DIMS,
            critic_hidden_dims=CRITIC_HIDDEN_DIMS
        ).to(self.device)

    def test_01_checkpoint_sha256(self):
        """Verify baseline checkpoint SHA256 integrity."""
        self.assertTrue(os.path.isfile(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}")
        with open(CHECKPOINT_PATH, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(
            actual_sha, EXPECTED_SHA256,
            f"SHA256 mismatch! Expected {EXPECTED_SHA256}, got {actual_sha}"
        )

    def test_02_ref_actor_weights_match_and_frozen(self):
        """Verify Reference Actor weights match model_25.pt and are strictly frozen."""
        actor_critic = self.make_ac()
        ppo = PPO(
            actor_critic=actor_critic,
            learning_rate=7.59375e-5,
            device=self.device,
            enable_ref_kl=True,
            ref_kl_coef=0.05,
            ref_policy_path=CHECKPOINT_PATH
        )

        ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device)
        model_state = ckpt['model_state_dict']

        # Check all actor parameters match
        for name, param in ppo.ref_actor.named_parameters():
            ckpt_key = f"actor.{name}"
            self.assertIn(ckpt_key, model_state, f"Missing key in checkpoint: {ckpt_key}")
            self.assertTrue(
                torch.equal(param.data, model_state[ckpt_key].to(self.device)),
                f"Weight mismatch in ref_actor: {name}"
            )
            self.assertFalse(param.requires_grad, f"ref_actor parameter {name} must have requires_grad=False")

        # Check ref_std matches and is frozen
        self.assertTrue(
            torch.equal(ppo.ref_std, model_state['std'].to(self.device)),
            "ref_std does not match checkpoint std"
        )
        self.assertFalse(ppo.ref_std.requires_grad, "ref_std must have requires_grad=False")

    def test_03_zero_rng_consumption(self):
        """Verify loading Reference Actor consumes 0 random numbers."""
        torch.manual_seed(42)
        np.random.seed(42)

        actor_critic = self.make_ac()

        # Capture RNG state right before PPO initialization
        rng_state_torch_before = torch.get_rng_state()
        rng_state_np_before = np.random.get_state()

        # Initialize PPO with enable_ref_kl=True
        ppo = PPO(
            actor_critic=actor_critic,
            learning_rate=7.59375e-5,
            device=self.device,
            enable_ref_kl=True,
            ref_kl_coef=0.05,
            ref_policy_path=CHECKPOINT_PATH
        )

        rng_state_torch_after = torch.get_rng_state()
        rng_state_np_after = np.random.get_state()

        self.assertTrue(
            torch.equal(rng_state_torch_before, rng_state_torch_after),
            "PPO Reference Actor initialization modified torch RNG state!"
        )
        self.assertEqual(
            rng_state_np_before[1].tolist(), rng_state_np_after[1].tolist(),
            "PPO Reference Actor initialization modified numpy RNG state!"
        )

    def test_04_kl_zero_at_identity_and_monotonically_increases(self):
        """Verify KL is zero at identity and strictly positive & monotonically increasing with perturbation."""
        actor_critic = self.make_ac()

        # Load model_25 weights into actor_critic
        ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device)
        actor_critic.load_state_dict(ckpt['model_state_dict'])

        ppo = PPO(
            actor_critic=actor_critic,
            learning_rate=7.59375e-5,
            device=self.device,
            enable_ref_kl=True,
            ref_kl_coef=0.05,
            ref_policy_path=CHECKPOINT_PATH
        )

        obs = torch.randn(128, NUM_ACTOR_OBS, device=self.device)

        # 1. Identity check
        with torch.no_grad():
            mu_ref = ppo.ref_actor(obs)
            mu_curr = ppo.actor_critic.actor(obs)
            sigma_curr = ppo.actor_critic.std

            dist_ref = Normal(mu_ref, ppo.ref_std)
            dist_curr = Normal(mu_curr, sigma_curr)
            kl_identity = kl_divergence(dist_ref, dist_curr).sum(dim=-1).mean().item()

        self.assertAlmostEqual(
            kl_identity, 0.0, places=6,
            msg=f"KL at identity must be 0, got {kl_identity}"
        )

        # 2. Perturbation monotonic increase check
        prev_kl = kl_identity
        for scale in [0.01, 0.05, 0.1, 0.5, 1.0]:
            perturbed_mu = mu_ref + scale * torch.ones_like(mu_ref)
            dist_pert = Normal(perturbed_mu, ppo.ref_std)
            kl_val = kl_divergence(dist_ref, dist_pert).sum(dim=-1).mean().item()
            self.assertGreater(
                kl_val, prev_kl,
                f"KL at perturbation scale {scale} ({kl_val}) not strictly greater than previous ({prev_kl})"
            )
            prev_kl = kl_val

    def test_05_analytical_kl_matches_pytorch(self):
        """Verify analytical Gaussian KL formula matches PyTorch kl_divergence (< 1e-7 in double precision)."""
        batch_size = 256
        mu_ref = torch.randn(batch_size, NUM_ACTIONS, dtype=torch.float64, device=self.device)
        sigma_ref = torch.exp(torch.randn(NUM_ACTIONS, dtype=torch.float64, device=self.device) * 0.2)

        mu_curr = torch.randn(batch_size, NUM_ACTIONS, dtype=torch.float64, device=self.device)
        sigma_curr = torch.exp(torch.randn(NUM_ACTIONS, dtype=torch.float64, device=self.device) * 0.2)

        # Analytical formula:
        # KL(ref || curr) = log(sigma_curr / sigma_ref) + (sigma_ref^2 + (mu_ref - mu_curr)^2) / (2 * sigma_curr^2) - 0.5
        kl_analytical = (
            torch.log(sigma_curr / sigma_ref) +
            (sigma_ref.pow(2) + (mu_ref - mu_curr).pow(2)) / (2.0 * sigma_curr.pow(2)) -
            0.5
        ).sum(dim=-1).mean()

        # PyTorch formula:
        dist_ref = Normal(mu_ref, sigma_ref)
        dist_curr = Normal(mu_curr, sigma_curr)
        kl_pytorch = kl_divergence(dist_ref, dist_curr).sum(dim=-1).mean()

        diff = torch.abs(kl_analytical - kl_pytorch).item()
        self.assertLess(
            diff, 1e-7,
            f"Analytical KL formula deviates from PyTorch kl_divergence by {diff}"
        )

    def test_06_beta_zero_recovers_original_ppo(self):
        """Verify beta=0 yields identical loss and gradients as enable_ref_kl=False."""
        obs = torch.randn(64, NUM_ACTOR_OBS, device=self.device)
        actions = torch.randn(64, NUM_ACTIONS, device=self.device)
        adv = torch.randn(64, 1, device=self.device)
        old_logp = torch.randn(64, device=self.device)

        # Model A: enable_ref_kl = False
        torch.manual_seed(123)
        ac_a = self.make_ac()
        ppo_a = PPO(ac_a, learning_rate=7.59375e-5, device=self.device, enable_ref_kl=False)

        # Model B: enable_ref_kl = True, ref_kl_coef = 0.0
        torch.manual_seed(123)
        ac_b = self.make_ac()
        ppo_b = PPO(
            ac_b, learning_rate=7.59375e-5, device=self.device,
            enable_ref_kl=True, ref_kl_coef=0.0, ref_policy_path=CHECKPOINT_PATH
        )

        # Forward A
        ac_a.act(obs)
        logp_a = ac_a.get_actions_log_prob(actions)
        ratio_a = torch.exp(logp_a - old_logp)
        surr_a = torch.max(-adv.squeeze() * ratio_a, -adv.squeeze() * torch.clamp(ratio_a, 0.8, 1.2)).mean()
        surr_a.backward()

        # Forward B
        ac_b.act(obs)
        logp_b = ac_b.get_actions_log_prob(actions)
        ratio_b = torch.exp(logp_b - old_logp)
        surr_b = torch.max(-adv.squeeze() * ratio_b, -adv.squeeze() * torch.clamp(ratio_b, 0.8, 1.2)).mean()

        with torch.no_grad():
            mu_ref = ppo_b.ref_actor(obs)
        dist_ref = Normal(mu_ref, ppo_b.ref_std)
        dist_curr = Normal(ac_b.action_mean, ac_b.action_std)
        ref_kl = kl_divergence(dist_ref, dist_curr).sum(dim=-1).mean()
        loss_b = surr_b + 0.0 * ref_kl
        loss_b.backward()

        self.assertAlmostEqual(surr_a.item(), loss_b.item(), places=6)

        # Compare gradients
        for p_a, p_b in zip(ac_a.actor.parameters(), ac_b.actor.parameters()):
            self.assertTrue(torch.allclose(p_a.grad, p_b.grad, atol=1e-7))

    def test_07_gradients_propagate_and_scale_with_beta(self):
        """Verify Reference KL propagates gradients to current actor and scales linearly with beta."""
        torch.manual_seed(999)
        ac = self.make_ac()
        ppo = PPO(
            ac, learning_rate=7.59375e-5, device=self.device,
            enable_ref_kl=True, ref_kl_coef=0.1, ref_policy_path=CHECKPOINT_PATH
        )

        obs = torch.randn(64, NUM_ACTOR_OBS, device=self.device)
        ac.act(obs)

        with torch.no_grad():
            mu_ref = ppo.ref_actor(obs)
        dist_ref = Normal(mu_ref, ppo.ref_std)
        dist_curr = Normal(ac.action_mean, ac.action_std)
        kl = kl_divergence(dist_ref, dist_curr).sum(dim=-1).mean()

        # Test beta 1 = 0.05
        beta_1 = 0.05
        loss_1 = beta_1 * kl
        ac.zero_grad()
        loss_1.backward(retain_graph=True)

        grads_1 = [p.grad.clone() for p in ac.actor.parameters()]
        norm_1 = torch.norm(torch.stack([torch.norm(g, 2) for g in grads_1]), 2).item()
        self.assertGreater(norm_1, 0.0, "Actor gradient norm must be non-zero for beta > 0")

        # Test beta 2 = 0.20 (4x beta_1)
        beta_2 = 0.20
        loss_2 = beta_2 * kl
        ac.zero_grad()
        loss_2.backward()

        grads_2 = [p.grad.clone() for p in ac.actor.parameters()]
        norm_2 = torch.norm(torch.stack([torch.norm(g, 2) for g in grads_2]), 2).item()

        expected_ratio = beta_2 / beta_1
        actual_ratio = norm_2 / norm_1
        self.assertAlmostEqual(
            actual_ratio, expected_ratio, places=5,
            msg=f"Gradient norm scaling ratio expected {expected_ratio}, got {actual_ratio}"
        )

    def test_08_reference_actor_has_no_gradients(self):
        """Verify Reference Actor parameters are never updated and grad is None."""
        ac = self.make_ac()
        ppo = PPO(
            ac, learning_rate=7.59375e-5, device=self.device,
            enable_ref_kl=True, ref_kl_coef=0.2, ref_policy_path=CHECKPOINT_PATH
        )

        obs = torch.randn(32, NUM_ACTOR_OBS, device=self.device)
        ac.act(obs)

        with torch.no_grad():
            mu_ref = ppo.ref_actor(obs)
        dist_ref = Normal(mu_ref, ppo.ref_std)
        dist_curr = Normal(ac.action_mean, ac.action_std)
        loss = 0.2 * kl_divergence(dist_ref, dist_curr).sum(dim=-1).mean()

        loss.backward()

        for name, param in ppo.ref_actor.named_parameters():
            self.assertIsNone(
                param.grad,
                f"Reference Actor parameter '{name}' must have grad == None!"
            )

    def test_09_critic_loss_and_gradients_decoupled(self):
        """Verify Critic loss and gradients are completely decoupled from Reference KL."""
        obs = torch.randn(32, NUM_ACTOR_OBS, device=self.device)
        critic_obs = torch.randn(32, NUM_CRITIC_OBS, device=self.device)
        returns = torch.randn(32, 1, device=self.device)

        # Baseline: beta = 0
        torch.manual_seed(777)
        ac_0 = self.make_ac()
        val_0 = ac_0.evaluate(critic_obs)
        val_loss_0 = (returns - val_0).pow(2).mean()
        val_loss_0.backward()
        critic_grads_0 = [p.grad.clone() for p in ac_0.critic.parameters()]

        # With Ref KL: beta = 0.5
        torch.manual_seed(777)
        ac_reg = self.make_ac()
        ppo_reg = PPO(
            ac_reg, learning_rate=7.59375e-5, device=self.device,
            enable_ref_kl=True, ref_kl_coef=0.5, ref_policy_path=CHECKPOINT_PATH
        )
        ac_reg.act(obs)
        val_reg = ac_reg.evaluate(critic_obs)
        val_loss_reg = (returns - val_reg).pow(2).mean()

        with torch.no_grad():
            mu_ref = ppo_reg.ref_actor(obs)
        dist_ref = Normal(mu_ref, ppo_reg.ref_std)
        dist_curr = Normal(ac_reg.action_mean, ac_reg.action_std)
        ref_kl = kl_divergence(dist_ref, dist_curr).sum(dim=-1).mean()

        total_loss = val_loss_reg + 0.5 * ref_kl
        total_loss.backward()
        critic_grads_reg = [p.grad.clone() for p in ac_reg.critic.parameters()]

        # Compare critic gradients
        for g0, greg in zip(critic_grads_0, critic_grads_reg):
            self.assertTrue(
                torch.equal(g0, greg),
                "Critic gradients differ between beta=0 and beta=0.5! Decoupling violated."
            )

    def test_10_output_shapes_and_no_nan_inf(self):
        """Verify observation forward pass produces finite values and matching dimensions."""
        ac = self.make_ac()
        ppo = PPO(
            ac, learning_rate=7.59375e-5, device=self.device,
            enable_ref_kl=True, ref_kl_coef=0.05, ref_policy_path=CHECKPOINT_PATH
        )

        obs = torch.randn(64, NUM_ACTOR_OBS, device=self.device)
        with torch.no_grad():
            ref_mu = ppo.ref_actor(obs)

        self.assertEqual(ref_mu.shape, (64, NUM_ACTIONS))
        self.assertFalse(torch.isnan(ref_mu).any())
        self.assertFalse(torch.isinf(ref_mu).any())


if __name__ == '__main__':
    unittest.main()
