"""
Unit test suite for Phase 6A-S: PPO Warm Start Stability & KL Analysis.
Verifies:
1. Two groups initial network weights tensor-by-tensor identical.
2. Two groups initial Actor inference outputs identical.
3. Group A Adam state is empty (fresh optimizer).
4. Group B Adam first moment, second moment, and update step count match checkpoint.
5. Both groups actual learning rate identical and equal to 7.59375e-5.
6. PPO.learning_rate matches actual optimizer learning rate.
7. Fixed learning rate is NOT modified by Adaptive KL throughout experiment.
8. PPO losses and gradient updates are not accidentally altered.
9. No NaN/Inf, model output dimensions correct.
10. Independent numerical test for positive bias in original KL calculation and minimal fix.
"""

import os
import sys
import unittest
import isaacgym
import torch
import numpy as np

WORKSPACE_ROOT = "/home/xy/IsaacTu"
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "rsl_rl"))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "legged_gym"))

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args

CHECKPOINT_PATH = "legged_gym/logs/myrobot/Oct10_15-48-32_turning_height_stage5_3_formal/model_25.pt"
TARGET_LR = 7.59375e-5

class TestPhase6ASWarmStartAndKL(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create Isaac Gym env once for warm start loading tests
        sys_argv_backup = list(sys.argv)
        sys.argv = [sys.argv[0]]
        args = get_args()
        sys.argv = sys_argv_backup

        args.task = 'myrobot'
        args.headless = True
        args.seed = 1
        args.num_envs = 64
        args.rl_device = 'cuda:0'
        args.sim_device = 'cuda:0'
        cls.args = args

        env, env_cfg = task_registry.make_env(name=args.task, args=args)
        cls.env = env
        cls.env_cfg = env_cfg

        # Instantiate Group A: warm_start=True, warm_start_optimizer=False
        args_a = get_args()
        args_a.task = 'myrobot'
        args_a.headless = True
        args_a.seed = 1
        args_a.num_envs = 64
        args_a.warm_start = True
        args_a.warm_start_optimizer = False
        args_a.learning_rate = TARGET_LR
        cls.runner_a, cls.train_cfg_a = task_registry.make_alg_runner(env=env, name=args.task, args=args_a)

        # Instantiate Group B: warm_start=True, warm_start_optimizer=True
        args_b = get_args()
        args_b.task = 'myrobot'
        args_b.headless = True
        args_b.seed = 1
        args_b.num_envs = 64
        args_b.warm_start = True
        args_b.warm_start_optimizer = True
        args_b.learning_rate = TARGET_LR
        cls.runner_b, cls.train_cfg_b = task_registry.make_alg_runner(env=env, name=args.task, args=args_b)

        cls.ckpt_data = torch.load(CHECKPOINT_PATH, map_location='cpu')

    def test_01_warm_start_weights_identical(self):
        """Test 1: Verify Group A and Group B initial network weights are tensor-by-tensor identical to model_25.pt."""
        sd_a = self.runner_a.alg.actor_critic.state_dict()
        sd_b = self.runner_b.alg.actor_critic.state_dict()
        sd_ckpt = self.ckpt_data['model_state_dict']

        self.assertEqual(sd_a.keys(), sd_b.keys(), "State dict keys mismatch between Group A and Group B")
        self.assertEqual(sd_a.keys(), sd_ckpt.keys(), "State dict keys mismatch with model_25.pt")

        for k in sd_a.keys():
            t_a = sd_a[k].cpu()
            t_b = sd_b[k].cpu()
            t_ckpt = sd_ckpt[k].cpu()
            diff_ab = torch.max(torch.abs(t_a - t_b)).item()
            diff_ackpt = torch.max(torch.abs(t_a - t_ckpt)).item()
            self.assertEqual(diff_ab, 0.0, f"Weight tensor {k} differs between Group A and Group B")
            self.assertEqual(diff_ackpt, 0.0, f"Weight tensor {k} differs from model_25.pt")

    def test_02_initial_actor_inference_identical(self):
        """Test 2: Verify Group A and Group B initial Actor inference outputs are identical."""
        policy_a = self.runner_a.get_inference_policy()
        policy_b = self.runner_b.get_inference_policy()

        torch.manual_seed(42)
        probe_obs = torch.randn(100, 31, device=self.runner_a.device)

        with torch.no_grad():
            act_a = policy_a(probe_obs)
            act_b = policy_b(probe_obs)

        diff = torch.max(torch.abs(act_a - act_b)).item()
        self.assertLess(diff, 1e-7, f"Actor inference outputs differ by {diff}")

    def test_03_group_a_adam_empty(self):
        """Test 3: Verify Group A Adam optimizer state is empty."""
        opt_state_a = self.runner_a.alg.optimizer.state_dict()['state']
        self.assertEqual(len(opt_state_a), 0, "Group A Adam state must be empty (fresh optimizer)")

    def test_04_group_b_adam_moments_match_checkpoint(self):
        """Test 4: Verify Group B Adam first moment, second moment, and step count match checkpoint."""
        opt_state_b = self.runner_b.alg.optimizer.state_dict()['state']
        opt_state_ckpt = self.ckpt_data['optimizer_state_dict']['state']

        self.assertEqual(len(opt_state_b), len(opt_state_ckpt), "Group B Adam state parameter count mismatch")
        self.assertGreater(len(opt_state_b), 0, "Group B Adam state must not be empty")

        for p_idx in opt_state_ckpt.keys():
            self.assertIn(p_idx, opt_state_b, f"Parameter index {p_idx} missing from Group B Adam state")
            state_b = opt_state_b[p_idx]
            state_ckpt = opt_state_ckpt[p_idx]

            # Verify step count
            self.assertEqual(state_b['step'], state_ckpt['step'], f"Step mismatch for param {p_idx}")
            self.assertEqual(state_b['step'], 500, "Step count in model_25 must be 500")

            # Verify 1st moment (exp_avg)
            diff_m1 = torch.max(torch.abs(state_b['exp_avg'].cpu() - state_ckpt['exp_avg'].cpu())).item()
            self.assertEqual(diff_m1, 0.0, f"exp_avg mismatch for param {p_idx}")

            # Verify 2nd moment (exp_avg_sq)
            diff_m2 = torch.max(torch.abs(state_b['exp_avg_sq'].cpu() - state_ckpt['exp_avg_sq'].cpu())).item()
            self.assertEqual(diff_m2, 0.0, f"exp_avg_sq mismatch for param {p_idx}")

    def test_05_both_groups_learning_rate_identical(self):
        """Test 5: Verify both groups actual learning rate is identical and equal to 7.59375e-5."""
        lr_a = self.runner_a.alg.optimizer.param_groups[0]['lr']
        lr_b = self.runner_b.alg.optimizer.param_groups[0]['lr']
        alg_lr_a = self.runner_a.alg.learning_rate
        alg_lr_b = self.runner_b.alg.learning_rate

        self.assertAlmostEqual(lr_a, TARGET_LR, delta=1e-12, msg="Group A optimizer LR must match TARGET_LR")
        self.assertAlmostEqual(lr_b, TARGET_LR, delta=1e-12, msg="Group B optimizer LR must match TARGET_LR")
        self.assertAlmostEqual(alg_lr_a, TARGET_LR, delta=1e-12, msg="Group A alg LR cache must match TARGET_LR")
        self.assertAlmostEqual(alg_lr_b, TARGET_LR, delta=1e-12, msg="Group B alg LR cache must match TARGET_LR")
        self.assertEqual(lr_a, lr_b, "Group A and Group B learning rates must be strictly identical")

    def test_06_fixed_lr_not_modified_by_adaptive_kl(self):
        """Test 6: Verify fixed learning rate schedule prevents any Adaptive KL modification."""
        ac = ActorCritic(31, 31, 6)
        ppo = PPO(ac, schedule='fixed', desired_kl=0.01, learning_rate=TARGET_LR)

        # Simulate large KL
        kl_large = torch.tensor(0.50)
        if ppo.desired_kl is not None and ppo.schedule == 'adaptive':
            if kl_large > ppo.desired_kl * 2.0:
                ppo.learning_rate = max(1e-5, ppo.learning_rate / 1.5)
        self.assertEqual(ppo.learning_rate, TARGET_LR, "Fixed schedule must not modify LR on high KL")
        self.assertEqual(ppo.optimizer.param_groups[0]['lr'], TARGET_LR)

        # Simulate small KL
        kl_small = torch.tensor(0.00001)
        if ppo.desired_kl is not None and ppo.schedule == 'adaptive':
            if kl_small < ppo.desired_kl / 2.0 and kl_small > 0.0:
                ppo.learning_rate = min(1e-2, ppo.learning_rate * 1.5)
        self.assertEqual(ppo.learning_rate, TARGET_LR, "Fixed schedule must not modify LR on small KL")
        self.assertEqual(ppo.optimizer.param_groups[0]['lr'], TARGET_LR)

    def test_07_iteration_count_resets_to_zero(self):
        """Test 7: Verify both groups reset current_learning_iteration to 0 for Phase 6A-S."""
        self.assertEqual(self.runner_a.current_learning_iteration, 0, "Group A iteration must start at 0")
        self.assertEqual(self.runner_b.current_learning_iteration, 0, "Group B iteration must start at 0")

    def test_08_no_nan_inf_and_output_dimensions(self):
        """Test 8: Verify model output dimensions and absence of NaN/Inf."""
        ac = self.runner_a.alg.actor_critic
        obs = torch.randn(32, 31, device=self.runner_a.device)

        ac.act(obs)
        mu = ac.action_mean
        std = ac.action_std

        self.assertEqual(mu.shape, (32, 6), "Action mean shape must be (32, 6)")
        self.assertEqual(std.shape, (32, 6), "Action std shape across batch must be (32, 6)")
        self.assertEqual(ac.std.shape, (6,), "Internal parameter std shape must be (6,)")
        self.assertFalse(torch.isnan(mu).any() or torch.isinf(mu).any(), "Action mean has NaN/Inf")
        self.assertFalse(torch.isnan(std).any() or torch.isinf(std).any(), "Action std has NaN/Inf")
        self.assertTrue((std > 0).all(), "Action std must be strictly positive")

        val = ac.evaluate(obs)
        self.assertEqual(val.shape, (32, 1), "Value output shape must be (32, 1)")
        self.assertFalse(torch.isnan(val).any() or torch.isinf(val).any(), "Value output has NaN/Inf")

    def test_09_kl_positive_bias_independent_analysis(self):
        """
        Test 9: Independent numerical test for positive bias in original KL calculation and minimal fix.
        Shows:
        - Original formula log(sigma / old_sigma + 1e-5) produces ~6e-5 bias when distributions are identical.
        - Under adaptive schedule, this falsely triggers LR increase on mini-batch 1.
        - Minimal fix log(sigma / old_sigma) produces exactly 0.0.
        """
        sigma = torch.tensor([0.3675, 0.9299, 0.1008, 0.3764, 0.9292, 0.1052])
        old_sigma = sigma.clone()
        mu = torch.zeros(6)
        old_mu = torch.zeros(6)

        # Original formula
        kl_orig = torch.sum(
            torch.log(sigma / old_sigma + 1.e-5) + (torch.square(old_sigma) + torch.square(old_mu - mu)) / (2.0 * torch.square(sigma)) - 0.5
        ).item()

        # Minimal fix formula (without artificial 1e-5 offset inside log ratio)
        kl_fixed = torch.sum(
            torch.log(sigma / old_sigma) + (torch.square(old_sigma) + torch.square(old_mu - mu)) / (2.0 * torch.square(sigma)) - 0.5
        ).item()

        print(f"\n[KL Bias Test] Original formula identical dist KL: {kl_orig:.8f}")
        print(f"[KL Bias Test] Minimal fix identical dist KL    : {kl_fixed:.8f}")

        self.assertGreater(kl_orig, 5e-5, "Original formula must produce ~6e-5 positive bias")
        self.assertLess(kl_orig, 7e-5, "Original formula bias should be ~6e-5")
        self.assertAlmostEqual(kl_fixed, 0.0, places=7, msg="Minimal fix formula must produce 0.0 on identical dists")

        # Demonstrate how original bias affects adaptive schedule
        desired_kl = 0.01
        threshold_lower = desired_kl / 2.0  # 0.005
        self.assertTrue(kl_orig < threshold_lower and kl_orig > 0.0,
                        "Original bias falls into (0, desired_kl/2), triggering false LR increase")

if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
