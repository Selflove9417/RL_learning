"""
Phase 6A: Unit and Integration Test Suite for PPO KL Early Stopping & Compatibility.

Tests:
7.1 Math Test: Gaussian KL divergence direction and analytical correctness.
7.2 Early Stop Test: Controlled trigger of KL early stopping, optimizer step cut-off,
    RolloutStorage cleanup, and subsequent update feasibility.
7.3 Compatibility Test: Bit-level loss and parameter update equivalence when early stopping is disabled.
7.4 Checkpoint Test: model_25.pt weight loading, tensor shapes, inference stability,
    and verification of baseline integrity (SHA256 check).
"""

import hashlib
import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in python path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "rsl_rl"))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "legged_gym"))

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


class TestPPOKLEarlyStopping(unittest.TestCase):

    def setUp(self):
        self.device = 'cpu'
        torch.manual_seed(42)
        np.random.seed(42)

    # -------------------------------------------------------------------------
    # 7.1 数学测试 (Math Test)
    # -------------------------------------------------------------------------
    def test_7_1_math_kl_divergence(self):
        """
        验证高斯策略分布 KL 计算方向与数值精度。
        公式实现:
        KL = sum( log(sigma_new / sigma_old + 1e-5) +
                  (sigma_old^2 + (mu_old - mu_new)^2) / (2 * sigma_new^2) - 0.5 )
        即 D_KL(pi_old || pi_new)
        """
        def compute_ppo_kl(old_mu, old_sigma, mu, sigma):
            return torch.sum(
                torch.log(sigma / old_sigma + 1e-5)
                + (torch.square(old_sigma) + torch.square(old_mu - mu)) / (2.0 * torch.square(sigma))
                - 0.5,
                dim=-1
            )

        # 1. 相同分布: KL 应接近 0 (仅由于 +1e-5 epsilon 有极微小的非负正值)
        mu = torch.randn(100, 6)
        sigma = torch.ones(100, 6) * 0.8
        kl_identical = compute_ppo_kl(mu, sigma, mu, sigma)
        self.assertTrue((kl_identical >= 0.0).all(), "KL 散度不能为负")
        self.assertLess(kl_identical.max().item(), 1e-4, "相同策略分布下的 KL 应接近 0")

        # 2. 均值偏移，标准差相同: D_KL = ||mu_old - mu_new||^2 / (2 * sigma^2)
        delta_mu = torch.ones(100, 6) * 0.2
        mu_new = mu + delta_mu
        kl_mean_shift = compute_ppo_kl(mu, sigma, mu_new, sigma)
        analytical_mean_shift = torch.sum(torch.square(delta_mu) / (2.0 * torch.square(sigma)), dim=-1)
        # 允许 epsilon 引起的微小误差
        self.assertTrue(torch.allclose(kl_mean_shift, analytical_mean_shift, atol=1e-4))

        # 3. 均值相同，标准差偏移: D_KL = sum( log(s_new/s_old) + s_old^2/(2*s_new^2) - 0.5 )
        sigma_new = sigma * 1.5
        kl_var_shift = compute_ppo_kl(mu, sigma, mu, sigma_new)
        analytical_var_shift = torch.sum(
            torch.log(sigma_new / sigma) + torch.square(sigma) / (2.0 * torch.square(sigma_new)) - 0.5,
            dim=-1
        )
        self.assertTrue(torch.allclose(kl_var_shift, analytical_var_shift, atol=1e-4))

        # 4. 非对称性验证: D_KL(p || q) != D_KL(q || p) 当方差不同时
        kl_forward = compute_ppo_kl(mu, sigma, mu, sigma_new)  # D_KL(old || new)
        kl_reverse = compute_ppo_kl(mu, sigma_new, mu, sigma)  # D_KL(new || old)
        self.assertFalse(torch.allclose(kl_forward, kl_reverse, atol=1e-2),
                         "KL 散度应具有非对称性")

        # 5. 单调性验证: 偏差越大，KL 越大
        kl_small_shift = compute_ppo_kl(mu, sigma, mu + 0.1, sigma)
        kl_large_shift = compute_ppo_kl(mu, sigma, mu + 0.5, sigma)
        self.assertTrue((kl_large_shift > kl_small_shift).all(),
                        "均值漂移越大，KL 散度应单调严格递增")

        print(">>> [Test 7.1 Math Test Passed]: 高斯策略 KL 散度数值与方向验证全部通过。")

    # -------------------------------------------------------------------------
    # 7.2 Early Stop 测试 (Early Stop Test)
    # -------------------------------------------------------------------------
    def test_7_2_early_stop_behavior(self):
        """
        人工构造 KL 超限情况，验证：
        1. B 组 (启用 early stopping) 确实在超限时停止剩余优化器更新。
        2. 不会继续执行多余的 optimizer.step()。
        3. Rollout Storage 能正确释放 (storage.step == 0)。
        4. 下一轮 update 可以正常执行。
        """
        num_envs = 4
        num_steps = 8
        obs_dim = 10
        act_dim = 4
        num_learning_epochs = 4
        num_mini_batches = 2
        planned_updates = num_learning_epochs * num_mini_batches  # 8

        # 辅助函数: 创建并填充 Dummy Rollout Storage
        def create_populated_ppo(enable_es, threshold_kl):
            ac = ActorCritic(obs_dim, obs_dim, act_dim, [32, 32], [32, 32], 'elu', 1.0)
            ppo = PPO(
                actor_critic=ac,
                num_learning_epochs=num_learning_epochs,
                num_mini_batches=num_mini_batches,
                clip_param=0.2,
                gamma=0.99,
                lam=0.95,
                learning_rate=1e-3,
                desired_kl=0.01,
                device=self.device,
                enable_early_stopping=enable_es,
                early_stop_kl=threshold_kl,
            )
            ppo.init_storage(num_envs, num_steps, [obs_dim], [obs_dim], [act_dim])

            # 填充 transitions
            for s in range(num_steps):
                obs = torch.randn(num_envs, obs_dim)
                crit_obs = obs
                ppo.act(obs, crit_obs)
                rewards = torch.ones(num_envs)
                dones = torch.zeros(num_envs, dtype=torch.uint8)
                ppo.process_env_step(rewards, dones, {})
            last_crit_obs = torch.randn(num_envs, obs_dim)
            ppo.compute_returns(last_crit_obs)
            return ppo

        # 场景 A: 阈值较高 (0.50)，不触发 Early Stop，应完成全部 8 次更新
        ppo_normal = create_populated_ppo(enable_es=True, threshold_kl=0.50)
        v_loss, s_loss = ppo_normal.update()
        diag_normal = ppo_normal.last_update_diagnostics
        self.assertEqual(diag_normal['updates_executed'], planned_updates,
                         f"正常更新应执行完计划的 {planned_updates} 次更新")
        self.assertEqual(diag_normal['early_stop_triggered'], 0)
        self.assertEqual(ppo_normal.storage.step, 0, "Rollout storage 必须在 update 后清空 (step==0)")

        # 场景 B: 阈值极低 (1e-6)，在第 1 次更新之后 (当网络参数改变产生微小 KL 时) 立即触发 Early Stop
        ppo_trigger = create_populated_ppo(enable_es=True, threshold_kl=1e-6)

        # 监控 optimizer.step 调用次数
        original_step = ppo_trigger.optimizer.step
        step_call_count = [0]
        def tracked_step(*args, **kwargs):
            step_call_count[0] += 1
            return original_step(*args, **kwargs)
        ppo_trigger.optimizer.step = tracked_step

        v_loss_b, s_loss_b = ppo_trigger.update()
        diag_trigger = ppo_trigger.last_update_diagnostics

        self.assertLess(diag_trigger['updates_executed'], planned_updates,
                        "触发 Early Stop 时执行更新次数必须严格小于计划次数")
        self.assertEqual(step_call_count[0], diag_trigger['updates_executed'],
                         "optimizer.step() 调用次数必须严格等于记录的更新次数")
        self.assertEqual(diag_trigger['early_stop_triggered'], 1,
                         "Early stop 标志位必须置为 1")
        self.assertIsNotNone(diag_trigger['early_stop_reason'],
                             "必须记录详细的 Early stop 原因")
        self.assertEqual(ppo_trigger.storage.step, 0,
                         "触发 Early stop 后 Rollout storage 必须正确清理 (step==0)")

        # 场景 C: 验证下一轮 update 能够正常执行
        for s in range(num_steps):
            obs = torch.randn(num_envs, obs_dim)
            crit_obs = obs
            ppo_trigger.act(obs, crit_obs)
            rewards = torch.ones(num_envs)
            dones = torch.zeros(num_envs, dtype=torch.uint8)
            ppo_trigger.process_env_step(rewards, dones, {})
        last_crit_obs = torch.randn(num_envs, obs_dim)
        ppo_trigger.compute_returns(last_crit_obs)

        # 放宽阈值至 0.50，下一轮应成功执行完毕
        ppo_trigger.early_stop_kl = 0.50
        v_next, s_next = ppo_trigger.update()
        self.assertEqual(ppo_trigger.last_update_diagnostics['updates_executed'], planned_updates,
                         "清理存储后下一轮 update 必须能正常完整执行")
        self.assertEqual(ppo_trigger.storage.step, 0)

        print(">>> [Test 7.2 Early Stop Behavior Passed]: 提前停止、步骤截断与存储清理验证全部通过。")

    # -------------------------------------------------------------------------
    # 7.3 兼容性测试 (Compatibility Test)
    # -------------------------------------------------------------------------
    def test_7_3_compatibility_with_vanilla(self):
        """
        关闭新功能时 (enable_early_stopping=False)，确认损失及梯度更新结果
        与原始 PPO 逻辑完全等价 (固定随机数种子与相同初始权重)。
        """
        obs_dim = 12
        act_dim = 4
        num_envs = 4
        num_steps = 10

        torch.manual_seed(12345)
        ac_1 = ActorCritic(obs_dim, obs_dim, act_dim, [64, 32], [64, 32], 'elu', 1.0)
        torch.manual_seed(12345)
        ac_2 = ActorCritic(obs_dim, obs_dim, act_dim, [64, 32], [64, 32], 'elu', 1.0)

        # 验证初始权重完全相同
        for p1, p2 in zip(ac_1.parameters(), ac_2.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        ppo_vanilla = PPO(
            actor_critic=ac_1,
            num_learning_epochs=3,
            num_mini_batches=2,
            clip_param=0.2,
            gamma=0.99,
            lam=0.95,
            learning_rate=2.5e-4,
            desired_kl=0.01,
            schedule='adaptive',
            enable_early_stopping=False,
        )
        ppo_test = PPO(
            actor_critic=ac_2,
            num_learning_epochs=3,
            num_mini_batches=2,
            clip_param=0.2,
            gamma=0.99,
            lam=0.95,
            learning_rate=2.5e-4,
            desired_kl=0.01,
            schedule='adaptive',
            enable_early_stopping=False,
            early_stop_kl=0.02,
        )

        ppo_vanilla.init_storage(num_envs, num_steps, [obs_dim], [obs_dim], [act_dim])
        ppo_test.init_storage(num_envs, num_steps, [obs_dim], [obs_dim], [act_dim])

        # 注入完全相同的经验数据
        torch.manual_seed(999)
        obs_data = [torch.randn(num_envs, obs_dim) for _ in range(num_steps)]
        rew_data = [torch.randn(num_envs) for _ in range(num_steps)]

        for s in range(num_steps):
            obs = obs_data[s]
            torch.manual_seed(s * 100)
            ppo_vanilla.act(obs, obs)
            ppo_vanilla.process_env_step(rew_data[s], torch.zeros(num_envs, dtype=torch.uint8), {})

            torch.manual_seed(s * 100)
            ppo_test.act(obs, obs)
            ppo_test.process_env_step(rew_data[s], torch.zeros(num_envs, dtype=torch.uint8), {})

        last_obs = torch.randn(num_envs, obs_dim)
        ppo_vanilla.compute_returns(last_obs)
        ppo_test.compute_returns(last_obs)

        # 运行更新
        torch.manual_seed(777)
        vl_1, sl_1 = ppo_vanilla.update()
        torch.manual_seed(777)
        vl_2, sl_2 = ppo_test.update()

        # 验证 Loss 浮点精确相等
        self.assertAlmostEqual(vl_1, vl_2, places=6, msg="Value loss 必须完全一致")
        self.assertAlmostEqual(sl_1, sl_2, places=6, msg="Surrogate loss 必须完全一致")

        # 验证更新后的全部模型权重浮点完全一致
        for p1, p2 in zip(ac_1.parameters(), ac_2.parameters()):
            self.assertTrue(torch.allclose(p1, p2, atol=1e-7),
                            "关闭新功能时，参数更新轨迹必须与原始 PPO 完全一致")

        print(">>> [Test 7.3 Compatibility Test Passed]: 原始 PPO 模式兼容性完全一致。")

    # -------------------------------------------------------------------------
    # 7.4 Checkpoint 测试 (Checkpoint Integrity & Warm Start Test)
    # -------------------------------------------------------------------------
    def test_7_4_checkpoint_model25_integrity(self):
        """
        验证主基线 model_25.pt 的完整性：
        1. SHA256 校验码准确匹配。
        2. 网络参数键名、维度结构正确加载。
        3. Actor 输入输出维度 (31 -> 6)，Critic 输入输出维度 (31 -> 1)。
        4. 推理前向计算数值稳定、无 NaN/Inf。
        5. 文件未被写入或破坏。
        """
        baseline_path = os.path.join(
            WORKSPACE_ROOT,
            "legged_gym/logs/myrobot/Oct10_15-48-32_turning_height_stage5_3_formal/model_25.pt"
        )
        expected_sha256 = "d67474d0e47f908a8c863e605b8d5de7ea1bd9f89f41d2c303fbef33d963536c"

        self.assertTrue(os.path.isfile(baseline_path), f"基线模型文件不存在: {baseline_path}")

        # 计算 SHA256
        hasher = hashlib.sha256()
        with open(baseline_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        calc_sha256 = hasher.hexdigest()
        self.assertEqual(calc_sha256, expected_sha256, "基线模型 SHA256 校验码不匹配！禁止篡改或覆盖基线。")

        # 加载权重
        loaded = torch.load(baseline_path, map_location='cpu')
        self.assertIn('model_state_dict', loaded, "Checkpoint 缺少 'model_state_dict'")
        self.assertIn('optimizer_state_dict', loaded, "Checkpoint 缺少 'optimizer_state_dict'")
        state_dict = loaded['model_state_dict']

        # 校验维度
        ac = ActorCritic(
            num_actor_obs=31,
            num_critic_obs=31,
            num_actions=6,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation='elu',
            init_noise_std=1.0
        )
        missing_keys, unexpected_keys = ac.load_state_dict(state_dict)
        self.assertEqual(len(missing_keys), 0, f"权重缺失键名: {missing_keys}")
        self.assertEqual(len(unexpected_keys), 0, f"权重多余键名: {unexpected_keys}")

        # 检查参数形状
        self.assertEqual(state_dict['actor.0.weight'].shape, (512, 31))
        self.assertEqual(state_dict['actor.6.weight'].shape, (6, 128))
        self.assertEqual(state_dict['critic.0.weight'].shape, (512, 31))
        self.assertEqual(state_dict['critic.6.weight'].shape, (1, 128))
        self.assertEqual(state_dict['std'].shape, (6,))

        # 检查推理输出稳定性
        ac.eval()
        dummy_obs = torch.randn(32, 31)
        with torch.no_grad():
            act_inference = ac.act_inference(dummy_obs)
            crit_eval = ac.evaluate(dummy_obs)

        self.assertEqual(act_inference.shape, (32, 6))
        self.assertEqual(crit_eval.shape, (32, 1))
        self.assertFalse(torch.isnan(act_inference).any(), "推理动作输出包含 NaN")
        self.assertFalse(torch.isinf(act_inference).any(), "推理动作输出包含 Inf")
        self.assertFalse(torch.isnan(crit_eval).any(), "价值评估输出包含 NaN")
        self.assertFalse(torch.isinf(crit_eval).any(), "价值评估输出包含 Inf")

        # 再次确认文件 hash 依然未被修改
        with open(baseline_path, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), expected_sha256,
                             "基线模型在测试期间被篡改或覆盖！")

        print(">>> [Test 7.4 Checkpoint Test Passed]: 主基线 model_25.pt 维度、推理与校验码完全正确。")


if __name__ == '__main__':
    unittest.main()
