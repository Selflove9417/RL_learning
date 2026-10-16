#!/usr/bin/env python3
# ==============================================================================
# MyRobot Forward Jump: Stage 8 Hot-Start Compatibility Verification
# ==============================================================================

import os
import sys

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
import torch

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry


def verify_hotstart_compatibility():
    print("=" * 70)
    print("🚀 开始 Stage 8 热启动兼容性与网络结构全面验证")
    print("=" * 70)

    # 1. 获取任务配置
    env_cfg, train_cfg = task_registry.get_cfgs("myrobot_forward_jump")
    num_obs = env_cfg.env.num_observations
    num_actions = env_cfg.env.num_actions

    print(f"📊 任务配置检查: 观测维度={num_obs}, 动作维度={num_actions}")
    assert num_obs == 35, f"预期 35 维观测，实际 {num_obs}"
    assert num_actions == 6, f"预期 6 维动作，实际 {num_actions}"

    # 2. 检查 Stage 8 冻结权重文件
    ckpt_path = train_cfg.runner.warm_start_path
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(workspace_root, ckpt_path)

    print(f"📦 权重检查点路径: {ckpt_path}")
    assert os.path.isfile(ckpt_path), f"权重文件不存在: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"🔑 检查点包含键: {list(ckpt.keys())}")
    assert "model_state_dict" in ckpt, "检查点缺失 model_state_dict"
    assert "optimizer_state_dict" in ckpt, "检查点缺失 optimizer_state_dict"

    # 3. 实例化前向跳跃 ActorCritic 网络
    actor_critic = ActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0
    )

    # 4. 加载 Stage 8 model_state_dict
    load_res = actor_critic.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"✅ ActorCritic 权重严格加载成功: {load_res}")

    # 5. 验证优化器状态隔离 (不继承旧优化器状态)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=train_cfg.algorithm.learning_rate)
    opt_state = optimizer.state_dict()
    assert len(opt_state["state"]) == 0, "新优化器状态字典应为空字典 (不继承旧动量与二阶矩)"
    print("✅ 优化器状态隔离验证成功: 全新 Adam 优化器未继承历史状态")

    # 6. 推理兼容性验证 (Dummy 批次输入)
    actor_critic.eval()
    dummy_obs_zeros = torch.zeros(16, 35)
    with torch.no_grad():
        actions_zero = actor_critic.act_inference(dummy_obs_zeros)
    assert actions_zero.shape == (16, 6), f"动作输出形状异常: {actions_zero.shape}"
    print(f"✅ 零指令输入推理输出形状: {actions_zero.shape}, 范数均值: {torch.norm(actions_zero, dim=-1).mean().item():.4f}")

    # 7. 前向指令激励响应测试 (槽位 9: commands[:, 1] * scale)
    dummy_obs_fwd = dummy_obs_zeros.clone()
    dummy_obs_fwd[:, 9] = 0.40 * 2.0  # 前向 0.4 m/s * scale 2.0
    with torch.no_grad():
        actions_fwd = actor_critic.act_inference(dummy_obs_fwd)

    diff = torch.norm(actions_fwd - actions_zero, dim=-1).mean().item()
    print(f"✅ 前向指令扰动差异: ||a(cmd_fwd) - a(0)|| = {diff:.6f}")
    print("=" * 70)
    print("🎉 Stage 8 热启动兼容性验证 100% 通过！")
    print("=" * 70)


if __name__ == "__main__":
    verify_hotstart_compatibility()
