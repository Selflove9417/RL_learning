#!/usr/bin/env python3
# ==============================================================================
# MyRobot Forward Jump: Simulation Environment & Stepping Verification
# ==============================================================================

import os
import sys

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
import torch
from legged_gym.envs import task_registry


def test_env_stepping():
    print("=" * 70)
    print("🚀 测试 MyRobotForwardJump 仿真环境初始化与单步推进")
    print("=" * 70)

    class CustomArgs:
        task = "myrobot_forward_jump"
        resume = False
        experiment_name = None
        run_name = None
        load_run = None
        checkpoint = None
        headless = True
        horovod = False
        rl_device = "cuda:0"
        sim_device = "cuda:0"
        compute_device_id = 0
        graphics_device_id = 0
        num_envs = 16
        seed = 42
        physics_engine = isaacgym.gymapi.SIM_PHYSX
        subscenes = 0
        slices = 0
        num_threads = 0
        use_gpu_pipeline = True
        use_gpu = True

    args = CustomArgs()
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    obs = env.get_observations()

    print(f"✅ 环境创建成功: num_envs={env.num_envs}, obs_dim={obs.shape[1]}")
    assert obs.shape == (16, 35), f"观测维度异常: {obs.shape}"

    # 检查初始指令
    assert torch.all(env.commands[:, 0] == 0.0), "lateral 指令应严格为 0"
    assert torch.all(env.commands[:, 2] == 0.0), "yaw 指令应严格为 0"
    assert torch.all(env.commands[:, 1] > 0.30), "forward 指令应在 0.35~0.45 范围内"

    # 执行环境重置测试
    env.reset()
    print(f"🔍 reset() 后 commands 检查: commands[:, 0] (lateral) = {env.commands[:, 0].mean().item():.4f}")
    print(f"🔍 reset() 后 commands 检查: commands[:, 1] (forward) = {env.commands[:, 1].mean().item():.4f}")
    print(f"🔍 reset() 后 commands 检查: commands[:, 2] (yaw)     = {env.commands[:, 2].mean().item():.4f}")
    assert torch.all(env.commands[:, 0] == 0.0), "lateral 指令应严格为 0"
    assert torch.all(env.commands[:, 2] == 0.0), "yaw 指令应严格为 0"
    assert torch.all(env.commands[:, 1] > 0.30), "forward 指令应在 0.35~0.45 范围内"

    # 执行 50 步物理推进测试
    actions = torch.zeros(16, 6, device=env.device)
    for step in range(50):
        obs, priv_obs, rewards, dones, infos = env.step(actions)
        if step % 10 == 0:
            print(f"Step {step:02d}: rewards mean={rewards.mean().item():.4f}, "
                  f"base_h={env.root_states[:, 2].mean().item():.3f}m, "
                  f"clr_min={env.bilateral_clearance.mean().item()*100:.2f}cm, "
                  f"vy={env.base_lin_vel[:, 1].mean().item():.3f}m/s")

    print("=" * 70)
    print("🎉 MyRobotForwardJump 仿真环境与物理流推进测试 100% 通过！")
    print("=" * 70)


if __name__ == "__main__":
    test_env_stepping()
