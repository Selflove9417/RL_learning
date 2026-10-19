#!/usr/bin/env python3
"""
Unit Test & Telemetry Inspector for MyRobot Running Jump Task
Verifies:
1. Cruising Warmup: Phase 15 accelerates the robot to v_y ~ 0.20 m/s before jump episode begins.
2. ZeroCmd Removal: Slot 9 retains command v_y^cmd = 0.20 m/s without zero-masking.
3. In-Motion Takeoff: Leg push-off happens while rolling forward.
4. Momentum & Clearance: Tracks max clearance and forward velocity retention.
"""

import os
import sys

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
import torch

from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from rsl_rl.modules import ActorCritic


def test_running_jump():
    sys.argv = ['test_running_jump', '--task', 'myrobot_running_jump', '--headless', '--num_envs', '4']
    args = get_args()

    env_cfg, train_cfg = task_registry.get_cfgs(name='myrobot_running_jump')
    env_cfg.env.num_envs = 4
    env_cfg.seed = 42

    print("\n" + "=" * 70)
    print(" 🚀 Initializing MyRobot Running Jump Environment (4 envs, headless)")
    print("=" * 70)

    env, _ = task_registry.make_env(name='myrobot_running_jump', args=args, env_cfg=env_cfg)

    print(f"✅ Environment created successfully!")
    print(f"   Num envs        : {env.num_envs}")
    print(f"   Num observations: {env.num_obs}")
    print(f"   Num actions     : {env.num_actions}")
    print(f"   Warmup steps    : {env.warmup_steps}")
    print(f"   Target fwd vel  : {env.target_forward_vel:.2f} m/s")

    # Load initial jump policy (hotstart from frozen stage 8 or pilot c)
    model_path = "/home/xy/IsaacTu_jump/evaluation_assets/frozen_stage8/model_40.pt"
    ac_jump = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ckpt = torch.load(model_path, map_location=env.device)
    ac_jump.load_state_dict(ckpt['model_state_dict'])
    ac_jump.eval()

    obs, _ = env.reset()

    # Verify Cruising State at Step 0
    vy_init = env.base_lin_vel[:, 1].cpu().numpy()
    wheel_w = env.dof_vel[:, env.wheel_indices].cpu().numpy()
    fn_L = env.contact_forces[:, env.left_wheel_body, 2].cpu().numpy()
    fn_R = env.contact_forces[:, env.right_wheel_body, 2].cpu().numpy()

    print("\n--- [Step 0 Physical State Post-Warmup] ---")
    for i in range(env.num_envs):
        print(f"   Env {i}: vy = {vy_init[i]:+.3f} m/s | Wheel omega = {wheel_w[i, 0]:+.2f}, {wheel_w[i, 1]:+.2f} rad/s | Fn = {fn_L[i]:.1f}N, {fn_R[i]:.1f}N")

    assert (env.base_lin_vel[:, 1] > 0.12).all(), f"Error: Robot not cruising! vy = {vy_init}"
    print("✅ Cruising state verification passed! Robot is rolling forward before jump.")

    # Check Slot 9 observation: should be v_y^cmd * 2.0 = 0.40
    slot9 = obs[:, 9].cpu().numpy()
    print(f"   Obs Slot 9 (Forward Cmd): {slot9} (Expected ~0.40, NO ZeroCmd)")
    assert (abs(slot9 - 0.40) < 0.01).all(), f"Error: Slot 9 not matching command! Slot 9 = {slot9}"
    print("✅ Observation Slot 9 verification passed! ZeroCmd is completely removed.")

    # Step through episode
    print("\n--- Stepping Episode (70 steps = 1.4s) ---")
    max_clr = torch.zeros(env.num_envs, device=env.device)
    vy_traj = []

    for step in range(70):
        with torch.no_grad():
            actions = ac_jump.act_inference(obs)

        obs, _, rews, dones, infos = env.step(actions)

        # Slot 9 check after potential touchdown
        slot9_step = obs[:, 9].cpu().numpy()
        assert (abs(slot9_step - 0.40) < 0.01).all(), f"Error: Slot 9 zeroed post-touchdown! Step {step}: {slot9_step}"

        max_clr = torch.maximum(max_clr, env.bilateral_clearance * 100.0)
        vy_cur = env.base_lin_vel[:, 1].cpu().numpy()
        vy_traj.append(vy_cur)

        if step % 15 == 0:
            print(f"   Step {step:2d} (t={step*env.dt:.2f}s) | Phase: {env.jump_phase[0].item():.2f} | vy: {vy_cur[0]:+.3f} m/s | Clr: {env.bilateral_clearance[0].item()*100:5.1f} cm (Max: {max_clr[0].item():5.1f} cm) | Rew: {rews[0].item():+.2f}")

    print("\n" + "=" * 70)
    print(" 📊 Pipeline Test Summary:")
    print(f"   Max clearance across envs: {[f'{c:.1f}cm' for c in max_clr.cpu().numpy()]}")
    print(f"   Takeoff vy: {env.takeoff_vy.cpu().numpy()}")
    print(f"   Touchdown vy: {env.touchdown_vy.cpu().numpy()}")
    print(f"   Flight distance: {[f'{d:.1f}cm' for d in env.flight_distance_y.cpu().numpy()]}")
    print(f"   Velocity retention: {[f'{r:.1%}' for r in env.velocity_retention_ratio.cpu().numpy()]}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_running_jump()
