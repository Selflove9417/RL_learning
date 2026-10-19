#!/usr/bin/env python3
# ==============================================================================
# Phase 3.2-T: Detailed 4-Timestamp & 5ms-Substep Contact Transient Audit
# Analyzes Seeds 100, 2, 17, 23 through:
#   1. Scene Creation (Fresh spawn)
#   2. After reset_idx (State A)
#   3. Substeps 1, 2, 3, 4 (5ms each) of self.step(zeros)
#   4. Before Policy Step 0 (State B / State C)
# ==============================================================================

import os
import sys
import json

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse
import torch
import numpy as np

from legged_gym.envs import task_registry
from legged_gym.utils import get_args

def audit_seed(seed):
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["audit", "--task", task_name, "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = seed

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.normalization.clip_actions = 100.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.noise.add_noise = False

    # 1. Scene Creation
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    
    print(f"\n================================================================================")
    print(f"  SEED {seed} SUBSTEP & TIMESTAMP AUDIT")
    print(f"================================================================================")

    # State right after make_env (which already called reset())
    # To see the clean State A -> State B progression, let's call reset_idx cleanly:
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    # Timestamp 2: State A
    r_pos = env.root_states[0, 0:3].cpu().numpy()
    r_vel = env.root_states[0, 7:10].cpu().numpy()
    b_vel = env.base_lin_vel[0].cpu().numpy()
    true_com_pos, true_com_vel = env._compute_true_com()
    c_vel = true_com_vel[0].cpu().numpy()
    w_L = env.dof_vel[0, env.left_wheel_idx].item()
    w_R = env.dof_vel[0, env.right_wheel_idx].item()
    r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
    roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
    pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi

    pos_L = env.rb_states[0, env.left_wheel_body, 2].item()
    pos_R = env.rb_states[0, env.right_wheel_body, 2].item()
    clr_L = pos_L - env.wheel_radius
    clr_R = pos_R - env.wheel_radius
    fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
    fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()

    print(f"[Timestamp 2: State A (After reset_idx, before 0.02s step)]:")
    print(f"  Root Pos: {r_pos}, Root Vel: {r_vel}, Base Vel: {b_vel}, CoM Vel: {c_vel}")
    print(f"  Pitch: {pitch:.2f} deg, Roll: {roll:.2f} deg, Wheel w: ({w_L:.2f}, {w_R:.2f})")
    print(f"  Wheel Bottom: L={clr_L*1000:+.2f} mm, R={clr_R*1000:+.2f} mm | Fn: ({fn_L:.1f}, {fn_R:.1f}) N")

    # Timestamp 3: 4 substeps (5ms each) of self.step(zeros)
    print(f"\n[Timestamp 3: 5ms Substep Breakdown of the 0.02s step]:")
    for substep in range(1, 5):
        # In LeggedRobot, compute torques with actions=0
        env.torques = env._compute_torques(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)

        t_ms = substep * 5
        r_p = env.root_states[0, 0:3].cpu().numpy()
        r_v = env.root_states[0, 7:10].cpu().numpy()
        r_w = env.root_states[0, 10:13].cpu().numpy()
        true_com_p, true_com_v = env._compute_true_com()
        c_v = true_com_v[0].cpu().numpy()
        
        pos_L = env.rb_states[0, env.left_wheel_body, 2].item()
        pos_R = env.rb_states[0, env.right_wheel_body, 2].item()
        clr_L = pos_L - env.wheel_radius
        clr_R = pos_R - env.wheel_radius
        fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
        fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()
        fy_L = env.contact_forces[0, env.left_wheel_body, 1].item()
        fy_R = env.contact_forces[0, env.right_wheel_body, 1].item()
        w_L = env.dof_vel[0, env.left_wheel_idx].item()
        w_R = env.dof_vel[0, env.right_wheel_idx].item()
        r_raw, p_raw, y_raw = get_euler_xyz(env.root_states[:, 3:7])
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi

        print(f"  Substep {substep} ({t_ms:2d}ms): z_root={r_p[2]:.5f}m, Vy_root={r_v[1]:+.4f}, Vz_root={r_v[2]:+.4f} | Vy_CoM={c_v[1]:+.4f}, Vz_CoM={c_v[2]:+.4f}")
        print(f"               clr: ({clr_L*1000:+.2f}, {clr_R*1000:+.2f}) mm | Fn: ({fn_L:6.1f}, {fn_R:6.1f}) N, Fy: ({fy_L:6.1f}, {fy_R:6.1f}) N | w: ({w_L:+.2f}, {w_R:+.2f}) | Pitch: {pitch:+.2f} deg, Roll: {roll:+.2f} deg")

    # Post physics step
    env.post_physics_step()
    # Timestamp 4: State B
    b_vel = env.base_lin_vel[0].cpu().numpy()
    _, true_com_v = env._compute_true_com()
    c_v = true_com_v[0].cpu().numpy()
    print(f"\n[Timestamp 4: State B (Completed 0.02s reset step, entering policy Step 0)]:")
    print(f"  Base Lin Vel: {b_vel}, CoM Lin Vel: {c_v}")
    print(f"  Left Wheel Contact Force: {env.contact_forces[0, env.left_wheel_body].cpu().numpy()}")
    print(f"  Right Wheel Contact Force: {env.contact_forces[0, env.right_wheel_body].cpu().numpy()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_seed", type=int, default=100)
    parsed, _ = parser.parse_known_args()
    audit_seed(parsed.audit_seed)
