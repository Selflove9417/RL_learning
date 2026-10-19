#!/usr/bin/env python3
# ==============================================================================
# MyRobot: Exact Whole-Body CoM Kinematic Closure Verification
# ==============================================================================

import os
import sys
import json
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate, get_euler_xyz
import torch

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args


def verify_closure():
    sys.argv = ["test_closure", "--task", "myrobot_forward_jump", "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = 42

    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_forward_jump")
    env_cfg.env.num_envs = 1
    env_cfg.seed = 42
    env_cfg.noise.add_noise = False
    env_cfg.env.enable_camera_sensors = False
    env, _ = task_registry.make_env(name="myrobot_forward_jump", args=args, env_cfg=env_cfg)

    trial_model_path = os.path.join(workspace_root, "evaluation_assets", "phase1_forward_jump", "models", "model_30_yaw_aligned.pt")
    ac = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], "elu", 1.0).to(env.device)
    ckpt = torch.load(trial_model_path, map_location=env.device)
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    policy = ac.act_inference

    props = env.gym.get_actor_rigid_body_properties(env.envs[0], 0)
    masses = torch.tensor([p.mass for p in props], device=env.device, dtype=torch.float)
    total_m = masses.sum().item()
    com_offsets = torch.tensor([[p.com.x, p.com.y, p.com.z] for p in props], device=env.device, dtype=torch.float)

    env.reset()

    history = []
    for step in range(150):
        t_sim = step * env.dt
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        # rb_states: (num_envs, num_bodies, 13)
        rb_pos = env.rb_states[0, :, 0:3]
        rb_quat = env.rb_states[0, :, 3:7]
        rb_lin_vel = env.rb_states[0, :, 7:10]
        rb_ang_vel = env.rb_states[0, :, 10:13]

        # True body COM positions: p_com = p_orig + R(q) * c
        r_com_offset = quat_rotate(rb_quat, com_offsets)
        true_body_pos = rb_pos + r_com_offset

        # True body COM velocities: v_com = v_orig + w x (R(q) * c)
        v_rot = torch.cross(rb_ang_vel, r_com_offset, dim=-1)
        true_body_vel = rb_lin_vel + v_rot

        # Whole-body true COM
        true_com_pos = (true_body_pos * masses.unsqueeze(-1)).sum(dim=0) / total_m
        true_com_vel = (true_body_vel * masses.unsqueeze(-1)).sum(dim=0) / total_m

        # Approximate origin-based COM (used in previous audit)
        approx_com_pos = (rb_pos * masses.unsqueeze(-1)).sum(dim=0) / total_m
        approx_com_vel = (rb_lin_vel * masses.unsqueeze(-1)).sum(dim=0) / total_m

        fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
        fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()
        is_air = bool(env.is_airborne[0].item())

        history.append({
            "step": step,
            "time": round(t_sim, 3),
            "is_air": is_air,
            "fn_L": fn_L, "fn_R": fn_R,
            "true_com_pos": true_com_pos.cpu().numpy().tolist(),
            "true_com_vel": true_com_vel.cpu().numpy().tolist(),
            "approx_com_pos": approx_com_pos.cpu().numpy().tolist(),
            "approx_com_vel": approx_com_vel.cpu().numpy().tolist(),
            "root_pos": env.root_states[0, 0:3].cpu().numpy().tolist(),
            "root_vel": env.root_states[0, 7:10].cpu().numpy().tolist(),
        })

        obs = env.get_observations()
        with torch.no_grad():
            actions = policy(obs)
        env.step(actions)

    print("=" * 80)
    print("🔬 运动学闭合深度验证：Jump 2 (Step 110 到 130)")
    print("=" * 80)

    # Frame-by-frame inspection during Jump 2 flight
    dt = 0.02
    print(f"{'Step':<5} {'Time':<6} {'Fn_L':<7} {'Fn_R':<7} {'True_Y':<9} {'dY(mm)':<9} {'True_Vy':<9} {'Vy*dt(mm)':<10} {'Residual(mm)':<12}")
    for s in range(110, 131):
        h = history[s]
        true_y = h['true_com_pos'][1]
        true_vy = h['true_com_vel'][1]
        fn_L = h['fn_L']
        fn_R = h['fn_R']
        if s > 110:
            dy = (true_y - history[s-1]['true_com_pos'][1]) * 1000.0
            # previous step velocity times dt
            vy_prev = history[s-1]['true_com_vel'][1]
            pred_dy = vy_prev * dt * 1000.0
            resid = dy - pred_dy
            print(f"{s:<5} {h['time']:<6.3f} {fn_L:<7.1f} {fn_R:<7.1f} {true_y:<9.4f} {dy:<+9.3f} {true_vy:<9.4f} {pred_dy:<10.3f} {resid:<+12.3f}")
        else:
            print(f"{s:<5} {h['time']:<6.3f} {fn_L:<7.1f} {fn_R:<7.1f} {true_y:<9.4f} {'-':<9} {true_vy:<9.4f} {'-':<10} {'-':<12}")

    # Summary integration
    h110 = history[110]
    h130 = history[130]

    # True COM integration
    true_delta_y = h130['true_com_pos'][1] - h110['true_com_pos'][1]
    true_int_vy_left = sum(history[k]['true_com_vel'][1] * dt for k in range(110, 130))
    true_int_vy_right = sum(history[k]['true_com_vel'][1] * dt for k in range(111, 131))
    true_int_vy_trap = 0.5 * (true_int_vy_left + true_int_vy_right)

    # Approx Origin integration
    approx_delta_y = h130['approx_com_pos'][1] - h110['approx_com_pos'][1]
    approx_int_vy = sum(history[k]['approx_com_vel'][1] * dt for k in range(110, 130))

    # Root integration
    root_delta_y = h130['root_pos'][1] - h110['root_pos'][1]
    root_int_vy = sum(history[k]['root_vel'][1] * dt for k in range(110, 130))

    print("\n" + "=" * 80)
    print("📊 统计对比结果汇总:")
    print("=" * 80)
    print(f"1. 真正整机质心 (True CoM):")
    print(f"   * 位置变化量 (Pos Delta) : {true_delta_y * 100:.3f} cm ({true_delta_y:.5f} m)")
    print(f"   * 速度积分值 (Left Sum)  : {true_int_vy_left * 100:.3f} cm ({true_int_vy_left:.5f} m)")
    print(f"   * 速度积分值 (Right Sum) : {true_int_vy_right * 100:.3f} cm ({true_int_vy_right:.5f} m)")
    print(f"   * 速度积分值 (Trap Sum)  : {true_int_vy_trap * 100:.3f} cm ({true_int_vy_trap:.5f} m)")
    print(f"   * 运动学闭合残差 (Residual): {abs(true_delta_y - true_int_vy_trap) * 1000.0:.3f} mm")

    print(f"\n2. 纯连接框架原点加权 (Approx Origin Weighted, 之前审计值):")
    print(f"   * 位置变化量 (Pos Delta) : {approx_delta_y * 100:.3f} cm")
    print(f"   * 速度积分值 (Vel Int)   : {approx_int_vy * 100:.3f} cm")
    print(f"   * 偏差原因               : 忽略了刚体局部质心偏置向量在旋转时的运动学投影 R(q)*c 及转动线速度 w x (R(q)*c)")

    print(f"\n3. 机身基座原点 (Base Link Root):")
    print(f"   * 位置变化量 (Pos Delta) : {root_delta_y * 100:.3f} cm")
    print(f"   * 速度积分值 (Vel Int)   : {root_int_vy * 100:.3f} cm")

    # Check Jump 1 as well
    h10 = history[10]
    h32 = history[32]
    j1_true_delta_y = h32['true_com_pos'][1] - h10['true_com_pos'][1]
    j1_true_int_vy = sum(history[k]['true_com_vel'][1] * dt for k in range(10, 32))
    j1_root_delta_y = h32['root_pos'][1] - h10['root_pos'][1]
    j1_root_int_vy = sum(history[k]['root_vel'][1] * dt for k in range(10, 32))

    print("\n" + "=" * 80)
    print("🔬 Jump 1 (静态起跳首跳 Step 10 到 32) 闭合验证:")
    print("=" * 80)
    print(f"   * 真正质心位移 : {j1_true_delta_y * 100:.3f} cm")
    print(f"   * 真正质心积分 : {j1_true_int_vy * 100:.3f} cm")
    print(f"   * 闭合残差     : {abs(j1_true_delta_y - j1_true_int_vy) * 1000.0:.3f} mm")
    print(f"   * 根节点位移   : {j1_root_delta_y * 100:.3f} cm")


if __name__ == "__main__":
    verify_closure()
