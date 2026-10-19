#!/usr/bin/env python3
import os
import sys

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse
import torch
import numpy as np

from rsl_rl.modules import ActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.envs.myrobot.myrobot_dual_env import compute_exact_cylinder_clearance

def run_mode_e_test(seed):
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["eval_mode_e", "--task", task_name, "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = seed

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.normalization.clip_actions = 100.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.noise.add_noise = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # Mode E: Ground-contact placement without artificial velocity wiping
    # Place root at z = 0.397739 so wheels are precisely on the ground plane (clr = 0)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.root_states[:, 2] = 0.397739
    env.root_states[:, 7:13] = 0.0
    env.dof_vel[:, :] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    # Physical settling step (natural contact interaction under actuators)
    # Step 1 step (0.02s) with zero actions (default PD)
    env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    # State before formal policy starts
    fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
    fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()
    pos_L = env.rb_states[0, env.left_wheel_body, 2].item() - env.wheel_radius
    pos_R = env.rb_states[0, env.right_wheel_body, 2].item() - env.wheel_radius
    base_vy = env.base_lin_vel[0, 1].item()
    base_vz = env.base_lin_vel[0, 2].item()
    _, true_com_vel = env._compute_true_com()
    com_vy = true_com_vel[0, 1].item()
    w_L = env.dof_vel[0, env.left_wheel_idx].item()
    w_R = env.dof_vel[0, env.right_wheel_idx].item()
    r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
    pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi
    roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi

    print(f"Seed {seed}: Before policy start:")
    print(f"  base_vy={base_vy:+.4f}, com_vy={com_vy:+.4f}, base_vz={base_vz:+.4f}")
    print(f"  fn=({fn_L:.1f}, {fn_R:.1f})N, clr=({pos_L*1000:+.2f}, {pos_R*1000:+.2f})mm")
    print(f"  w=({w_L:.2f}, {w_R:.2f})rad/s, pitch={pitch:.2f}deg, roll={roll:.2f}deg")

    # Start jump cycle with phi=0 and fresh clock
    env.jump_timer[:] = 0.0
    env.jump_phase[:] = 0.0
    env.episode_length_buf[:] = 0
    env.filtered_actions[:] = 0.0
    env.compute_observations()
    obs = env.obs_buf.clone()

    # Load Champion
    ac = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    model_path = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_1_forward_distance/models/model_champion.pt"
    ckpt = torch.load(model_path, map_location=env.device)
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    policy = ac.act_inference

    total_steps = int(2.8 / env.dt)
    takeoff_step = None
    touchdown_step = None
    took_off = False
    landed = False
    max_clearance_cm = 0.0
    takeoff_com_y = 0.0
    touchdown_com_y = 0.0
    touchdown_pitch = 0.0
    touchdown_roll = 0.0
    recovery_vys = []
    abnormal_resets = 0

    step5_com_vy = 0.0
    takeoff_com_vy = 0.0
    takeoff_com_vz = 0.0

    for step in range(total_steps):
        t_sim = step * env.dt
        true_com_pos, true_com_vel = env._compute_true_com()
        base_lin_vel_curr = env.base_lin_vel[0].cpu().numpy()
        com_vel_curr = true_com_vel[0].cpu().numpy()
        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        roll_curr = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item() * 180.0 / np.pi
        pitch_curr = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item() * 180.0 / np.pi

        if step == 5:
            step5_com_vy = float(com_vel_curr[1])

        clr_min, _, _ = compute_exact_cylinder_clearance(env)
        clr_min_cm = clr_min[0].item() * 100.0
        is_air = bool(env.is_airborne[0].item())
        has_took = bool(env.has_taken_off[0].item())

        if is_air and has_took:
            max_clearance_cm = max(max_clearance_cm, clr_min_cm)

        if is_air and has_took and not took_off:
            took_off = True
            takeoff_step = step
            takeoff_com_y = float(true_com_pos[0, 1].item())
            takeoff_com_vy = float(com_vel_curr[1])
            takeoff_com_vz = float(com_vel_curr[2])

        if took_off and (not is_air) and (not landed):
            landed = True
            touchdown_step = step
            touchdown_com_y = float(true_com_pos[0, 1].item())
            touchdown_pitch = pitch_curr
            touchdown_roll = roll_curr

        if landed and t_sim >= 1.0:
            recovery_vys.append(base_lin_vel_curr[1])

        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _, _ = env.step(actions)
        if env.reset_buf[0].item() != 0 and step < total_steps - 5:
            abnormal_resets += 1

    fwd_dist_cm = (touchdown_com_y - takeoff_com_y) * 100.0 if (took_off and landed) else 0.0
    post_rec_vy = float(np.mean(recovery_vys[-10:])) if len(recovery_vys) >= 10 else 0.0
    print(f"  Result: fwd_dist={fwd_dist_cm:.2f}cm, max_clr={max_clearance_cm:.2f}cm, step5_com_vy={step5_com_vy:+.4f}, to_vy={takeoff_com_vy:+.4f}, to_vz={takeoff_com_vz:+.4f}, TD_roll={touchdown_roll:.2f}deg, TD_pitch={touchdown_pitch:.2f}deg, rec_vy={post_rec_vy:+.4f}, abnormal={abnormal_resets}")

if __name__ == "__main__":
    for s in [42, 100, 200, 17, 23, 2, 7, 11]:
        run_mode_e_test(s)
