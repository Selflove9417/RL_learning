"""
Phase 2.5 Per-Jump Evaluation and Physical Reliability Verification Script
--------------------------------------------------------------------------
Tracks:
1. Per-jump (cycle 1, 2, 3) clearance and >= 20cm success rate:
   - Simplified formula: min(pos_L_z - R, pos_R_z - R)
   - Exact 3D cylinder collision geometry support function: min(z_min_L, z_min_R)
2. Touchdown horizontal displacement (in-flight drift)
3. Post-touchdown ground rolling displacement
4. Touchdown settling time (time to v_xy < 0.10 m/s)
5. 6.0s session net horizontal displacement vs cumulative path distance
6. Actuator saturation (peak hip torque, knee torque, knee angular velocity)
7. Non-wheel ground contacts and premature resets
"""

import os
import sys
import json
import isaacgym
from isaacgym import gymtorch
from isaacgym.torch_utils import get_euler_xyz
import torch
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args

def wrap_to_pi(angles):
    angles = angles % (2.0 * np.pi)
    return np.where(angles > np.pi, angles - 2.0 * np.pi, angles)

def compute_exact_cylinder_clearance(env, R=0.070, L=0.0885):
    """
    Computes exact minimum vertical distance from 3D cylinder wheel collision geometry to ground.
    Collision geometries:
      link_004 (left wheel): offset [0.03825, 0, 0], cylinder axis along local X
      link_007 (right wheel): offset [0.01925, 0, 0], cylinder axis along local X
    """
    # Left wheel
    q_L = env.rb_states[:, env.left_wheel_body, 3:7]  # x, y, z, w
    p_L = env.rb_states[:, env.left_wheel_body, 0:3]
    uz_L = 2.0 * (q_L[:, 0]*q_L[:, 2] - q_L[:, 3]*q_L[:, 1])
    cz_L = p_L[:, 2] + 0.03825 * uz_L
    zmin_L = cz_L - (R * torch.sqrt(torch.clamp(1.0 - uz_L**2, min=0.0)) + 0.5 * L * torch.abs(uz_L))
    
    # Right wheel
    q_R = env.rb_states[:, env.right_wheel_body, 3:7]
    p_R = env.rb_states[:, env.right_wheel_body, 0:3]
    uz_R = 2.0 * (q_R[:, 0]*q_R[:, 2] - q_R[:, 3]*q_R[:, 1])
    cz_R = p_R[:, 2] + 0.01925 * uz_R
    zmin_R = cz_R - (R * torch.sqrt(torch.clamp(1.0 - uz_R**2, min=0.0)) + 0.5 * L * torch.abs(uz_R))
    
    exact_clr_L = torch.clamp(zmin_L, min=0.0)
    exact_clr_R = torch.clamp(zmin_R, min=0.0)
    exact_bilateral = torch.minimum(exact_clr_L, exact_clr_R)
    return exact_bilateral, exact_clr_L, exact_clr_R

def run_evaluation(model_path, seed=1, num_envs=16, zero_vel_reset=True):
    sys.argv = ['eval', '--headless', '--task', 'myrobot_jump']
    args = get_args()
    args.seed = seed
    env_cfg, train_cfg = task_registry.get_cfgs(name='myrobot_jump')
    env_cfg.env.num_envs = num_envs
    env_cfg.seed = seed
    env, _ = task_registry.make_env(name='myrobot_jump', args=args, env_cfg=env_cfg)

    ckpt = torch.load(model_path, map_location=env.device)
    from rsl_rl.modules import ActorCritic
    ac = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(ckpt['model_state_dict'])
    ac.eval()

    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]

    if zero_vel_reset:
        env.root_states[:, 7:13] = 0.0
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
        obs = env.get_observations()

    jump_clrs_simple = [[] for _ in range(3)]
    jump_clrs_exact = [[] for _ in range(3)]
    jump_flight_times = [[] for _ in range(3)]
    jump_disp_td = [[] for _ in range(3)]
    jump_post_td_disp = [[] for _ in range(3)]
    jump_settle_times = [[] for _ in range(3)]
    jump_post_td_speeds = [[] for _ in range(3)]
    jump_td_vx = [[] for _ in range(3)]
    jump_td_vy = [[] for _ in range(3)]
    jump_td_pitch = [[] for _ in range(3)]
    recovery_vels = [[] for _ in range(3)]
    recovery_pitches = [[] for _ in range(3)]
    max_hip_t = torch.zeros(num_envs, device=env.device)
    max_knee_t = torch.zeros(num_envs, device=env.device)
    max_knee_v = torch.zeros(num_envs, device=env.device)
    non_wheel_contacts = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    start_pos = env.root_states[:, 0:2].clone()
    last_step_pos = start_pos.clone()
    cumul_path_dist = torch.zeros(num_envs, device=env.device)

    for cycle in range(3):
        max_c_simp = torch.zeros(num_envs, device=env.device)
        max_c_exac = torch.zeros(num_envs, device=env.device)
        flight_t = torch.zeros(num_envs, device=env.device)
        cycle_start_pos = env.root_states[:, 0:2].clone()
        td_recorded = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        td_pos = torch.zeros(num_envs, 2, device=env.device)
        td_step = torch.full((num_envs,), -1, dtype=torch.long, device=env.device)
        settled_recorded = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        settle_t = torch.full((num_envs,), -1.0, device=env.device)
        disp_td = torch.zeros(num_envs, 2, device=env.device)
        td_v = torch.zeros(num_envs, 2, device=env.device)
        td_pitch_buf = torch.zeros(num_envs, device=env.device)
        prev_air = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

        cycle_rec_vels = []
        cycle_rec_pitches = []
        cycle_post_td_vels = []

        for step in range(100):
            with torch.no_grad(): act = ac.act_inference(obs)
            obs, _, _, dones, _ = env.step(act)

            curr_pos = env.root_states[:, 0:2]
            cumul_path_dist += torch.norm(curr_pos - last_step_pos, dim=-1)
            last_step_pos = curr_pos.clone()

            clr_simp = env.bilateral_clearance
            clr_exac, _, _ = compute_exact_cylinder_clearance(env)
            max_c_simp = torch.maximum(max_c_simp, clr_simp)
            max_c_exac = torch.maximum(max_c_exac, clr_exac)
            air = env.is_airborne
            flight_t += air.float() * env.dt

            roll, pitch, yaw = get_euler_xyz(env.base_quat)

            # Touchdown detection
            just_td = (~air) & prev_air & (env.jump_phase > 0.22)
            for i in range(num_envs):
                if just_td[i].item() and not td_recorded[i].item():
                    td_recorded[i] = True
                    td_step[i] = step
                    td_pos[i] = curr_pos[i]
                    disp_td[i] = curr_pos[i] - cycle_start_pos[i]
                    td_v[i] = env.base_lin_vel[i, 0:2]
                    td_pitch_buf[i] = pitch[i]

                # Settling detection: after touchdown, when horizontal speed drops below 0.10 m/s
                if td_recorded[i].item() and not settled_recorded[i].item():
                    v_horiz = torch.norm(env.base_lin_vel[i, 0:2]).item()
                    if v_horiz < 0.10 and not air[i].item():
                        settled_recorded[i] = True
                        settle_t[i] = (step - td_step[i].item()) * env.dt

            prev_air[:] = air

            # Post-touchdown stance tracking
            grounded_post_td = td_recorded & (~air)
            if torch.any(grounded_post_td):
                g_idx = torch.where(grounded_post_td)[0]
                cycle_post_td_vels.append(env.base_lin_vel[g_idx, 0:2].cpu().numpy())

            # Recovery phase tracking (phase >= 0.35 and grounded)
            in_recovery = (env.jump_phase >= 0.35) & (~air) & td_recorded
            if torch.any(in_recovery):
                rec_idx = torch.where(in_recovery)[0]
                cycle_rec_vels.append(env.base_lin_vel[rec_idx, 0:2].cpu().numpy())
                cycle_rec_pitches.append(pitch[rec_idx].cpu().numpy())

            th = torch.maximum(torch.abs(env.torques[:, env.left_hip_idx]), torch.abs(env.torques[:, env.right_hip_idx]))
            tk = torch.maximum(torch.abs(env.torques[:, env.left_knee_idx]), torch.abs(env.torques[:, env.right_knee_idx]))
            vk = torch.maximum(torch.abs(env.dof_vel[:, env.left_knee_idx]), torch.abs(env.dof_vel[:, env.right_knee_idx]))
            max_hip_t = torch.maximum(max_hip_t, th)
            max_knee_t = torch.maximum(max_knee_t, tk)
            max_knee_v = torch.maximum(max_knee_v, vk)

            if len(env.termination_contact_indices) > 0:
                link_f = torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1)
                non_wheel_contacts |= torch.any(link_f > 15.0, dim=1)

        # Displacement on ground from touchdown to end of cycle
        post_td_disp = torch.norm(env.root_states[:, 0:2] - td_pos, dim=-1)

        jump_clrs_simple[cycle] = max_c_simp.cpu().numpy()
        jump_clrs_exact[cycle] = max_c_exac.cpu().numpy()
        jump_flight_times[cycle] = flight_t.cpu().numpy()
        jump_disp_td[cycle] = torch.norm(disp_td, dim=-1).cpu().numpy()
        jump_post_td_disp[cycle] = post_td_disp.cpu().numpy()

        settle_t_np = settle_t.cpu().numpy()
        rem_cycle_t = (100 - td_step.cpu().numpy()) * 0.02
        settle_t_filled = np.where(settle_t_np >= 0, settle_t_np, rem_cycle_t)
        jump_settle_times[cycle] = settle_t_filled

        jump_td_vx[cycle] = td_v[:, 0].cpu().numpy()
        jump_td_vy[cycle] = td_v[:, 1].cpu().numpy()
        jump_td_pitch[cycle] = wrap_to_pi(td_pitch_buf.cpu().numpy())

        if len(cycle_post_td_vels) > 0:
            jump_post_td_speeds[cycle] = np.linalg.norm(np.concatenate(cycle_post_td_vels, axis=0), axis=1)
        else:
            jump_post_td_speeds[cycle] = np.zeros(1)

        if len(cycle_rec_vels) > 0:
            recovery_vels[cycle] = np.concatenate(cycle_rec_vels, axis=0)
            recovery_pitches[cycle] = wrap_to_pi(np.concatenate(cycle_rec_pitches, axis=0))
        else:
            recovery_vels[cycle] = np.zeros((0, 2))
            recovery_pitches[cycle] = np.zeros(0)

    net_disp_6s = torch.norm(env.root_states[:, 0:2] - start_pos, dim=-1).cpu().numpy()
    path_dist_6s = cumul_path_dist.cpu().numpy()
    
    all_jump_clrs_simp = np.concatenate(jump_clrs_simple) * 100.0
    all_jump_clrs_exac = np.concatenate(jump_clrs_exact) * 100.0
    session_max_clrs_exac = np.maximum.reduce(jump_clrs_exact) * 100.0

    all_rec_v = np.concatenate(recovery_vels, axis=0) if sum(len(v) for v in recovery_vels) > 0 else np.zeros((1, 2))
    all_rec_p = np.concatenate(recovery_pitches, axis=0) if sum(len(p) for p in recovery_pitches) > 0 else np.zeros(1)

    cycles_data = []
    for c in range(3):
        c_clrs_s = jump_clrs_simple[c] * 100.0
        c_clrs_e = jump_clrs_exact[c] * 100.0
        s_pass = int(np.sum(c_clrs_s >= 20.0))
        e_pass = int(np.sum(c_clrs_e >= 20.0))
        total_in_cycle = int(len(c_clrs_e))
        c_rec_v = recovery_vels[c]
        c_rec_p = recovery_pitches[c]
        cycles_data.append({
            'cycle': c + 1,
            'reach_20cm_count_exact': e_pass,
            'reach_20cm_count_simple': s_pass,
            'total_jumps': total_in_cycle,
            'reach_20cm_rate_exact_pct': float(e_pass / total_in_cycle * 100.0),
            'reach_20cm_rate_simple_pct': float(s_pass / total_in_cycle * 100.0),
            'mean_clr_exact_cm': float(np.mean(c_clrs_e)),
            'min_clr_exact_cm': float(np.min(c_clrs_e)),
            'max_clr_exact_cm': float(np.max(c_clrs_e)),
            'mean_clr_simple_cm': float(np.mean(c_clrs_s)),
            'mean_flight_time_s': float(np.mean(jump_flight_times[c])),
            'mean_td_disp_m': float(np.mean(jump_disp_td[c])),
            'mean_post_td_roll_disp_m': float(np.mean(jump_post_td_disp[c])),
            'mean_settle_time_s': float(np.mean(jump_settle_times[c])),
            'mean_td_vx_mps': float(np.mean(jump_td_vx[c])),
            'mean_td_vy_mps': float(np.mean(jump_td_vy[c])),
            'mean_td_pitch_deg': float(np.mean(jump_td_pitch[c]) * 180.0 / np.pi),
            'post_td_speed_mean_mps': float(np.mean(jump_post_td_speeds[c])),
            'recovery_speed_mean_mps': float(np.mean(np.linalg.norm(c_rec_v, axis=1))) if len(c_rec_v) > 0 else 0.0,
            'recovery_speed_std_mps': float(np.std(np.linalg.norm(c_rec_v, axis=1))) if len(c_rec_v) > 0 else 0.0,
            'recovery_pitch_std_deg': float(np.std(c_rec_p) * 180.0 / np.pi) if len(c_rec_p) > 0 else 0.0,
        })

    total_success_exact = int(np.sum(all_jump_clrs_exac >= 20.0))
    total_success_simple = int(np.sum(all_jump_clrs_simp >= 20.0))
    total_jump_count = int(len(all_jump_clrs_exac))
    session_success_count = int(np.sum(session_max_clrs_exac >= 20.0))
    total_env_count = int(len(session_max_clrs_exac))

    td_disp_all = np.concatenate(jump_disp_td)
    mean_td_disp = float(np.mean(td_disp_all))
    mean_net_disp_6s = float(np.mean(net_disp_6s))
    mean_path_dist_6s = float(np.mean(path_dist_6s))
    mean_post_td_roll = float(np.mean(np.concatenate(jump_post_td_disp)))
    mean_settle_t = float(np.mean(np.concatenate(jump_settle_times)))

    clr_diff_cm = all_jump_clrs_simp - all_jump_clrs_exac

    return {
        'model_path': model_path,
        'seed': seed,
        'num_envs': num_envs,
        'zero_vel_reset': zero_vel_reset,
        'total_reach_20cm_count_exact': total_success_exact,
        'total_reach_20cm_count_simple': total_success_simple,
        'total_jumps': total_jump_count,
        'per_jump_20cm_rate_exact_pct': float(total_success_exact / total_jump_count * 100.0),
        'per_jump_20cm_rate_simple_pct': float(total_success_simple / total_jump_count * 100.0),
        'session_reach_20cm_count': session_success_count,
        'total_envs': total_env_count,
        'per_env_session_20cm_rate_pct': float(session_success_count / total_env_count * 100.0),
        'per_jump_clr_exact_mean_cm': float(np.mean(all_jump_clrs_exac)),
        'per_jump_clr_exact_min_cm': float(np.min(all_jump_clrs_exac)),
        'per_jump_clr_exact_max_cm': float(np.max(all_jump_clrs_exac)),
        'per_jump_clr_simple_mean_cm': float(np.mean(all_jump_clrs_simp)),
        'mean_formula_diff_cm': float(np.mean(clr_diff_cm)),
        'session_max_clr_mean_cm': float(np.mean(session_max_clrs_exac)),
        'touchdown_disp_mean_m': mean_td_disp,
        'post_td_roll_disp_mean_m': mean_post_td_roll,
        'time_touchdown_to_settled_s': mean_settle_t,
        'net_displacement_6s_mean_m': mean_net_disp_6s,
        'cumul_path_dist_6s_mean_m': mean_path_dist_6s,
        'recovery_speed_mean_mps': float(np.mean(np.linalg.norm(all_rec_v, axis=1))),
        'recovery_speed_var': float(np.var(all_rec_v)),
        'recovery_pitch_std_deg': float(np.std(all_rec_p) * 180.0 / np.pi),
        'max_hip_torque_Nm': float(torch.max(max_hip_t).item()),
        'max_knee_torque_Nm': float(torch.max(max_knee_t).item()),
        'max_knee_vel_rads': float(torch.max(max_knee_v).item()),
        'crashed_envs': int(torch.sum(non_wheel_contacts).item()),
        'cycles': cycles_data
    }

if __name__ == '__main__':
    model_p = sys.argv[1]
    out_p = sys.argv[2]
    eval_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    zero_vel = (sys.argv[4].lower() == 'true') if len(sys.argv) > 4 else True
    res = run_evaluation(model_p, seed=eval_seed, num_envs=16, zero_vel_reset=zero_vel)
    with open(out_p, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(f"Saved to {out_p}")
    print(f"  Seed {eval_seed} (zero_vel={zero_vel}):")
    print(f"    Per-Jump 20cm (Exact 3D Cylinder): {res['total_reach_20cm_count_exact']}/{res['total_jumps']} ({res['per_jump_20cm_rate_exact_pct']:.2f}%), mean clr: {res['per_jump_clr_exact_mean_cm']:.2f}cm")
    print(f"    Per-Jump 20cm (Simple pos_z-R):     {res['total_reach_20cm_count_simple']}/{res['total_jumps']} ({res['per_jump_20cm_rate_simple_pct']:.2f}%), mean clr: {res['per_jump_clr_simple_mean_cm']:.2f}cm")
    print(f"    Formula discrepancy (Simple - Exact): {res['mean_formula_diff_cm']:.3f}cm ({res['mean_formula_diff_cm']*10:.2f}mm)")
    for c in res['cycles']:
        print(f"      Cycle {c['cycle']}: Exact {c['reach_20cm_count_exact']}/{c['total_jumps']} ({c['reach_20cm_rate_exact_pct']:.1f}%), clr: {c['mean_clr_exact_cm']:.2f}cm, roll: {c['mean_post_td_roll_disp_m']:.3f}m, settle: {c['mean_settle_time_s']:.3f}s")
    print(f"    TD disp: {res['touchdown_disp_mean_m']:.3f}m | Post-TD roll: {res['post_td_roll_disp_mean_m']:.3f}m | Net 6s: {res['net_displacement_6s_mean_m']:.3f}m | Path 6s: {res['cumul_path_dist_6s_mean_m']:.3f}m")
    print(f"    Actuators: Hip T={res['max_hip_torque_Nm']:.1f}Nm, Knee T={res['max_knee_torque_Nm']:.1f}Nm, Knee V={res['max_knee_vel_rads']:.2f}rad/s, Crashes: {res['crashed_envs']}")
