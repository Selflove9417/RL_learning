import os
import sys
import numpy as np

# isaacgym must be imported before torch
import isaacgym
from isaacgym import gymapi
import torch
import json
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from rsl_rl.modules import ActorCritic

def run_termination_diagnostics():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    model_path = "evaluation_assets/champion_v6/models/champion_v6_m115.pt"
    if not os.path.exists(model_path):
        model_path = "evaluation_assets/phase14/models/phase14_exp1_m115.pt"

    output_path = "evaluation_assets/phase14/results/termination_diagnosis_4cases.json"

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    print(f"\n================ 4-Scenario Termination Diagnosis ================")
    print(f"Model Path: {model_path}")
    print(f"Num Envs: {args.num_envs}")

    loaded = torch.load(model_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    pol = ac.act_inference

    # 4 focus scenarios
    focus_cases = [
        ("g3_cruise_squat",       0.15,  0.00, 0.36),
        ("g3_nominal_cruise",     0.20,  0.00, 0.39774),
        ("g4_exp_fwd_left_squat", 0.16,  0.30, 0.36),
        ("g4_exp_fwd_left_rise",  0.16,  0.30, 0.44),
    ]

    body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    print(f"Rigid bodies in robot: {body_names}")
    term_contact_indices = env.termination_contact_indices.cpu().numpy()
    term_body_names = [body_names[idx] for idx in term_contact_indices]
    print(f"Termination contact bodies: {term_body_names}")

    results = {}

    protocols = ["protocol_A_original", "protocol_B_stabilized_settling"]

    for proto in protocols:
        results[proto] = {}
        print(f"\n----------------- Testing Protocol: {proto} -----------------")

        for case_name, cmd_vy, cmd_wz, target_h in focus_cases:
            print(f"\n>>> Running Scenario: {case_name} (vy={cmd_vy}, wz={cmd_wz}, H={target_h})")
            torch.manual_seed(42)
            np.random.seed(42)
            env.reset()

            total_steps = 300
            step_cmd_start = 100

            termination_records = []
            env_done_seen = set()

            h_ref = target_h
            delta_h = -0.005 * float(np.clip((0.39774 - h_ref) / (0.39774 - 0.360), 0.0, 1.0))
            h_ctrl_final = h_ref + delta_h

            current_vy_cmd = 0.0
            current_wz_cmd = 0.0
            current_h_cmd = 0.39774

            dt = env.dt # 0.02s
            max_accel_y = 0.8
            max_accel_w = 2.0
            max_speed_h = 0.04

            current_step = -1
            orig_reset_idx = env.reset_idx

            def hooked_reset_idx(env_ids):
                if current_step >= 0 and len(env_ids) > 0:
                    if isinstance(env_ids, torch.Tensor):
                        eids = env_ids.tolist()
                    else:
                        eids = list(env_ids)

                    for env_idx in eids:
                        if env_idx not in env_done_seen:
                            env_done_seen.add(env_idx)

                            # 1. 详细分解终止原因 (在真正 reset 之前读取真实物理量)
                            c_forces = torch.norm(env.contact_forces[env_idx, env.termination_contact_indices, :], dim=-1).cpu().numpy()
                            contact_triggered = np.any(c_forces > 1.0)
                            violating_bodies = []
                            for idx_c, f_val in enumerate(c_forces):
                                if f_val > 1.0:
                                    violating_bodies.append({
                                        "body_name": term_body_names[idx_c],
                                        "force_N": float(f_val)
                                    })

                            proj_g_z = env.projected_gravity[env_idx, 2].item()
                            tilt_triggered = (proj_g_z > -0.70)

                            base_z = env.root_states[env_idx, 2].item()
                            height_triggered = (base_z < 0.25)
                            time_out_triggered = bool(env.episode_length_buf[env_idx].item() > env.max_episode_length)

                            roll = np.arctan2(env.projected_gravity[env_idx, 1].item(), -env.projected_gravity[env_idx, 2].item()) * 180.0 / np.pi
                            pitch = np.arctan2(-env.projected_gravity[env_idx, 0].item(), -env.projected_gravity[env_idx, 2].item()) * 180.0 / np.pi

                            vy = env.base_lin_vel[env_idx, 1].item()
                            wz = env.base_ang_vel[env_idx, 2].item()

                            record = {
                                "env_id": int(env_idx),
                                "step": int(current_step),
                                "time_s": float(current_step * dt),
                                "stage": "initialization" if current_step < 10 else ("pre_cmd" if current_step < step_cmd_start else "active_cmd"),
                                "contact_reset": bool(contact_triggered),
                                "violating_bodies": violating_bodies,
                                "tilt_reset": bool(tilt_triggered),
                                "proj_g_z": float(proj_g_z),
                                "pitch_deg": float(pitch),
                                "roll_deg": float(roll),
                                "height_reset": bool(height_triggered),
                                "time_out_reset": bool(time_out_triggered),
                                "base_z_m": float(base_z),
                                "base_vy_mps": float(vy),
                                "base_wz_radps": float(wz),
                                "cmd_vy_mps": float(cmd_vy),
                                "cmd_wz_radps": float(cmd_wz),
                                "cmd_h_m": float(curr_h_ctrl if proto == "protocol_A_original" else current_h_cmd),
                                "knee_torque_nm": [float(env.torques[env_idx, env.left_knee_idx].item()),
                                                   float(env.torques[env_idx, env.right_knee_idx].item())],
                                "wheel_torque_nm": [float(env.torques[env_idx, env.left_wheel_idx].item()),
                                                    float(env.torques[env_idx, env.right_wheel_idx].item())]
                            }
                            termination_records.append(record)
                            print(f"  [PRE-RESET TERMINATION] Env {env_idx} at step {current_step} ({current_step * dt:.2f}s): "
                                  f"Contact={contact_triggered} (bodies={violating_bodies}), "
                                  f"Tilt={tilt_triggered} (proj_g_z={proj_g_z:.3f}, pitch={pitch:.1f}°, roll={roll:.1f}°), "
                                  f"HeightLow={height_triggered} (z={base_z:.3f}m), "
                                  f"TimeOut={time_out_triggered}")

                return orig_reset_idx(env_ids)

            env.reset_idx = hooked_reset_idx

            for step in range(total_steps):
                current_step = step
                time_s = step * dt

                if proto == "protocol_A_original":
                    # 原始 21 工况协议: 速度在 step 0 即瞬态下发，高度在 step 100 阶跃下发
                    curr_h_ctrl = 0.39774 if step < step_cmd_start else h_ctrl_final
                    env.commands[:, 0] = 0.0
                    env.commands[:, 1] = cmd_vy
                    env.commands[:, 2] = cmd_wz
                    env.height_cmd_buf[:] = curr_h_ctrl
                else:
                    # 协议 B: 前 100 步 (2.0s) 原地稳定站立 (vy=0, wz=0, h=0.39774)
                    # step 100 之后通过平滑斜坡发生器 (4 cm/s) 过渡到目标速度和目标高度
                    if step < step_cmd_start:
                        target_vy_step = 0.0
                        target_wz_step = 0.0
                        target_h_step = 0.39774
                    else:
                        target_vy_step = cmd_vy
                        target_wz_step = cmd_wz
                        target_h_step = h_ctrl_final

                    # 速度平滑
                    dvy = target_vy_step - current_vy_cmd
                    step_vy = max_accel_y * dt
                    if abs(dvy) <= step_vy:
                        current_vy_cmd = target_vy_step
                    else:
                        current_vy_cmd += np.sign(dvy) * step_vy

                    dwz = target_wz_step - current_wz_cmd
                    step_wz = max_accel_w * dt
                    if abs(dwz) <= step_wz:
                        current_wz_cmd = target_wz_step
                    else:
                        current_wz_cmd += np.sign(dwz) * step_wz

                    dh = target_h_step - current_h_cmd
                    step_h = max_speed_h * dt
                    if abs(dh) <= step_h:
                        current_h_cmd = target_h_step
                    else:
                        current_h_cmd += np.sign(dh) * step_h

                    env.commands[:, 0] = 0.0
                    env.commands[:, 1] = current_vy_cmd
                    env.commands[:, 2] = current_wz_cmd
                    env.height_cmd_buf[:] = current_h_cmd
                    if hasattr(env, 'target_height_buf'):
                        env.target_height_buf[:] = current_h_cmd

                env.compute_observations()
                obs = env.get_observations()
                with torch.no_grad():
                    actions = pol(obs)

                obs, _, _, dones, _ = env.step(actions)

            # 恢复原始 reset_idx
            env.reset_idx = orig_reset_idx

            surv_rate = 100.0 * (1.0 - len(env_done_seen) / args.num_envs)
            print(f"  --> Case {case_name} Finished: SurvRate={surv_rate:.2f}%, NumTerminations={len(termination_records)}")

            results[proto][case_name] = {
                "num_envs": args.num_envs,
                "survival_rate": surv_rate,
                "terminated_count": len(env_done_seen),
                "records": termination_records
            }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDiagnosis results saved to: {output_path}")

if __name__ == '__main__':
    run_termination_diagnostics()
