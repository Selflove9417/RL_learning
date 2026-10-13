"""
Phase 8A: Evaluation of High-difficulty Compound Motion Curriculum Expansion Model.
Model: Phase 8A iter1 (curriculum ext: |vy| in [0.08, 0.16], |wz| in [0.15, 0.30], ref_kl beta=0.20)
Checkpoint: legged_gym/logs/myrobot/Oct12_15-05-39_phase8a_curriculum_ext_beta020_iter1/model_1.pt

Evaluation Interface:
    Continuous -5 mm height feedforward compensation:
        delta_h = -0.005 * clamp((0.39774 - h_ref) / (0.39774 - 0.360), 0.0, 1.0)
        h_ctrl = h_ref + delta_h

Rules:
    1. 21 standard benchmark scenarios (Phase 7C aligned protocol).
    2. All pass rates and errors evaluated strictly against h_ref.
    3. Output saved to evaluation_assets/phase8a/results/eval_phase8a_results.json.
"""

import isaacgym
import torch
import os
import json
import numpy as np

from isaacgym import gymtorch
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def run_phase8a_evaluation():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    # Rigid body states tensor
    body_states_tensor = env.gym.acquire_rigid_body_state_tensor(env.sim)
    bs = gymtorch.wrap_tensor(body_states_tensor).view(args.num_envs, env.num_bodies, 13)

    model_path = "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct12_15-05-39_phase8a_curriculum_ext_beta020_iter1/model_1.pt"

    models_to_test = {
        "phase8a_curriculum_ext_m1": model_path,
    }

    scenarios = {
        # 第一组：重点下蹲测试 (H=0.36m)
        "group1_squat": [
            ("g1_fwd_left_squat",     0.10,  0.20, 0.36),
            ("g1_bwd_left_squat",    -0.10,  0.20, 0.36),
            ("g1_fwd_right_squat",    0.10, -0.20, 0.36),
            ("g1_bwd_right_squat",   -0.10, -0.20, 0.36),
            ("g1_fwd_straight_squat", 0.10,  0.00, 0.36),
        ],
        # 第二组：升高与恢复能力 (H=0.44m / 0.39774m)
        "group2_rise": [
            ("g2_fwd_left_rise",      0.10,  0.20, 0.44),
            ("g2_fwd_right_rise",     0.10, -0.20, 0.44),
            ("g2_bwd_left_rise",     -0.10,  0.20, 0.44),
            ("g2_bwd_right_rise",    -0.10, -0.20, 0.44),
            ("g2_recover_nominal",    0.10,  0.20, 0.39774),
        ],
        # 第三组：旧任务保持 (防遗忘)
        "group3_regression": [
            ("g3_inplace_squat",      0.00,  0.00, 0.36),
            ("g3_cruise_squat",       0.15,  0.00, 0.36),
            ("g3_nominal_cruise",     0.20,  0.00, 0.39774),
            ("g3_nominal_compound",   0.10,  0.20, 0.39774),
            ("g3_nominal_stand",      0.00,  0.00, 0.39774),
        ],
        # 第四组：扩展域压力测试 (原 Stage 2 首段范围)
        "group4_stress": [
            ("g4_exp_fwd_left_squat",   0.16,  0.30, 0.36),
            ("g4_exp_fwd_right_squat",  0.16, -0.30, 0.36),
            ("g4_exp_bwd_left_squat",  -0.16,  0.30, 0.36),
            ("g4_exp_bwd_right_squat", -0.16, -0.30, 0.36),
            ("g4_exp_fwd_left_rise",    0.16,  0.30, 0.44),
            ("g4_exp_fwd_right_rise",   0.16, -0.30, 0.44),
        ]
    }

    # 评价标准与公差
    tol_h = 0.010       # 1cm
    tol_vy = 0.030      # 0.03 m/s
    tol_wz = 0.050      # 0.05 rad/s
    hold_steps_req = 25 # 0.5s at 50Hz
    rad_to_deg = 180.0 / np.pi
    R_wheel = 0.070
    r_vec = np.array([0.0, 0.0, -R_wheel])

    total_steps = 300
    step_cmd_start = 100
    steady_start = 200 # 200~300 (4.0~6.0s) 为稳态统计窗口

    eval_data = {}

    for model_name, ckpt_path in models_to_test.items():
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint {ckpt_path} not found! Skipping...")
            continue
        print(f"\n================ Evaluating Checkpoint: {model_name} ================")
        loaded = torch.load(ckpt_path, map_location=env.device)
        ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
        ac.load_state_dict(loaded['model_state_dict'])
        ac.eval()
        pol = ac.act_inference

        model_results = {}

        for group_name, cases in scenarios.items():
            group_metrics = {}
            for case_name, cmd_vy, cmd_wz, target_h in cases:
                h_ref = target_h # 原始基准任务期望高度

                # 计算补偿后的实际控制指令 h_ctrl
                delta_h = -0.005 * float(np.clip((0.39774 - h_ref) / (0.39774 - 0.360), 0.0, 1.0))
                h_ctrl = h_ref + delta_h

                torch.manual_seed(42)
                np.random.seed(42)
                env.reset()

                # 记录容器 (误差严格相对 h_ref)
                h_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
                vy_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
                wz_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
                early_done_count = 0
                done_env_flags = np.zeros(args.num_envs, dtype=bool)

                # 稳态统计容器
                steady_actual_h = []
                steady_h_signed = []
                steady_h_abs = []
                steady_h_sq = []
                steady_vy_sq = []
                steady_wz_sq = []
                steady_pitch_abs = []
                steady_pitch_max = []
                steady_roll_abs = []
                steady_roll_max = []
                knee_tau_mean_list = []
                knee_tau_max_list = []
                true_slip_mean_list = []
                true_slip_max_list = []
                wheel_sat_steps = np.zeros(args.num_envs)

                for step in range(total_steps):
                    # 阶跃前标称高度仍为 0.39774m，第 100 步阶跃下发 h_ctrl
                    curr_h_ctrl = 0.39774 if step < step_cmd_start else h_ctrl
                    env.commands[:, 0] = 0.0
                    env.commands[:, 1] = cmd_vy
                    env.commands[:, 2] = cmd_wz
                    env.height_cmd_buf[:] = curr_h_ctrl

                    obs = env.get_observations()
                    with torch.no_grad():
                        actions = pol(obs)
                    obs, _, _, dones, _ = env.step(actions)

                    d_arr = dones.cpu().numpy()
                    if np.any(d_arr):
                        early_done_count += np.sum(d_arr)
                        done_env_flags |= d_arr

                    # 检查轮端力矩饱和 (|tau| >= 6.95Nm，限幅7.0Nm)
                    tau_wheels = torch.abs(env.torques[:, env.wheel_indices]).cpu().numpy()
                    is_sat = np.any(tau_wheels >= 6.95, axis=1)
                    wheel_sat_steps += is_sat.astype(float)

                    if step >= step_cmd_start:
                        idx = step - step_cmd_start
                        base_h = env.root_states[:, 2].cpu().numpy()
                        base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                        base_wz = env.base_ang_vel[:, 2].cpu().numpy()

                        # 达标率严格相对原始任务目标 h_ref 计算！
                        h_err_traj[:, idx] = np.abs(base_h - h_ref)
                        vy_err_traj[:, idx] = np.abs(base_vy - cmd_vy)
                        wz_err_traj[:, idx] = np.abs(base_wz - cmd_wz)

                    if step >= steady_start:
                        base_h = env.root_states[:, 2].cpu().numpy()
                        h_signed_err = base_h - h_ref
                        h_err = np.abs(h_signed_err)

                        steady_actual_h.append(base_h)
                        steady_h_signed.append(h_signed_err)
                        steady_h_abs.append(h_err)
                        steady_h_sq.append(np.square(h_err))

                        base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                        base_wz = env.base_ang_vel[:, 2].cpu().numpy()
                        steady_vy_sq.append(np.square(base_vy - cmd_vy))
                        steady_wz_sq.append(np.square(base_wz - cmd_wz))

                        # 姿态
                        pitch = (torch.asin(torch.clamp(env.projected_gravity[:, 1], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
                        roll = (torch.asin(torch.clamp(env.projected_gravity[:, 0], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
                        steady_pitch_abs.append(np.abs(pitch))
                        steady_pitch_max.append(np.max(np.abs(pitch)))
                        steady_roll_abs.append(np.abs(roll))
                        steady_roll_max.append(np.max(np.abs(roll)))

                        # 膝关节力矩
                        tau_knee_L = np.abs(env.torques[:, env.left_knee_idx].cpu().numpy())
                        tau_knee_R = np.abs(env.torques[:, env.right_knee_idx].cpu().numpy())
                        knee_tau_mean_list.append(0.5 * (tau_knee_L + tau_knee_R))
                        knee_tau_max_list.append(np.maximum(tau_knee_L, tau_knee_R))

                        # 真实切向接触点物理滑动速度
                        env.gym.refresh_rigid_body_state_tensor(env.sim)
                        slips_step = []
                        for b_idx in [3, 6]:
                            lin_v = bs[:, b_idx, 7:10].cpu().numpy()
                            ang_w = bs[:, b_idx, 10:13].cpu().numpy()
                            v_c = lin_v + np.cross(ang_w, r_vec)
                            slip_xy = np.linalg.norm(v_c[:, :2], axis=1)
                            slips_step.append(slip_xy)
                        mean_slip = 0.5 * (slips_step[0] + slips_step[1])
                        true_slip_mean_list.append(mean_slip)
                        true_slip_max_list.append(np.maximum(slips_step[0], slips_step[1]))

                # 达标率与调节时间
                pass_h_mat = (h_err_traj <= tol_h)
                pass_vy_mat = (vy_err_traj <= tol_vy)
                pass_wz_mat = (wz_err_traj <= tol_wz)
                pass_joint_mat = pass_h_mat & pass_vy_mat & pass_wz_mat

                steady_slice = slice(100, 200)
                sample_pass_h = float(np.mean(pass_h_mat[:, steady_slice]) * 100.0)
                sample_pass_vy = float(np.mean(pass_vy_mat[:, steady_slice]) * 100.0)
                sample_pass_wz = float(np.mean(pass_wz_mat[:, steady_slice]) * 100.0)
                sample_pass_joint = float(np.mean(pass_joint_mat[:, steady_slice]) * 100.0)

                settling_times = []
                failed_envs = 0
                for e in range(args.num_envs):
                    if done_env_flags[e]:
                        failed_envs += 1
                        continue
                    j_seq = pass_joint_mat[e]
                    cur_consec = 0
                    found_time = None
                    for t_idx in range(len(j_seq)):
                        if j_seq[t_idx]:
                            cur_consec += 1
                            if cur_consec >= hold_steps_req:
                                found_time = (t_idx - hold_steps_req + 1) * 0.02
                                break
                        else:
                            cur_consec = 0
                    if found_time is not None:
                        settling_times.append(found_time)
                    else:
                        failed_envs += 1

                env_pass_rate = float((args.num_envs - failed_envs) / args.num_envs * 100.0)
                p50_settle = float(np.percentile(settling_times, 50)) if len(settling_times) > 0 else -1.0
                p95_settle = float(np.percentile(settling_times, 95)) if len(settling_times) > 0 else -1.0
                surv_rate = float((args.num_envs - np.sum(done_env_flags)) / args.num_envs * 100.0)

                case_res = {
                    "h_ref_target_m": round(float(h_ref), 4),
                    "h_ctrl_cmd_m": round(float(h_ctrl), 4),
                    "steady_actual_h_m": round(float(np.mean(steady_actual_h)), 4),
                    "h_signed_err_cm": round(float(np.mean(steady_h_signed) * 100.0), 3),
                    "h_steady_err_cm": round(float(np.mean(steady_h_abs) * 100.0), 3),
                    "h_rms_cm": round(float(np.sqrt(np.mean(steady_h_sq)) * 100.0), 3),
                    "h_pass_rate": round(sample_pass_h, 1),
                    "vy_rms_mps": round(float(np.sqrt(np.mean(steady_vy_sq))), 4),
                    "vy_pass_rate": round(sample_pass_vy, 1),
                    "wz_rms_radps": round(float(np.sqrt(np.mean(steady_wz_sq))), 4),
                    "wz_pass_rate": round(sample_pass_wz, 1),
                    "joint_pass_rate": round(sample_pass_joint, 1),
                    "env_pass_rate": round(env_pass_rate, 1),
                    "failed_envs": int(failed_envs),
                    "settle_p50_s": round(p50_settle, 2),
                    "settle_p95_s": round(p95_settle, 2),
                    "pitch_rms_deg": round(float(np.mean(steady_pitch_abs)), 2),
                    "pitch_peak_deg": round(float(np.max(steady_pitch_max)), 2),
                    "roll_rms_deg": round(float(np.mean(steady_roll_abs)), 2),
                    "roll_peak_deg": round(float(np.max(steady_roll_max)), 2),
                    "knee_tau_mean_Nm": round(float(np.mean(knee_tau_mean_list)), 2),
                    "knee_tau_max_Nm": round(float(np.max(knee_tau_max_list)), 2),
                    "wheel_sat_rate": round(float(np.mean(wheel_sat_steps) / total_steps * 100.0), 2),
                    "true_slip_mean_mps": round(float(np.mean(true_slip_mean_list)), 4),
                    "true_slip_max_mps": round(float(np.max(true_slip_max_list)), 4),
                    "early_dones": int(early_done_count),
                    "episode_surv_rate": round(surv_rate, 2),
                }
                group_metrics[case_name] = case_res
                print(f"[{group_name}] {case_name:22s} | H_ref:{h_ref:.3f} H_act:{case_res['steady_actual_h_m']:.4f} | H_err:{case_res['h_steady_err_cm']:4.2f}cm H_pass:{case_res['h_pass_rate']:4.1f}% | Jnt:{case_res['joint_pass_rate']:4.1f}% Env:{case_res['env_pass_rate']:4.1f}% | Surv:{case_res['episode_surv_rate']:5.1f}%")

            model_results[group_name] = group_metrics
        eval_data[model_name] = model_results

    out_dir = "/home/xy/IsaacTu/evaluation_assets/phase8a/results"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "eval_phase8a_results.json")
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 8A Curriculum Extension evaluation results saved to {out_file}!\n")

if __name__ == '__main__':
    run_phase8a_evaluation()
