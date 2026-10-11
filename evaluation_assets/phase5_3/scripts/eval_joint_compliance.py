import isaacgym
import torch
import os
import json
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def run_joint_compliance_eval():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, _ = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    models_to_test = {
        "stage1_model25": "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_15-48-32_turning_height_stage5_3_formal/model_25.pt",
        "stage2_model10": "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_16-13-43_turning_height_stage5_3_stage2_seg1/model_10.pt",
        "stage2_model25": "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_16-13-43_turning_height_stage5_3_stage2_seg1/model_25.pt",
    }

    scenarios = {
        # Suite 1: Stage 1 训练分布内 (vy=0.10, wz=0.20, H in [0.36, 0.44])
        "suite1_stage1_indist": [
            ("s1_fwd_left_squat",     0.10,  0.20, 0.36),
            ("s1_fwd_right_squat",    0.10, -0.20, 0.36),
            ("s1_fwd_straight_squat", 0.10,  0.00, 0.36),
            ("s1_bwd_left_squat",    -0.10,  0.20, 0.36),
            ("s1_bwd_right_squat",   -0.10, -0.20, 0.36),
            ("s1_fwd_left_rise",      0.10,  0.20, 0.44),
            ("s1_fwd_right_rise",     0.10, -0.20, 0.44),
            ("s1_bwd_left_rise",     -0.10,  0.20, 0.44),
            ("s1_bwd_right_rise",    -0.10, -0.20, 0.44),
        ],
        # Suite 2: Stage 2 扩展范围 (vy=0.16, wz=0.30, H in [0.36, 0.44])
        "suite2_stage2_expanded": [
            ("s2_exp_fwd_left_squat",   0.16,  0.30, 0.36),
            ("s2_exp_fwd_right_squat",  0.16, -0.30, 0.36),
            ("s2_exp_bwd_left_squat",  -0.16,  0.30, 0.36),
            ("s2_exp_bwd_right_squat", -0.16, -0.30, 0.36),
            ("s2_exp_fwd_left_rise",    0.16,  0.30, 0.44),
            ("s2_exp_fwd_right_rise",   0.16, -0.30, 0.44),
            ("s2_exp_bwd_left_rise",   -0.16,  0.30, 0.44),
            ("s2_exp_bwd_right_rise",  -0.16, -0.30, 0.44),
        ]
    }

    # 判据阈值 (Stage 1 既定标准)
    tol_h = 0.010       # 高度误差 <= 0.010 m
    tol_vy = 0.030      # 纵向速度误差 <= 0.030 m/s
    tol_wz = 0.050      # 偏航角速度误差 <= 0.050 rad/s
    hold_steps_req = 25 # 连续保持 0.5s (25步 * 0.02s = 0.5s)
    total_steps = 300
    step_cmd_start = 100 # 第100步下发指令

    results = {}

    for model_name, ckpt_path in models_to_test.items():
        print(f"\n================ Running Compliance Eval: {model_name} ================")
        loaded = torch.load(ckpt_path, map_location=env.device)
        ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
        ac.load_state_dict(loaded['model_state_dict'])
        ac.eval()
        pol = ac.act_inference

        model_results = {}

        for suite_name, cases in scenarios.items():
            suite_metrics = {}
            for case_name, cmd_vy, cmd_wz, target_h in cases:
                torch.manual_seed(42)
                env.reset()

                # 每个环境的时间序列记录 (32, 200) -> 从 step 100 到 300
                h_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
                vy_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
                wz_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
                early_done_count = 0
                done_env_flags = np.zeros(args.num_envs, dtype=bool)

                for step in range(total_steps):
                    curr_h = 0.39774 if step < step_cmd_start else target_h
                    env.commands[:, 0] = 0.0
                    env.commands[:, 1] = cmd_vy
                    env.commands[:, 2] = cmd_wz
                    env.height_cmd_buf[:] = curr_h

                    obs = env.get_observations()
                    with torch.no_grad():
                        actions = pol(obs)
                    obs, _, _, dones, _ = env.step(actions)

                    d_arr = dones.cpu().numpy()
                    if np.any(d_arr):
                        early_done_count += np.sum(d_arr)
                        done_env_flags |= d_arr

                    if step >= step_cmd_start:
                        idx = step - step_cmd_start
                        base_h = env.root_states[:, 2].cpu().numpy()
                        base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                        base_wz = env.base_ang_vel[:, 2].cpu().numpy()

                        h_err_traj[:, idx] = np.abs(base_h - target_h)
                        vy_err_traj[:, idx] = np.abs(base_vy - cmd_vy)
                        wz_err_traj[:, idx] = np.abs(base_wz - cmd_wz)

                # 逐环境达标判定
                # 单项及联合布尔矩阵 (32, 200)
                pass_h_mat = (h_err_traj <= tol_h)
                pass_vy_mat = (vy_err_traj <= tol_vy)
                pass_wz_mat = (wz_err_traj <= tol_wz)
                pass_joint_mat = pass_h_mat & pass_vy_mat & pass_wz_mat

                # 计算稳态期 (第 200~300步，对应 trajectory 内 100~200) 样本级达标率
                steady_slice = slice(100, 200)
                sample_pass_h = np.mean(pass_h_mat[:, steady_slice]) * 100.0
                sample_pass_vy = np.mean(pass_vy_mat[:, steady_slice]) * 100.0
                sample_pass_wz = np.mean(pass_wz_mat[:, steady_slice]) * 100.0
                sample_pass_joint = np.mean(pass_joint_mat[:, steady_slice]) * 100.0

                # 逐环境稳定调节时间: 寻找最早连续满足 hold_steps_req 的起始步
                settling_times = []
                failed_env_count = 0

                for e in range(args.num_envs):
                    if done_env_flags[e]:
                        # 发生了提前终止，标记为失败
                        failed_env_count += 1
                        continue

                    j_seq = pass_joint_mat[e]
                    # 寻找连续 hold_steps_req 个 True
                    found_step = None
                    cur_consec = 0
                    for t_idx in range(len(j_seq)):
                        if j_seq[t_idx]:
                            cur_consec += 1
                            if cur_consec >= hold_steps_req:
                                # 达标调节时间 = 从第 100 步阶跃起的秒数
                                found_step = (t_idx - hold_steps_req + 1) * 0.02
                                break
                        else:
                            cur_consec = 0

                    if found_step is not None:
                        settling_times.append(found_step)
                    else:
                        failed_env_count += 1

                env_pass_rate = (args.num_envs - failed_env_count) / args.num_envs * 100.0
                p50_settle = float(np.percentile(settling_times, 50)) if len(settling_times) > 0 else -1.0
                p95_settle = float(np.percentile(settling_times, 95)) if len(settling_times) > 0 else -1.0

                case_res = {
                    "sample_pass_rate_h": round(float(sample_pass_h), 1),
                    "sample_pass_rate_vy": round(float(sample_pass_vy), 1),
                    "sample_pass_rate_wz": round(float(sample_pass_wz), 1),
                    "sample_pass_rate_joint": round(float(sample_pass_joint), 1),
                    "env_joint_pass_rate": round(float(env_pass_rate), 1),
                    "failed_envs": int(failed_env_count),
                    "early_dones": int(early_done_count),
                    "settle_time_p50_s": round(p50_settle, 2),
                    "settle_time_p95_s": round(p95_settle, 2),
                }
                suite_metrics[case_name] = case_res
                print(f"[{suite_name}] {case_name:24s} | H:{case_res['sample_pass_rate_h']:4.1f}% Vy:{case_res['sample_pass_rate_vy']:4.1f}% Wz:{case_res['sample_pass_rate_wz']:4.1f}% Joint:{case_res['sample_pass_rate_joint']:4.1f}% | EnvPass:{case_res['env_joint_pass_rate']:4.1f}% Fail:{case_res['failed_envs']} P50:{case_res['settle_time_p50_s']}s P95:{case_res['settle_time_p95_s']}s")

            model_results[suite_name] = suite_metrics
        results[model_name] = model_results

    out_file = "/home/xy/.gemini/antigravity/brain/853aaed2-8860-4930-be3f-1bb608c17795/scratch/compliance_joint_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved compliance metrics to {out_file}!\n")

if __name__ == '__main__':
    run_joint_compliance_eval()
