import isaacgym
import torch
import os, json
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic
from isaacgym import gymtorch

args = get_args()
args.task = 'myrobot'
args.headless = True
args.num_envs = 32

env, env_cfg = task_registry.make_env(name=args.task, args=args)
env._resample_commands = lambda env_ids: None
env.cfg.domain_rand.push_robots = False

body_states_tensor = env.gym.acquire_rigid_body_state_tensor(env.sim)
bs = gymtorch.wrap_tensor(body_states_tensor).view(args.num_envs, env.num_bodies, 13)

base_ckpt = "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_15-48-32_turning_height_stage5_3_formal/model_25.pt"
dir_A = "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_18-20-47_phase5_3_height_deadzone_A_015"
dir_low_lr = "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_19-00-42_phase5_3_low_lr_1e4_015"

models_to_test = {
    "baseline_m25": base_ckpt,
    "groupA_m5":   os.path.join(dir_A, "model_5.pt"),
    "groupA_m10":  os.path.join(dir_A, "model_10.pt"),
    "groupA_m15":  os.path.join(dir_A, "model_15.pt"),
    "low_lr_m5":   os.path.join(dir_low_lr, "model_5.pt"),
    "low_lr_m10":  os.path.join(dir_low_lr, "model_10.pt"),
    "low_lr_m15":  os.path.join(dir_low_lr, "model_15.pt"),
}

scenarios = [
    ("g1_fwd_left_squat",     0.10,  0.20, 0.36),
    ("g1_bwd_left_squat",    -0.10,  0.20, 0.36),
    ("g2_fwd_right_rise",     0.10, -0.20, 0.44),
    ("g2_bwd_right_rise",    -0.10, -0.20, 0.44),
    ("g3_inplace_squat",      0.00,  0.00, 0.36),
    ("g3_cruise_squat",       0.15,  0.00, 0.36),
    ("g3_nominal_compound",   0.15,  0.25, 0.39774),
    ("g3_nominal_stand",      0.00,  0.00, 0.39774),
    ("g4_exp_fwd_left_squat", 0.16,  0.30, 0.36),
]

total_steps = 1000 # 20.0s
step_cmd_start = 100 # 2.0s
steady_start = 200 # 4.0s

results = {}
for m_name, m_path in models_to_test.items():
    print(f"\n================ Evaluating Checkpoint: {m_name} ================")
    loaded = torch.load(m_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    pol = ac.act_inference

    m_res = {}
    for case_name, cmd_vy, cmd_wz, target_h in scenarios:
        torch.manual_seed(42)
        env.reset()

        h_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
        vy_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
        wz_err_traj = np.zeros((args.num_envs, total_steps - step_cmd_start))
        early_dones = 0
        done_flags = np.zeros(args.num_envs, dtype=bool)

        steady_h_abs, steady_vy_sq, steady_wz_sq = [], [], []
        steady_pitch_abs, steady_roll_abs = [], []
        knee_tau_list = []
        wheel_sat_steps = np.zeros(args.num_envs)

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
                early_dones += int(np.sum(d_arr))
                done_flags |= d_arr

            tau_wheels = torch.abs(env.torques[:, env.wheel_indices]).cpu().numpy()
            is_sat = np.any(tau_wheels >= 6.95, axis=1)
            wheel_sat_steps += is_sat.astype(float)

            if step >= step_cmd_start:
                idx = step - step_cmd_start
                base_h = env.root_states[:, 2].cpu().numpy()
                base_vy = env.base_lin_vel[:, 1].cpu().numpy()
                base_wz = env.base_ang_vel[:, 2].cpu().numpy()

                h_err_traj[:, idx] = np.abs(base_h - target_h)
                vy_err_traj[:, idx] = np.abs(base_vy - cmd_vy)
                wz_err_traj[:, idx] = np.abs(base_wz - cmd_wz)

            if step >= steady_start:
                steady_h_abs.append(np.abs(env.root_states[:, 2].cpu().numpy() - target_h))
                steady_vy_sq.append(np.square(env.base_lin_vel[:, 1].cpu().numpy() - cmd_vy))
                steady_wz_sq.append(np.square(env.base_ang_vel[:, 2].cpu().numpy() - cmd_wz))

                pitch_rad = env.root_states[:, 4].cpu().numpy()
                roll_rad = env.root_states[:, 3].cpu().numpy()
                steady_pitch_abs.append(np.abs(np.degrees(pitch_rad)))
                steady_roll_abs.append(np.abs(np.degrees(roll_rad)))

                knee_taus = torch.abs(env.torques[:, [env.left_knee_idx, env.right_knee_idx]]).cpu().numpy()
                knee_tau_list.append(knee_taus)

        # Metrics
        h_steady_cm = float(np.mean(steady_h_abs) * 100.0)
        h_rms_cm = float(np.sqrt(np.mean(np.square(steady_h_abs))) * 100.0)
        vy_rms_mps = float(np.sqrt(np.mean(steady_vy_sq)))
        wz_rms_radps = float(np.sqrt(np.mean(steady_wz_sq)))

        h_pass = float(np.mean(np.array(steady_h_abs) <= 0.010) * 100.0)
        vy_pass = float(np.mean(np.sqrt(steady_vy_sq) <= 0.030) * 100.0)
        wz_pass = float(np.mean(np.sqrt(steady_wz_sq) <= 0.050) * 100.0)

        joint_pass_samples = (np.array(steady_h_abs) <= 0.010) & (np.sqrt(steady_vy_sq) <= 0.030) & (np.sqrt(steady_wz_sq) <= 0.050)
        joint_pass_rate = float(np.mean(joint_pass_samples) * 100.0)

        # Env pass: continuous 25 steps (0.5s)
        window = 25
        all_passed_traj = (h_err_traj <= 0.010) & (vy_err_traj <= 0.030) & (wz_err_traj <= 0.050)
        env_passed_flags = np.zeros(args.num_envs, dtype=bool)
        for e in range(args.num_envs):
            if done_flags[e]: continue
            traj_e = all_passed_traj[e]
            conv = np.convolve(traj_e.astype(int), np.ones(window, dtype=int), mode='valid')
            if np.any(conv == window):
                env_passed_flags[e] = True
        env_pass_rate = float(np.mean(env_passed_flags) * 100.0)
        surv_rate = float((1.0 - np.mean(done_flags)) * 100.0)

        m_res[case_name] = {
            'h_steady_cm': round(h_steady_cm, 3),
            'h_rms_cm': round(h_rms_cm, 3),
            'h_pass': round(h_pass, 1),
            'vy_rms': round(vy_rms_mps, 4),
            'vy_pass': round(vy_pass, 1),
            'wz_rms': round(wz_rms_radps, 4),
            'wz_pass': round(wz_pass, 1),
            'joint_pass': round(joint_pass_rate, 1),
            'env_pass': round(env_pass_rate, 1),
            'surv_rate': round(surv_rate, 1),
            'early_dones': early_dones,
            'knee_tau_mean': round(float(np.mean(knee_tau_list)), 2),
            'knee_tau_max': round(float(np.max(knee_tau_list)), 2),
            'pitch_rms': round(float(np.sqrt(np.mean(np.square(steady_pitch_abs)))), 2),
            'roll_rms': round(float(np.sqrt(np.mean(np.square(steady_roll_abs)))), 3),
            'wheel_sat_rate': round(float(np.mean(wheel_sat_steps) / total_steps * 100.0), 3)
        }
    results[m_name] = m_res

out_path = "/home/xy/.gemini/antigravity/brain/853aaed2-8860-4930-be3f-1bb608c17795/scratch/eval_phase5_3_low_lr_results.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {out_path}")
