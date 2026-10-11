import isaacgym
import torch
import os
import json
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def audit_contact_events():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 20

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None

    m150_path = "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_10-32-49_in_place_height_stage5_1_formal/model_150.pt"
    loaded = torch.load(m150_path, map_location=env.device)
    ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(loaded['model_state_dict'])
    ac.eval()
    policy = ac.act_inference

    rad_to_deg = 180.0 / np.pi

    modes = [
        ("step", [
            (0, 100, 0.39774, "nominal"),
            (100, 200, 0.35000, "squat"),
            (200, 300, 0.45000, "stand"),
            (300, 400, 0.39774, "recover"),
        ]),
        ("ramp", [
            (0, 100, (0.39774, 0.35000), "ramp_down"),
            (100, 200, (0.35000, 0.35000), "hold_low"),
            (200, 300, (0.35000, 0.45000), "ramp_up"),
            (300, 400, (0.45000, 0.39774), "ramp_recover"),
        ])
    ]

    contact_audit_results = {}

    for mode_name, stages in modes:
        print(f"\n--- Running Contact Audit on {mode_name} mode ---")
        torch.manual_seed(42)
        env.reset()
        env.commands[:, :] = 0.0

        lost_contact_events = []
        total_samples = 400 * env.num_envs
        both_contact_count = 0
        left_contact_count = 0
        right_contact_count = 0

        # Track consecutive non-contact steps per env
        consecutive_lost = np.zeros(env.num_envs, dtype=int)
        max_consecutive_lost = np.zeros(env.num_envs, dtype=int)

        for step in range(400):
            if mode_name == "step":
                for s_start, s_end, s_target, s_label in stages:
                    if s_start <= step < s_end:
                        curr_h = s_target
                        stage_label = s_label
                        break
            else:
                for s_start, s_end, (h_s, h_e), s_label in stages:
                    if s_start <= step < s_end:
                        alpha = (step - s_start) / float(s_end - s_start)
                        curr_h = h_s + alpha * (h_e - h_s)
                        stage_label = s_label
                        break

            env.height_cmd_buf[:] = curr_h
            obs = env.get_observations()
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, dones, _ = env.step(actions)

            fL = env.contact_forces[:, env.feet_indices[0], 2].cpu().numpy()
            fR = env.contact_forces[:, env.feet_indices[1], 2].cpu().numpy()
            pitch = (torch.asin(torch.clamp(env.projected_gravity[:, 1], -1.0, 1.0)) * rad_to_deg).cpu().numpy()
            pitch_rate = (env.base_ang_vel[:, 0] * rad_to_deg).cpu().numpy()
            wheel_tau_L = torch.abs(env.torques[:, env.left_wheel_idx]).cpu().numpy()
            wheel_tau_R = torch.abs(env.torques[:, env.right_wheel_idx]).cpu().numpy()

            cL = fL > 1.0
            cR = fR > 1.0
            both = cL & cR

            both_contact_count += np.sum(both)
            left_contact_count += np.sum(cL)
            right_contact_count += np.sum(cR)

            for i in range(env.num_envs):
                if not both[i]:
                    consecutive_lost[i] += 1
                    if consecutive_lost[i] > max_consecutive_lost[i]:
                        max_consecutive_lost[i] = consecutive_lost[i]
                    lost_contact_events.append({
                        "step": step,
                        "time_s": round(step * 0.02, 2),
                        "env_id": i,
                        "stage": stage_label,
                        "fL": float(round(fL[i], 3)),
                        "fR": float(round(fR[i], 3)),
                        "pitch_deg": float(round(pitch[i], 2)),
                        "pitch_rate_degps": float(round(pitch_rate[i], 2)),
                        "wheel_tau_Nm": float(round(max(wheel_tau_L[i], wheel_tau_R[i]), 2))
                    })
                else:
                    consecutive_lost[i] = 0

        pct_both = (both_contact_count / total_samples) * 100.0
        pct_left = (left_contact_count / total_samples) * 100.0
        pct_right = (right_contact_count / total_samples) * 100.0

        contact_audit_results[mode_name] = {
            "total_samples": total_samples,
            "both_contact_count": int(both_contact_count),
            "pct_both": round(float(pct_both), 2),
            "pct_left": round(float(pct_left), 2),
            "pct_right": round(float(pct_right), 2),
            "num_lost_events": len(lost_contact_events),
            "max_consecutive_steps_lost": int(np.max(max_consecutive_lost)),
            "max_consecutive_duration_s": round(float(np.max(max_consecutive_lost) * 0.02), 3),
            "lost_events_sample": lost_contact_events[:15], # First 15 events
        }

        print(f"[{mode_name}] Both: {pct_both:.2f}% | Left: {pct_left:.2f}% | Right: {pct_right:.2f}%")
        print(f"[{mode_name}] Total non-contact env-steps: {len(lost_contact_events)} / {total_samples}")
        print(f"[{mode_name}] Max consecutive lost duration: {np.max(max_consecutive_lost)} steps ({np.max(max_consecutive_lost)*0.02:.3f} s)")

    out_file = "/home/xy/.gemini/antigravity/brain/853aaed2-8860-4930-be3f-1bb608c17795/scratch/contact_events_deep_audit.json"
    with open(out_file, "w") as f:
        json.dump(contact_audit_results, f, indent=2)
    print(f"Saved contact audit to: {out_file}")

if __name__ == '__main__':
    audit_contact_events()
