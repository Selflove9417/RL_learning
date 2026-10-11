import isaacgym
import torch
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

def audit_terminations():
    args = get_args()
    args.task = 'myrobot'
    args.headless = True
    args.num_envs = 32

    env, _ = task_registry.make_env(name=args.task, args=args)
    env._resample_commands = lambda env_ids: None
    env.cfg.domain_rand.push_robots = False

    test_cases = [
        ("stage1_model25", "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_15-48-32_turning_height_stage5_3_formal/model_25.pt", "s4_nominal_cruise_fwd", 0.20, 0.0, 0.39774),
        ("stage2_model10", "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_16-13-43_turning_height_stage5_3_stage2_seg1/model_10.pt", "s4_nominal_cruise_fwd", 0.20, 0.0, 0.39774),
        ("stage2_model25", "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_16-13-43_turning_height_stage5_3_stage2_seg1/model_25.pt", "s1_fwd_left_squat", 0.10, 0.20, 0.36),
        ("stage2_model25", "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_16-13-43_turning_height_stage5_3_stage2_seg1/model_25.pt", "s1_fwd_right_rise", 0.10, -0.20, 0.44),
        ("stage2_model25", "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_16-13-43_turning_height_stage5_3_stage2_seg1/model_25.pt", "s4_nominal_stand", 0.00, 0.00, 0.39774),
        ("stage2_model25", "/home/xy/IsaacTu/legged_gym/logs/myrobot/Oct10_16-13-43_turning_height_stage5_3_stage2_seg1/model_25.pt", "s4_nominal_cruise_fwd", 0.20, 0.0, 0.39774),
    ]

    termination_log = []

    for m_name, ckpt, c_name, cmd_vy, cmd_wz, target_h in test_cases:
        loaded = torch.load(ckpt, map_location=env.device)
        ac = ActorCritic(31, 31, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
        ac.load_state_dict(loaded['model_state_dict'])
        ac.eval()
        pol = ac.act_inference

        torch.manual_seed(42)
        env.reset()

        case_dones = []
        for step in range(300):
            curr_h = 0.39774 if step < 100 else target_h
            env.commands[:, 0] = 0.0
            env.commands[:, 1] = cmd_vy
            env.commands[:, 2] = cmd_wz
            env.height_cmd_buf[:] = curr_h

            obs = env.get_observations()
            with torch.no_grad():
                actions = pol(obs)

            # 提前检查重置触发标志
            if len(env.termination_contact_indices) > 0:
                c_reset = torch.any(torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1) > 1., dim=1)
            else:
                c_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            t_reset = env.projected_gravity[:, 2] > -0.70
            h_reset = env.root_states[:, 2] < 0.25

            obs, _, _, dones, _ = env.step(actions)

            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).flatten().cpu().numpy()
                for env_id in done_ids:
                    reason = []
                    if c_reset[env_id]:
                        reason.append("contact_non_foot")
                    if t_reset[env_id]:
                        reason.append(f"tilt_excessive(grav_z={env.projected_gravity[env_id, 2]:.2f})")
                    if h_reset[env_id]:
                        reason.append(f"height_too_low(h={env.root_states[env_id, 2]:.2f}m)")
                    info = {
                        "model": m_name,
                        "case": c_name,
                        "step": step,
                        "env_id": int(env_id),
                        "reason": ", ".join(reason) if reason else "timeout_or_other"
                    }
                    case_dones.append(info)
                    print(f"[{m_name} | {c_name}] Step {step} Env {env_id} TERMINATED: {info['reason']}")

        termination_log.extend(case_dones)

    import json
    out_file = "/home/xy/.gemini/antigravity/brain/853aaed2-8860-4930-be3f-1bb608c17795/scratch/termination_reasons.json"
    with open(out_file, "w") as f:
        json.dump(termination_log, f, indent=2)
    print(f"Termination causes audit saved to {out_file}!")

if __name__ == '__main__':
    audit_terminations()
