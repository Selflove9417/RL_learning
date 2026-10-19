# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Diagnostic Attribution Experiment: Deconstructing Takeoff Deceleration
# Evaluates Seeds 11, 13, 15, 20 from Jump Trigger (Step 38) to Liftoff

import os
import sys
import math
import json
import subprocess

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))


def run_single_diagnostic():
    import isaacgym
    from isaacgym import gymtorch
    from isaacgym.torch_utils import get_euler_xyz
    import torch
    import numpy as np
    from legged_gym.envs import MyRobotRunningJump
    from legged_gym.utils import task_registry, get_args
    from rsl_rl.modules import ActorCritic

    args = get_args()
    args.headless = True
    seed = args.seed
    mask_actor_wheel = (os.environ.get("MASK_ACTOR_WHEEL") == "1")
    model_path = args.model_path if (hasattr(args, 'model_path') and args.model_path is not None) else 'evaluation_assets/frozen_stage8/model_40.pt'

    env_cfg, train_cfg = task_registry.get_cfgs(name="myrobot_running_jump")
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed

    env, _ = task_registry.make_env(name="myrobot_running_jump", args=args, env_cfg=env_cfg)
    env.reset()

    ckpt = torch.load(model_path, map_location=env.device)
    policy = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()

    obs = env.get_observations()

    step_records = []
    prev_vy = None

    for step_i in range(140):
        with torch.no_grad():
            actions = policy.act_inference(obs)

        # Condition B: Mask actor wheel actions during JUMP phase only
        if mask_actor_wheel and env.fsm_state[0].item() in [env.STATE_JUMP, env.STATE_RECOVER]:
            actions[:, env.left_wheel_idx] = 0.0
            actions[:, env.right_wheel_idx] = 0.0

        # Step simulation
        obs, _, rews, dones, infos = env.step(actions)

        # Collect high-resolution telemetry around takeoff window (steps 38 to 60)
        vy = env.base_lin_vel[0, 1].item()
        vz = env.base_lin_vel[0, 2].item()
        ay = (vy - prev_vy) / env.dt if prev_vy is not None else 0.0
        prev_vy = vy

        fsm = env.fsm_state[0].item()
        air = env.is_airborne[0].item()
        took_off = env.has_taken_off[0].item()
        clr = env.bilateral_clearance[0].item()

        # Wheel speeds & torques
        w_L = env.dof_vel[0, env.left_wheel_idx].item()
        w_R = env.dof_vel[0, env.right_wheel_idx].item()
        tau_w_L = env.torques[0, env.left_wheel_idx].item()
        tau_w_R = env.torques[0, env.right_wheel_idx].item()

        # Leg torques & positions
        tau_hip_L = env.torques[0, env.left_hip_idx].item()
        tau_knee_L = env.torques[0, env.left_knee_idx].item()
        tau_hip_R = env.torques[0, env.right_hip_idx].item()
        tau_knee_R = env.torques[0, env.right_knee_idx].item()

        q_hip_L = env.dof_pos[0, env.left_hip_idx].item()
        q_knee_L = env.dof_pos[0, env.left_knee_idx].item()

        # Actor raw actions on wheels & legs
        act_w_L = actions[0, env.left_wheel_idx].item()
        act_w_R = actions[0, env.right_wheel_idx].item()
        act_hip_L = actions[0, env.left_hip_idx].item()
        act_knee_L = actions[0, env.left_knee_idx].item()

        # Contact forces (Z component on wheels)
        f_contact_L = env.contact_forces[0, env.feet_indices[0], 2].item()
        f_contact_R = env.contact_forces[0, env.feet_indices[1], 2].item()

        # Body attitude
        r_raw, p_raw, _ = get_euler_xyz(env.base_quat)
        pitch = torch.atan2(torch.sin(r_raw), torch.cos(r_raw)).item()
        pitch_rate = env.base_ang_vel[0, 0].item()

        record = {
            "step": step_i,
            "fsm": fsm,
            "vy": vy,
            "vz": vz,
            "ay": ay,
            "pitch_deg": math.degrees(pitch),
            "pitch_rate_dps": math.degrees(pitch_rate),
            "act_w_L": act_w_L,
            "act_w_R": act_w_R,
            "act_hip_L": act_hip_L,
            "act_knee_L": act_knee_L,
            "w_L": w_L,
            "w_R": w_R,
            "tau_w_L": tau_w_L,
            "tau_w_R": tau_w_R,
            "tau_hip_L": tau_hip_L,
            "tau_knee_L": tau_knee_L,
            "q_hip_L": q_hip_L,
            "q_knee_L": q_knee_L,
            "f_contact_L": f_contact_L,
            "f_contact_R": f_contact_R,
            "clr_cm": clr * 100.0,
            "air": air,
            "took_off": took_off,
            "done": dones[0].item()
        }

        if 38 <= step_i <= 55:
            step_records.append(record)

        if took_off and step_i > 45:
            if step_i > 55:
                break

    output_data = {
        "seed": seed,
        "mask_actor_wheel": mask_actor_wheel,
        "records": step_records
    }
    print("\n===DIAG_START===")
    print(json.dumps(output_data))
    print("===DIAG_END===")


def run_worker(seed, mask_actor_wheel):
    cmd = [
        sys.executable,
        "-u",
        __file__,
        "--seed", str(seed)
    ]
    sub_env = os.environ.copy()
    sub_env["RUN_DIAG_WORKER"] = "1"
    sub_env["MASK_ACTOR_WHEEL"] = "1" if mask_actor_wheel else "0"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=sub_env)
    stdout, stderr = proc.communicate()

    if "===DIAG_START===" in stdout:
        block = stdout.split("===DIAG_START===")[1].split("===DIAG_END===")[0].strip()
        return json.loads(block)
    else:
        raise RuntimeError(f"Diagnostic failed for seed {seed} (mask={mask_actor_wheel}):\n{stderr[-500:]}")


def main():
    if os.environ.get("RUN_DIAG_WORKER") == "1":
        run_single_diagnostic()
        return

    test_seeds = [11, 13, 15, 20]
    print("=" * 120)
    print("  MYROBOT RUNNING JUMP: TAKEOFF DECELERATION ATTRIBUTION EXPERIMENT")
    print("  * Target Seeds: 11, 13, 15, 20")
    print("  * Condition A: Baseline Control (Stage 8 policy active on legs & wheels)")
    print("  * Condition B: Ablation (Actor wheel action masked to 0, wheel speed damping feedback active)")
    print("=" * 120)

    all_data_A = {}
    all_data_B = {}

    for s in test_seeds:
        print(f"Running Seed {s:2d} Condition A (Baseline)...", flush=True)
        all_data_A[s] = run_worker(s, mask_actor_wheel=False)
        print(f"Running Seed {s:2d} Condition B (Wheel Action Masked)...", flush=True)
        all_data_B[s] = run_worker(s, mask_actor_wheel=True)

    # Detailed Step-by-Step Printout for Seed 11
    print("\n" + "=" * 125)
    print(" STEP-BY-STEP TELEMETRY: SEED 11 (CONDITION A: BASELINE)")
    print("=" * 125)
    print(f"| Step | FSM | vy (m/s) | ay (m/s2) | Pitch(deg) | Act_w(L/R)    | Tau_w(L/R) (Nm)  | Wheel w(L/R) (rad/s) | Leg Tau(Hip/Knee)| Contact_Fz(L/R) | Clr(cm)| Air |")
    print(f"|------|-----|----------|-----------|------------|---------------|------------------|----------------------|------------------|-----------------|--------|-----|")
    for r in all_data_A[11]["records"]:
        act_w_str = f"{r['act_w_L']:+.2f}/{r['act_w_R']:+.2f}"
        tau_w_str = f"{r['tau_w_L']:+.1f}/{r['tau_w_R']:+.1f}"
        w_str = f"{r['w_L']:+.1f}/{r['w_R']:+.1f}"
        leg_str = f"{r['tau_hip_L']:+.1f}/{r['tau_knee_L']:+.1f}"
        fz_str = f"{r['f_contact_L']:4.0f}/{r['f_contact_R']:4.0f}"
        air_str = "YES" if r["air"] else "no "
        print(f"| {r['step']:4d} | {r['fsm']:3d} | {r['vy']:8.3f} | {r['ay']:9.2f} | {r['pitch_deg']:10.1f} | {act_w_str:13s} | {tau_w_str:16s} | {w_str:20s} | {leg_str:16s} | {fz_str:15s} | {r['clr_cm']:6.1f} | {air_str} |")

    print("\n" + "=" * 125)
    print(" STEP-BY-STEP TELEMETRY: SEED 11 (CONDITION B: ACTOR WHEEL MASKED)")
    print("=" * 125)
    print(f"| Step | FSM | vy (m/s) | ay (m/s2) | Pitch(deg) | Act_w(L/R)    | Tau_w(L/R) (Nm)  | Wheel w(L/R) (rad/s) | Leg Tau(Hip/Knee)| Contact_Fz(L/R) | Clr(cm)| Air |")
    print(f"|------|-----|----------|-----------|------------|---------------|------------------|----------------------|------------------|-----------------|--------|-----|")
    for r in all_data_B[11]["records"]:
        act_w_str = f"{r['act_w_L']:+.2f}/{r['act_w_R']:+.2f}"
        tau_w_str = f"{r['tau_w_L']:+.1f}/{r['tau_w_R']:+.1f}"
        w_str = f"{r['w_L']:+.1f}/{r['w_R']:+.1f}"
        leg_str = f"{r['tau_hip_L']:+.1f}/{r['tau_knee_L']:+.1f}"
        fz_str = f"{r['f_contact_L']:4.0f}/{r['f_contact_R']:4.0f}"
        air_str = "YES" if r["air"] else "no "
        print(f"| {r['step']:4d} | {r['fsm']:3d} | {r['vy']:8.3f} | {r['ay']:9.2f} | {r['pitch_deg']:10.1f} | {act_w_str:13s} | {tau_w_str:16s} | {w_str:20s} | {leg_str:16s} | {fz_str:15s} | {r['clr_cm']:6.1f} | {air_str} |")

    # Comparative Summary Across Seeds 11, 13, 15, 20
    print("\n" + "=" * 115)
    print(" ATTRIBUTION SUMMARY COMPARISON ACROSS SEEDS 11, 13, 15, 20")
    print("=" * 115)
    print(f"| Seed | Condition             | Pre-Jump vy | Min vy in Push | Liftoff vy | Deceleration Delta | Max Clearance |")
    print(f"|------|-----------------------|-------------|----------------|------------|--------------------|---------------|")

    for s in test_seeds:
        rec_A = all_data_A[s]["records"]
        rec_B = all_data_B[s]["records"]

        vy_pre_A = rec_A[1]["vy"] # Step 39
        vy_pre_B = rec_B[1]["vy"]

        push_vy_A = [r["vy"] for r in rec_A if 40 <= r["step"] <= 48]
        push_vy_B = [r["vy"] for r in rec_B if 40 <= r["step"] <= 48]

        min_vy_A = min(push_vy_A) if push_vy_A else 0.0
        min_vy_B = min(push_vy_B) if push_vy_B else 0.0

        lo_r_A = next((r for r in rec_A if r["took_off"]), rec_A[-1])
        lo_r_B = next((r for r in rec_B if r["took_off"]), rec_B[-1])

        clr_max_A = max(r["clr_cm"] for r in rec_A)
        clr_max_B = max(r["clr_cm"] for r in rec_B)

        delta_A = lo_r_A["vy"] - vy_pre_A
        delta_B = lo_r_B["vy"] - vy_pre_B

        print(f"| {s:4d} | A (Baseline)          | {vy_pre_A:9.3f}m/s | {min_vy_A:12.3f}m/s | {lo_r_A['vy']:8.3f}m/s | {delta_A:16.3f}m/s | {clr_max_A:11.1f}cm |")
        print(f"|      | B (Actor Wheel Masked)| {vy_pre_B:9.3f}m/s | {min_vy_B:12.3f}m/s | {lo_r_B['vy']:8.3f}m/s | {delta_B:16.3f}m/s | {clr_max_B:11.1f}cm |")
        print(f"|------|-----------------------|-------------|----------------|------------|--------------------|---------------|")


if __name__ == '__main__':
    main()
