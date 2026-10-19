# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Final Minimal Diagnostic Attribution Experiment:
# Kinematic Decomposition (World vs Body Frame), Liftoff Timing,
# Wheel Slip Dynamics, and Actor Loss Mask Validation.
# Target Seeds: 11, 13, 15, 20

import os
import sys
import math
import json
import csv
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
    condition_label = "B_MaskedWheel" if mask_actor_wheel else "A_Baseline"

    for step_i in range(140):
        with torch.no_grad():
            actions = policy.act_inference(obs)

        # Condition B: Mask actor wheel actions during JUMP phase only
        if mask_actor_wheel and env.fsm_state[0].item() in [env.STATE_JUMP, env.STATE_RECOVER]:
            actions[:, env.left_wheel_idx] = 0.0
            actions[:, env.right_wheel_idx] = 0.0

        # Step simulation
        obs, _, rews, dones, infos = env.step(actions)

        # 1. World-frame kinematics from root_states
        v_world_x = env.root_states[0, 7].item()
        v_world_y = env.root_states[0, 8].item()
        v_world_z = env.root_states[0, 9].item()
        pos_z = env.root_states[0, 2].item()

        # 2. Body-frame kinematics from base_lin_vel
        v_body_x = env.base_lin_vel[0, 0].item()
        v_body_y = env.base_lin_vel[0, 1].item()
        v_body_z = env.base_lin_vel[0, 2].item()

        # 3. Attitude: Sagittal pitch around wheel axle X, Coronal roll around Y
        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        pitch = torch.atan2(torch.sin(r_raw), torch.cos(r_raw)).item()
        roll = torch.atan2(torch.sin(p_raw), torch.cos(p_raw)).item()
        pitch_rate_dps = math.degrees(env.base_ang_vel[0, 0].item())

        # Mathematical projection of world velocity onto body forward axis:
        # v_body_y ~ cos(pitch) * v_world_y + sin(pitch) * v_world_z
        proj_vy = math.cos(pitch) * v_world_y + math.sin(pitch) * v_world_z
        tilt_artifact = v_body_y - v_world_y

        # 4. FSM and Airborne State
        fsm = env.fsm_state[0].item()
        air = env.is_airborne[0].item()
        took_off = env.has_taken_off[0].item()
        touched_down = env.has_touched_down[0].item()

        # 5. Ground Contact & Clearance
        f_contact_L = env.contact_forces[0, env.feet_indices[0], 2].item()
        f_contact_R = env.contact_forces[0, env.feet_indices[1], 2].item()
        clr_min = env.bilateral_clearance[0].item() * 100.0
        clr_L = env.wheel_clearance[0, 0].item() * 100.0
        clr_R = env.wheel_clearance[0, 1].item() * 100.0

        # True bilateral contact loss: normal contact force < 1.0 N on both wheels
        is_contact_lost = (f_contact_L < 1.0) and (f_contact_R < 1.0)

        # 6. Wheel Speed, Slip and Torques
        w_L = env.dof_vel[0, env.left_wheel_idx].item()
        w_R = env.dof_vel[0, env.right_wheel_idx].item()
        R_w = env.wheel_radius
        v_surf_L = - w_L * R_w  # Forward surface speed (w < 0 is forward roll)
        v_surf_R = - w_R * R_w
        slip_L_world = v_surf_L - v_world_y
        slip_R_world = v_surf_R - v_world_y

        tau_w_L = env.torques[0, env.left_wheel_idx].item()
        tau_w_R = env.torques[0, env.right_wheel_idx].item()

        act_w_L = actions[0, env.left_wheel_idx].item()
        act_w_R = actions[0, env.right_wheel_idx].item()

        # 7. Leg Dynamics
        tau_hip_L = env.torques[0, env.left_hip_idx].item()
        tau_knee_L = env.torques[0, env.left_knee_idx].item()
        q_hip_L = env.dof_pos[0, env.left_hip_idx].item()
        q_knee_L = env.dof_pos[0, env.left_knee_idx].item()

        # 8. Reward terms
        rew_retention = env._reward_forward_velocity_retention()[0].item()
        rew_takeoff_vz = env._reward_takeoff_velocity()[0].item()
        rew_pitch_stab = env._reward_pitch_stability()[0].item()
        sigma = env_cfg.rewards.tracking_sigma
        rew_retention_hypothetical_world = math.exp(-((v_world_y - 0.20)**2) / (2.0 * sigma**2))

        record = {
            "seed": seed,
            "condition": condition_label,
            "step": step_i,
            "fsm": fsm,
            "v_world_y": v_world_y,
            "v_world_z": v_world_z,
            "v_body_y": v_body_y,
            "v_body_z": v_body_z,
            "proj_vy": proj_vy,
            "tilt_artifact": tilt_artifact,
            "pitch_deg": math.degrees(pitch),
            "pitch_rate_dps": pitch_rate_dps,
            "roll_deg": math.degrees(roll),
            "pos_z": pos_z,
            "f_contact_L": f_contact_L,
            "f_contact_R": f_contact_R,
            "is_contact_lost": is_contact_lost,
            "is_airborne": air,
            "has_taken_off": took_off,
            "has_touched_down": touched_down,
            "clr_min_cm": clr_min,
            "clr_L_cm": clr_L,
            "clr_R_cm": clr_R,
            "w_L": w_L,
            "w_R": w_R,
            "v_surf_L": v_surf_L,
            "v_surf_R": v_surf_R,
            "slip_L_world": slip_L_world,
            "slip_R_world": slip_R_world,
            "act_w_L": act_w_L,
            "act_w_R": act_w_R,
            "tau_w_L": tau_w_L,
            "tau_w_R": tau_w_R,
            "tau_hip_L": tau_hip_L,
            "tau_knee_L": tau_knee_L,
            "q_hip_L": q_hip_L,
            "q_knee_L": q_knee_L,
            "rew_retention": rew_retention,
            "rew_retention_hypothetical_world": rew_retention_hypothetical_world,
            "rew_takeoff_vz": rew_takeoff_vz,
            "rew_pitch_stab": rew_pitch_stab,
            "done": dones[0].item()
        }

        if 38 <= step_i <= 55:
            step_records.append(record)

        if took_off and step_i > 55:
            break

    output_data = {
        "seed": seed,
        "condition": condition_label,
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
        raise RuntimeError(f"Diagnostic failed for seed {seed} (mask={mask_actor_wheel}):\n{stderr[-600:]}")


def main():
    if os.environ.get("RUN_DIAG_WORKER") == "1":
        run_single_diagnostic()
        return

    test_seeds = [11, 13, 15, 20]
    print("=" * 125)
    print(" MYROBOT RUNNING JUMP: FINAL MINIMAL DIAGNOSTIC ATTRIBUTION EXPERIMENT")
    print(" Evaluates Seeds 11, 13, 15, 20: Step 38~55")
    print(" - Synchronous World-Frame (root_states[:, 8]) vs Body-Frame (base_lin_vel[:, 1])")
    print(" - True Bilateral Contact Loss vs has_taken_off Frame Delay")
    print(" - Slip Velocity and Wheel Torque Direction Analysis")
    print(" - Velocity Retention Reward vs Coordinate Transformation Artifact")
    print("=" * 125)

    all_data_A = {}
    all_data_B = {}
    all_flat_records = []

    for s in test_seeds:
        print(f"Executing Seed {s:2d} Condition A (Baseline)...", flush=True)
        res_A = run_worker(s, mask_actor_wheel=False)
        all_data_A[s] = res_A
        all_flat_records.extend(res_A["records"])

        print(f"Executing Seed {s:2d} Condition B (Actor Wheel Masked)...", flush=True)
        res_B = run_worker(s, mask_actor_wheel=True)
        all_data_B[s] = res_B
        all_flat_records.extend(res_B["records"])

    # 1. Save Full Telemetry to CSV
    csv_path = os.path.join(workspace_root, "evaluation_assets", "running_jump", "takeoff_attribution_telemetry.csv")
    fieldnames = list(all_flat_records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_flat_records)
    print(f"\n[Artifact] Full high-precision telemetry exported to:\n  {csv_path} ({len(all_flat_records)} rows)")

    # 2. Detailed Step 39~47 Comparison Table for Seed 11
    print("\n" + "=" * 140)
    print(" SEED 11 SYNCHRONOUS TELEMETRY (STEPS 39~47): CONDITION A (BASELINE) VS CONDITION B (MASKED WHEEL)")
    print("=" * 140)
    print("| Cond | Step | v_world_Y | v_world_Z | v_body_y | Pitch(deg) | Proj_Diff(m/s) | Contact_Fz(L/R) | Clr_min | Contact_Lost | has_taken_off | Tau_w(L/R) (Nm)  | Wheel w(L/R) (rad/s) | Slip_Y(L/R) (m/s) | Rew_Ret(Body/World) |")
    print("|------|------|-----------|-----------|----------|------------|----------------|-----------------|---------|--------------|---------------|------------------|----------------------|-------------------|---------------------|")

    for cond_code, cond_data in [("A", all_data_A[11]["records"]), ("B", all_data_B[11]["records"])]:
        for r in cond_data:
            if 39 <= r["step"] <= 47:
                proj_diff = r["v_body_y"] - r["v_world_y"]
                fz_str = f"{r['f_contact_L']:3.0f}/{r['f_contact_R']:3.0f}"
                tau_w_str = f"{r['tau_w_L']:+4.1f}/{r['tau_w_R']:+4.1f}"
                w_str = f"{r['w_L']:+5.1f}/{r['w_R']:+5.1f}"
                slip_str = f"{r['slip_L_world']:+4.2f}/{r['slip_R_world']:+4.2f}"
                c_lost_str = "YES" if r["is_contact_lost"] else "no "
                t_off_str = "TRUE " if r["has_taken_off"] else "false"
                rew_str = f"{r['rew_retention']:.2f}/{r['rew_retention_hypothetical_world']:.2f}"

                print(f"|  {cond_code}   |  {r['step']:2d}  |  {r['v_world_y']:+7.3f}  |  {r['v_world_z']:+7.3f}  |  {r['v_body_y']:+7.3f} |   {r['pitch_deg']:+6.1f}   |    {proj_diff:+7.3f}     | {fz_str:15s} | {r['clr_min_cm']:5.1f}cm |     {c_lost_str}      |     {t_off_str}     | {tau_w_str:16s} | {w_str:20s} | {slip_str:17s} | {rew_str:19s} |")

    # 3. Liftoff Timing & Frame Delay Analysis across Seeds 11, 13, 15, 20
    print("\n" + "=" * 125)
    print(" LIFTOFF TIMING & FRAME DELAY AUDIT ACROSS SEEDS 11, 13, 15, 20")
    print("=" * 125)
    print("| Seed | Condition   | T_contact_lost | T_has_taken_off | Frame Delay (Steps) | v_world_Y at Contact Loss | v_body_y at Contact Loss | Pitch at Contact Loss |")
    print("|------|-------------|----------------|-----------------|---------------------|---------------------------|--------------------------|-----------------------|")

    for s in test_seeds:
        for cond_label, recs in [("A (Baseline)", all_data_A[s]["records"]), ("B (Masked)  ", all_data_B[s]["records"])]:
            t_lost = next((r["step"] for r in recs if r["is_contact_lost"]), -1)
            t_taken = next((r["step"] for r in recs if r["has_taken_off"]), -1)
            delay = (t_taken - t_lost) if (t_lost != -1 and t_taken != -1) else -1

            r_lost = next((r for r in recs if r["step"] == t_lost), None)
            vw_lost = r_lost["v_world_y"] if r_lost else 0.0
            vb_lost = r_lost["v_body_y"] if r_lost else 0.0
            pitch_lost = r_lost["pitch_deg"] if r_lost else 0.0

            print(f"|  {s:2d}  | {cond_label} |    Step {t_lost:2d}    |     Step {t_taken:2d}     |      {delay:+2d} step(s)        |         {vw_lost:+7.3f} m/s       |        {vb_lost:+7.3f} m/s       |       {pitch_lost:+5.1f} deg        |")

    # 4. True World-Frame Momentum Retention vs Body-Frame Deceleration
    print("\n" + "=" * 135)
    print(" WORLD-FRAME MOMENTUM RETENTION VS BODY-FRAME PROJECTION ARTIFACT")
    print("=" * 135)
    print("| Seed | Condition   | v_world_Y(Pre) | v_world_Y(Liftoff) | True Momentum Loss | v_body_y(Pre) | v_body_y(Liftoff) | Apparent Body Loss | Coordinate Tilt Artifact |")
    print("|------|-------------|----------------|--------------------|--------------------|---------------|-------------------|--------------------|--------------------------|")

    for s in test_seeds:
        for cond_label, recs in [("A (Baseline)", all_data_A[s]["records"]), ("B (Masked)  ", all_data_B[s]["records"])]:
            r_pre = next(r for r in recs if r["step"] == 39)
            t_lost = next(r["step"] for r in recs if r["is_contact_lost"])
            r_lost = next(r for r in recs if r["step"] == t_lost)

            vw_pre = r_pre["v_world_y"]
            vw_post = r_lost["v_world_y"]
            delta_vw = vw_post - vw_pre

            vb_pre = r_pre["v_body_y"]
            vb_post = r_lost["v_body_y"]
            delta_vb = vb_post - vb_pre

            tilt_contribution = (vb_post - vw_post) - (vb_pre - vw_pre)

            print(f"|  {s:2d}  | {cond_label} |   {vw_pre:+7.3f} m/s  |     {vw_post:+7.3f} m/s    |     {delta_vw:+7.3f} m/s    |  {vb_pre:+7.3f} m/s  |    {vb_post:+7.3f} m/s    |     {delta_vb:+7.3f} m/s   |         {tilt_contribution:+7.3f} m/s        |")


if __name__ == '__main__':
    main()
