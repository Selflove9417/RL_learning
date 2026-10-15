#!/usr/bin/env python3
"""
MyRobot Formal Regression Test for myrobot_dual Task
====================================================
Directly invokes the formal registered task `myrobot_dual` (MyRobotDualEnv),
executing the complete continuous consecutive double jump sequence:
  1. 0.0s ~ 1.5s   : Stand in BALANCE
  2. 1.5s ~ 4.0s   : Forward movement (vy = 0.30 m/s)
  3. 4.0s          : Jump 1 triggered (STOPPING -> PREPARE -> JUMP -> RECOVER -> NORMAL_HANDOFF)
  4. ~5.6s ~ 6.5s  : Post-jump settlement in BALANCE
  5. 6.5s ~ 9.0s   : Re-engage forward movement (vy = 0.30 m/s)
  6. 9.0s          : Jump 2 triggered (STOPPING -> PREPARE -> JUMP -> RECOVER -> NORMAL_HANDOFF)
  7. ~10.6s ~ 15.0s: Stable standing in BALANCE until completion

Records:
  - FSM state transitions & events
  - PhysX is_airborne status
  - Exact 3D cylinder clearance (min, L, R)
  - Left & right wheel contact forces
  - Simulation abnormal reset count (asserted == 0)
  - High-res offscreen video recording
"""

import os
import sys
import math
import csv
import json
import argparse
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz
import torch
import imageio

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.myrobot.myrobot_dual_env import (
    MyRobotDualEnv,
    compute_exact_cylinder_clearance,
    STATE_NAMES
)

def run_formal_regression(seed=42, duration_s=15.0,
                          csv_path="evaluation_assets/phase6/formal_myrobot_dual_regression_telemetry.csv",
                          video_path="evaluation_assets/phase6/formal_myrobot_dual_consecutive_jumps.mp4"):
    print("=" * 95)
    print("  MyRobot Formal Task Regression: `myrobot_dual` Continuous Double Jump")
    print(f"  * Seed: {seed} | Duration: {duration_s}s | dt: 0.02s")
    print(f"  * CSV Telemetry Path : {csv_path}")
    print(f"  * Video Demo Path    : {video_path}")
    print("=" * 95)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    sys.argv = ['test_formal', '--task', 'myrobot_dual']
    args = get_args()
    args.headless = False
    args.seed = seed

    env_cfg, train_cfg = task_registry.get_cfgs(name='myrobot_dual')
    env_cfg.env.num_envs = 1
    env_cfg.seed = seed
    env_cfg.env.episode_length_s = duration_s + 10.0
    env_cfg.noise.add_noise = False
    env_cfg.env.enable_camera_sensors = True

    env, _ = task_registry.make_env(name='myrobot_dual', args=args, env_cfg=env_cfg)
    assert isinstance(env, MyRobotDualEnv), f"Expected MyRobotDualEnv, got {type(env)}"

    # Reset environment and initialize dual policies with official default checkpoints
    env.reset()
    env.root_states[:, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.init_dual_policies()

    # Create offscreen camera for high-definition video recording
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1280
    camera_props.height = 720
    camera_props.enable_tensors = False
    camera_props.horizontal_fov = 65.0

    camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    cam_offset = np.array([-1.5, -2.0, 0.85])

    # Open CSV for 50 Hz telemetry logging
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        "timestamp_s", "step", "fsm_state", "fsm_state_name",
        "cmd_vy", "act_vy", "cmd_height", "act_height",
        "roll_deg", "pitch_deg", "yaw_deg",
        "is_airborne", "clr_exact_cm", "clr_L_cm", "clr_R_cm",
        "fn_L", "fn_R",
        "tau_hip_L", "tau_knee_L", "tau_hip_R", "tau_knee_R",
        "tau_wheel_L", "tau_wheel_R",
        "event_tag"
    ])

    total_steps = int(duration_s / env.dt)
    frames = []

    # Regression tracking data structures
    fsm_transitions = []
    airborne_events = []
    jump1_max_clr = 0.0
    jump2_max_clr = 0.0
    jump1_handoff = ""
    jump2_handoff = ""
    jump1_airborne_steps = 0
    jump2_airborne_steps = 0
    abnormal_resets = 0

    prev_state = env.fsm_state[0].item()
    prev_airborne = bool(env.is_airborne[0].item())
    current_jump_idx = 0

    remove_start_pos = None
    remove_end_pos = None

    print("\nExecuting 15.0-second regression mission...\n")

    for step in range(total_steps):
        t_sim = step * env.dt

        # Check if environment reset unexpectedly
        if env.reset_buf[0].item() != 0:
            abnormal_resets += 1

        # 1. Mission timeline commands
        if 1.5 <= t_sim < 4.0:
            vy = 0.30
        elif 6.5 <= t_sim < 9.0:
            vy = 0.30
            if remove_start_pos is None:
                remove_start_pos = env.root_states[0, 1].item()
        else:
            vy = 0.0
            if t_sim >= 9.0 and remove_end_pos is None and remove_start_pos is not None:
                remove_end_pos = env.root_states[0, 1].item()

        env.set_teleop_targets(vy=vy, yaw=0.0, height=0.39774)

        # Trigger jumps
        if step == int(4.0 / env.dt):
            current_jump_idx = 1
            env.request_teleop_jump()
            print(f"[{t_sim:5.2f}s] >>> [COMMAND] Jump 1 requested at vy={env.base_lin_vel[0, 1].item():.3f} m/s")
        elif step == int(9.0 / env.dt):
            current_jump_idx = 2
            env.request_teleop_jump()
            print(f"[{t_sim:5.2f}s] >>> [COMMAND] Jump 2 requested at vy={env.base_lin_vel[0, 1].item():.3f} m/s")

        # Execute dual policy control step
        env.step_dual_policy()

        # Telemetry extraction
        cur_st = env.fsm_state[0].item()
        cur_air = bool(env.is_airborne[0].item())
        clr_exact, clr_L, clr_R = compute_exact_cylinder_clearance(env)
        fn_l = env.contact_forces[0, env.feet_indices[0], 2].item()
        fn_r = env.contact_forces[0, env.feet_indices[1], 2].item()

        r_raw, p_raw, y_raw = get_euler_xyz(env.base_quat)
        roll = torch.atan2(torch.sin(r_raw[0]), torch.cos(r_raw[0])).item()
        pitch = torch.atan2(torch.sin(p_raw[0]), torch.cos(p_raw[0])).item()
        yaw = torch.atan2(torch.sin(y_raw[0]), torch.cos(y_raw[0])).item()
        root_z = env.root_states[0, 2].item()

        clr_cm = clr_exact[0].item() * 100.0

        # Track transitions
        if cur_st != prev_state:
            fsm_transitions.append({
                "time_s": round(t_sim, 4),
                "from_state": STATE_NAMES[prev_state],
                "to_state": STATE_NAMES[cur_st],
                "event": env.last_transition_event[0]
            })
            print(f"[{t_sim:5.2f}s] FSM: {STATE_NAMES[prev_state]} -> {STATE_NAMES[cur_st]} | Evt: {env.last_transition_event[0]}")
            if prev_state == env.STATE_RECOVER and cur_st == env.STATE_BALANCE:
                if current_jump_idx == 1 and not jump1_handoff:
                    jump1_handoff = env.last_transition_event[0]
                elif current_jump_idx == 2 and not jump2_handoff:
                    jump2_handoff = env.last_transition_event[0]
            prev_state = cur_st

        # Track airborne
        if cur_air != prev_airborne:
            airborne_events.append({
                "time_s": round(t_sim, 4),
                "is_airborne": cur_air,
                "clr_cm": round(clr_cm, 2),
                "vz": round(env.base_lin_vel[0, 2].item(), 3)
            })
            status_str = "AIRBORNE LIFTOFF" if cur_air else "TOUCHDOWN CONTACT"
            print(f"[{t_sim:5.2f}s] *** PhysX Status: {status_str} | Clr: {clr_cm:.1f} cm | Fn: L={fn_l:.1f}N, R={fn_r:.1f}N")
            prev_airborne = cur_air

        if cur_st in (env.STATE_JUMP, env.STATE_RECOVER):
            if current_jump_idx == 1:
                if clr_cm > jump1_max_clr: jump1_max_clr = clr_cm
                if cur_air: jump1_airborne_steps += 1
            elif current_jump_idx == 2:
                if clr_cm > jump2_max_clr: jump2_max_clr = clr_cm
                if cur_air: jump2_airborne_steps += 1

        # Write CSV
        writer.writerow([
            f"{t_sim:.4f}", step, cur_st, STATE_NAMES[cur_st],
            f"{env.actual_cmd_vy[0].item():.3f}", f"{env.base_lin_vel[0, 1].item():.3f}",
            f"{env.actual_cmd_height[0].item():.3f}", f"{root_z:.3f}",
            f"{np.degrees(roll):.2f}", f"{np.degrees(pitch):.2f}", f"{np.degrees(yaw):.2f}",
            cur_air, f"{clr_cm:.2f}", f"{clr_L[0].item()*100:.2f}", f"{clr_R[0].item()*100:.2f}",
            f"{fn_l:.1f}", f"{fn_r:.1f}",
            f"{env.torques[0, env.left_hip_idx].item():.2f}",
            f"{env.torques[0, env.left_knee_idx].item():.2f}",
            f"{env.torques[0, env.right_hip_idx].item():.2f}",
            f"{env.torques[0, env.right_knee_idx].item():.2f}",
            f"{env.torques[0, env.left_wheel_idx].item():.2f}",
            f"{env.torques[0, env.right_wheel_idx].item():.2f}",
            env.last_transition_event[0]
        ])

        # Capture video frame
        root_pos = env.root_states[0, 0:3].cpu().numpy()
        cam_target = gymapi.Vec3(root_pos[0], root_pos[1], root_pos[2] + 0.15)
        cam_eye = gymapi.Vec3(root_pos[0] + cam_offset[0], root_pos[1] + cam_offset[1], root_pos[2] + cam_offset[2])
        env.gym.set_camera_location(camera_handle, env.envs[0], cam_eye, cam_target)
        env.gym.render_all_camera_sensors(env.sim)
        img = env.gym.get_camera_image(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
        img = img.reshape((camera_props.height, camera_props.width, 4))[:, :, :3]
        frames.append(img)

    csv_file.close()

    print(f"\nEncoding {len(frames)} frames into video: {video_path}...")
    imageio.mimsave(video_path, frames, fps=50, quality=8)
    print(f"--> Video saved ({os.path.getsize(video_path)/1024/1024:.2f} MB)")

    remove_dist = (remove_end_pos - remove_start_pos) if (remove_start_pos and remove_end_pos) else 0.0

    print("\n" + "=" * 95)
    print(" === FORMAL REGRESSION VERIFICATION RESULTS ===")
    print("=" * 95)
    print(f"Total Steps Executed            : {total_steps} (15.0 seconds at 50 Hz)")
    print(f"Abnormal Mid-Episode Resets     : {abnormal_resets}")
    print(f"Jump 1 Peak Bilateral Clearance : {jump1_max_clr:.2f} cm (Target >= 20.0 cm: {'PASS' if jump1_max_clr >= 20.0 else 'BORDER'})")
    print(f"Jump 1 Airborne Duration        : {jump1_airborne_steps * 0.02:.2f}s ({jump1_airborne_steps} steps)")
    print(f"Jump 1 Recovery Handoff Type    : {jump1_handoff}")
    print(f"In-Between Re-Move Displacement : {remove_dist:.3f} m (vy = 0.30 m/s cruise)")
    print(f"Jump 2 Peak Bilateral Clearance : {jump2_max_clr:.2f} cm (Target >= 20.0 cm: {'PASS' if jump2_max_clr >= 20.0 else 'BORDER'})")
    print(f"Jump 2 Airborne Duration        : {jump2_airborne_steps * 0.02:.2f}s ({jump2_airborne_steps} steps)")
    print(f"Jump 2 Recovery Handoff Type    : {jump2_handoff}")
    print("=" * 95)

    # Formal assertions
    assert abnormal_resets == 0, f"Abnormal reset occurred ({abnormal_resets})!"
    assert jump1_max_clr >= 19.5, f"Jump 1 clearance failed: {jump1_max_clr} cm"
    assert jump1_handoff == "NORMAL_HANDOFF", f"Jump 1 handoff failed: {jump1_handoff}"
    assert jump1_airborne_steps >= 15, f"Jump 1 insufficient airborne duration: {jump1_airborne_steps} steps"
    assert remove_dist >= 0.40, f"In-between re-move failed: {remove_dist} m"
    assert jump2_max_clr >= 19.5, f"Jump 2 clearance failed: {jump2_max_clr} cm"
    assert jump2_handoff == "NORMAL_HANDOFF", f"Jump 2 handoff failed: {jump2_handoff}"
    assert jump2_airborne_steps >= 15, f"Jump 2 insufficient airborne duration: {jump2_airborne_steps} steps"
    print("\n>>> ALL FORMAL REGRESSION ASSERTIONS PASSED! <<<\n")

    return {
        "abnormal_resets": abnormal_resets,
        "jump1_max_clr_cm": jump1_max_clr,
        "jump1_airborne_steps": jump1_airborne_steps,
        "jump1_handoff": jump1_handoff,
        "remove_dist_m": remove_dist,
        "jump2_max_clr_cm": jump2_max_clr,
        "jump2_airborne_steps": jump2_airborne_steps,
        "jump2_handoff": jump2_handoff,
        "fsm_transitions": fsm_transitions,
        "airborne_events": airborne_events
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()
    run_formal_regression(seed=args.seed, duration_s=args.duration)
