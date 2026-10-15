"""
MyRobot Jump Physics Feasibility Test Suite
-------------------------------------------
Evaluates whether the MyRobot dual-wheel-legged robot can achieve genuine wheel lift-off
under strictly enforced URDF torque, velocity, joint limits, and PhysX contact dynamics.

No neural network or PPO training is involved.
Tests open-loop and closed-loop physical push-off control regimes.

Saves telemetry CSV files to evaluation_assets/jump_feasibility/
"""

import os
import sys
import math
import csv
import numpy as np

# Ensure proper Isaac Gym and torch import order
import isaacgym
from isaacgym import gymtorch, gymapi
import torch

from legged_gym.envs import *
from legged_gym.utils import task_registry


def quat_to_euler(q):
    """Convert quaternion [x, y, z, w] to euler angles [roll, pitch, yaw] in radians."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class JumpFeasibilityRunner:
    def __init__(self, output_dir="evaluation_assets/jump_feasibility"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Get configurations
        self.env_cfg, _ = task_registry.get_cfgs(name="myrobot")
        self.env_cfg.env.num_envs = 1
        self.env_cfg.sim.physx.num_threads = 4
        self.env_cfg.domain_rand.push_robots = False
        self.env_cfg.domain_rand.randomize_friction = False
        self.env_cfg.domain_rand.randomize_base_mass = False
        self.env_cfg.noise.add_noise = False

        # Create environment
        self.env, _ = task_registry.make_env(name="myrobot", env_cfg=self.env_cfg)

        # Acquire rigid body state tensor
        _rb_tensor = self.env.gym.acquire_rigid_body_state_tensor(self.env.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_tensor).view(self.env.num_envs, -1, 13)

        self.wheel_radius = 0.070  # meters
        self.left_wheel_body = self.env.feet_indices[0].item()
        self.right_wheel_body = self.env.feet_indices[1].item()

        # Joint indices
        self.left_hip = self.env.left_hip_idx
        self.left_knee = self.env.left_knee_idx
        self.right_hip = self.env.right_hip_idx
        self.right_knee = self.env.right_knee_idx
        self.left_wheel = self.env.left_wheel_idx
        self.right_wheel = self.env.right_wheel_idx

        self.torque_limits = self.env.torque_limits.cpu().numpy()
        self.dof_vel_limits = np.array([5.0, 5.0, 30.0, 5.0, 5.0, 30.0])  # rad/s from URDF

    def reset_to_crouch(self, crouch_height=0.36):
        """Reset the robot and settle into a stable crouch configuration."""
        self.env.reset()
        self.env.height_cmd_buf[:] = crouch_height
        self.env.target_height_buf[:] = crouch_height

        # Settle for 1.5 seconds (75 control steps at 50Hz)
        zero_actions = torch.zeros(self.env.num_envs, self.env.num_actions, device=self.env.device)
        for _ in range(75):
            self.env.step(zero_actions)

    def run_scenario(self, scenario_name, control_fn, duration_s=2.5):
        """
        Runs a simulation scenario.
        control_fn(t, step_i, env, rb_states) returns:
            custom_torques (Tensor of shape [num_envs, 6]) or None
            actions (Tensor of shape [num_envs, 6]) if custom_torques is None
        """
        print(f"\n==========================================")
        print(f"RUNNING SCENARIO: {scenario_name}")
        print(f"==========================================")

        self.reset_to_crouch(crouch_height=0.36)

        num_steps = int(duration_s / self.env.dt)
        dt = self.env.dt

        telemetry = []
        lift_off_detected = False
        lift_off_start_t = None
        max_clearance_L = 0.0
        max_clearance_R = 0.0
        flight_duration = 0.0
        takeoff_vz = 0.0
        peak_impact_force = 0.0
        terminal_failure = False
        failure_reason = "None"

        for step_i in range(num_steps):
            t = step_i * dt

            # Refresh states
            self.env.gym.refresh_rigid_body_state_tensor(self.env.sim)
            self.env.gym.refresh_dof_state_tensor(self.env.sim)
            self.env.gym.refresh_net_contact_force_tensor(self.env.sim)

            # Wheel kinematics
            pos_L = self.rb_states[0, self.left_wheel_body, :3].cpu().numpy()
            pos_R = self.rb_states[0, self.right_wheel_body, :3].cpu().numpy()
            root_pos = self.env.root_states[0, :3].cpu().numpy()
            root_quat = self.env.root_states[0, 3:7].cpu().numpy()
            root_lin_vel = self.env.root_states[0, 7:10].cpu().numpy()
            root_ang_vel = self.env.root_states[0, 10:13].cpu().numpy()

            roll, pitch, yaw = quat_to_euler(root_quat)

            clearance_L = max(0.0, float(pos_L[2] - self.wheel_radius))
            clearance_R = max(0.0, float(pos_R[2] - self.wheel_radius))
            min_clearance = min(clearance_L, clearance_R)

            # Contact forces on wheels
            f_contact_L = float(self.env.contact_forces[0, self.left_wheel_body, 2].item())
            f_contact_R = float(self.env.contact_forces[0, self.right_wheel_body, 2].item())

            # Non-wheel contacts
            non_wheel_force = 0.0
            for idx in self.env.termination_contact_indices:
                non_wheel_force += float(torch.norm(self.env.contact_forces[0, idx, :]).item())

            # DOF states
            dof_pos = self.env.dof_pos[0].cpu().numpy()
            dof_vel = self.env.dof_vel[0].cpu().numpy()

            # Execute control
            custom_torques, actions = control_fn(t, step_i, self.env, self.rb_states)

            if custom_torques is not None:
                # Strictly clamp to URDF torque limits
                applied_torques = torch.clamp(custom_torques, -self.env.torque_limits, self.env.torque_limits)
                # Step simulation sub-steps manually
                for _ in range(self.env.cfg.control.decimation):
                    self.env.torques = applied_torques.clone()
                    self.env.gym.set_dof_actuation_force_tensor(self.env.sim, gymtorch.unwrap_tensor(applied_torques))
                    self.env.gym.simulate(self.env.sim)
                    self.env.gym.refresh_dof_state_tensor(self.env.sim)
                self.env.post_physics_step()
                torques_record = applied_torques[0].cpu().numpy()
            else:
                self.env.step(actions)
                torques_record = self.env.torques[0].cpu().numpy()

            # Track flight & lift-off
            # Lift-off condition: both wheels have clearance > 5mm AND normal contact force < 1.0 N
            is_airborne = (clearance_L > 0.005 and clearance_R > 0.005 and f_contact_L < 1.0 and f_contact_R < 1.0)

            if is_airborne:
                if not lift_off_detected:
                    lift_off_detected = True
                    lift_off_start_t = t
                    takeoff_vz = float(root_lin_vel[2])
                    print(f"  [Take-off Detected] t = {t:.3f} s, v_z = {takeoff_vz:.3f} m/s")
                flight_duration += dt
                max_clearance_L = max(max_clearance_L, clearance_L)
                max_clearance_R = max(max_clearance_R, clearance_R)
            elif lift_off_detected and flight_duration > 0.02:
                # Landed
                if f_contact_L > 5.0 or f_contact_R > 5.0:
                    peak_impact_force = max(peak_impact_force, max(f_contact_L, f_contact_R))

            # Failure checks
            if non_wheel_force > 1.0 and not terminal_failure:
                terminal_failure = True
                failure_reason = f"Non-wheel ground contact ({non_wheel_force:.1f} N)"
            if abs(pitch) > math.radians(45.0) and not terminal_failure:
                terminal_failure = True
                failure_reason = f"Severe pitch tilt ({math.degrees(pitch):.1f} deg)"
            if abs(roll) > math.radians(45.0) and not terminal_failure:
                terminal_failure = True
                failure_reason = f"Severe roll tilt ({math.degrees(roll):.1f} deg)"
            if root_pos[2] < 0.20 and not terminal_failure:
                terminal_failure = True
                failure_reason = f"Chassis collapsed (height {root_pos[2]:.3f} m < 0.20m)"

            # Record telemetry
            telemetry.append({
                "time": t,
                "root_z": root_pos[2],
                "root_vz": root_lin_vel[2],
                "roll_deg": math.degrees(roll),
                "pitch_deg": math.degrees(pitch),
                "yaw_deg": math.degrees(yaw),
                "pitch_rate": root_ang_vel[0],
                "wheel_clearance_L": clearance_L,
                "wheel_clearance_R": clearance_R,
                "contact_force_L": f_contact_L,
                "contact_force_R": f_contact_R,
                "is_airborne": int(is_airborne),
                "hip_pos_L": dof_pos[self.left_hip],
                "knee_pos_L": dof_pos[self.left_knee],
                "hip_vel_L": dof_vel[self.left_hip],
                "knee_vel_L": dof_vel[self.left_knee],
                "hip_tau_L": torques_record[self.left_hip],
                "knee_tau_L": torques_record[self.left_knee],
                "wheel_tau_L": torques_record[self.left_wheel],
                "non_wheel_contact": non_wheel_force,
            })

        # Save CSV
        csv_file = os.path.join(self.output_dir, f"{scenario_name}.csv")
        if telemetry:
            keys = list(telemetry[0].keys())
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(telemetry)
            print(f"  Saved telemetry log to: {csv_file}")

        # Classify outcome
        max_clearance = max(max_clearance_L, max_clearance_R)
        if max_clearance < 0.005 or not lift_off_detected:
            classification = "Case 1: Only body rises, wheels stay on ground"
        elif terminal_failure:
            classification = f"Case 2: Brief lift-off ({max_clearance*1000:.1f} mm), but landing unstable ({failure_reason})"
        else:
            classification = f"Case 3: Genuine lift-off ({max_clearance*1000:.1f} mm), and successful landing recovery"

        summary = {
            "scenario": scenario_name,
            "classification": classification,
            "max_clearance_mm": max_clearance * 1000.0,
            "flight_duration_ms": flight_duration * 1000.0,
            "takeoff_vz_mps": takeoff_vz,
            "peak_impact_N": peak_impact_force,
            "terminal_failure": terminal_failure,
            "failure_reason": failure_reason,
        }

        print(f"  --> Result: {classification}")
        print(f"      Max Clearance: {summary['max_clearance_mm']:.1f} mm")
        print(f"      Flight Time:   {summary['flight_duration_ms']:.1f} ms")
        print(f"      Take-off Vz:   {summary['takeoff_vz_mps']:.3f} m/s")
        print(f"      Peak Impact:   {summary['peak_impact_N']:.1f} N")

        return summary


def run_all_tests():
    runner = JumpFeasibilityRunner()

    results = []

    # =========================================================================
    # Scenario 1: Existing Baseline Controller (Height Ramp 0.04m/s, Kp=40, Kd=2.0)
    # =========================================================================
    def ctrl_scen1(t, step_i, env, rb):
        # Target height set to 0.52m at t = 0.2s, let existing ramp handle it
        if t >= 0.2:
            env.target_height_buf[:] = 0.52
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        return None, actions

    res1 = runner.run_scenario("scenario_1_baseline_height_ramp", ctrl_scen1, duration_s=2.0)
    results.append(res1)

    # =========================================================================
    # Scenario 2: Existing PD with Full Residual Action (Kp=40, Kd=2.0, action=1.0)
    # =========================================================================
    def ctrl_scen2(t, step_i, env, rb):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        if 0.2 <= t < 0.4:
            # Full positive knee action
            actions[:, env.left_knee_idx] = 1.0
            actions[:, env.right_knee_idx] = 1.0
        return None, actions

    res2 = runner.run_scenario("scenario_2_existing_pd_full_action", ctrl_scen2, duration_s=2.0)
    results.append(res2)

    # =========================================================================
    # Scenario 3: Instantaneous Step IK Target with Existing PD (Kp=40, Kd=2.0)
    # =========================================================================
    def ctrl_scen3(t, step_i, env, rb):
        if t >= 0.2:
            # Bypass slow ramp by setting height_cmd_buf directly to 0.58m
            env.height_cmd_buf[:] = 0.58
            env.target_height_buf[:] = 0.58
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        return None, actions

    res3 = runner.run_scenario("scenario_3_instant_step_ik_existing_pd", ctrl_scen3, duration_s=2.0)
    results.append(res3)

    # =========================================================================
    # Scenario 4: High-Gain PD Extension (Kp=200, Kd=5.0, strict URDF limits)
    # =========================================================================
    def ctrl_scen4(t, step_i, env, rb):
        torques = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        kp_leg = 200.0
        kd_leg = 5.0

        if t < 0.2:
            h_des = 0.36
        elif t < 0.5:
            h_des = 0.58
        else:
            h_des = 0.40  # Land / recover height

        q1_des, q2_des = env.compute_leg_ik(torch.tensor([h_des], device=env.device))
        targets = {
            env.left_hip_idx: q1_des,
            env.right_hip_idx: q1_des,
            env.left_knee_idx: q2_des,
            env.right_knee_idx: q2_des,
        }

        for idx in [env.left_hip_idx, env.left_knee_idx, env.right_hip_idx, env.right_knee_idx]:
            tgt = targets[idx]
            pos_err = tgt - env.dof_pos[:, idx]
            vel = env.dof_vel[:, idx]
            torques[:, idx] = kp_leg * pos_err - kd_leg * vel

        # Wheel velocity damping
        for widx in [env.left_wheel_idx, env.right_wheel_idx]:
            torques[:, widx] = -0.5 * env.dof_vel[:, widx]

        return torques, None

    res4 = runner.run_scenario("scenario_4_high_gain_pd_extension", ctrl_scen4, duration_s=2.0)
    results.append(res4)

    # =========================================================================
    # Scenario 5: Impulsive Feedforward Torque Push-off (Strict URDF limits: Knee 55 Nm)
    # =========================================================================
    def ctrl_scen5(t, step_i, env, rb):
        torques = torch.zeros(env.num_envs, env.num_actions, device=env.device)

        if t < 0.2:
            # Stand at crouch 0.36m
            q1_des, q2_des = env.compute_leg_ik(torch.tensor([0.36], device=env.device))
            for idx, q_d in [(env.left_hip_idx, q1_des), (env.right_hip_idx, q1_des),
                             (env.left_knee_idx, q2_des), (env.right_knee_idx, q2_des)]:
                torques[:, idx] = 120.0 * (q_d - env.dof_pos[:, idx]) - 4.0 * env.dof_vel[:, idx]
        elif 0.2 <= t < 0.30:  # 100 ms impulsive push-off
            # Push-off: knee extends with 55 Nm (within 60 Nm URDF limit)
            # Knee extension is negative q2 velocity / negative torque
            torques[:, env.left_knee_idx] = -55.0
            torques[:, env.right_knee_idx] = -55.0
            # Hip maintains sagittal balance: PD on pitch
            pitch_rate = env.base_ang_vel[:, 0]
            pitch_err = env.projected_gravity[:, 1]
            tau_hip_balance = -40.0 * pitch_err - 2.0 * pitch_rate
            torques[:, env.left_hip_idx] = tau_hip_balance
            torques[:, env.right_hip_idx] = tau_hip_balance
        elif 0.30 <= t < 0.50:  # Airborne phase: retract legs slightly to 0.45m
            q1_air, q2_air = env.compute_leg_ik(torch.tensor([0.45], device=env.device))
            for idx, q_d in [(env.left_hip_idx, q1_air), (env.right_hip_idx, q1_air),
                             (env.left_knee_idx, q2_air), (env.right_knee_idx, q2_air)]:
                torques[:, idx] = 80.0 * (q_d - env.dof_pos[:, idx]) - 3.0 * env.dof_vel[:, idx]
            # Active wheel spin in air to stabilize pitch (if not blocked by CACM)
            pitch_rate = env.base_ang_vel[:, 0]
            torques[:, env.left_wheel_idx] = torch.clamp(-0.5 * pitch_rate, -2.0, 2.0)
            torques[:, env.right_wheel_idx] = torch.clamp(-0.5 * pitch_rate, -2.0, 2.0)
        else:  # Landing & recovery phase: soft touchdown compliance
            q1_land, q2_land = env.compute_leg_ik(torch.tensor([0.38], device=env.device))
            for idx, q_d in [(env.left_hip_idx, q1_land), (env.right_hip_idx, q1_land),
                             (env.left_knee_idx, q2_land), (env.right_knee_idx, q2_land)]:
                torques[:, idx] = 100.0 * (q_d - env.dof_pos[:, idx]) - 6.0 * env.dof_vel[:, idx]
            # Wheel damping for landing stability
            for widx in [env.left_wheel_idx, env.right_wheel_idx]:
                torques[:, widx] = -0.5 * env.dof_vel[:, widx]

        return torques, None

    res5 = runner.run_scenario("scenario_5_impulsive_feedforward_pushoff", ctrl_scen5, duration_s=2.0)
    results.append(res5)

    print("\n" + "="*80)
    print("PHYSICAL FEASIBILITY TEST SUITE SUMMARY")
    print("="*80)
    print(f"{'Scenario':<38} | {'Classification':<42} | {'Max Clr (mm)':<12} | {'Vz (m/s)':<8} | {'T_air (ms)':<10}")
    print("-" * 115)
    for r in results:
        print(f"{r['scenario']:<38} | {r['classification']:<42} | {r['max_clearance_mm']:<12.1f} | {r['takeoff_vz_mps']:<8.2f} | {r['flight_duration_ms']:<10.1f}")
    print("="*80)


if __name__ == '__main__':
    run_all_tests()
