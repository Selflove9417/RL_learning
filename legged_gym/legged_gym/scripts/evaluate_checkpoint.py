"""Deterministic, headless checkpoint evaluation for the myrobot task."""

import json
import math
import os

import isaacgym  # noqa: F401  # Must be imported before legged_gym modules.
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403  # Registers tasks.
from legged_gym.utils import get_args, get_load_path, task_registry


SCENARIOS = {"straight", "stand", "ramp", "stop", "reverse", "turn", "teleop"}


def _targets(scenario, time_s):
    """Return target forward velocity and yaw rate for a repeatable scenario."""
    if scenario == "straight":
        return 0.39, 0.0
    if scenario == "stand":
        return 0.0, 0.0
    if scenario == "ramp":
        return (0.0, 0.0) if time_s < 1.0 else (0.40, 0.0)
    if scenario == "stop":
        if time_s < 1.0:
            return 0.0, 0.0
        return (0.40, 0.0) if time_s < 6.0 else (0.0, 0.0)
    if scenario == "reverse":
        return (0.0, 0.0) if time_s < 1.0 else (-0.30, 0.0)
    if scenario == "turn":
        return (0.0, 0.0) if time_s < 1.0 else (0.0, 0.50)
    # Teleop-like sequence: stand, forward, stop, reverse, stop, turn, stop.
    if time_s < 1.0:
        return 0.0, 0.0
    if time_s < 6.0:
        return 0.40, 0.0
    if time_s < 9.0:
        return 0.0, 0.0
    if time_s < 13.0:
        return -0.30, 0.0
    if time_s < 15.0:
        return 0.0, 0.0
    if time_s < 18.0:
        return 0.0, 0.50
    return 0.0, 0.0


def _slew(current, target, max_rate, dt):
    delta = target - current
    return current + max(-max_rate * dt, min(max_rate * dt, delta))


def _percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[index]


def _pearson_corr(x, y):
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - x_mean) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - y_mean) ** 2 for yi in y))
    if den_x * den_y < 1e-9:
        return 0.0
    return num / (den_x * den_y)


def evaluate(args):
    if args.eval_scenario not in SCENARIOS:
        raise ValueError(f"Unknown --eval_scenario={args.eval_scenario!r}; choose from {sorted(SCENARIOS)}")
    if args.eval_duration <= 0.0:
        raise ValueError("--eval_duration must be positive")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = max(env_cfg.env.episode_length_s, args.eval_duration + 1.0)
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    # Evaluation always measures the final deployable actuator mapping, never
    # the temporary training curriculum limit.
    env_cfg.control.wheel_action_limit_curriculum = False
    if hasattr(env_cfg.control, 'wheel_damping_curriculum'):
        env_cfg.control.wheel_damping_curriculum = False
    if hasattr(env_cfg.control, 'wheel_torque_curriculum'):
        env_cfg.control.wheel_torque_curriculum = False
    if hasattr(env_cfg.control, 'wheel_velocity_damping_curriculum'):
        env_cfg.control.wheel_velocity_damping_curriculum = False
    if hasattr(env_cfg.control, 'wheel_yaw_damping_curriculum'):
        env_cfg.control.wheel_yaw_damping_curriculum = False
    if hasattr(env_cfg.control, 'wheel_diff_damping_curriculum'):
        env_cfg.control.wheel_diff_damping_curriculum = False
    if hasattr(env_cfg.rewards, 'tracking_lin_vel_curriculum'):
        env_cfg.rewards.tracking_lin_vel_curriculum = False
    env_cfg.commands.resampling_time = 1.0e6
    env_cfg.seed = args.seed if args.seed is not None else train_cfg.seed
    train_cfg.seed = env_cfg.seed
    args.headless = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    checkpoint_path = get_load_path(log_root, args.load_run, args.checkpoint)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.device)

    observations = env.get_observations()
    command_v = 0.0
    command_yaw = 0.0
    steps = int(math.ceil(args.eval_duration / env.dt))
    warmup_steps = int(math.ceil(args.eval_warmup / env.dt))
    samples = {name: [] for name in (
        "velocity_error", "yaw_error", "pitch_deg", "height_m", "wheel_speed_diff",
        "contact_imbalance", "conditional_contact_imbalance", "action_rate", "action_jerk",
        "wheel_action_requested", "wheel_action_applied", "wheel_action_diff",
        "wheel_torque_diff", "wheel_max_abs_vel", "wheel_counterdrive",
        "forward_vel", "roll_deg", "roll_rate_rad_s",
        "wheel_l_contact", "wheel_r_contact",
        "both_wheels_contact", "single_wheel_contact", "no_wheel_contact",
        "wheel_action_comm", "wheel_torque_comm", "wheel_comm_same_sign",
        "wheel_comm_saturated", "vel_tracking_error",
        "cruise_act_comm", "cruise_throttle_exceed", "diff_torque_clipped"
    )}
    actual_reward = 0.0
    wheel_saturated = 0.0
    wheel_requested_saturated = 0.0
    leg_saturated = 0.0
    wheel_torque_sq_sum = 0.0
    wheel_high_load = 0.0
    current_consecutive_high_load = 0
    max_consecutive_high_load_steps = 0
    measured_steps = 0
    early_terminations = 0

    for step in range(steps):
        time_s = step * env.dt
        target_v, target_yaw = _targets(args.eval_scenario, time_s)
        if args.eval_scenario == "straight":
            command_v, command_yaw = target_v, target_yaw
        else:
            command_v = _slew(command_v, target_v, 1.0, env.dt)
            command_yaw = _slew(command_yaw, target_yaw, 2.5, env.dt)

        env.commands[:, 0] = 0.0
        env.commands[:, 1] = command_v
        env.commands[:, 2] = command_yaw
        env.compute_observations()
        observations = env.get_observations()
        actions = policy(observations.detach())
        observations, _, rewards, dones, _ = env.step(actions.detach())

        if bool(dones[0].item()):
            early_terminations += 1
            break
        if step < warmup_steps:
            continue

        measured_steps += 1
        actual_reward += rewards[0].item()
        current_vel = env.base_lin_vel[0, 1].item()
        velocity_error = current_vel - command_v
        yaw_error = env.base_ang_vel[0, 2].item() - command_yaw
        gravity_y = max(-1.0, min(1.0, env.projected_gravity[0, 1].item()))
        pitch_deg = math.degrees(math.asin(gravity_y))
        gravity_x = max(-1.0, min(1.0, env.projected_gravity[0, 0].item()))
        roll_deg = math.degrees(math.asin(gravity_x))
        roll_rate_rad_s = env.base_ang_vel[0, 1].item()
        left_force = env.contact_forces[0, env.feet_indices[0], 2].item()
        right_force = env.contact_forces[0, env.feet_indices[1], 2].item()
        force_total = abs(left_force) + abs(right_force)

        is_left_contact = left_force > 1.0
        is_right_contact = right_force > 1.0

        samples["forward_vel"].append(current_vel)
        samples["velocity_error"].append(velocity_error)
        samples["yaw_error"].append(yaw_error)
        samples["pitch_deg"].append(pitch_deg)
        samples["roll_deg"].append(roll_deg)
        samples["roll_rate_rad_s"].append(roll_rate_rad_s)
        samples["height_m"].append(env.root_states[0, 2].item())
        samples["wheel_speed_diff"].append(
            env.dof_vel[0, env.left_wheel_idx].item() - env.dof_vel[0, env.right_wheel_idx].item()
        )
        samples["contact_imbalance"].append(
            abs(left_force - right_force) / force_total if force_total > 1.0e-6 else 1.0
        )
        if force_total >= 10.0:
            samples["conditional_contact_imbalance"].append(
                abs(left_force - right_force) / force_total
            )
        samples["wheel_l_contact"].append(1.0 if is_left_contact else 0.0)
        samples["wheel_r_contact"].append(1.0 if is_right_contact else 0.0)
        samples["both_wheels_contact"].append(1.0 if (is_left_contact and is_right_contact) else 0.0)
        samples["single_wheel_contact"].append(1.0 if (is_left_contact ^ is_right_contact) else 0.0)
        samples["no_wheel_contact"].append(1.0 if (not is_left_contact and not is_right_contact) else 0.0)
        samples["action_rate"].append(
            torch.linalg.vector_norm(env.filtered_actions[0] - env.last_filtered_actions[0]).item()
        )
        samples["action_jerk"].append(
            torch.linalg.vector_norm(
                env.filtered_actions[0] - 2.0 * env.last_filtered_actions[0] + env.last_last_filtered_actions[0]
            ).item()
        )
        samples["wheel_action_requested"].append(
            torch.max(torch.abs(env.requested_actions[0, env.wheel_indices])).item()
        )
        samples["wheel_action_applied"].append(
            torch.max(torch.abs(env.filtered_actions[0, env.wheel_indices])).item()
        )
        samples["wheel_action_diff"].append(
            abs(env.filtered_actions[0, env.left_wheel_idx].item() - env.filtered_actions[0, env.right_wheel_idx].item())
        )
        samples["wheel_torque_diff"].append(
            abs(env.torques[0, env.left_wheel_idx].item() - env.torques[0, env.right_wheel_idx].item())
        )
        samples["wheel_max_abs_vel"].append(
            max(abs(env.dof_vel[0, env.left_wheel_idx].item()), abs(env.dof_vel[0, env.right_wheel_idx].item()))
        )
        left_act = env.filtered_actions[0, env.left_wheel_idx].item()
        right_act = env.filtered_actions[0, env.right_wheel_idx].item()
        left_trq = env.torques[0, env.left_wheel_idx].item()
        right_trq = env.torques[0, env.right_wheel_idx].item()
        is_straight = abs(command_yaw) < 0.1
        samples["wheel_counterdrive"].append(
            1.0 if (is_straight and (left_act * right_act < -1.0e-4)) else 0.0
        )

        wheel_limits = env.torque_limits[env.wheel_indices]
        act_comm = 0.5 * (left_act + right_act)
        trq_comm = 0.5 * (left_trq + right_trq)
        v_err = command_v - current_vel
        is_same_sign = 1.0 if (abs(command_v) > 0.05 and act_comm * command_v > 0) else (1.0 if abs(act_comm) < 0.1 else 0.0)
        is_comm_sat = 1.0 if abs(trq_comm) >= (wheel_limits[0].item() - 1.0e-3) else 0.0
        samples["wheel_action_comm"].append(act_comm)
        samples["wheel_torque_comm"].append(trq_comm)
        samples["vel_tracking_error"].append(v_err)
        samples["wheel_comm_same_sign"].append(is_same_sign)
        samples["wheel_comm_saturated"].append(is_comm_sat)

        cmd_sign = 1.0 if command_v > 0 else (-1.0 if command_v < 0 else 0.0)
        is_nonzero_cmd = abs(command_v) > 0.05
        v_deficit = (command_v - current_vel) * cmd_sign if is_nonzero_cmd else 0.0
        is_cruise = is_nonzero_cmd and (v_deficit <= 0.10)
        if is_cruise:
            samples["cruise_act_comm"].append(abs(act_comm))
            samples["cruise_throttle_exceed"].append(1.0 if abs(act_comm) > 0.70 else 0.0)

        # 差动力矩被 URDF 裁剪判定
        is_diff_clipped = env.wheel_diff_clipped[0].item() if hasattr(env, "wheel_diff_clipped") else (
            torch.any(torch.abs(env.unclipped_torques[0, env.wheel_indices]) >= wheel_limits - 1.0e-3).item()
        )
        samples["diff_torque_clipped"].append(1.0 if is_diff_clipped else 0.0)

        leg_limits = env.torque_limits[env.leg_indices]
        wheel_saturated += (torch.abs(env.torques[0, env.wheel_indices]) >= wheel_limits - 1.0e-3).float().mean().item()
        wheel_requested_saturated += (torch.abs(env.unclipped_torques[0, env.wheel_indices]) >= wheel_limits).float().mean().item()
        leg_saturated += (torch.abs(env.torques[0, env.leg_indices]) >= leg_limits - 1.0e-3).float().mean().item()

        wheel_norm_sq = torch.mean(torch.square(env.torques[0, env.wheel_indices] / wheel_limits)).item()
        wheel_torque_sq_sum += wheel_norm_sq
        is_high_load = (torch.abs(env.torques[0, env.wheel_indices]) >= 0.85 * wheel_limits)
        wheel_high_load += is_high_load.float().mean().item()
        if torch.any(is_high_load).item():
            current_consecutive_high_load += 1
            if current_consecutive_high_load > max_consecutive_high_load_steps:
                max_consecutive_high_load_steps = current_consecutive_high_load
        else:
            current_consecutive_high_load = 0

    denominator = max(1, measured_steps)
    abs_yaw = [abs(value) for value in samples["yaw_error"]]
    abs_pitch = [abs(value) for value in samples["pitch_deg"]]
    abs_roll = [abs(value) for value in samples["roll_deg"]]
    abs_roll_rate = [abs(value) for value in samples["roll_rate_rad_s"]]
    metrics = {
        "checkpoint": checkpoint_path,
        "seed": env_cfg.seed,
        "scenario": args.eval_scenario,
        "duration_s": args.eval_duration,
        "warmup_s": args.eval_warmup,
        "measured_steps": measured_steps,
        "early_terminations": early_terminations,
        "actual_reward_per_second": actual_reward / (denominator * env.dt),
        "velocity_rmse_m_s": math.sqrt(sum(value * value for value in samples["velocity_error"]) / denominator),
        "velocity_error_peak_to_peak_m_s": (
            max(samples["velocity_error"]) - min(samples["velocity_error"])
            if samples["velocity_error"] else 0.0
        ),
        "vel_mean_m_s": sum(samples["forward_vel"]) / denominator if samples["forward_vel"] else 0.0,
        "vel_p2p_m_s": (max(samples["forward_vel"]) - min(samples["forward_vel"])) if samples["forward_vel"] else 0.0,
        "vel_min_m_s": min(samples["forward_vel"], default=0.0),
        "vel_max_m_s": max(samples["forward_vel"], default=0.0),
        "yaw_error_p95_rad_s": _percentile(abs_yaw, 0.95),
        "yaw_error_max_rad_s": max(abs_yaw, default=0.0),
        "pitch_p95_deg": _percentile(abs_pitch, 0.95),
        "pitch_max_deg": max(abs_pitch, default=0.0),
        "roll_p95_deg": _percentile(abs_roll, 0.95),
        "roll_max_deg": max(abs_roll, default=0.0),
        "roll_rate_p95_rad_s": _percentile(abs_roll_rate, 0.95),
        "roll_rate_max_rad_s": max(abs_roll_rate, default=0.0),
        "height_min_m": min(samples["height_m"], default=0.0),
        "height_max_m": max(samples["height_m"], default=0.0),
        "wheel_speed_diff_p95_rad_s": _percentile([abs(v) for v in samples["wheel_speed_diff"]], 0.95),
        "contact_imbalance_p95": _percentile(samples["contact_imbalance"], 0.95),
        "conditional_contact_imbalance_p95": (
            _percentile(samples["conditional_contact_imbalance"], 0.95)
            if samples["conditional_contact_imbalance"] else 0.0
        ),
        "conditional_contact_imbalance_mean": (
            sum(samples["conditional_contact_imbalance"]) / len(samples["conditional_contact_imbalance"])
            if samples["conditional_contact_imbalance"] else 0.0
        ),
        "conditional_contact_imbalance_count": len(samples["conditional_contact_imbalance"]),
        "wheel_l_contact_duty": sum(samples["wheel_l_contact"]) / denominator,
        "wheel_r_contact_duty": sum(samples["wheel_r_contact"]) / denominator,
        "both_wheels_contact_fraction": sum(samples["both_wheels_contact"]) / denominator,
        "single_wheel_contact_fraction": sum(samples["single_wheel_contact"]) / denominator,
        "no_wheel_contact_fraction": sum(samples["no_wheel_contact"]) / denominator,
        "wheel_torque_saturation_fraction": wheel_saturated / denominator,
        "wheel_requested_saturation_fraction": wheel_requested_saturated / denominator,
        "wheel_torque_rms_fraction": math.sqrt(wheel_torque_sq_sum / denominator),
        "wheel_high_load_fraction": wheel_high_load / denominator,
        "wheel_max_consecutive_high_load_s": max_consecutive_high_load_steps * env.dt,
        "leg_torque_saturation_fraction": leg_saturated / denominator,
        "action_rate_p95": _percentile(samples["action_rate"], 0.95),
        "action_jerk_p95": _percentile(samples["action_jerk"], 0.95),
        "wheel_action_requested_p95": _percentile(samples["wheel_action_requested"], 0.95),
        "wheel_action_requested_max": max(samples["wheel_action_requested"], default=0.0),
        "wheel_action_applied_p95": _percentile(samples["wheel_action_applied"], 0.95),
        "wheel_action_applied_max": max(samples["wheel_action_applied"], default=0.0),
        "wheel_action_diff_p95": _percentile(samples["wheel_action_diff"], 0.95),
        "wheel_torque_diff_p95": _percentile(samples["wheel_torque_diff"], 0.95),
        "wheel_max_abs_vel": max(samples["wheel_max_abs_vel"], default=0.0),
        "wheel_counterdrive_fraction": (
            sum(samples["wheel_counterdrive"]) / denominator
            if samples["wheel_counterdrive"] else 0.0
        ),
        "wheel_action_comm_mean": (
            sum(samples["wheel_action_comm"]) / denominator
            if samples["wheel_action_comm"] else 0.0
        ),
        "wheel_action_comm_p95": _percentile([abs(v) for v in samples["wheel_action_comm"]], 0.95),
        "wheel_torque_comm_mean": (
            sum(samples["wheel_torque_comm"]) / denominator
            if samples["wheel_torque_comm"] else 0.0
        ),
        "wheel_torque_comm_p95": _percentile([abs(v) for v in samples["wheel_torque_comm"]], 0.95),
        "wheel_comm_same_sign_fraction": (
            sum(samples["wheel_comm_same_sign"]) / denominator
            if samples["wheel_comm_same_sign"] else 0.0
        ),
        "wheel_comm_saturation_fraction": (
            sum(samples["wheel_comm_saturated"]) / denominator
            if samples["wheel_comm_saturated"] else 0.0
        ),
        "corr_vel_err_trq_comm": _pearson_corr(samples["vel_tracking_error"], samples["wheel_torque_comm"]),
        "cruise_throttle_exceed_fraction": (
            sum(samples["cruise_throttle_exceed"]) / len(samples["cruise_throttle_exceed"])
            if samples["cruise_throttle_exceed"] else 0.0
        ),
        "cruise_action_comm_p95": (
            _percentile(samples["cruise_act_comm"], 0.95)
            if samples["cruise_act_comm"] else 0.0
        ),
        "diff_torque_clipped_fraction": (
            sum(samples["diff_torque_clipped"]) / denominator
            if samples["diff_torque_clipped"] else 0.0
        ),
    }

    rendered = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(rendered)
    if args.eval_output:
        output_path = os.path.abspath(args.eval_output)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    parsed_args = get_args()
    if parsed_args.load_run is None or parsed_args.checkpoint is None:
        raise ValueError("Pass both --load_run and --checkpoint.")
    evaluate(parsed_args)
