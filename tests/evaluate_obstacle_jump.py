#!/usr/bin/env python3
"""Physical obstacle-clearance evaluation for the frozen jump policy.

This replaces `benchmark_obstacle_jump.py` (kept for history, never overwritten).
It differs in five ways that all changed the verdict:

1. GEOMETRY. The old script measured the wheel front as `pos_y + wheel_radius`,
   omitting the ~0.114 m longitudinal offset of the wheel centres and the forward
   extent of the legs. It under-reported the front edge by up to 0.22 m, so the
   trigger distance and both margins were computed against a phantom robot. Here
   every distance comes from `obstacle_geometry`: analytic cylinder/capsule
   intervals for the wheels, URDF support-point envelopes for the other links,
   clipped to geometry that can actually reach the box top.

2. PRE-RESET SNAPSHOT. `step()` clears the per-episode flags on the step the
   episode ends, so reading them afterwards describes the NEXT episode. The old
   script reported `touchdown_pos_y = 0.0` and `flight_dist = -0.106 m` from
   exactly that mistake. Here all episode outcomes come from
   `env.episode_snapshot`, captured moments before reset.

3. COLLISION EVIDENCE. Isaac Gym's contact list is speculative: a record appears
   as soon as the volumes are within the 20 mm contact offset. A genuine near-miss
   with 9.9 mm of clearance produced a record with `lambda = 0`. A collision is
   only counted on positive evidence -- a real impulse, an exact analytic overlap,
   or force on the obstacle.

4. RECOVERY CRITERION. Phase-2 success required `vel_retention_pct >= 65`. The
   task requires 70 % of the commanded speed sustained over the observation
   window with no persistent regression.

5. NO OBSTACLE MEANS NO OBSTACLE. The old "without obstacle" arm moved the box to
   y = 10 m, leaving it in the scene. Here the obstacle actor is moved out of the
   robot's path AND the trigger is disabled, so the free-run arm measures the
   policy, not a distant box.

Usage (from the repository root, after `source ./activate_jump.sh`):

    python tests/evaluate_obstacle_jump.py --pipeline cpu
    python tests/evaluate_obstacle_jump.py --seeds 11 13 --steps 200
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(WORKSPACE, 'legged_gym'))
sys.path.insert(0, os.path.join(WORKSPACE, 'rsl_rl'))

DEFAULT_MODEL = ('legged_gym/logs/myrobot_running_jump/'
                 'Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt')
DEFAULT_SEEDS = [11, 13, 15, 20, 25, 27, 31, 35]
RECOVERY_WINDOW = 40           # steps of post-touchdown observation
RECOVERY_SPEED_FRACTION = 0.70  # of the commanded forward speed


def _sha256_of(path, chunk=1 << 20):
    """SHA256 of a checkpoint, so a report can prove WHICH network it measured.

    Two checkpoints in this project have identical file sizes; without the digest
    a swapped path would be undetectable after the fact.
    """
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as fh:
            for block in iter(lambda: fh.read(chunk), b''):
                h.update(block)
        return h.hexdigest()
    except OSError as exc:
        return 'unavailable: %s' % exc


# ---------------------------------------------------------------------------
# Worker: one seed, one condition
# ---------------------------------------------------------------------------

def run_worker():
    """Run a single (seed, condition) rollout and print one JSON line."""
    import isaacgym  # noqa: F401  (must precede torch)
    import torch
    # Same decomposition the env's fall test uses (see myrobot_running_jump_env:
    # r_raw is pitch about the wheel axle, p_raw is roll about the travel axis).
    # The trace reports attitude so a 40-degree impact pitch cannot be quoted as
    # a calm upright pose.
    from isaacgym.torch_utils import get_euler_xyz
    from legged_gym.envs import MyRobotObstacleJump  # noqa: F401
    from legged_gym.envs.myrobot import obstacle_geometry as geo
    from legged_gym.envs.myrobot.myrobot_dual_env import STATE_NAMES
    from legged_gym.utils import get_args, task_registry
    from rsl_rl.modules import ActorCritic

    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--task', default='myrobot_obstacle_jump')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--pipeline', default='cpu')
    p.add_argument('--num_envs', type=int, default=1)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--model_path', default=DEFAULT_MODEL)
    p.add_argument('--with_obstacle', type=int, default=1)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--action_mode', default='deterministic',
                   choices=['deterministic', 'stochastic'])
    known, _ = p.parse_known_args()

    sys.argv = [sys.argv[0], '--task', known.task, '--headless',
                '--seed', str(known.seed), '--num_envs', '1',
                '--pipeline', known.pipeline]
    args = get_args()
    cfg, _ = task_registry.get_cfgs(name=known.task)
    cfg.env.num_envs = 1
    cfg.seed = known.seed
    cfg.env.episode_length_s = 4.0
    cfg.trigger.mode = 'spatial'
    with_obstacle = bool(known.with_obstacle)
    action_mode = known.action_mode

    if not with_obstacle:
        # CONTROL ARM: no obstacle in the world, but the SAME spatial trigger
        # principle, so the two arms differ only in whether the box is physically
        # present. The box is parked at y=60 m (it must not touch the robot), and
        # the env trigger is disabled; the worker re-implements the identical
        # spatial criterion against a virtual plane at the nominal box front.
        # Both the trigger distance and the virtual plane use the URDF-derived
        # front edge, never `pos_y + wheel_radius`.
        cfg.obstacle.pos_y = 60.0
        cfg.trigger.mode = 'step'
        cfg.trigger.step_trigger = 10 ** 9
        cfg.env.auto_jump = False

    env, _ = task_registry.make_env(name=known.task, args=args, env_cfg=cfg)
    env.reset()

    ckpt = torch.load(known.model_path, map_location=env.device)
    policy = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128],
                         'elu', 1.0).to(env.device)
    policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()

    cmd_vy = float(env.target_forward_vel)
    obstacles = env.obstacle_root_states[:, 0:3].clone()
    # Nominal box front used by the control arm's virtual trigger plane.
    virtual_front_y = 0.300 - 0.5 * float(env.obstacle_w)
    trigger_dist = float(env.spatial_trigger_dist)
    want_trigger = (not with_obstacle) and trigger_dist > 0.0

    history = []          # per-step trace, pre-reset
    recovery_vels = []
    recovery_curve_row = []   # raw env-side ring buffer, filled at episode end
    pre_jump_vel = None
    first_contact = None
    outcome = None
    obs = env.get_observations() if hasattr(env, 'get_observations') else env.obs_buf

    # One extra step: a 200-step episode (episode_length_s=4.0) only reports
    # `dones` on the step AFTER the counter reaches max_episode_length, so a
    # loop of exactly 200 steps would never observe the timeout reset.
    for step_i in range(known.steps + 1):
        # Control-arm trigger: identical spatial criterion, virtual plane.
        # The distance is measured with LAYER 1 (`approach_envelope`), the SAME
        # envelope the real env uses to fire its spatial trigger. Using the old
        # single-envelope helper here would fire the control arm from a different
        # front edge than the obstacle arm, which is exactly the mismatch the
        # control arm exists to rule out. `approach_envelope` has no root fallback,
        # so an undefined front simply does not fire.
        if want_trigger:
            appr = geo.approach_envelope(env)
            if bool(appr['could_reach'][0]):
                dist = float(virtual_front_y - appr['front_y'][0])
                if (trigger_dist >= dist >= -0.02
                        and int(env.fsm_state[0]) == 0
                        and int(env.episode_length_buf[0]) >= 20
                        and float(env.root_states[0, 8]) > 0.10):
                    env.request_teleop_jump(torch.tensor([True], device=env.device))
                    want_trigger = False
                    print('  [control] virtual trigger fired at step %d, dist=%.4f'
                          % (step_i, dist))

        with torch.no_grad():
            if action_mode == 'deterministic':
                # BENCHMARK MODE: the deterministic mean action. This is the only
                # mode the acceptance numbers are computed from, because sampling
                # injects run-to-run variance that has nothing to do with the
                # policy's competence and would make cross-run comparison
                # meaningless.
                actions = policy.act_inference(obs)
            else:
                # STOCHASTIC MODE: the sampled action from the Gaussian policy.
                # Kept as an INDEPENDENT test condition (robustness / variance
                # study). Its numbers are NEVER averaged together with the
                # deterministic ones.
                actions = policy.act(obs)
        obs, _, _, dones, infos = env.step(actions)

        # Forward speed at the moment the FSM commits to a jump. Captured live
        # because the FSM state is reset on the epilogue step.
        if pre_jump_vel is None and int(env.fsm_state[0]) == 3:
            pre_jump_vel = float(env.root_states[0, 8])

        # ---- pre-reset metrics: everything below is still the live episode ----
        #
        # THREE-LAYER GEOMETRY (Phase D, defect A). The old row read ONE envelope,
        # `robot_forward_interval`, and that envelope had two defects: its rear
        # edge was the rear of the LOW parts (about +0.047 m relative to the root,
        # i.e. ahead of the root) rather than the rear of the robot (about
        # -0.209 m), and while airborne it silently fell back to the root Y. Both
        # are quoted verbatim below so the trace can never be read as if
        # `rear_y` meant "the robot's tail".
        #
        # Every layer is now published by the env itself in `obstacle_layers` and
        # read back unmodified, so the trace cannot drift from the exact numbers
        # the env terminated and paid reward on. Re-deriving them here with a
        # second implementation would be a silent-divergence risk with no upside.
        layers = getattr(env, 'obstacle_layers', {})
        box_front_y, box_rear_y, box_top_z = geo.obstacle_y_interval(env)
        gaps = geo.wheel_box_signed_gap(env)
        clrs = geo.wheel_clearance_above_box_top(env)

        def layer_1(key, default=None):
            t = layers.get(key)
            return default if t is None else float(t[0].item())

        def layer_2(key):
            t = layers.get(key)
            if t is None:
                return None
            return [round(float(t[0, 0].item()), 5), round(float(t[0, 1].item()), 5)]

        def layer_2b(key):
            t = layers.get(key)
            if t is None:
                return None
            return [bool(t[0, 0].item()), bool(t[0, 1].item())]

        # Attitude, from the same Euler decomposition the fall test uses
        # (`myrobot_running_jump_env.py`: r_raw = pitch about the wheel axle,
        # p_raw = roll about the travel axis). Recomputed rather than read from
        # `env` because the env keeps no attitude buffer.
        r_raw, p_raw, _ = get_euler_xyz(env.base_quat)
        roll_rad = torch.atan2(torch.sin(p_raw), torch.cos(p_raw))
        pitch_rad = torch.atan2(torch.sin(r_raw), torch.cos(r_raw))
        # Heading-projected forward speed -- the SAME quantity the recovery gate
        # uses (`_get_world_forward_velocity`), so "cruise at 70%" means one thing
        # in the trace and in the criterion.
        v_fwd_now = float(env._get_world_forward_velocity()[0].item())

        row = {
            'step': step_i,
            # The env's own step counter. It is the clock every latched event
            # (trigger / first contact / liftoff / touchdown / pass) is stamped
            # with, and it is NOT equal to the loop index: it is incremented
            # inside env.step() and starts one ahead. Phase lookups therefore go
            # through this value, never through `step`.
            'common_step': int(env.common_step_counter),
            'fsm': int(env.fsm_state[0]),
            'fsm_name': STATE_NAMES.get(int(env.fsm_state[0].item()), '?'),
            'root_y': round(float(env.root_states[0, 1]), 5),
            'root_z': round(float(env.root_states[0, 2]), 5),
            'vz': round(float(env.root_states[0, 9]), 4),
            # --- LAYER 1: approach envelope (height-limited, no root fallback) --
            # This is the only layer valid for "is the robot close enough to hit
            # the box". `could_reach` False means nothing on the robot is low
            # enough to touch it, so `front_y`/`rear_y` are NaN by construction --
            # an explicit "undefined", never a believable stand-in number.
            'approach_front_y': (lambda v: None if v is None or not math.isfinite(v)
                                 else round(v, 5))(layer_1('approach_front_y')),
            'approach_rear_y': (lambda v: None if v is None or not math.isfinite(v)
                                else round(v, 5))(layer_1('approach_rear_y')),
            'approach_could_reach': (None if layers.get('approach_could_reach') is None
                                     else bool(layers['approach_could_reach'][0].item())),
            'dist_to_box': (None if layers.get('approach_front_y') is None
                            or not math.isfinite(layer_1('approach_front_y'))
                            else round(float(box_front_y[0].item())
                                       - layer_1('approach_front_y'), 5)),
            # --- LAYER 2: exact analytic per-wheel envelopes --------------------
            # `passed_rear` / `danger` are the two per-wheel facts the tier-1
            # "wheels cleared it" claim rests on. `danger` = over the footprint
            # AND below the top plane, i.e. the only predicate that can mean "this
            # wheel could hit the box while above it".
            'wheel_front_y': layer_2('wheel_front_y'),
            'wheel_rear_y': layer_2('wheel_rear_y'),
            'wheel_bottom_z': layer_2('wheel_bottom_z'),
            'wheel_over_footprint': layer_2b('wheel_over_footprint'),
            'wheel_danger': layer_2b('wheel_danger'),
            'wheel_pass_clean': layer_2b('wheel_pass_clean'),
            'landing_safe': (None if layers.get('landing_safe') is None
                             else bool(layers['landing_safe'][0].item())),
            # PhysX-derived wheel/box signed gap (independent geometric source).
            'wheel_gap': round(float(torch.min(gaps[0]).item()), 5),
            # Footprint-limited vertical clearance. The old `wheel_bottom_zs -
            # top_z` reported -0.0477 m for a robot standing NEXT TO the box,
            # because a wheel's lowest point sits directly under its centre and
            # therefore never actually above the footprint. `None` means the
            # wheels cannot reach the footprint at all.
            'wheel_clearance_above_top': (
                lambda v: None if not math.isfinite(v) else round(v, 5))(
                    float(torch.min(clrs[0]).item())),
            # --- LAYER 3: whole-body envelope (full hull, no clip) --------------
            'whole_body_front_y': (lambda v: None if v is None else round(v, 5))(
                layer_1('whole_body_front_y')),
            'whole_body_rear_y': (lambda v: None if v is None else round(v, 5))(
                layer_1('whole_body_rear_y')),
            'whole_body_rear_margin': (lambda v: None if v is None else round(v, 5))(
                layer_1('whole_body_rear_margin')),
            'whole_body_passed': (None if layers.get('whole_body_passed') is None
                                  else bool(layers['whole_body_passed'][0].item())),
            # --- Kinematic cross-check: NOT an envelope -------------------------
            # root Y plus a CONSTANT rear offset. Reported only so a reader can see
            # how far the constant-offset estimate is from the real envelope; it
            # must never be substituted for `whole_body_rear_margin`.
            'root_estimated_rear_y': (lambda v: None if v is None else round(v, 5))(
                layer_1('root_estimated_rear_y')),
            'root_estimated_margin': (lambda v: None if v is None else round(v, 5))(
                layer_1('root_estimated_margin')),
            # --- Independent physical evidence (no envelope involved) ----------
            'collision': int(env.has_obstacle_collision[0]),
            'peak_obstacle_force_N': (lambda v: None if v is None else round(v, 2))(
                layer_1('peak_obstacle_contact_force_N')),
            # --- attitude and heading-projected speed ---------------------------
            'roll_rad': round(float(roll_rad[0].item()), 4),
            'pitch_rad': round(float(pitch_rad[0].item()), 4),
            'v_fwd': round(v_fwd_now, 5),
            # --- task-owned latches (defect B: survive FSM transitions) ---------
            # Read from the live buffer, not from `obstacle_layers`: the three
            # layers are published as one dict but these latches are not in it,
            # and reading `layers.get(...)` here silently produced null columns.
            'task_took_off': bool(env.task_took_off[0].item()),
            'takeoff_step': int(env.task_takeoff_step[0].item()),
            'touchdown_step': int(env.task_touchdown_step[0].item()),
            'taken_off': int(env.has_taken_off[0]),
            'touched_down': int(env.has_touched_down[0]),
            # `obstacle_cleared` is retained ONLY so the historic column keeps its
            # meaning. It is a tier-1 latch and NO phase-D verdict uses it: the
            # task book explicitly forbids judging success from it.
            'cleared': int(env.obstacle_cleared[0]),
            'task_failed': int(env.task_failed[0]),
            # --- recovery / handoff state --------------------------------------
            'recovery_outcome': env.recovery_outcome[0],
            'recovery_hold_steps': int(env.recovery_hold_steps[0].item()),
            'recovery_best_hold_steps': int(env.recovery_best_hold_steps[0].item()),
            'handoff_step': int(env.handoff_step[0].item()),
            'abort_step': int(env.abort_step[0].item()),
            'last_event': env.last_transition_event[0],
        }
        history.append(row)

        # Post-touchdown cruise speed, restricted to the observation window that
        # starts at touchdown. Reading to the end of the episode would mix in
        # motion from a later FSM phase (and, in practice, a reverse roll-back
        # that is a different failure mode).
        if (bool(env.has_touched_down[0].item())
                and int(env.touchdown_step[0].item()) >= 0):
            td_step = int(env.touchdown_step[0].item())
            if td_step <= step_i < td_step + RECOVERY_WINDOW:
                recovery_vels.append(float(env.root_states[0, 8]))

        if bool(dones[0].item()):
            snap = env.episode_snapshot
            if bool(snap['valid'][0].item()):
                outcome = {
                    'reset_step': int(snap['reset_step'][0].item()),
                    'reason': snap['failure_reason'][0],
                    'last_event': snap['last_transition_event'][0],
                    'collision': bool(snap['has_obstacle_collision'][0].item()),
                    'task_failed': bool(snap['task_failed'][0].item()),
                    'fell': bool(snap['fell'][0].item()),
                    'taken_off': bool(snap['has_taken_off'][0].item()),
                    'touched_down': bool(snap['has_touched_down'][0].item()),
                    'cleared': bool(snap['obstacle_cleared'][0].item()),
                    'takeoff_step': int(snap['takeoff_step'][0].item()),
                    'touchdown_step': int(snap['touchdown_step'][0].item()),
                    'takeoff_y': round(float(snap['takeoff_pos_y'][0]), 5),
                    'touchdown_y': round(float(snap['touchdown_pos_y'][0]), 5),
                    'takeoff_front_y': round(float(snap['takeoff_front_y'][0]), 5),
                    'takeoff_rear_y': round(float(snap['takeoff_rear_y'][0]), 5),
                    'min_clearance_m': round(float(snap['min_obstacle_clearance'][0]), 5),
                    'peak_force_N': round(float(snap['peak_obstacle_contact_force'][0]), 1),
                    'takeoff_margin_m': round(float(snap['takeoff_margin_m'][0]), 5),
                    'landing_margin_m': round(float(snap['landing_margin_m'][0]), 5),
                    'first_contact_step': int(snap['first_contact_step'][0].item()),
                    'first_contact_body': int(snap['first_contact_body'][0].item()),
                    'first_contact_source': snap['first_contact_source'][0],
                    'trigger_step': int(snap['trigger_step'][0].item()),
                    'pass_step': int(snap['pass_step'][0].item()),
                    'root_y': round(float(snap['root_y'][0]), 5),
                    'root_z': round(float(snap['root_z'][0]), 5),
                    # --- recovery evidence (task 2) -------------------------
                    # Task-owned 2 s recovery survives the FSM handoff.
                    'mission_completed': bool(snap['mission_completed'][0].item()),
                    'fast_recovered': bool(snap['fast_recovered'][0].item()),
                    'e2e_best_hold_steps': int(snap['e2e_best_hold_steps'][0].item()),
                    'e2e_success_step': int(snap['e2e_success_step'][0].item()),
                    'recovery_outcome': snap['recovery_outcome'][0],
                    'recovery_best_hold_steps': int(
                        snap['recovery_best_hold_steps'][0].item()),
                    'recovery_regression_steps': int(
                        snap['recovery_regression_steps'][0].item()),
                    'recovery_grounded_steps': int(
                        snap['recovery_grounded_steps'][0].item()),
                    'recovered_sustained': bool(
                        snap['recovered_sustained'][0].item()),
                    # --- THREE-LAYER passage record (defect A) --------------
                    # Phase C computed "passed the obstacle" from one envelope,
                    # `robot_rear_margin`, which only describes parts BELOW the
                    # box top. On this robot that envelope's rear sits at about
                    # +0.047 m relative to the root (ahead of it) while the whole
                    # machine's rear sits at -0.209 m: a 0.25 m disagreement. Five
                    # of eight Phase C runs were therefore reported as "crossed
                    # without liftoff" AFTER genuinely clearing the box, and three
                    # were reported as clean passes that a whole-body test rejects.
                    # The three tiers are recorded separately here and NONE of them
                    # is a success verdict on its own.
                    'task_took_off': bool(snap['task_took_off'][0].item()),
                    'takeoff_front_y_true': round(float(snap['takeoff_front_y'][0]), 5),
                    'takeoff_rear_y_true': round(float(snap['takeoff_rear_y'][0]), 5),
                    'takeoff_clearance_above_obs': round(
                        float(snap['takeoff_clearance_above_obs'][0]), 5),
                    # LAYER 2, per wheel [lead, trail].
                    'wheel_pass_clean': [bool(snap['wheel_pass_clean'][0, i].item())
                                         for i in range(2)],
                    'wheel_danger_seen': [bool(snap['wheel_danger_seen'][0, i].item())
                                          for i in range(2)],
                    'wheel_both_pass_clean': bool(
                        snap['wheel_pass_clean'][0].all().item()),
                    # LAYER 3, whole body.
                    'landing_safe': bool(snap['landing_safe'][0].item()),
                    'whole_body_passed': bool(snap['whole_body_passed'][0].item()),
                    'whole_body_rear_margin': round(
                        float(snap['whole_body_rear_margin'][0]), 5),
                    # Independent physical evidence, no envelope involved: the
                    # only source of truth for "did it touch the box".
                    'obstacle_contact': bool(snap['has_obstacle_collision'][0].item()),
                    # --- attitude and speed at the terminating frame ---------
                    # Captured by the env in `_last_geometry` on the last live
                    # step; without it the terminating row can only be rebuilt
                    # from root_y/root_z and reads as a pristine stand-still.
                    'terminating_roll_rad': round(float(snap['roll_rad'][0]), 4),
                    'terminating_pitch_rad': round(float(snap['pitch_rad'][0]), 4),
                    'terminating_forward_vel': round(float(snap['forward_vel'][0]), 5),
                    # --- handoff bookkeeping (defect B / task 4) -------------
                    # `handoff_step` is stamped when the FSM leaves RECOVER for
                    # BALANCE; `abort_step` when it leaves by the abort branch.
                    # The difference between them and `touchdown_step` is the
                    # BALANCE-takeover latency the task book asks for.
                    'handoff_step': int(snap['handoff_step'][0].item())
                    if snap.get('handoff_step') is not None else -1,
                    'abort_step': int(snap['abort_step'][0].item())
                    if snap.get('abort_step') is not None else -1,
                }
                # Raw forward-velocity curve in time order. The ring buffer is
                # head-indexed and NaN-padded, so it is rotated and trimmed here
                # rather than handed over raw: the acceptance requirement is that
                # the REAL curve is kept, not a summary of it.
                curve = snap.get('recovery_curve')
                if curve is not None:
                    row = curve[0].tolist()
                    head = 0
                    if 'recovery_curve_head' in snap:
                        head = int(snap['recovery_curve_head'][0].item())
                    if head > 0:
                        row = row[head:] + row[:head]
                    recovery_curve_row = [round(float(v), 5) for v in row
                                          if math.isfinite(float(v))]
                else:
                    recovery_curve_row = []
                # The authoritative first-contact record also lives in the
                # snapshot: `env.first_contact_*` has already been cleared by
                # reset_idx when step() returns.
                if outcome['first_contact_step'] >= 0:
                    fb = outcome['first_contact_body']
                    first_contact = {
                        'step': outcome['first_contact_step'],
                        'body': fb,
                        'body_name': (env.link_names[fb]
                                      if 0 <= fb < len(env.link_names) else 'unknown'),
                        'world_pos': [round(float(v), 5)
                                      for v in snap['first_contact_pos'][0]],
                        # Both Newtons, different questions. `pair_force_N` is the
                        # solver's lambda for THIS contact pair; `net_force_N` is
                        # the net force on the whole obstacle body (all its pairs
                        # summed). lambda was measured to be a force, not an
                        # impulse: tests/test_contact_force_units.py sees it match
                        # m*g for a settled known mass at three different dt.
                        'pair_force_N': round(float(snap['first_contact_force'][0]), 1),
                        'net_force_N': round(float(snap['first_contact_net_force'][0]), 1),
                        'force_unit': 'N',
                        'source': outcome['first_contact_source'],
                        'before_liftoff': outcome['first_contact_step'] < outcome['takeoff_step'],
                    }

                # The trace row appended a moment ago describes the NEXT episode:
                # reset_idx ran inside this very step, so root_y is back to 0.0 and
                # the wheels are 9.8 cm from the box again. Phase markers that land
                # on the final step (it is usually the first-contact step, because
                # any contact terminates) would then show a pristine stand-still
                # instead of the impact. Rebuild that row's geometry from the
                # snapshot, which was captured before the reset.
                if history and history[-1]['common_step'] == outcome['reset_step']:
                    fixed = history[-1]
                    fixed['post_reset'] = True
                    fixed['fsm_name'] = 'RESET(%s)' % outcome['reason']
                    for key, snap_key in (('root_y', 'root_y'), ('root_z', 'root_z'),
                                          ('front_y', 'front_y'), ('rear_y', 'rear_y')):
                        fixed[key] = round(float(snap[snap_key][0]), 5)
                    gap = float(snap['wheel_gap_min'][0])
                    fixed['wheel_gap'] = round(gap, 5)
                    clr = float(snap['wheel_clearance_min'][0])
                    fixed['wheel_above_top'] = round(clr, 5) if math.isfinite(clr) else None
                    fixed['dist_to_box'] = round(
                        float(snap['obstacle_front_y'][0]) - fixed['front_y'], 5)
                    fixed['taken_off'] = int(outcome['takeoff_step'] >= 0)
                    fixed['touched_down'] = int(outcome['touchdown_step'] >= 0)
                    fixed['collision'] = int(outcome['collision'])
                    fixed['cleared'] = int(outcome['cleared'])
                    fixed['task_failed'] = int(outcome['task_failed'])
                    # The terminating step is the one that matters most for
                    # attribution -- it is usually the impact -- so it must carry
                    # the same THREE-LAYER columns as every live row. Leaving the
                    # layers at their reset-time values here is how a 40-degree
                    # impact pose gets reported as a stand-still, and how a wheel
                    # whose gap was measured pre-reset gets written into the row
                    # that documents the crash.
                    fixed['roll_rad'] = round(float(snap['roll_rad'][0]), 5)
                    fixed['pitch_rad'] = round(float(snap['pitch_rad'][0]), 5)
                    fixed['v_fwd'] = round(float(snap['forward_vel'][0]), 5)
                    snap_layers = snap.get('obstacle_layers') or {}

                    def snap_1(key):
                        t = snap_layers.get(key)
                        return None if t is None else round(float(t[0].item()), 5)

                    def snap_b(key):
                        t = snap_layers.get(key)
                        return None if t is None else bool(t[0].item())

                    for key in ('approach_front_y', 'approach_rear_y',
                                'whole_body_front_y', 'whole_body_rear_y',
                                'whole_body_rear_margin', 'root_estimated_rear_y',
                                'root_estimated_margin',
                                'peak_obstacle_contact_force_N'):
                        value = snap_1(key)
                        if value is not None and not math.isnan(value):
                            fixed[key] = value
                    for key in ('approach_could_reach', 'whole_body_passed',
                                'landing_safe'):
                        value = snap_b(key)
                        if value is not None:
                            fixed[key] = value
                    for key in ('wheel_front_y', 'wheel_rear_y', 'wheel_bottom_z'):
                        t = snap_layers.get(key)
                        if t is not None:
                            fixed[key] = [round(float(t[0, 0].item()), 5),
                                          round(float(t[0, 1].item()), 5)]
                    for key in ('wheel_over_footprint', 'wheel_danger',
                                'wheel_pass_clean'):
                        t = snap_layers.get(key)
                        if t is not None:
                            fixed[key] = [bool(t[0, 0].item()),
                                          bool(t[0, 1].item())]
            break

    # ---- derived metrics ---------------------------------------------------
    derived = {}
    if outcome:
        # Flight distance only exists if the robot actually landed. A robot that
        # collided mid-air never touched down, and reporting
        # `touchdown_y - takeoff_y` there would fabricate a negative distance
        # out of the reset-time zero.
        if outcome['touched_down']:
            flight = outcome['touchdown_y'] - outcome['takeoff_y']
            derived['flight_dist_m'] = round(flight, 5)
        else:
            flight = None
            derived['flight_dist_m'] = None

        # Forward displacement still owed at liftoff, measured on the SAME point of
        # the body as the flight distance (the root): the robot has cleared the box
        # only when its REAR envelope passes the rear face, so it must advance
        # (box_rear - takeoff_rear_y) of root travel. Only defined for a run with a
        # real obstacle in front AND an actual liftoff: a run that never took off
        # leaves `takeoff_rear_y == 0` (reset default), which would fabricate a
        # plausible-looking requirement for a robot that never jumped.
        took_off = outcome['takeoff_step'] >= 0
        if with_obstacle and took_off:
            box_rear_y = float(obstacles[0, 1]) + 0.5 * float(env.obstacle_w)
            required = box_rear_y - outcome['takeoff_rear_y']
            derived['required_dist_m'] = round(required, 5)
            derived['shortfall_m'] = (None if flight is None
                                      else round(required - flight, 5))
        else:
            derived['required_dist_m'] = None
            derived['shortfall_m'] = None
        if not with_obstacle:
            # Both margins are measured against the box face; with the box 60 m
            # away they describe nothing, so they are dropped rather than
            # reported as if they meant something.
            outcome['takeoff_margin_m'] = None
            outcome['landing_margin_m'] = None

        # Set before the metric block below, which reads it for the approach
        # summary. It is the speed at the moment the jump was requested, NOT the
        # separation speed: this line used to sit after the metrics and the
        # ordering mattered.
        derived['pre_jump_speed_mps'] = (None if pre_jump_vel is None
                                         else round(pre_jump_vel, 4))

        # Shared with the PPO success gate: safe sustained cruise by TD + 2 s.
        derived['recovery_sustained'] = bool(outcome['recovered_sustained'])
        derived['recovery_verdict_source'] = 'env._recovery_succeeded'

        # =====================================================================
        # THE TEN ACCEPTANCE METRICS (Phase D task 3), recorded item by item.
        # The list is the task book's, in its order, so the report can be checked
        # against the request without a mapping table. Each entry is a FACT, not
        # a verdict; the verdicts live in `accounts` below and are deliberately
        # NOT collapsible into one number.
        # =====================================================================
        td = outcome['touchdown_step']
        tk = outcome['takeoff_step']
        rows_by_cs = {row['common_step']: row for row in history}

        def rows_between(lo, hi):
            """Live trace rows with lo <= common_step < hi, in time order."""
            return [rows_by_cs[k] for k in sorted(rows_by_cs)
                    if lo <= k < hi and k in rows_by_cs]

        # `takeoff_clearance_above_obs` stores 0.0 as a SENTINEL for "+inf, no
        # wheel was over the box footprint at liftoff", so the JSON stays finite.
        # Reporting that sentinel as a measured 0 m clearance would be a lie; the
        # applicability is therefore resolved from the traced footprint overlap.
        tk_row = rows_by_cs.get(tk) if tk >= 0 else None
        tk_over = (tk_row or {}).get('wheel_over_footprint')
        lift_clr_defined = bool(tk_over is not None and any(tk_over))
        LIFT_CLR_NOTE = (
            'defined: a wheel was over the box footprint at liftoff, so this is '
            'the measured gap above the box top. undefined: no wheel was over '
            'the footprint (takeoff before the box); the env stores 0.0 as a '
            'finite sentinel for +inf and that 0.0 is NOT a measurement.')

        # ---- (1) complete approach and takeoff -----------------------------
        # The approach is only meaningful with the height-limited LAYER-1
        # envelope: `could_reach` False means no part of the robot is low enough
        # to touch the box, so a distance there is undefined, not large.
        approach_rows = rows_between(0, tk) if tk >= 0 else rows_between(0, 10 ** 9)
        reachable = [r for r in approach_rows if r.get('approach_could_reach')]
        derived['m1_approach'] = {
            'n_rows': len(approach_rows),
            'n_rows_with_defined_envelope': len(reachable),
            'trigger_step': outcome['trigger_step'],
            'takeoff_step': tk,
            'steps_from_trigger_to_liftoff': (None if tk < 0
                                              or outcome['trigger_step'] < 0
                                              else tk - outcome['trigger_step']),
            'speed_before_jump_mps': derived.get('pre_jump_speed_mps'),
            'takeoff_margin_m': outcome['takeoff_margin_m'],
            'takeoff_clearance_defined': lift_clr_defined,
            'takeoff_clearance_above_box_m': (
                outcome['takeoff_clearance_above_obs'] if lift_clr_defined else None),
            'takeoff_clearance_note': LIFT_CLR_NOTE,
            # How far the true rear envelope was from the box front at liftoff.
            # Negative means the tail was still over the box.
            'takeoff_rear_y_true': outcome['takeoff_rear_y_true'],
            'first_approach_dist_to_box_m': (round(reachable[0]['dist_to_box'], 5)
                                             if reachable
                                             and reachable[0]['dist_to_box'] is not None
                                             else None),
            'min_approach_dist_to_box_m': (
                round(min(r['dist_to_box'] for r in reachable
                          if r['dist_to_box'] is not None), 5)
                if any(r['dist_to_box'] is not None for r in reachable) else None),
        }

        # ---- (2) airborne velocity and displacement ------------------------
        # The horizontal speed is differenced from the traced root Y, NOT taken
        # from the policy's commanded speed and NOT from the root velocity
        # channel: the question is how far the machine actually travelled while
        # airborne. `root_y` is a root point, so this is a root displacement and
        # must never be presented as a body-length clearance.
        air_rows = ([rows_by_cs[k] for k in sorted(rows_by_cs) if tk <= k < td]
                    if (tk >= 0 and td >= 0) else [])
        if len(air_rows) >= 2:
            dy = air_rows[-1]['root_y'] - air_rows[0]['root_y']
            n_air = air_rows[-1]['common_step'] - air_rows[0]['common_step']
            air_speed = dy / (n_air * 0.02) if n_air > 0 else None
            peak_z = max(r['root_z'] for r in air_rows)
        else:
            dy, n_air, air_speed, peak_z = None, 0, None, None
        derived['m2_airborne'] = {
            'n_airborne_rows': len(air_rows),
            'airborne_steps': n_air,
            'airborne_time_s': (None if not n_air else round(n_air * 0.02, 4)),
            'root_displacement_m': None if dy is None else round(dy, 5),
            'mean_horizontal_speed_mps': (None if air_speed is None
                                          else round(air_speed, 4)),
            'peak_root_z_m': None if peak_z is None else round(peak_z, 5),
            'takeoff_root_y': outcome['takeoff_y'],
            'touchdown_root_y': outcome['touchdown_y'],
            'flight_dist_m': derived['flight_dist_m'],
            'required_dist_m': derived['required_dist_m'],
            'shortfall_m': derived['shortfall_m'],
            'note': ('root_displacement_m is a ROOT displacement. It is not a '
                     'clearance and must not be reported as one.'),
        }

        # ---- (3) per-wheel safe passage ------------------------------------
        derived['m3_wheels'] = {
            'pass_clean': outcome['wheel_pass_clean'],
            'danger_seen': outcome['wheel_danger_seen'],
            'both_wheels_clean': outcome['wheel_both_pass_clean'],
            'wheel_names': ['lead(link_004)', 'trail(link_007)'],
            'lift_off_clearance_defined': lift_clr_defined,
            'lift_off_clearance_above_box_m': (
                outcome['takeoff_clearance_above_obs'] if lift_clr_defined else None),
            'lift_off_clearance_note': LIFT_CLR_NOTE,
        }

        # ---- (4) actual landing position -----------------------------------
        box_rear_for_margins = (None if not with_obstacle
                                else float(obstacles[0, 1]) + 0.5 * float(env.obstacle_w))
        derived['m4_landing'] = {
            'touchdown_step': td,
            'touchdown_root_y': outcome['touchdown_y'],
            'touchdown_relative_to_box_rear_m': (
                None if (box_rear_for_margins is None or td < 0)
                else round(outcome['touchdown_y'] - box_rear_for_margins, 5)),
            'landing_margin_m': outcome['landing_margin_m'],
            'landing_safe_no_wheel_over_footprint': outcome['landing_safe'],
            'terminating_roll_rad': outcome['terminating_roll_rad'],
            'terminating_pitch_rad': outcome['terminating_pitch_rad'],
            'terminating_forward_vel_mps': outcome['terminating_forward_vel'],
        }

        # ---- (5) real contact throughout -----------------------------------
        # 1.0 is the env's sentinel for "never came near the height band"; in the
        # free-run arm the box is 60 m away so it means nothing at all. Reporting
        # it as a 1 m clearance would put a fabricated measurement in a table.
        clr_is_defined = bool(with_obstacle)
        derived['m5_contact'] = {
            'any_obstacle_contact': outcome['obstacle_contact'],
            'cross_check_collision_latch': outcome['collision'],
            'first_contact_step': outcome['first_contact_step'],
            'first_contact_body': outcome['first_contact_body'],
            'first_contact_source': outcome['first_contact_source'],
            'peak_obstacle_force_N': outcome['peak_force_N'],
            'min_obstacle_clearance_defined': clr_is_defined,
            'min_obstacle_clearance_m': (outcome['min_clearance_m']
                                         if clr_is_defined else None),
            'min_obstacle_clearance_note': (
                'Defined only in the obstacle arm; the free-run arm has no box in '
                'the path and the env keeps its 1.0 sentinel there.'),
            'clearance_resolution_note': (
                'The wheel/box gap is sampled at 65 points along the axle; '
                'h=1.383 mm so a sampled minimum OVERSTATES the true minimum by '
                'at most h/2=0.691 mm. Values inside that band support only the '
                'SIGN of the gap, never a penetration depth.'),
        }

        # ---- (6) whole body finally clear ----------------------------------
        derived['m6_whole_body'] = {
            'whole_body_passed': outcome['whole_body_passed'],
            'whole_body_rear_margin_m': outcome['whole_body_rear_margin'],
            'pass_step': outcome['pass_step'],
            'rear_margin_definition': (
                'rear of the FULL URDF hull (all links, no height clip) minus the '
                'box rear face. Positive = the whole machine is past the box.'),
        }

        # ---- (7) the three handoffs ----------------------------------------
        # TIMEOUT_HANDOFF is a takeover by ABANDONING the recovery objective; it
        # is recorded here as data because it is the mechanism by which every
        # seed in Phase C eventually started cruising again.
        derived['m7_handoff'] = {
            'recovery_outcome': outcome['recovery_outcome'],
            'fast_recovery_within_040s': outcome['fast_recovered'],
            'normal_handoff': outcome['recovery_outcome'] == 'NORMAL_HANDOFF',
            'timeout_handoff': outcome['recovery_outcome'] == 'TIMEOUT_HANDOFF',
            'recover_abort': outcome['recovery_outcome'] == 'RECOVER_ABORT',
            'handoff_step': outcome['handoff_step'],
            'abort_step': outcome['abort_step'],
            'touchdown_step': td,
            'balance_takeover_latency_s': (
                None if (td < 0 or outcome['handoff_step'] < 0)
                else round((outcome['handoff_step'] - td) * 0.02, 3)),
            'abort_latency_s': (
                None if (td < 0 or outcome['abort_step'] < 0)
                else round((outcome['abort_step'] - td) * 0.02, 3)),
            'best_hold_steps': outcome['recovery_best_hold_steps'],
            'best_hold_time_s': round(
                outcome['recovery_best_hold_steps'] * 0.02, 3),
            'grounded_steps_in_window': outcome['recovery_grounded_steps'],
            'regression_steps': outcome['recovery_regression_steps'],
        }

        # ---- (8) the 1.0 s AFTER landing: real speed and attitude ----------
        # This is the window the task book names explicitly. It is measured from
        # the raw trace, NOT from the env's recovery counters, because the point
        # is to show what the machine actually did in that second -- including
        # the case where it was still rolling BACKWARDS.
        post_rows = (rows_between(td, td + 50) if td >= 0 else [])
        if post_rows:
            speeds = [r['v_fwd'] for r in post_rows]
            rolls = [r['roll_rad'] for r in post_rows]
            pitches = [r['pitch_rad'] for r in post_rows]
            threshold = RECOVERY_SPEED_FRACTION * cmd_vy
            # Longest CONSECUTIVE run at or above the threshold, and when the
            # first such run of RECOVERY_HOLD_STEPS steps began. This is the
            # honest "did it re-cruise" measurement: it does not care which
            # handoff closed the RECOVER state, which is exactly what makes it
            # comparable to the strict verdict rather than a restatement of it.
            best_run = run = 0
            first_sustained_step = None
            for r in post_rows:
                if r['v_fwd'] >= threshold:
                    run += 1
                    best_run = max(best_run, run)
                    if run == 10 and first_sustained_step is None:
                        first_sustained_step = r['common_step']
                else:
                    run = 0
            derived['m8_post_landing_1s'] = {
                'n_rows': len(post_rows),
                'window_seconds': round(len(post_rows) * 0.02, 3),
                'speed_threshold_mps': round(threshold, 4),
                'speed_min_mps': round(min(speeds), 4),
                'speed_max_mps': round(max(speeds), 4),
                'speed_mean_mps': round(sum(speeds) / len(speeds), 4),
                'backslide_m': round(sum(max(-v, 0.0) for v in speeds) * 0.02, 5),
                'n_steps_negative_speed': sum(1 for v in speeds if v < 0),
                'max_abs_roll_deg': round(max(abs(math.degrees(v)) for v in rolls), 2),
                'max_abs_pitch_deg': round(max(abs(math.degrees(v)) for v in pitches), 2),
                'best_consecutive_steps_at_70pct': best_run,
                'first_sustained_cruise_step': first_sustained_step,
                'time_to_sustained_cruise_s': (
                    None if first_sustained_step is None
                    else round((first_sustained_step - td) * 0.02, 3)),
                'vels': [round(v, 4) for v in speeds],
            }
        else:
            derived['m8_post_landing_1s'] = {
                'n_rows': 0, 'window_seconds': 0.0,
                'speed_threshold_mps': round(RECOVERY_SPEED_FRACTION * cmd_vy, 4),
                'note': 'no touchdown, so there is no post-landing window',
            }

        # ---- (4b) RECOVERY AFTER THE BALANCE TAKEOVER ----------------------
        # The task book asks explicitly: did the machine actually recover after
        # BALANCE took over, and how long did that take? The 1.0 s window above
        # cannot answer it -- in this task line the cruise reliably starts about
        # 0.9 s AFTER touchdown, i.e. only after RECOVER has already given up
        # (TIMEOUT_HANDOFF at +0.38 s). So this scan runs over the WHOLE episode
        # and reports both reference points. It is a separate entry from the
        # strict verdict on purpose: it is the mechanism, not an excuse.
        all_post_rows = rows_between(td, 10 ** 9) if td >= 0 else []
        scan_threshold = RECOVERY_SPEED_FRACTION * cmd_vy
        run = 0
        first_cruise = None
        first_cruise_run_start = None
        for r in all_post_rows:
            if r['v_fwd'] >= scan_threshold:
                run += 1
                if run == 10 and first_cruise is None:
                    first_cruise = r['common_step']
                    first_cruise_run_start = r['common_step'] - 9
            else:
                run = 0
        derived['recovery_after_balance_takeover'] = {
            'scanned_steps': len(all_post_rows),
            'scan_window_seconds': round(len(all_post_rows) * 0.02, 3),
            'threshold_mps': round(scan_threshold, 4),
            'handoff_closed_recover': outcome['recovery_outcome'],
            'handoff_step': outcome['handoff_step'],
            'first_sustained_cruise_step': first_cruise,
            'first_sustained_cruise_run_start_step': first_cruise_run_start,
            'recovered_at_all': first_cruise is not None,
            'recovered_only_after_handoff': bool(
                first_cruise is not None
                and outcome['handoff_step'] >= 0
                and first_cruise > outcome['handoff_step']),
            'seconds_from_touchdown': (
                None if first_cruise is None
                else round((first_cruise - td) * 0.02, 3)),
            'seconds_from_handoff': (
                None if (first_cruise is None or outcome['handoff_step'] < 0)
                else round((first_cruise - outcome['handoff_step']) * 0.02, 3)),
            'strict_gate_satisfied': derived['recovery_sustained'],
            'note': ('This legacy speed-only scan is diagnostic. Formal recovery '
                     'requires safe sustained cruise within 2 s, recorded separately.'),
        }

        # Small helper for the accounts below: the shared kinematic facts, each
        # named once so the three verdicts cannot drift apart.
        took_off_ok = outcome['task_took_off'] or tk >= 0
        landed_ok = td >= 0
        no_contact = not outcome['obstacle_contact']
        wheels_ok = outcome['wheel_both_pass_clean']
        body_ok = outcome['whole_body_passed']

        # ---- (9) STRICT VERDICT --------------------------------------------
        # Every clause is a separate physical fact and the recovery clause is the
        # env's own safe 2 s recovery verdict. Nothing here can be satisfied
        # by `cleared`, which is exactly what Phase C did by mistake.
        strict_criteria = {
            'real_liftoff': took_off_ok,
            'real_touchdown': landed_ok,
            'both_wheels_cleared_clean': wheels_ok,
            'landing_was_clear': outcome['landing_safe'],
            'whole_body_left_the_box': body_ok,
            'no_obstacle_contact': no_contact,
            'safe_sustained_recovery_within_2s': derived['recovery_sustained'],
            'task_did_not_fail': not outcome['task_failed'],
            'whole_body_still_past_box': outcome['whole_body_rear_margin'] > 0.005,
            'did_not_fall': not outcome['fell'],
        }
        assert all(strict_criteria.values()) == outcome['mission_completed'], (
            'Evaluator and PPO physical success contracts disagree', strict_criteria)
        derived['m9_strict'] = {
            'criteria': strict_criteria,
            'success': all(strict_criteria.values()),
        }

        # ---- (10) END-TO-END RE-CRUISE, independent of the handoff ---------
        # Two readings of the same question, both reported because they disagree
        # in this task line and the disagreement is the finding:
        #   * within_1s  -- the task book's 1.0 s window (item 8 of the list);
        #   * whole_episode -- does the machine re-cruise at ANY point after
        #     touching down, which is what "end to end" actually asks.
        post = derived['m8_post_landing_1s']
        reconv = derived['recovery_after_balance_takeover']
        derived['m10_end_to_end_recruise'] = {
            'formal_within_2s': {
                'success': derived['recovery_sustained'],
                'best_consecutive_safe_steps': outcome['e2e_best_hold_steps'],
                'success_step': outcome['e2e_success_step'],
                'window_seconds': 2.0,
            },
            'within_1s': {
                'reached_70pct_for_10_consecutive_steps': bool(
                    post.get('best_consecutive_steps_at_70pct', 0) >= 10),
                'best_consecutive_steps': post.get(
                    'best_consecutive_steps_at_70pct', 0),
                'time_to_sustained_cruise_s': post.get(
                    'time_to_sustained_cruise_s'),
                'window_seconds': post.get('window_seconds'),
            },
            'whole_episode': {
                'reached_70pct_for_10_consecutive_steps': reconv['recovered_at_all'],
                'first_sustained_cruise_step': reconv[
                    'first_sustained_cruise_step'],
                'seconds_from_touchdown': reconv['seconds_from_touchdown'],
                'seconds_from_handoff': reconv['seconds_from_handoff'],
                'scanned_steps': reconv['scanned_steps'],
            },
            'measured_on': ('raw trace v_fwd; does NOT require NORMAL_HANDOFF, '
                            'and says nothing about whether the recovery state '
                            'itself succeeded'),
        }

        # ---- THE THREE ACCOUNTS (task 4) ----------------------------------
        # Three different questions that Phase C answered with one number.
        # Collapsing them again would hide exactly the disagreement they exist to
        # expose: in this task line the machine can clear the box, avoid all
        # contact, and still never satisfy the recovery gate -- because the
        # cruise begins only after the RECOVER state has been abandoned.
        derived['accounts'] = {
            'strict_obstacle_clearance': {
                'value': outcome['mission_completed'],
                'definition': ('liftoff + touchdown + both wheels clean + clear '
                               'landing + whole body past the box + zero '
                               'obstacle contact + safe sustained 70% cruise within '
                               '2 s of touchdown + no fall + no task failure'),
                'uses_cleared_latch': False,
            },
            'collision_free_safe_pass': {
                'value': bool(wheels_ok and outcome['landing_safe'] and body_ok
                              and no_contact),
                'definition': ('both wheels clean + landing clear + whole body '
                               'past the box + zero obstacle contact. No '
                               'requirement on speed, recovery or attitude.'),
                'uses_cleared_latch': False,
            },
            'end_to_end_recruise': {
                'value': derived['recovery_sustained'],
                'definition': ('safe >=70% commanded forward speed for ten consecutive '
                               'steps completed within 2 s after touchdown; both wheels '
                               'loaded, attitude and rates stable, no fall; any handoff'),
                'uses_cleared_latch': False,
            },
        }

        # The env's own verdict was already recorded above (the strict block
        # depends on it); this section adds the SUPPORTING mean that must never be
        # mistaken for it.
        vels = recovery_vels[-RECOVERY_WINDOW:] if recovery_vels else []
        if vels:
            mean_v = sum(vels) / len(vels)
            derived['recovery_mean_vel'] = round(mean_v, 4)
            derived['recovery_ratio'] = round(mean_v / cmd_vy, 4)
            derived['recovery_declining_steps'] = sum(
                1 for a, b in zip(vels, vels[1:]) if b < a - 1e-3)
            # The Phase A heuristic, retained under an honest name so the two
            # numbers can be compared side by side instead of one silently
            # standing in for the other.
            derived['recovery_mean_window_heuristic'] = bool(
                mean_v >= RECOVERY_SPEED_FRACTION * cmd_vy
                and derived['recovery_declining_steps'] <= len(vels) // 4)
            derived['recovery_window_vels'] = [round(v, 5) for v in vels]
        else:
            derived['recovery_mean_vel'] = 0.0
            derived['recovery_ratio'] = 0.0
            derived['recovery_declining_steps'] = 0
            derived['recovery_mean_window_heuristic'] = False
            derived['recovery_window_vels'] = []
        # Raw curve straight from the env's ring buffer (task 2 requirement: keep
        # the original velocity curve, not just a statistic about it).
        derived['recovery_curve'] = recovery_curve_row
        derived['recovery_curve_samples'] = len(recovery_curve_row)

        # The historical `criteria` / `success` pair is REMOVED, not deprecated.
        # Phase C computed `cleared_obstacle` from the tier-1 latch, and that latch
        # is satisfied by a pass over the box's height band alone -- it says
        # nothing about the wheels, the landing or the whole-body rear margin.
        # Keeping the old key around with a stricter value would let a downstream
        # reader believe the two numbers are the same measurement, and keeping it
        # with the old value would let Phase C's false positives reappear. The
        # replacement is `m9_strict` above, and `derived['success']` now aliases
        # it so that anything still reading the old name gets the honest answer.
        derived['success'] = derived['m9_strict']['success']
        # Explicit audit trail: which criteria the pre-Phase-D code would have
        # used, so the delta between the two verdicts is inspectable rather than
        # asserted. Values are recorded; they are never ANDed into `success`.
        derived['deprecated_criteria'] = {
            '_note': ('Phase A/B/C criteria, kept for forensic comparison only. '
                      '`cleared_obstacle` is the tier-1 pass latch and is NOT a '
                      'success condition. Never combine these into a verdict.'),
            'took_off': outcome['takeoff_step'] >= 0,
            'touched_down': outcome['touchdown_step'] >= 0,
            'recovery_70pct_sustained': derived['recovery_sustained'],
            'no_task_failure': not outcome['task_failed'],
            'not_reset_by_fall': not outcome['fell'],
            'pre_jump_speed': bool(pre_jump_vel is not None and pre_jump_vel >= 0.15),
            'no_obstacle_contact': (not outcome['collision']) if with_obstacle else None,
            'cleared_obstacle': outcome['cleared'] if with_obstacle else None,
            'positive_clearance': ((outcome['min_clearance_m'] > 0.0)
                                   if with_obstacle else None),
        }

        # ---- phase markers (task 5.3) --------------------------------------
        # Five checkpoints the acceptance report compares between the two arms:
        #   before first contact / at trigger / at real liftoff / at box passage /
        #   after touchdown.
        # Every entry is a pre-reset trace row at that instant. The terminating
        # step's row is repaired from the episode snapshot right after the step
        # loop (see `post_reset` handling above), because reset_idx runs inside
        # that step and would otherwise leave a pristine stand-still pose in it.
        # Lookup is by `common_step` (the env's own clock, which is what the
        # latched steps are stamped with), not by the loop index.
        by_common_step = {row['common_step']: row for row in history}

        def at(step):
            if step is None or step < 0:
                return None
            return by_common_step.get(int(step))

        contact_step = outcome['first_contact_step']
        derived['phase_markers'] = {
            'before_first_contact': at((contact_step - 1) if contact_step > 0 else 0),
            'at_trigger': at(outcome['trigger_step']),
            'at_first_contact': at(contact_step),
            'at_liftoff': at(outcome['takeoff_step']),
            'at_box_passage': at(outcome['pass_step']),
            'at_touchdown': at(outcome['touchdown_step']),
            'after_touchdown': at((outcome['touchdown_step'] + 10)
                                  if outcome['touchdown_step'] >= 0 else -1),
            'at_reset': at(outcome['reset_step'] - 1) if outcome['reset_step'] > 0 else None,
        }

    result = {
        'seed': known.seed,
        'with_obstacle': with_obstacle,
        'pipeline': known.pipeline,
        'contact_pairs_available': bool(env.contact_pairs_available),
        'commanded_speed': cmd_vy,
        'obstacle': {
            'height_m': float(env.obstacle_h),
            'width_m': float(env.obstacle_w),
            'centre_y': round(float(obstacles[0, 1]), 5),
        },
        'model_path': known.model_path,
        'outcome': outcome,
        'derived': derived,
        'first_contact': first_contact,
        'history': history,
    }
    print('===EVAL_RESULT===')
    print(json.dumps(result))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_all(seeds, steps, model_path, pipeline, action_mode='deterministic'):
    results = []
    for seed in seeds:
        for with_obs in (1, 0):
            label = 'obstacle' if with_obs else 'free-run'
            print('  seed %-3d %-9s ... ' % (seed, label), end='', flush=True)
            cmd = [sys.executable, os.path.abspath(__file__), '--worker',
                   '--seed', str(seed), '--steps', str(steps),
                   '--model_path', model_path, '--pipeline', pipeline,
                   '--with_obstacle', str(with_obs),
                   '--action_mode', action_mode,
                   '--task', 'myrobot_obstacle_jump', '--headless']
            proc = subprocess.run(cmd, cwd=WORKSPACE, capture_output=True, text=True)
            payload = None
            marker = '===EVAL_RESULT==='
            if marker in proc.stdout:
                blob = proc.stdout.split(marker, 1)[1].strip().splitlines()[0]
                payload = json.loads(blob)
            if payload is None:
                print('FAILED')
                print(proc.stdout[-2000:], proc.stderr[-2000:])
                results.append({'seed': seed, 'with_obstacle': bool(with_obs),
                                'error': 'worker produced no result'})
                continue
            snap = payload.get('outcome') or {}
            deriv = payload.get('derived') or {}
            clr = deriv.get('m5_contact', {}).get('min_obstacle_clearance_m')
            print('reason=%-22s cleared=%s clr=%s flight=%s' % (
                snap.get('reason', '-'), snap.get('cleared'),
                'n/a' if clr is None else '%+.4f' % clr,
                'n/a' if deriv.get('flight_dist_m') is None
                else '%.3f' % deriv['flight_dist_m']))
            results.append(payload)
    return results


def summarise(results):
    """Aggregate one arm, keeping the three verdicts apart (Phase D task 4).

    The pre-Phase-D version returned a single `success` count taken from the
    tier-1 `cleared` latch. That number is gone from this function: the three
    accounts below answer three different questions and are never summed into a
    total. A run that clears the box with all wheels clean but never satisfies the
    recovery gate appears in `collision_free_safe_pass` and not in
    `strict_obstacle_clearance` -- which is the whole point of reporting them
    separately, and is exactly what Phase C reported as a contradiction.
    """

    def agg(pred):
        rows = [r for r in results if 'outcome' in r and r['outcome'] and pred(r)]
        if not rows:
            return {'n': 0}
        d = [r['derived'] for r in rows]

        def mean_of(key):
            """Average over runs where the metric exists (None = not applicable)."""
            vals = [x[key] for x in d if x.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        def count_of(key):
            """How many runs actually contributed to `mean_of(key)`.

            Reported alongside every mean on purpose: `mean_flight_m` averages
            only the runs that landed (2 of 8 here) while `mean_required_m`
            averages the runs that took off (6 of 8). Quoting those two numbers
            side by side without their denominators invites reading a difference
            that no single run exhibits.
            """
            return sum(1 for x in d if x.get(key) is not None)

        def metric_of(key):
            """One of the ten task-book metrics, per run; None where undefined."""
            return [x.get(key) for x in d]

        def sum_field(name, getter):
            vals = [getter(x) for x in d if getter(x) is not None]
            return round(sum(vals), 4) if vals else None

        def avg(getter, ndigits=4):
            """Mean over the runs that supply a value, or None when none do.

            Returning 0.0 for an empty set is the most misleading thing a summary
            can do here: `takeover=0.000 s (n=0)` reads as "it took over
            instantly" when it actually means RECOVER never handed off at all.
            """
            vals = [v for v in (getter(x) for x in d) if v is not None]
            return round(sum(vals) / len(vals), ndigits) if vals else None

        def n_of(getter):
            """Denominator that belongs to the matching `avg` call."""
            return sum(1 for x in d if getter(x) is not None)

        # ---- the three accounts, each with its own denominator -------------
        def account(name):
            return {'count': sum(1 for x in d
                                 if x.get('accounts', {}).get(name, {}).get('value')),
                    'n': len(d)}

        return {
            'n': len(rows),
            # THE THREE ACCOUNTS. Never add these together, never report one as
            # "the" success rate.
            'account_strict_obstacle_clearance': account('strict_obstacle_clearance'),
            'account_collision_free_safe_pass': account('collision_free_safe_pass'),
            'account_end_to_end_recruise': account('end_to_end_recruise'),
            # ---- supporting counts (facts, not verdicts) -------------------
            # `cleared` is the tier-1 latch and is listed as a diagnostic only.
            'collision': sum(1 for r in rows if r['outcome']['collision']),
            'cleared_tier1_latch': sum(1 for r in rows if r['outcome']['cleared']),
            'took_off': sum(1 for r in rows if r['outcome']['takeoff_step'] >= 0),
            'touched_down': sum(1 for r in rows if r['outcome']['touched_down']),
            'both_wheels_pass_clean': sum(
                1 for r in rows if r['outcome'].get('wheel_both_pass_clean')),
            'landing_safe': sum(1 for r in rows if r['outcome'].get('landing_safe')),
            'whole_body_passed': sum(
                1 for r in rows if r['outcome'].get('whole_body_passed')),
            'no_obstacle_contact': sum(
                1 for r in rows if not r['outcome'].get('obstacle_contact', True)),
            # ---- metric 2: airborne ---------------------------------------
            'mean_flight_m': mean_of('flight_dist_m'),
            'n_flight_measured': count_of('flight_dist_m'),
            'mean_required_m': mean_of('required_dist_m'),
            'n_required_measured': count_of('required_dist_m'),
            'mean_shortfall_m': mean_of('shortfall_m'),
            'n_shortfall_measured': count_of('shortfall_m'),
            'mean_airborne_speed_mps': avg(
                lambda x: x.get('m2_airborne', {}).get(
                    'mean_horizontal_speed_mps')),
            'n_airborne_speed_measured': n_of(
                lambda x: x.get('m2_airborne', {}).get(
                    'mean_horizontal_speed_mps')),
            'mean_airborne_time_s': avg(
                lambda x: x.get('m2_airborne', {}).get('airborne_time_s'),
                4),
            # ---- metric 4/8: landing and the second after it --------------
            'mean_min_clearance_m': avg(
                lambda x: x.get('m5_contact', {}).get('min_obstacle_clearance_m')),
            'n_min_clearance_measured': n_of(
                lambda x: x.get('m5_contact', {}).get('min_obstacle_clearance_m')),
            'mean_peak_force_N': round(
                sum(r['outcome']['peak_force_N'] for r in rows) / len(rows), 1),
            'mean_post_landing_speed_mps': avg(
                lambda x: x.get('m8_post_landing_1s', {}).get('speed_mean_mps')),
            'n_post_landing_measured': n_of(
                lambda x: x.get('m8_post_landing_1s', {}).get('speed_mean_mps')),
            'mean_backslide_m': avg(
                lambda x: (x.get('m8_post_landing_1s') or {}).get('backslide_m')
                if (x.get('m8_post_landing_1s') or {}).get('n_rows') else None,
                5),
            'mean_max_abs_pitch_deg': avg(
                lambda x: x.get('m8_post_landing_1s', {}).get(
                    'max_abs_pitch_deg'), 2),
            # ---- metric 7: handoff census + takeover latency --------------
            'handoff_counts': {k: sum(
                1 for r in rows if r['outcome'].get('recovery_outcome') == k)
                for k in ('NORMAL_HANDOFF', 'TIMEOUT_HANDOFF', 'RECOVER_ABORT',
                          'NONE')},
            'mean_balance_takeover_latency_s': avg(
                lambda x: x.get('m7_handoff', {}).get(
                    'balance_takeover_latency_s'), 3),
            'n_takeover_measured': n_of(
                lambda x: x.get('m7_handoff', {}).get(
                    'balance_takeover_latency_s')),
            'mean_best_hold_steps': avg(
                lambda x: x.get('m7_handoff', {}).get('best_hold_steps'), 2),
            # ---- recovery verdicts (strict gate, kept for comparison) -----
            'mean_recovery_ratio': mean_of('recovery_ratio'),
            'fast_recovery_040s': sum(1 for r in rows if r['outcome']['fast_recovered']),
            'recovery_70pct': sum(1 for x in d if x.get('recovery_sustained')),
            # ---- task 4 extra: did it actually re-cruise after the handoff
            # These three are NOT verdicts and must never be read as a success
            # rate: they answer "did the machine end up cruising again", which
            # the strict gate deliberately refuses to accept when RECOVER was
            # abandoned. Reported separately precisely so the refusal stays
            # visible next to the fact.
            'recruited_after_takeover': sum(
                1 for x in d
                if x.get('recovery_after_balance_takeover', {}).get(
                    'recovered_at_all')),
            'n_recruise_scan_measured': sum(
                1 for x in d
                if x.get('recovery_after_balance_takeover', {}).get(
                    'scanned_steps')),
            'recruited_only_after_handoff': sum(
                1 for x in d
                if x.get('recovery_after_balance_takeover', {}).get(
                    'recovered_only_after_handoff')),
            'mean_seconds_touchdown_to_cruise': avg(
                lambda x: x.get('recovery_after_balance_takeover', {}).get(
                    'seconds_from_touchdown'), 3),
            'n_seconds_to_cruise_measured': n_of(
                lambda x: x.get('recovery_after_balance_takeover', {}).get(
                    'seconds_from_touchdown')),
            'mean_seconds_handoff_to_cruise': avg(
                lambda x: x.get('recovery_after_balance_takeover', {}).get(
                    'seconds_from_handoff'), 3),
            # m10, both readings, side by side and never averaged together
            'recruise_within_1s': sum(
                1 for x in d
                if x.get('m10_end_to_end_recruise', {}).get(
                    'within_1s', {}).get(
                        'reached_70pct_for_10_consecutive_steps')),
            'recruise_whole_episode': sum(
                1 for x in d
                if x.get('m10_end_to_end_recruise', {}).get(
                    'whole_episode', {}).get(
                        'reached_70pct_for_10_consecutive_steps')),
            'reasons': {k: sum(1 for r in rows if r['outcome']['reason'] == k)
                        for k in sorted({r['outcome']['reason'] for r in rows})},
        }

    return {
        'obstacle': agg(lambda r: r['with_obstacle']),
        'free_run': agg(lambda r: not r['with_obstacle']),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--task', default='myrobot_obstacle_jump')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--pipeline', default='cpu')
    ap.add_argument('--num_envs', type=int, default=1)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--model_path', default=DEFAULT_MODEL)
    ap.add_argument('--with_obstacle', type=int, default=1)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    ap.add_argument('--action_mode', default='deterministic',
                    choices=['deterministic', 'stochastic'],
                    help='deterministic = act_inference (benchmark); '
                         'stochastic = sampled act(), an independent test mode '
                         'whose numbers are never averaged with deterministic ones')
    ap.add_argument('--overwrite', action='store_true',
                    help='allow replacing an existing results file')
    ap.add_argument('--label', default=None,
                    help='tag inserted into the default output filename, e.g. '
                         '"phaseD_before"; keeps a new campaign from landing on '
                         'top of Phase A/B/C data')
    ap.add_argument('--model_label', default=None,
                    help='human label for the checkpoint, e.g. "BEFORE(model_3)"; '
                         'travels with the numbers so a merged table cannot '
                         'silently attribute a run to the wrong network')
    ap.add_argument('--out', default=None,
                    help='output JSON; defaults to a name that encodes the '
                         'action mode so one mode can never overwrite the other')
    args = ap.parse_args()

    if args.worker:
        run_worker()
        return 0

    if args.out is None:
        # Phase A was measured with sampled actions. Encoding the mode in the
        # filename guarantees the new deterministic data lands in a NEW file and
        # the historical numbers stay readable side by side. The label does the
        # same for successive campaigns: Phase D must not land on the Phase B
        # file, and a bare mode name would do exactly that.
        stem = ('obstacle_jump_%s_%s' % (args.label, args.action_mode)
                if args.label else 'obstacle_jump_reeval_%s' % args.action_mode)
        args.out = 'evaluation_assets/%s_results.json' % stem
    out_path = os.path.join(WORKSPACE, args.out)
    if os.path.exists(out_path) and not args.overwrite:
        print('refusing to overwrite existing results: %s' % out_path)
        print('pass --overwrite if you really mean it, or choose another --out')
        return 2

    print('=' * 78)
    print('obstacle-jump physical evaluation (no training, no weight update)')
    print('  model  : %s' % args.model_path)
    print('  label  : %s' % (args.model_label or '(unlabelled)'))
    print('  seeds  : %s' % args.seeds)
    print('  pipe   : %s' % args.pipeline)
    print('  action : %s' % args.action_mode)
    print('=' * 78)
    results = run_all(args.seeds, args.steps, args.model_path, args.pipeline,
                      args.action_mode)

    summary = summarise(results)
    payload = {
        'evaluation': 'obstacle_jump_phaseD1_safe_recovery',
        'note': ('Pre-reset snapshots; three-layer URDF geometry (approach / '
                 'wheel crossing / whole-body); collision evidence from the '
                 'contact pair AND an independent geometric overlap source; '
                 'safe 2 s sustained recovery gate. Phase D.1 '
                 'reads no tier-1 `cleared` latch as a verdict.'),
        'model_path': args.model_path,
        'model_label': args.model_label,
        'model_sha256': _sha256_of(args.model_path),
        'seeds': args.seeds,
        # The action mode is a property of the measurement, not a detail: a
        # sampled-action run and a deterministic run are NOT samples of the same
        # distribution, so the mode must travel with the numbers. Never merge
        # summaries produced under different modes.
        'action_mode': args.action_mode,
        'action_mode_note': (
            'deterministic = policy.act_inference(obs), the acceptance mode; '
            'stochastic = policy.act(obs) sampling, kept as an independent '
            'test condition only. Results from the two modes must not be '
            'mixed in a single mean.'),
        'commanded_speed': next((r.get('commanded_speed') for r in results
                                 if 'commanded_speed' in r), None),
        'recovery_speed_fraction': RECOVERY_SPEED_FRACTION,
        'recovery_window_steps': RECOVERY_WINDOW,
        'summary': summary,
        'runs': [{k: v for k, v in r.items() if k != 'history'} for r in results],
        'traces': {('seed%d_%s' % (r['seed'], 'obs' if r['with_obstacle'] else 'free')):
                   r.get('history', []) for r in results},
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(payload, fh, indent=2)

    print('\n' + '=' * 78)
    for arm, s in summary.items():
        if not s.get('n'):
            print('%-10s no completed episodes' % arm)
            continue

        def fmt(key, spec='%.3f'):
            v = s.get(key)
            return 'n/a' if v is None else (spec % v)

        print('%-10s n=%d  collision=%d  tier1_latch=%d  '
              'took_off=%d  landed=%d' % (
                  arm, s['n'], s['collision'], s['cleared_tier1_latch'],
                  s['took_off'], s['touched_down']))
        # THE THREE ACCOUNTS, printed as three separate fractions. No total line
        # exists on purpose: the task book forbids merging them, and a reader who
        # wants "the" success rate has to choose which question they are asking.
        for key, label in (('account_strict_obstacle_clearance',
                            'strict clearance  '),
                           ('account_collision_free_safe_pass',
                            'collision-free    '),
                           ('account_end_to_end_recruise',
                            're-cruise (e2e)   ')):
            acc = s.get(key) or {'count': 0, 'n': s['n']}
            print('           %s %d/%d' % (label, acc['count'], acc['n']))
        print('           tiers: wheels_clean=%d  landing_safe=%d  '
              'whole_body_passed=%d  no_contact=%d' % (
                  s['both_wheels_pass_clean'], s['landing_safe'],
                  s['whole_body_passed'], s['no_obstacle_contact']))
        # Each mean is printed with its own denominator: they differ, and the
        # difference is the whole point (see `count_of`).
        print('           flight=%s m (n=%d)  required=%s m (n=%d)  '
              'shortfall=%s m (n=%d)' % (
                  fmt('mean_flight_m'), s['n_flight_measured'],
                  fmt('mean_required_m'), s['n_required_measured'],
                  fmt('mean_shortfall_m'), s['n_shortfall_measured']))
        print('           air_speed=%s m/s (n=%d)  air_time=%s s  '
              'post_land_speed=%s m/s (n=%d)  backslide=%s m' % (
                  fmt('mean_airborne_speed_mps'), s['n_airborne_speed_measured'],
                  fmt('mean_airborne_time_s'),
                  fmt('mean_post_landing_speed_mps'), s['n_post_landing_measured'],
                  fmt('mean_backslide_m', '%.4f')))
        print('           handoff=%s  takeover=%s s (n=%d)  best_hold=%s steps' % (
            s['handoff_counts'], fmt('mean_balance_takeover_latency_s'),
            s['n_takeover_measured'], fmt('mean_best_hold_steps', '%.1f')))
        ratio = s.get('mean_recovery_ratio')
        clr = s.get('mean_min_clearance_m')
        print('           min_clearance=%s m (n=%d)  peak_force=%.0f N  '
              'recovery_mean=%s (%d/%d strictly sustained)' % (
                  'n/a' if clr is None else '%+.4f' % clr,
                  s['n_min_clearance_measured'], s['mean_peak_force_N'],
                  'n/a' if ratio is None else '%.0f%%' % (100.0 * ratio),
                  s['recovery_70pct'], s['n']))
        # Task 4's extra question, printed next to the strict refusal so the two
        # facts cannot be confused for one another.
        print('           RECRUISE after handoff: any=%d/%d  only-after=%d  '
              'td->cruise=%s s (n=%d)  handoff->cruise=%s s' % (
                  s['recruited_after_takeover'], s['n'],
                  s['recruited_only_after_handoff'],
                  fmt('mean_seconds_touchdown_to_cruise'),
                  s['n_seconds_to_cruise_measured'],
                  fmt('mean_seconds_handoff_to_cruise')))
        print('           RECRUISE m10: within_1s=%d/%d  whole_episode=%d/%d' % (
            s['recruise_within_1s'], s['n'],
            s['recruise_whole_episode'], s['n']))
        print('           reasons=%s' % s['reasons'])
    print('\nwrote %s' % out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
