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


# ---------------------------------------------------------------------------
# Worker: one seed, one condition
# ---------------------------------------------------------------------------

def run_worker():
    """Run a single (seed, condition) rollout and print one JSON line."""
    import isaacgym  # noqa: F401  (must precede torch)
    import torch
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
    pre_jump_vel = None
    first_contact = None
    outcome = None
    obs = env.get_observations() if hasattr(env, 'get_observations') else env.obs_buf

    # One extra step: a 200-step episode (episode_length_s=4.0) only reports
    # `dones` on the step AFTER the counter reaches max_episode_length, so a
    # loop of exactly 200 steps would never observe the timeout reset.
    for step_i in range(known.steps + 1):
        # Control-arm trigger: identical spatial criterion, virtual plane.
        # The z-ceiling is taken from the (far away) box so that both arms gate
        # the front edge with exactly the same envelope definition.
        if want_trigger:
            _, _, ceil_z = geo.obstacle_y_interval(env)
            envl_t = geo.robot_forward_interval(env, z_ceiling=ceil_z)
            dist = float(virtual_front_y - envl_t['front_y'][0])
            if (trigger_dist >= dist >= -0.02
                    and int(env.fsm_state[0]) == 0
                    and int(env.episode_length_buf[0]) >= 20
                    and float(env.root_states[0, 8]) > 0.10):
                env.request_teleop_jump(torch.tensor([True], device=env.device))
                want_trigger = False
                print('  [control] virtual trigger fired at step %d, dist=%.4f' % (step_i, dist))

        with torch.no_grad():
            actions = policy.act(obs)
        obs, _, _, dones, infos = env.step(actions)

        # Forward speed at the moment the FSM commits to a jump. Captured live
        # because the FSM state is reset on the epilogue step.
        if pre_jump_vel is None and int(env.fsm_state[0]) == 3:
            pre_jump_vel = float(env.root_states[0, 8])

        # ---- pre-reset metrics: everything below is still the live episode ----
        front_y, rear_y, top_z = geo.obstacle_y_interval(env)
        envl = geo.robot_forward_interval(env, z_ceiling=top_z)
        gaps = geo.wheel_box_signed_gap(env)
        clrs = geo.wheel_clearance_above_box_top(env)
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
            'front_y': round(float(envl['front_y'][0]), 5),
            'rear_y': round(float(envl['rear_y'][0]), 5),
            'front_body': int(envl['front_body'][0]),
            'wheel_gap': round(float(torch.min(gaps[0]).item()), 5),
            'dist_to_box': round(float(front_y[0] - envl['front_y'][0]), 5),
            # Footprint-limited clearance (see obstacle_geometry.…). The old
            # `wheel_bottom_zs - top_z` reported -0.0477 m for a robot standing
            # next to the box, because a wheel's lowest point is directly under
            # its centre and therefore never actually above the footprint.
            # `None` means the wheels cannot reach the footprint at all.
            'wheel_above_top': (lambda v: None if not math.isfinite(v) else round(v, 5))(
                float(torch.min(clrs[0]).item())),
            'taken_off': int(env.has_taken_off[0]),
            'touched_down': int(env.has_touched_down[0]),
            'collision': int(env.has_obstacle_collision[0]),
            'cleared': int(env.obstacle_cleared[0]),
            'task_failed': int(env.task_failed[0]),
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
                }
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
                        'force_N': round(float(snap['first_contact_force'][0]), 1),
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

        vels = recovery_vels[-RECOVERY_WINDOW:] if recovery_vels else []
        if vels:
            mean_v = sum(vels) / len(vels)
            derived['recovery_mean_vel'] = round(mean_v, 4)
            derived['recovery_ratio'] = round(mean_v / cmd_vy, 4)
            derived['recovery_declining_steps'] = sum(
                1 for a, b in zip(vels, vels[1:]) if b < a - 1e-3)
            derived['recovery_sustained'] = bool(
                mean_v >= RECOVERY_SPEED_FRACTION * cmd_vy
                and derived['recovery_declining_steps'] <= len(vels) // 4)
        else:
            derived['recovery_mean_vel'] = 0.0
            derived['recovery_ratio'] = 0.0
            derived['recovery_sustained'] = False

        criteria = {
            # Derived from the latched step counters, not from the live flags:
            # the FSM clears `has_taken_off` when the balance policy resumes, so a
            # robot that jumped, landed and handed off ends its episode with
            # has_taken_off == False and would look like it never left the ground.
            'took_off': outcome['takeoff_step'] >= 0,
            'touched_down': outcome['touchdown_step'] >= 0,
            'recovery_70pct_sustained': derived['recovery_sustained'],
            'no_task_failure': not outcome['task_failed'],
            'not_reset_by_fall': not outcome['fell'],
        }
        if with_obstacle:
            # Criteria that only make sense when there is something to clear.
            criteria['pre_jump_speed'] = bool(pre_jump_vel is not None
                                              and pre_jump_vel >= 0.15)
            criteria['no_obstacle_contact'] = not outcome['collision']
            criteria['cleared_obstacle'] = outcome['cleared']
            criteria['positive_clearance'] = outcome['min_clearance_m'] > 0.0
        derived['pre_jump_speed_mps'] = None if pre_jump_vel is None else round(pre_jump_vel, 4)
        derived['criteria'] = criteria
        derived['success'] = all(criteria.values())

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

def run_all(seeds, steps, model_path, pipeline):
    results = []
    for seed in seeds:
        for with_obs in (1, 0):
            label = 'obstacle' if with_obs else 'free-run'
            print('  seed %-3d %-9s ... ' % (seed, label), end='', flush=True)
            cmd = [sys.executable, os.path.abspath(__file__), '--worker',
                   '--seed', str(seed), '--steps', str(steps),
                   '--model_path', model_path, '--pipeline', pipeline,
                   '--with_obstacle', str(with_obs),
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
            print('reason=%-22s cleared=%s clr=%+.3f flight=%s' % (
                snap.get('reason', '-'), snap.get('cleared'),
                snap.get('min_clearance_m', 0.0),
                'n/a' if deriv.get('flight_dist_m') is None
                else '%.3f' % deriv['flight_dist_m']))
            results.append(payload)
    return results


def summarise(results):
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

        return {
            'n': len(rows),
            'success': sum(1 for x in d if x.get('success')),
            'collision': sum(1 for r in rows if r['outcome']['collision']),
            'cleared': sum(1 for r in rows if r['outcome']['cleared']),
            'took_off': sum(1 for r in rows if r['outcome']['takeoff_step'] >= 0),
            'touched_down': sum(1 for r in rows if r['outcome']['touched_down']),
            'mean_flight_m': mean_of('flight_dist_m'),
            'n_flight_measured': count_of('flight_dist_m'),
            'mean_required_m': mean_of('required_dist_m'),
            'n_required_measured': count_of('required_dist_m'),
            'mean_shortfall_m': mean_of('shortfall_m'),
            'n_shortfall_measured': count_of('shortfall_m'),
            'mean_min_clearance_m': round(
                sum(r['outcome']['min_clearance_m'] for r in rows) / len(rows), 4),
            'mean_recovery_ratio': mean_of('recovery_ratio'),
            'recovery_70pct': sum(1 for x in d if x.get('recovery_sustained')),
            'mean_peak_force_N': round(
                sum(r['outcome']['peak_force_N'] for r in rows) / len(rows), 1),
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
    ap.add_argument('--out', default='evaluation_assets/obstacle_jump_reeval_results.json')
    args = ap.parse_args()

    if args.worker:
        run_worker()
        return 0

    print('=' * 78)
    print('obstacle-jump physical evaluation (frozen model_3.pt, no training)')
    print('  model  : %s' % args.model_path)
    print('  seeds  : %s' % args.seeds)
    print('  pipe   : %s' % args.pipeline)
    print('=' * 78)
    results = run_all(args.seeds, args.steps, args.model_path, args.pipeline)

    summary = summarise(results)
    payload = {
        'evaluation': 'obstacle_jump_reeval',
        'note': ('Pre-reset snapshots; URDF-derived geometric envelope; '
                 'impulse-or-overlap collision evidence; 70% recovery gate. '
                 'Does not overwrite any historical benchmark.'),
        'model_path': args.model_path,
        'seeds': args.seeds,
        'commanded_speed': next((r.get('commanded_speed') for r in results
                                 if 'commanded_speed' in r), None),
        'recovery_speed_fraction': RECOVERY_SPEED_FRACTION,
        'recovery_window_steps': RECOVERY_WINDOW,
        'summary': summary,
        'runs': [{k: v for k, v in r.items() if k != 'history'} for r in results],
        'traces': {('seed%d_%s' % (r['seed'], 'obs' if r['with_obstacle'] else 'free')):
                   r.get('history', []) for r in results},
    }
    out_path = os.path.join(WORKSPACE, args.out)
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

        print('%-10s n=%d  success=%d  collision=%d  cleared=%d  '
              'took_off=%d  landed=%d' % (
                  arm, s['n'], s['success'], s['collision'], s['cleared'],
                  s['took_off'], s['touched_down']))
        # Each mean is printed with its own denominator: they differ, and the
        # difference is the whole point (see `count_of`).
        print('           flight=%s m (n=%d)  required=%s m (n=%d)  '
              'shortfall=%s m (n=%d)' % (
                  fmt('mean_flight_m'), s['n_flight_measured'],
                  fmt('mean_required_m'), s['n_required_measured'],
                  fmt('mean_shortfall_m'), s['n_shortfall_measured']))
        ratio = s.get('mean_recovery_ratio')
        print('           min_clearance=%+.3f m  peak_force=%.0f N  recovery=%s (%d/%d sustained)' % (
            s['mean_min_clearance_m'], s['mean_peak_force_N'],
            'n/a' if ratio is None else '%.0f%%' % (100.0 * ratio),
            s['recovery_70pct'], s['n']))
        print('           reasons=%s' % s['reasons'])
    print('\nwrote %s' % out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
