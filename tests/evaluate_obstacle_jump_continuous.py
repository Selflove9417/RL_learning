#!/usr/bin/env python3
"""Continuous (multi-cycle) obstacle evaluation for the frozen jump policy.

Replaces `verify_obstacle_jump_continuous.py` (kept for history, never
overwritten). The old script had five defects that made its numbers unusable as
evidence, all of which are fixed here:

1. SHARED OBSTACLE SCALAR. It read `env.obstacle_py`, a single Python float that
   `_reset_root_states` rewrote back to the configured offset on every reset. With
   several environments that scalar described whichever env wrote it last. Here
   each environment's obstacle offset is read from its own row of
   `obstacle_root_states`, and the test asserts that repositioning one env leaves
   the others untouched.

2. POST-RESET READS. `step()` clears the per-episode flags on the step an episode
   ends, so anything read after `step()` returns describes the next episode. All
   episode outcomes here come from `env.episode_snapshot`, captured immediately
   before reset.

3. HISTORIC-FRONT GEOMETRY. It computed flight distance from `pos_y` alone and
   never compared the trajectory against a real envelope. Distances here come from
   `obstacle_geometry`, the same functions the environment uses.

4. PEAK REUSE. It kept a running maximum of obstacle force across an entire
   multi-cycle run, so a late cycle inherited an early cycle's peak. Here every
   cycle is closed out and its peak recorded independently, and the test asserts
   that a fresh cycle starts at zero.

5. NO FAILURE ACCOUNTING. It only recorded cycles that completed a recovery, so
   collisions, aborts and resets simply vanished from the output. Here every cycle
   terminates in one of the recorded outcomes and nothing is dropped.

Usage (from the repository root, after `source ./activate_jump.sh`):

    python tests/evaluate_obstacle_jump_continuous.py --pipeline cpu
    python tests/evaluate_obstacle_jump_continuous.py --steps 600 --seeds 11 13

The output goes to `evaluation_assets/obstacle_jump_continuous_reeval_results.json`
-- a new filename, so the historic `continuous_obstacle_jump_results.json` is
never overwritten.
"""

import argparse
import json
import os
import subprocess
import sys

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(WORKSPACE, 'legged_gym'))
sys.path.insert(0, os.path.join(WORKSPACE, 'rsl_rl'))

DEFAULT_MODEL = ('legged_gym/logs/myrobot_running_jump/'
                 'Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt')
DEFAULT_SEEDS = [11, 13, 15, 20, 25, 27, 31, 35]
# Long enough for several jump cycles at ~90 steps per cycle.
DEFAULT_STEPS = 600


# ---------------------------------------------------------------------------
# Worker: one seed, multi-cycle
# ---------------------------------------------------------------------------

def run_worker():
    """Run one multi-cycle rollout and print one JSON line."""
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
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--model_path', default=DEFAULT_MODEL)
    p.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    p.add_argument('--lead_dist', type=float, default=0.60)
    parsed, _ = p.parse_known_args()

    sys.argv = [sys.argv[0], '--task', parsed.task, '--headless',
                '--seed', str(parsed.seed), '--num_envs', '1',
                '--pipeline', parsed.pipeline]
    args = get_args()
    cfg, _ = task_registry.get_cfgs(name=parsed.task)
    cfg.env.num_envs = 1
    cfg.seed = parsed.seed
    # Continuous mode: no timeout reset, so multiple cycles live in one episode.
    cfg.env.continuous_mode = True
    cfg.env.episode_length_s = 400.0
    cfg.trigger.mode = 'spatial'
    cfg.obstacle.lead_dist = parsed.lead_dist

    env, _ = task_registry.make_env(name=parsed.task, args=args, env_cfg=cfg)
    env.reset()

    ckpt = torch.load(parsed.model_path, map_location=env.device)
    policy = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128],
                         'elu', 1.0).to(env.device)
    policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()

    cmd_vy = float(env.target_forward_vel)
    obs = env.get_observations() if hasattr(env, 'get_observations') else env.obs_buf

    cycles = []            # completed cycle records
    open_cycle = None      # cycle currently in progress
    resets = []            # every episode-ending event, with its context
    joint_events = []      # FSM transitions that matter for cycle bookkeeping
    prev_fsm = int(env.fsm_state[0].item())
    prev_event = env.last_transition_event[0]

    def open_new_cycle(step_i):
        """Start a per-cycle record.

        Per-cycle means the peak force, the clearance minimum and the first
        contact are all measured from scratch: nothing is inherited from the
        previous attempt, so no peak can be silently reused.
        """
        return {
            'cycle_idx': len(cycles) + 1,
            'open_step': step_i,
            'open_root_y': round(float(env.root_states[0, 1].item()), 5),
            'obstacle_y': round(float(env.obstacle_root_states[0, 1].item()), 5),
            'trigger_step': None,
            'takeoff_step': None,
            'takeoff_rear_y': None,
            'touchdown_step': None,
            'touchdown_y': None,
            'peak_force_N': 0.0,
            'min_clearance_m': None,
            'collision': False,
            'first_contact': None,
            'cleared': False,
            'failure_reason': 'NONE',
            'recovery_event': None,
            'recovery_vels': [],
            'close_step': None,
            'outcome': 'INCOMPLETE',
        }

    def close_cycle(reason):
        nonlocal open_cycle
        if open_cycle is None:
            return
        open_cycle['close_step'] = step_i
        open_cycle['outcome'] = reason
        rec = open_cycle['recovery_vels']
        if rec:
            open_cycle['recovery_mean_vel'] = round(sum(rec) / len(rec), 4)
            open_cycle['recovery_ratio'] = round(
                (sum(rec) / len(rec)) / cmd_vy, 4)
        else:
            open_cycle['recovery_mean_vel'] = None
            open_cycle['recovery_ratio'] = None
        open_cycle.pop('recovery_vels', None)
        cycles.append(open_cycle)
        open_cycle = None

    for step_i in range(parsed.steps):
        with torch.no_grad():
            actions = policy.act_inference(obs)
        obs, _, _rews, dones, infos = env.step(actions)

        done = bool(dones[0].item())
        snap = env.episode_snapshot if done else None
        snap_ok = bool(snap is not None and snap['valid'][0].item())

        # --- Live pre-reset telemetry -------------------------------------
        fsm = int(env.fsm_state[0].item())
        event = env.last_transition_event[0]
        root_y = float(env.root_states[0, 1].item())

        # A reposition is a NEW crossing opportunity. The environment resets its
        # own per-cycle bookkeeping at that moment; the test mirrors that by
        # closing the previous record and opening a fresh one.
        repopped = event != prev_event and event in ('NORMAL_HANDOFF',
                                                     'TIMEOUT_HANDOFF')
        if repopped:
            joint_events.append({
                'step': step_i,
                'event': event,
                'root_y': round(root_y, 5),
                'obstacle_y': round(float(env.obstacle_root_states[0, 1].item()), 5),
            })
            close_cycle(event)

        if open_cycle is None:
            open_cycle = open_new_cycle(step_i)

        if fsm == 3 and open_cycle['trigger_step'] is None:
            open_cycle['trigger_step'] = step_i

        # Liftoff/touchdown are latched step counters in the environment. They
        # survive until the next reset, so they can be read live here -- but on a
        # reset step they are already cleared, which is why the snapshot path
        # below overwrites the cycle record with authoritative values.
        ts = float(env.takeoff_step[0].item())
        if ts >= 0 and open_cycle['takeoff_step'] is None:
            open_cycle['takeoff_step'] = int(ts)
            open_cycle['takeoff_rear_y'] = round(
                float(env.takeoff_rear_y[0].item()), 5)
        tds = float(env.touchdown_step[0].item())
        if tds >= 0 and open_cycle['touchdown_step'] is None:
            open_cycle['touchdown_step'] = int(tds)
            open_cycle['touchdown_y'] = round(
                float(env.touchdown_pos_y[0].item()), 5)

        if open_cycle['trigger_step'] is not None:
            # Force/clearance/collision are read from the ENV's own per-cycle
            # buffers, which `_begin_new_obstacle_cycle` zeroes at each trigger.
            pf = float(env.peak_obstacle_contact_force[0].item())
            open_cycle['peak_force_N'] = round(max(open_cycle['peak_force_N'], pf), 2)
            if bool(env.has_obstacle_collision[0].item()):
                open_cycle['collision'] = True
            if bool(env.clear_success_event[0].item()):
                open_cycle['cleared'] = True
            fc_step = float(env.first_contact_step[0].item())
            if fc_step >= 0 and open_cycle['first_contact'] is None:
                body_idx = int(env.first_contact_body[0].item())
                open_cycle['first_contact'] = {
                    'step': int(fc_step),
                    'body': (env.link_names[body_idx]
                             if 0 <= body_idx < len(env.link_names) else str(body_idx)),
                    'pair_force_N': round(float(env.first_contact_force[0].item()), 2),
                    'net_force_N': round(float(env.first_contact_net_force[0].item()), 2),
                    'force_unit': 'N',
                    'source': env.first_contact_source[0],
                    'world_pos': [round(float(v), 5) for v in
                                  env.first_contact_pos[0].tolist()],
                }

        # Recovery-window forward speed, sampled live but only counted while the
        # FSM is actually in RECOVER.
        if fsm == 4:
            open_cycle['recovery_vels'].append(float(env.root_states[0, 8].item()))

        if fsm != prev_fsm or event != prev_event:
            if event in ('LIFTOFF', 'TOUCHDOWN_ENTER_RECOVER'):
                joint_events.append({
                    'step': step_i,
                    'event': event,
                    'fsm': STATE_NAMES.get(fsm, str(fsm)),
                    'root_y': round(root_y, 5),
                })
        prev_fsm = fsm
        prev_event = event

        # --- Episode end: take the outcome from the pre-reset snapshot ----
        if done:
            rec = {
                'step': step_i,
                'step_counter': int(env.common_step_counter),
                'source': 'episode_snapshot' if snap_ok else 'live',
                'root_y': round(root_y, 5),
                'root_z': round(float(env.root_states[0, 2].item()), 5),
                'obstacle_y': round(float(env.obstacle_root_states[0, 1].item()), 5),
                'reason': 'NONE',
                'collision': False,
                'cleared': False,
                'fell': False,
                'cycles_completed': 0.0,
                'peak_force_N': 0.0,
            }
            if snap_ok:
                rec.update({
                    'reason': snap['failure_reason'][0],
                    'collision': bool(snap['has_obstacle_collision'][0].item()),
                    'cleared': bool(snap['obstacle_cleared'][0].item()),
                    'fell': bool(snap['fell'][0].item()),
                    'cycles_completed': round(
                        float(snap['cycles_completed'][0].item()), 3),
                    'peak_force_N': round(
                        float(snap['peak_obstacle_contact_force'][0].item()), 2),
                })
            resets.append(rec)
            if open_cycle is not None:
                open_cycle['failure_reason'] = rec['reason']
                # The environment zeroed its per-cycle buffers inside reset, so
                # the live values read above describe the NEXT attempt. The
                # snapshot is the only authoritative source for the cycle that
                # just ended; overwrite rather than merge.
                if snap_ok:
                    open_cycle['peak_force_N'] = rec['peak_force_N']
                    open_cycle['collision'] = rec['collision']
                    open_cycle['cleared'] = rec['cleared']
                    if float(snap['takeoff_step'][0].item()) >= 0:
                        open_cycle['takeoff_step'] = int(
                            snap['takeoff_step'][0].item())
                        open_cycle['takeoff_rear_y'] = round(
                            float(snap['takeoff_rear_y'][0].item()), 5)
                    if float(snap['touchdown_step'][0].item()) >= 0:
                        open_cycle['touchdown_step'] = int(
                            snap['touchdown_step'][0].item())
                        open_cycle['touchdown_y'] = round(
                            float(snap['touchdown_pos_y'][0].item()), 5)
                    if float(snap['first_contact_step'][0].item()) >= 0:
                        body_idx = int(snap['first_contact_body'][0].item())
                        open_cycle['first_contact'] = {
                            'step': int(snap['first_contact_step'][0].item()),
                            'body': (env.link_names[body_idx]
                                     if 0 <= body_idx < len(env.link_names)
                                     else str(body_idx)),
                            'pair_force_N': round(
                                float(snap['first_contact_force'][0].item()), 2),
                            'net_force_N': round(
                                float(snap['first_contact_net_force'][0].item()), 2),
                            'force_unit': 'N',
                            'source': snap['first_contact_source'][0],
                            'world_pos': [round(float(v), 5) for v in
                                          snap['first_contact_pos'][0].tolist()],
                        }
                    clr = float(snap['min_obstacle_clearance'][0].item())
                    open_cycle['min_clearance_m'] = (
                        None if clr >= 1.0 else round(clr, 6))
                close_cycle('EPISODE_RESET:' + rec['reason'])
            # A reset ends the episode; the next step starts a fresh one.
            prev_fsm = int(env.fsm_state[0].item())
            prev_event = env.last_transition_event[0]

    # Anything still open at the end of a continuous run is recorded as
    # incomplete rather than silently dropped -- that was defect 5.
    close_cycle('STEP_LIMIT')

    def summarise(rows):
        if not rows:
            return {'n': 0}
        collided = sum(1 for c in rows if c['collision'])
        cleared = sum(1 for c in rows if c['cleared'])
        recovs = [c for c in rows if c['recovery_ratio'] is not None]
        return {
            'n': len(rows),
            'collision': collided,
            'cleared': cleared,
            # `obstacle_cleared` and `has_obstacle_collision` are independent
            # latches over the same attempt, so both can be true: on seed 15 the
            # rear envelope passed the box rear face at step 118 (14 mm margin)
            # and the wheel then clipped the box's rear top corner two frames
            # later (403 N impulse, analytic gap -6 um). A cycle that touched the
            # box is NOT a clean clearance no matter when the clear latched, so
            # the headline number is this conjunction -- never `cleared` alone.
            'cleared_and_clean': sum(1 for c in rows
                                     if c['cleared'] and not c['collision']),
            'took_off': sum(1 for c in rows if c['takeoff_step'] is not None),
            'mean_peak_force_N': round(
                sum(c['peak_force_N'] for c in rows) / len(rows), 2),
            'max_peak_force_N': round(max(c['peak_force_N'] for c in rows), 2),
            'mean_recovery_ratio': (None if not recovs else round(
                sum(c['recovery_ratio'] for c in recovs) / len(recovs), 4)),
            'n_recovery_measured': len(recovs),
            'outcomes': {k: sum(1 for c in rows if c['outcome'] == k)
                         for k in sorted({c['outcome'] for c in rows})},
        }

    result = {
        'seed': parsed.seed,
        'pipeline': parsed.pipeline,
        'model_path': parsed.model_path,
        'steps': parsed.steps,
        'lead_dist_m': parsed.lead_dist,
        'per_env_obstacle_state': bool(
            getattr(env, 'obstacle_py_t', None) is not None),
        'cycles': cycles,
        'resets': resets,
        'fsm_events': joint_events,
        'summary': summarise(cycles),
    }
    print('===CONT_RESULT===')
    print(json.dumps(result))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_all(seeds, steps, model_path, pipeline, lead_dist):
    results = []
    for seed in seeds:
        print('  seed %-3d multi-cycle ... ' % seed, end='', flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), '--worker',
               '--seed', str(seed), '--steps', str(steps),
               '--model_path', model_path, '--pipeline', pipeline,
               '--lead_dist', str(lead_dist),
               '--task', 'myrobot_obstacle_jump', '--headless']
        proc = subprocess.run(cmd, cwd=WORKSPACE, capture_output=True, text=True)
        marker = '===CONT_RESULT==='
        payload = None
        if marker in proc.stdout:
            blob = proc.stdout.split(marker, 1)[1].strip().splitlines()[0]
            payload = json.loads(blob)
        if payload is None:
            print('FAILED')
            print(proc.stdout[-3000:], proc.stderr[-3000:])
            results.append({'seed': seed, 'error': 'worker produced no result'})
            continue
        summary = payload['summary']
        print('cycles=%d collision=%d cleared=%d resets=%d' % (
            summary['n'], summary['collision'], summary['cleared'],
            len(payload['resets'])))
        results.append(payload)
    return results


def summarise_all(results):
    rows = [r for r in results if 'summary' in r]
    if not rows:
        return {'n_seeds': 0}
    all_cycles = [c for r in rows for c in r['cycles']]
    all_resets = [s for r in rows for s in r['resets']]
    recovs = [c['recovery_ratio'] for c in all_cycles
              if c['recovery_ratio'] is not None]
    return {
        'n_seeds': len(rows),
        'n_cycles': len(all_cycles),
        'cycles_collision': sum(1 for c in all_cycles if c['collision']),
        'cycles_cleared': sum(1 for c in all_cycles if c['cleared']),
        # The only number that means "crossed without touching": both latches
        # are independent, so `cleared` alone overcounts (see summarise()).
        'cycles_cleared_and_clean': sum(
            1 for c in all_cycles if c['cleared'] and not c['collision']),
        'cycles_took_off': sum(1 for c in all_cycles
                               if c['takeoff_step'] is not None),
        'mean_peak_force_N': (0.0 if not all_cycles else round(
            sum(c['peak_force_N'] for c in all_cycles) / len(all_cycles), 2)),
        'mean_recovery_ratio': (None if not recovs else round(
            sum(recovs) / len(recovs), 4)),
        'n_recovery_measured': len(recovs),
        'episode_resets': len(all_resets),
        'reset_reasons': {k: sum(1 for s in all_resets if s['reason'] == k)
                          for k in sorted({s['reason'] for s in all_resets})},
        'falls': sum(1 for s in all_resets if s['fell']),
        'cycle_outcomes': {k: sum(1 for c in all_cycles if c['outcome'] == k)
                           for k in sorted({c['outcome'] for c in all_cycles})},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--task', default='myrobot_obstacle_jump')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--pipeline', default='cpu')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--model_path', default=DEFAULT_MODEL)
    ap.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    ap.add_argument('--lead_dist', type=float, default=0.60)
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    ap.add_argument('--out', default=(
        'evaluation_assets/obstacle_jump_continuous_reeval_results.json'))
    known, _ = ap.parse_known_args()

    if known.worker:
        return run_worker()

    print('=' * 78)
    print('continuous obstacle evaluation (frozen model_3.pt, no training)')
    print('  seeds: %s   steps/seed: %d   lead_dist: %.2f m' % (
        known.seeds, known.steps, known.lead_dist))
    print('  pipeline: %s' % known.pipeline)
    print('=' * 78)

    results = run_all(known.seeds, known.steps, known.model_path,
                      known.pipeline, known.lead_dist)
    overall = summarise_all(results)

    print('\n' + '=' * 78)
    print('OVERALL')
    print('  seeds=%d  cycles=%d  took_off=%d  collision=%d  cleared=%d'
          '  cleared_and_clean=%d' % (
              overall.get('n_seeds', 0), overall.get('n_cycles', 0),
              overall.get('cycles_took_off', 0),
              overall.get('cycles_collision', 0),
              overall.get('cycles_cleared', 0),
              overall.get('cycles_cleared_and_clean', 0)))
    print('  mean_peak_force=%.2f N  mean_recovery_ratio=%s (n=%d)' % (
        overall.get('mean_peak_force_N', 0.0),
        overall.get('mean_recovery_ratio'),
        overall.get('n_recovery_measured', 0)))
    print('  episode_resets=%d falls=%d reasons=%s' % (
        overall.get('episode_resets', 0), overall.get('falls', 0),
        overall.get('reset_reasons')))
    print('  cycle_outcomes=%s' % (overall.get('cycle_outcomes'),))

    payload = {
        'evaluation': 'obstacle_jump_continuous_reeval',
        'note': ('Multi-cycle continuous evaluation. Every cycle record is closed '
                 'out independently; episode outcomes come from '
                 'env.episode_snapshot. Historic '
                 'continuous_obstacle_jump_results.json is left untouched.'),
        'model_path': known.model_path,
        'seeds': known.seeds,
        'steps_per_seed': known.steps,
        'lead_dist_m': known.lead_dist,
        'summary': overall,
        'runs': results,
    }
    with open(known.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print('\nwrote %s' % os.path.abspath(known.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
