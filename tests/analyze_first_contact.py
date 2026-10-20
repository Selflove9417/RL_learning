#!/usr/bin/env python3
"""Attribute the first obstacle contact to a robot rigid body, with world pose.

The frozen `paired_obstacle_comparison_results.json` established that on all 8
evaluation seeds the first obstacle contact happens BEFORE the true liftoff
steps 56-58 vs liftoff 62-65), but it could not say which body touched the box:
it only had `peak_obstacle_contact_force` from the obstacle-side net force
tensor, which is a scalar sum over every contact the box has.

This script answers that question with two independent evidence sources:

  contact_pair
      Isaac Gym's rigid-contact list. Gives `body0`/`body1` indices and the
      contact `lambda`, so the body is read off directly. REQUIRES the CPU
      pipeline: on the GPU pipeline `gym.get_env_rigid_contacts()` raises
      "Function GymGetEnvRigidContacts cannot be used with the GPU pipeline
      after simulation starts. Please use the tensor API if possible."

  analytic_geometry
      Falls back to the URDF-derived capsule/box test when no impulse exists.
      This is EXACT for the wheels (the box is axis aligned and
      `replace_cylinder_with_capsule` is on, so the wheel collision volume is a
      swept sphere) but it CANNOT speak for the non-wheel links, which are
      triangle meshes with no penetration depth available.

Whatever the source, the contact is only reported when there is positive
evidence: a real impulse, or a genuine analytic overlap. A PhysX proximity
record with `lambda == 0` and `minDist == contact_offset` is a speculative
contact, not a collision, and is explicitly excluded.

Usage (from the repository root, after `source ./activate_jump.sh`):

    python tests/analyze_first_contact.py --pipeline cpu
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


def run_worker():
    import isaacgym  # noqa: F401  (must precede torch)
    import torch
    from legged_gym.envs import MyRobotObstacleJump  # noqa: F401
    from legged_gym.envs.myrobot import obstacle_geometry as geo
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

    env, _ = task_registry.make_env(name=known.task, args=args, env_cfg=cfg)
    env.reset()

    ckpt = torch.load(known.model_path, map_location=env.device)
    policy = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128],
                         'elu', 1.0).to(env.device)
    policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()

    obs = env.get_observations() if hasattr(env, 'get_observations') else env.obs_buf
    evt = {'liftoff': None, 'touchdown': None, 'contact': None}
    track = []
    final_snapshot = None

    for step_i in range(known.steps + 1):
        with torch.no_grad():
            # Deterministic actions, so the attributed frames belong to the same
            # trajectory that evaluate_obstacle_jump.py measures. Sampling here
            # would explain a run nobody is being asked to accept.
            actions = policy.act_inference(obs)
        obs, _, _, dones, _ = env.step(actions)
        done = bool(dones[0].item())

        # IMPORTANT: on the terminal step `reset_idx` has already run inside
        # step(), which clears first_contact_* / takeoff_step / touchdown_step.
        # Reading those live buffers after step() returns therefore reports "no
        # contact" on exactly the episodes that DID collide -- the first obstacle
        # contact in this benchmark almost always coincides with the fatal one.
        # The pre-reset `episode_snapshot` is the authoritative copy.
        snap = env.episode_snapshot if done else None
        if snap is not None and not bool(snap['valid'][0].item()):
            snap = None

        def live_or_snap(tensor_key, scalar_key):
            """Return (value, ) preferring the pre-reset snapshot when present."""
            if snap is not None:
                return snap[tensor_key][0]
            return getattr(env, scalar_key)[0]

        if evt['liftoff'] is None:
            tstep = int(live_or_snap('takeoff_step', 'takeoff_step').item())
            if tstep >= 0:
                evt['liftoff'] = {
                    'step': tstep,
                    'root_y': round(float(live_or_snap(
                        'takeoff_pos_y', 'takeoff_pos_y').item()), 5),
                    'front_y': round(float(live_or_snap(
                        'takeoff_front_y', 'takeoff_front_y').item()), 5),
                    'clearance_above_box': round(float(live_or_snap(
                        'takeoff_clearance_above_obs',
                        'takeoff_clearance_above_obs').item()), 5),
                    'margin_to_box': round(float(live_or_snap(
                        'takeoff_margin_m', 'takeoff_margin_m').item()), 5),
                }
        if evt['touchdown'] is None:
            tdstep = int(live_or_snap('touchdown_step', 'touchdown_step').item())
            if tdstep >= 0:
                evt['touchdown'] = {
                    'step': tdstep,
                    'root_y': round(float(live_or_snap(
                        'touchdown_pos_y', 'touchdown_pos_y').item()), 5),
                }
        if evt['contact'] is None:
            cstep = int(live_or_snap('first_contact_step',
                                     'first_contact_step').item())
            if cstep >= 0:
                cbody = int(live_or_snap('first_contact_body',
                                         'first_contact_body').item())
                cforce = float(live_or_snap('first_contact_force',
                                            'first_contact_force').item())
                cnet = float(live_or_snap('first_contact_net_force',
                                          'first_contact_net_force').item())
                cpos = live_or_snap('first_contact_pos', 'first_contact_pos')
                csrc = (snap['first_contact_source'][0] if snap is not None
                        else env.first_contact_source[0])
                evt['contact'] = {
                    'step': cstep,
                    'body_index': cbody,
                    'pair_force_N': round(cforce, 1),
                    'net_force_N': round(cnet, 1),
                    'force_unit': 'N',
                    'source': csrc,
                    'world_pos': [round(float(v), 5) for v in cpos],
                }

        gaps = geo.wheel_box_signed_gap(env)
        fr, rr, tz = geo.obstacle_y_interval(env)
        envl = geo.robot_forward_interval(env, z_ceiling=tz)
        track.append({
            'step': step_i,
            'fsm': int(env.fsm_state[0]),
            'root_y': round(float(env.root_states[0, 1]), 5),
            'root_z': round(float(env.root_states[0, 2]), 5),
            'vz': round(float(env.root_states[0, 9]), 4),
            'front_y': round(float(envl['front_y'][0]), 5),
            'front_body': int(envl['front_body'][0]),
            'left_gap': round(float(gaps[0, 0]), 5),
            'right_gap': round(float(gaps[0, 1]), 5),
            'dist_to_box': round(float(fr[0] - envl['front_y'][0]), 5),
            'taken_off': int(env.has_taken_off[0]),
            'touched_down': int(env.has_touched_down[0]),
            'collision': int(env.has_obstacle_collision[0]),
        })

        if bool(dones[0].item()):
            # Captured before reset_idx ran, inside step(); survives the reset.
            if bool(env.episode_snapshot['valid'][0].item()):
                final_snapshot = env.episode_snapshot
            break

    for key, rec in evt.items():
        if rec and 'body_index' in rec:
            names = env.link_names
            rec['body_name'] = (names[rec['body_index']]
                                if 0 <= rec['body_index'] < len(names) else 'unknown')

    # Answer the question the historic report left open.
    answer = {}
    if evt['contact'] and evt['liftoff']:
        answer['first_contact_before_liftoff'] = (
            evt['contact']['step'] < evt['liftoff']['step'])
        answer['contact_vs_liftoff_steps'] = (evt['liftoff']['step']
                                              - evt['contact']['step'])
    elif evt['contact']:
        answer['first_contact_before_liftoff'] = True   # never took off at all
        answer['contact_vs_liftoff_steps'] = None
    else:
        answer['first_contact_before_liftoff'] = None
        answer['contact_vs_liftoff_steps'] = None

    # Final outcome, taken from the pre-reset snapshot captured on the terminal
    # step (see `_capture_episode_snapshot`). Falls back to live buffers only
    # when the loop ended without a terminal step (e.g. --steps ran out).
    final = {
        'reason': 'UNKNOWN',
        'collision': False,
        'cleared': False,
        'min_clearance_m': None,
        'peak_force_N': None,
        'took_off': False,
        'touched_down': False,
        'fell': False,
        'source': 'live',
    }
    if final_snapshot is not None:
        final = {
            'reason': final_snapshot['failure_reason'][0],
            'collision': bool(final_snapshot['has_obstacle_collision'][0].item()),
            'cleared': bool(final_snapshot['obstacle_cleared'][0].item()),
            'min_clearance_m': round(
                float(final_snapshot['min_obstacle_clearance'][0]), 5),
            'peak_force_N': round(
                float(final_snapshot['peak_obstacle_contact_force'][0]), 1),
            'took_off': int(final_snapshot['takeoff_step'][0].item()) >= 0,
            'touched_down': int(final_snapshot['touchdown_step'][0].item()) >= 0,
            'fell': bool(final_snapshot['fell'][0].item()),
            'source': 'episode_snapshot',
        }
    else:
        final['reason'] = env.failure_reason[0]
        final['collision'] = bool(env.has_obstacle_collision[0].item())
        final['cleared'] = bool(env.obstacle_cleared[0].item())
        final['min_clearance_m'] = round(float(env.min_obstacle_clearance[0]), 5)
        final['peak_force_N'] = round(float(env.peak_obstacle_contact_force[0]), 1)

    result = {
        'seed': known.seed,
        'pipeline': known.pipeline,
        'contact_pairs_available': bool(env.contact_pairs_available),
        'events': evt,
        'answer': answer,
        'final': final,
        'track': track,
    }
    print('===CONTACT_RESULT===')
    print(json.dumps(result))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--task', default='myrobot_obstacle_jump')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--pipeline', default='cpu')
    ap.add_argument('--num_envs', type=int, default=1)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--model_path', default=DEFAULT_MODEL)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    ap.add_argument('--out', default='evaluation_assets/first_contact_attribution.json')
    args = ap.parse_args()

    if args.worker:
        run_worker()
        return 0

    print('=' * 78)
    print('first-obstacle-contact attribution (frozen model_3.pt, no training)')
    print('  seeds: %s   pipeline: %s' % (args.seeds, args.pipeline))
    print('=' * 78)

    rows = []
    for seed in args.seeds:
        print('  seed %-3d ... ' % seed, end='', flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), '--worker',
               '--seed', str(seed), '--steps', str(args.steps),
               '--model_path', args.model_path, '--pipeline', args.pipeline,
               '--task', args.task, '--headless']
        proc = subprocess.run(cmd, cwd=WORKSPACE, capture_output=True, text=True)
        payload = None
        if '===CONTACT_RESULT===' in proc.stdout:
            blob = proc.stdout.split('===CONTACT_RESULT===', 1)[1].strip().splitlines()[0]
            payload = json.loads(blob)
        if payload is None:
            print('FAILED')
            print(proc.stdout[-1500:], proc.stderr[-1500:])
            rows.append({'seed': seed, 'error': 'worker produced no result'})
            continue
        c = payload['events']['contact']
        l = payload['events']['liftoff']
        if c:
            print('contact step %-3d body=%-9s src=%-15s before_liftoff=%s' % (
                c['step'], c.get('body_name', '?'), c['source'],
                payload['answer']['first_contact_before_liftoff']))
        elif l:
            print('no contact; liftoff step %d' % l['step'])
        else:
            print('no contact, no liftoff')
        rows.append(payload)

    ok = [r for r in rows if 'events' in r]
    summary = {
        'n_runs': len(rows),
        'n_with_contact': sum(1 for r in ok if r['events']['contact']),
        'n_contact_before_liftoff': sum(
            1 for r in ok if r['answer'].get('first_contact_before_liftoff')),
        'bodies': {},
        'sources': {},
        'n_liftoff': sum(1 for r in ok if r['events']['liftoff']),
        'n_cleared': sum(1 for r in ok if r['final']['cleared']),
        'n_collision': sum(1 for r in ok if r['final']['collision']),
    }
    for r in ok:
        c = r['events']['contact']
        if not c:
            continue
        summary['bodies'][c.get('body_name', '?')] = \
            summary['bodies'].get(c.get('body_name', '?'), 0) + 1
        summary['sources'][c['source']] = summary['sources'].get(c['source'], 0) + 1

    payload = {
        'evaluation': 'first_contact_attribution',
        'note': ('contact_pair requires the CPU pipeline (GPU pipeline rejects '
                 'get_env_rigid_contacts after simulation starts). analytic_geometry '
                 'is exact only for the wheels; non-wheel links are triangle meshes.'),
        'model_path': args.model_path,
        'seeds': args.seeds,
        'summary': summary,
        'runs': [{k: v for k, v in r.items() if k != 'track'} for r in rows],
        'tracks': {'seed%d' % r['seed']: r.get('track', []) for r in rows},
    }
    out_path = os.path.join(WORKSPACE, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(payload, fh, indent=2)

    print('\n' + '=' * 78)
    print('runs=%d  with_contact=%d  contact_before_liftoff=%d  liftoff=%d' % (
        summary['n_runs'], summary['n_with_contact'],
        summary['n_contact_before_liftoff'], summary['n_liftoff']))
    print('first-contact bodies : %s' % summary['bodies'])
    print('evidence sources     : %s' % summary['sources'])
    print('collision=%d  cleared=%d' % (summary['n_collision'], summary['n_cleared']))
    print('wrote %s' % out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
