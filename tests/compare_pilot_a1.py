#!/usr/bin/env python3
"""Phase C step 4 -- frozen before/after comparison for the A1 pilot.

Reads the two deterministic 8-seed result files produced by
``tests/evaluate_obstacle_jump.py`` and reports the eight quantities the task
book asks for, side by side, with every denominator printed.

Three definitions are kept deliberately separate because conflating them is the
exact failure the task book warns about:

``lowparts``
    the env's own ``robot_forward_interval(z_ceiling=obstacle_top)`` envelope.
    Only the parts of the robot that can physically hit a 5 cm box are inside
    it, so its rear edge sits ~4.7 cm AHEAD of the root.  ``cleared_lowparts``
    therefore means "the low parts have passed the box", not "the robot has
    passed the box".

``wholebody``
    the full URDF support envelope with no z filter, whose rear edge sits
    ~20.9 cm BEHIND the root.  This is the strict reading of "the robot has
    completely passed the obstacle", and it is recomputed here from the trace
    root position and the measured stance offset, because the env never
    records it.

``success``
    ``derived.success``, i.e. the TIER-2 mission success.  Never inferred here.

Usage:
    python tests/compare_pilot_a1.py --before <json> --after <json>
"""

import argparse
import json
import sys

DT = 0.02                 # control period, seconds
COMMANDED_SPEED = 0.35    # cfg.commands.target_forward_vel
RECOVERY_FRACTION = 0.70
HOLD_STEPS = 10           # == myrobot_dual_env.RECOVERY_HOLD_STEPS
BOX_FRONT = 0.280
BOX_REAR = 0.320
# Measured from obstacle_geometry.robot_forward_interval WITHOUT the z-ceiling
# at a nominal stance: rear_y - base_y = -0.2089 m.  It moves with the joint
# pose, so treat it as an approximation of the strict requirement, not a
# measurement.
WHOLEBODY_REAR_OFFSET = -0.2089


def rows_of(payload, seed, with_obstacle):
    key = 'seed%d_%s' % (seed, 'obs' if with_obstacle else 'free')
    rows = payload.get('traces', {}).get(key) or []
    return [r for r in rows if not r.get('post_reset')]


def forward_speed(prev, cur):
    return (cur['root_y'] - prev['root_y']) / DT


def first_true(rows, key):
    for i, r in enumerate(rows):
        if r.get(key):
            return i
    return -1


def data_done_reason(run, rows):
    """The real cause of the episode end.

    ``outcome.reason`` cannot be taken at face value for the CROSSED_WITHOUT_LIFTOFF
    cases: the latch is evaluated on a step where the env has already been reset
    home while ``robot_rear_margin`` still holds the previous episode's value, so
    the reason is recorded for a robot that had just completed a full jump.  A
    run is therefore only counted as a real crossing failure when the trace shows
    the robot never left the ground.
    """
    reason = (run.get('outcome') or {}).get('reason', 'NONE')
    took_off = (run.get('outcome') or {}).get('takeoff_step', -1) >= 0
    if reason == 'CROSSED_WITHOUT_LIFTOFF' and took_off:
        return 'SPURIOUS_STALE_LATCH'
    return reason


def summarise(payload, run):
    seed, obs = run['seed'], bool(run['with_obstacle'])
    rows = rows_of(payload, seed, obs)
    out = run.get('outcome') or {}
    der = run.get('derived') or {}
    rec = {
        'seed': seed,
        'with_obstacle': obs,
        'reason': data_done_reason(run, rows),
        'success': bool(der.get('success')),
        'recovery_outcome': out.get('recovery_outcome'),
        'collision': bool(out.get('collision')),
        # The env's own `cleared` flag: "the parts that sit below the 5 cm box
        # top have passed the box rear edge".  This is the authoritative loose
        # passage count -- `passed_lowparts` below is a recomputation from the
        # trace and is unreliable for the AFTER arm because the env recorded
        # takeoff_front_y == takeoff_rear_y == root_y while airborne.
        'cleared_lowparts': bool(out.get('cleared')),
        'touched_down': out.get('touchdown_step', -1) >= 0,
        'took_off': out.get('takeoff_step', -1) >= 0,
    }
    take_i = first_true(rows, 'taken_off')
    touch_i = first_true(rows, 'touched_down')
    rec['takeoff_step_trace'] = take_i
    rec['touchdown_step_trace'] = touch_i

    # 1. forward speed at liftoff
    rec['liftoff_speed'] = forward_speed(rows[take_i - 1], rows[take_i]) if take_i > 0 else None

    # 2. airborne time / displacement / in-flight speed
    if take_i >= 0 and touch_i > take_i:
        rec['flight_steps'] = touch_i - take_i
        rec['flight_time_s'] = (touch_i - take_i) * DT
        rec['flight_dy'] = rows[touch_i]['root_y'] - rows[take_i]['root_y']
        sp = [forward_speed(rows[j - 1], rows[j]) for j in range(take_i + 1, touch_i + 1)]
        rec['flight_mean_speed'] = sum(sp) / len(sp) if sp else None
    else:
        for k in ('flight_steps', 'flight_time_s', 'flight_dy', 'flight_mean_speed'):
            rec[k] = None

    # low-parts requirement, as the env itself defines it (rear edge at liftoff)
    if take_i >= 0:
        rec['takeoff_rear_lowparts'] = rows[take_i]['rear_y']
        rec['takeoff_y_trace'] = rows[take_i]['root_y']
        rec['required_lowparts'] = BOX_REAR - rec['takeoff_rear_lowparts']
        rec['required_wholebody'] = (BOX_REAR - (rec['takeoff_y_trace']
                                                 + WHOLEBODY_REAR_OFFSET))
        if rec['flight_dy'] is not None:
            rec['passed_lowparts'] = rec['flight_dy'] >= rec['required_lowparts']
            rec['passed_wholebody'] = rec['flight_dy'] >= rec['required_wholebody']
        else:
            rec['passed_lowparts'] = rec['passed_wholebody'] = False
    else:
        for k in ('takeoff_rear_lowparts', 'takeoff_y_trace', 'required_lowparts',
                  'required_wholebody'):
            rec[k] = None
        rec['passed_lowparts'] = rec['passed_wholebody'] = False

    # 7. post-landing stable cruise measured from the trace
    rec['post_land_hold_steps'] = None
    rec['post_land_cruise_steps'] = None
    if touch_i >= 0 and touch_i + 1 < len(rows):
        sp = [forward_speed(rows[j - 1], rows[j]) for j in range(touch_i + 1, len(rows))]
        good = [v >= RECOVERY_FRACTION * COMMANDED_SPEED for v in sp]
        cur = start = -1
        for i, g in enumerate(good):
            cur = cur + 1 if g else 0
            if cur >= HOLD_STEPS:
                start = i - HOLD_STEPS + 1
                break
        rec['post_land_hold_steps'] = start
        if start >= 0:
            n = 0
            for g in good[start:]:
                if not g:
                    break
                n += 1
            rec['post_land_cruise_steps'] = n
    return rec


def mean_of(vals):
    v = [x for x in vals if x is not None]
    return (sum(v) / len(v), len(v)) if v else (None, 0)


def f(v, nd=4):
    return 'n/a' if v is None else ('%.*f' % (nd, v))


def report(label, per):
    n = len(per)
    print('=' * 76)
    print('%s   (n=%d)' % (label, n))
    print('=' * 76)
    k = sum(1 for p in per if p['success'])
    print('8. TIER-2 mission success      %d/%d = %.1f%%' % (k, n, 100.0 * k / n))
    k = sum(1 for p in per if p['touched_down'])
    print('5. real touchdown              %d/%d = %.1f%%' % (k, n, 100.0 * k / n))
    k = sum(1 for p in per if p['took_off'])
    print('   liftoff                     %d/%d = %.1f%%' % (k, n, 100.0 * k / n))
    k = sum(1 for p in per if p['passed_wholebody'])
    print('3a. passage, WHOLE BODY (strict) %d/%d = %.1f%%' % (k, n, 100.0 * k / n))
    k = sum(1 for p in per if p['passed_lowparts'])
    print('3b. passage, low parts (env def) %d/%d = %.1f%%' % (k, n, 100.0 * k / n))
    k = sum(1 for p in per if p['cleared_lowparts'])
    print('3c. env-recorded loose passage (outcome.cleared) %d/%d = %.1f%%'
          % (k, n, 100.0 * k / n))
    print('   (3a/3b are recomputed from the trace; 3c is what the env itself latched.'
          ' For the AFTER arm 3b under-reports because takeoff_rear_y was recorded as'
          ' root_y while airborne, so 3c is the honest loose count there.)')
    k = sum(1 for p in per if p['collision'])
    print('4. obstacle contact            %d/%d = %.1f%%' % (k, n, 100.0 * k / n))
    v, m = mean_of([p['liftoff_speed'] for p in per])
    print('1. liftoff forward speed       %s m/s (n=%d, cmd %.2f, ratio %s)'
          % (f(v), m, COMMANDED_SPEED, f(v / COMMANDED_SPEED, 3) if v is not None else 'n/a'))
    v, m = mean_of([p['flight_time_s'] for p in per])
    s, m2 = mean_of([p['flight_steps'] for p in per])
    d, m3 = mean_of([p['flight_dy'] for p in per])
    q, m4 = mean_of([p['flight_mean_speed'] for p in per])
    print('2. airborne                    %s s / %s steps (n=%d)  dy=%s m (n=%d)  mean speed %s m/s (n=%d)'
          % (f(v, 3), f(s, 2), m, f(d), m3, f(q), m4))
    o = {}
    for p in per:
        o[p['recovery_outcome'] or 'NONE'] = o.get(p['recovery_outcome'] or 'NONE', 0) + 1
    print('6. recovery outcomes           %s' % o)
    landed = [p for p in per if p['post_land_hold_steps'] is not None]
    held = [p for p in landed if p['post_land_hold_steps'] >= 0]
    print('7. post-landing cruise         %d/%d analysable landed runs reached a %d-step hold'
          % (len(held), len(landed), HOLD_STEPS))
    print('   (analysable = the trace has a touchdown row with steps left after it;'
          ' item 5 counts touchdowns from outcome)')
    if held:
        v, m = mean_of([p['post_land_hold_steps'] * DT for p in held])
        print('   touchdown -> sustained cruise %s s (n=%d)' % (f(v, 3), m))
        v, m = mean_of([p['post_land_cruise_steps'] * DT for p in held if p['post_land_cruise_steps']])
        print('   sustained cruise duration     %s s (n=%d)' % (f(v, 3), m))
    r = {}
    for p in per:
        r[p['reason']] = r.get(p['reason'], 0) + 1
    print('   end reason                     %s' % r)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', required=True)
    ap.add_argument('--after', required=True)
    ap.add_argument('--json_out', default=None)
    a = ap.parse_args()
    b = json.load(open(a.before, encoding='utf-8'))
    c = json.load(open(a.after, encoding='utf-8'))
    print('BEFORE %s' % b.get('model_path'))
    print('AFTER  %s' % c.get('model_path'))
    print('action_mode before=%s after=%s' % (b.get('action_mode'), c.get('action_mode')))
    print()
    bundle = {}
    for arm, flag in (('obstacle', True), ('free_run', False)):
        print('#' * 76)
        print('# ARM %s' % arm)
        print('#' * 76)
        pb = [summarise(b, r) for r in b['runs'] if bool(r['with_obstacle']) == flag]
        pa = [summarise(c, r) for r in c['runs'] if bool(r['with_obstacle']) == flag]
        report('BEFORE frozen model_3.pt', pb)
        report('AFTER  pilot model_10.pt', pa)
        bundle[arm] = {'before': pb, 'after': pa}
    if a.json_out:
        with open(a.json_out, 'w', encoding='utf-8') as fh:
            json.dump(bundle, fh, indent=2, ensure_ascii=False)
        print('wrote %s' % a.json_out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
