"""Pure admission decision for the 64-env, 25-step GPU startup probe.

Exit codes: 0 PASS, 1 FAIL (observed defect), 2 INCOMPLETE (missing evidence).
Initial settling is allowed. Three consecutive leg-only support frames, an early
termination, or no three-frame bilateral wheel support by step 25 veto admission.
"""

EXIT_CODES = {'PASS': 0, 'FAIL': 1, 'INCOMPLETE': 2}


def evaluate(evidence):
    required = ('pipeline', 'num_envs', 'initial_state', 'frames',
                'policy_weights_unchanged', 'optimizer_steps_before',
                'optimizer_steps_after', 'reset_calls', 'validation_errors')
    missing = [key for key in required if key not in evidence]
    if missing or evidence.get('runtime_error'):
        return {'status': 'INCOMPLETE', 'reasons': missing or [evidence['runtime_error']]}
    frames = evidence['frames']
    frame_keys = ('step', 'done_env_ids', 'leg_only_env_ids', 'supported_env_ids',
                  'storage_step', 'root_states', 'observations', 'actions',
                  'rewards', 'actor_loss_mask', 'time_outs', 'stored_actor_masks',
                  'stored_rewards', 'wheel_normal_forces', 'leg_loaded', 'terminal')
    if len(frames) != 25 or any(any(k not in f for k in frame_keys) for f in frames):
        return {'status': 'INCOMPLETE', 'reasons': ['expected 25 complete rollout frames']}
    if (not isinstance(evidence['initial_state'], dict)
            or len(evidence['initial_state'].get('root_states', [])) != 64):
        return {'status': 'INCOMPLETE', 'reasons': ['missing 64-env initial state']}
    for frame in frames:
        for key, width in (('root_states', 13), ('observations', 35),
                           ('actions', 6), ('wheel_normal_forces', 2)):
            value = frame[key]
            if (not isinstance(value, list) or len(value) != 64
                    or any(not isinstance(row, list) or len(row) != width for row in value)):
                return {'status': 'INCOMPLETE', 'reasons': ['missing full array: ' + key]}
        for key in ('rewards', 'actor_loss_mask', 'time_outs', 'stored_actor_masks',
                    'stored_rewards', 'leg_loaded'):
            if not isinstance(frame[key], list) or len(frame[key]) != 64:
                return {'status': 'INCOMPLETE', 'reasons': ['missing full vector: ' + key]}

    reasons = list(evidence['validation_errors'])
    if evidence['pipeline'] != 'gpu' or evidence['num_envs'] != 64:
        reasons.append('requires GPU pipeline and 64 environments')
    if evidence['reset_calls'] != 1:
        reasons.append('expected exactly one runner initialization reset')
    if evidence['policy_weights_unchanged'] is not True:
        reasons.append('policy weights changed')
    if evidence['optimizer_steps_after'] != evidence['optimizer_steps_before']:
        reasons.append('optimizer updated')

    leg_runs, support_runs = [0] * 64, [0] * 64
    supported, leg_only, terminated = set(), set(), set()
    for index, frame in enumerate(frames):
        if frame['step'] != index or frame['storage_step'] != index + 1:
            reasons.append('rollout order/storage count mismatch at step %d' % index)
        terminated.update(frame['done_env_ids'])
        for env_id in range(64):
            leg_runs[env_id] = leg_runs[env_id] + 1 if env_id in frame['leg_only_env_ids'] else 0
            support_runs[env_id] = support_runs[env_id] + 1 if env_id in frame['supported_env_ids'] else 0
            if leg_runs[env_id] >= 3:
                leg_only.add(env_id)
            if support_runs[env_id] >= 3:
                supported.add(env_id)
    if terminated:
        reasons.append('terminated within startup window: %s' % sorted(terminated))
    if leg_only:
        reasons.append('sustained leg-only support: %s' % sorted(leg_only))
    unsupported = sorted(set(range(64)) - supported)
    if unsupported:
        reasons.append('no sustained bilateral wheel support: %s' % unsupported)
    return {'status': 'FAIL' if reasons else 'PASS', 'reasons': reasons,
            'terminated_env_ids': sorted(terminated),
            'leg_only_env_ids': sorted(leg_only), 'unsupported_env_ids': unsupported}


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence')
    args = parser.parse_args()
    try:
        with open(args.evidence) as fh:
            verdict = evaluate(json.load(fh))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        verdict = {'status': 'INCOMPLETE', 'reasons': [str(exc)]}
    print(json.dumps(verdict))
    return EXIT_CODES[verdict['status']]


if __name__ == '__main__':
    raise SystemExit(main())
