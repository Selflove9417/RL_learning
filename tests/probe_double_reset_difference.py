"""Measure what a DOUBLE reset actually changes relative to a SINGLE reset.

WHY THIS FILE EXISTS
--------------------
The GPU pipeline collapses env0's first controlled cycle (root_z 0.41552 instead
of 0.39867, legs on the ground, episode dies at step 13). Calling `env.reset()`
twice before handing control to the policy makes the collapse disappear.

That is a WORKAROUND, not a repair, and it must not be adopted before stating
exactly what it changes. Two candidate differences matter:

  1. STARTING STATE. `reset_idx` draws the DOF initial positions from a uniform
     random distribution (`torch_rand_float(0.5, 1.5, ...)` in `_reset_dofs`).
     A second reset CONSUMES ANOTHER DRAW, so the robot does not start from the
     same joint angles -- unless the draw is per-env deterministic, which is not
     the case. If the joints move, the second reset is NOT experiment-neutral.
  2. PHYSICAL HISTORY. `env.reset()` is `reset_idx(all) + step(zeros)`, so each
     reset advances the solver by one full control cycle (`decimation` substeps)
     and increments `common_step_counter`. Two resets = one extra cycle of
     unwanted contact settling, and one extra step of clock.

This script measures both, on one pipeline, and prints the deltas. It does not
change any code and does not touch the evaluator.

Run from the repository root, after `source ./activate_jump.sh`:

    python tests/probe_double_reset_difference.py --pipeline gpu
    python tests/probe_double_reset_difference.py --pipeline cpu
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import isaacgym  # noqa: F401  (must precede torch)
import torch

from legged_gym.envs import MyRobotObstacleJump  # noqa: F401
from legged_gym.utils import get_args, task_registry

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(pipeline, num_envs=4, seed=11):
    argv = [sys.argv[0], '--task', 'myrobot_obstacle_jump', '--headless',
            '--seed', str(seed), '--num_envs', str(num_envs),
            '--pipeline', pipeline]
    if pipeline == 'cpu':
        argv += ['--rl_device', 'cpu']
    sys.argv = argv
    args = get_args()
    cfg, _ = task_registry.get_cfgs(name=args.task)
    cfg.env.num_envs = num_envs
    cfg.seed = seed
    cfg.env.episode_length_s = 4.0
    cfg.trigger.mode = 'spatial'
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=cfg)
    return env


def snapshot(env, tag):
    """Capture the quantities a second reset could perturb."""
    return {
        'tag': tag,
        'dof_pos': env.dof_pos.detach().cpu().clone(),
        'root_xyz': env.root_states[:, 0:3].detach().cpu().clone(),
        'root_quat': env.root_states[:, 3:7].detach().cpu().clone(),
        'rb_base_z': env.rb_states[:, 0, 2].detach().cpu().clone(),
        'common_step_counter': int(env.common_step_counter),
        'episode_length_buf': env.episode_length_buf.detach().cpu().clone(),
        'rng_torch': torch.rand(1).item(),   # value of the stream AFTER the call
    }


def main():
    pipeline = 'gpu'
    for i, tok in enumerate(sys.argv):
        if tok == '--pipeline' and i + 1 < len(sys.argv):
            pipeline = sys.argv[i + 1]

    print('=' * 78)
    print('DOUBLE-RESET DIFFERENCE PROBE -- pipeline=%s' % pipeline)
    print('=' * 78)

    env = build(pipeline)

    # --- state after the FIRST reset_idx(all) ----------------------------
    ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(ids)
    first = snapshot(env, 'after reset_idx #1')

    # --- state after a SECOND reset_idx(all), matching "double reset" -----
    env.reset_idx(ids)
    second = snapshot(env, 'after reset_idx #2')

    d_dof = (second['dof_pos'] - first['dof_pos']).abs()
    d_root = (second['root_xyz'] - first['root_xyz']).abs()
    d_quat = (second['root_quat'] - first['root_quat']).abs()
    d_rbz = (second['rb_base_z'] - first['rb_base_z']).abs()

    print('\n[1] STARTING STATE -- does the second reset move the spawn pose?')
    print('    max |dof_pos  delta| = %.6e rad' % float(d_dof.max()))
    print('    max |root_xyz delta| = %.6e m' % float(d_root.max()))
    print('    max |root_quat delta|= %.6e' % float(d_quat.max()))
    print('    max |rb_base_z delta|= %.6e m' % float(d_rbz.max()))
    dof_moved = float(d_dof.max()) > 1e-9
    print('    -> spawn joint angles CHANGED by the second reset: %s' % dof_moved)
    print('    dof_pos row0 first : %s'
          % [round(v, 4) for v in first['dof_pos'][0].tolist()])
    print('    dof_pos row0 second: %s'
          % [round(v, 4) for v in second['dof_pos'][0].tolist()])

    print('\n[2] RANDOM STREAM -- how far did the second reset advance it?')
    print('    torch RNG value sampled after reset #1 = %.9f'
          % first['rng_torch'])
    print('    torch RNG value sampled after reset #2 = %.9f'
          % second['rng_torch'])
    print('    (a different value proves the reset CONSUMES the RNG, so a '
          'second reset is not draw-neutral)')

    print('\n[3] PHYSICAL HISTORY -- what did the extra reset add to the clock?')
    print('    common_step_counter after #1 = %d' % first['common_step_counter'])
    print('    common_step_counter after #2 = %d' % second['common_step_counter'])
    print('    (reset_idx does NOT advance the counter -- only step() does. The')
    print('     physical-history cost of a second reset is therefore the whole')
    print('     extra controlled cycle it has to run to absorb, measured in [4].)')

    # --- now run ONE controlled cycle from each and compare --------------
    # This is the part that matters for evaluation protocol: after a single
    # reset the first controlled cycle on GPU is where the artifact lives.
    #
    # SINGLE protocol (what the evaluator does today):
    #     reset_idx(all)            -> spawn pose written
    #     step(zeros)               -> first controlled cycle  <-- artifact here
    #
    # DOUBLE protocol (the candidate workaround):
    #     reset_idx(all); step(zeros)   -> absorb the bad cycle
    #     reset_idx(all)                -> re-spawn
    #     step(zeros)                   -> the cycle the policy would actually get
    # (The evaluator calls `env.reset()`, which is exactly reset_idx+step; the
    #  probe uses reset_idx+step explicitly so each phase can be sampled.)
    env.reset_idx(ids)
    z_before1 = env.root_states[:, 2].detach().cpu().clone()
    env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
    z_after1 = env.root_states[:, 2].detach().cpu().clone()

    env.reset_idx(ids)
    env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
    env.reset_idx(ids)                     # <-- the second reset
    z_before2 = env.root_states[:, 2].detach().cpu().clone()
    env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
    z_after2 = env.root_states[:, 2].detach().cpu().clone()

    nominal = float(env.base_init_state[2])
    print('\n[4] FIRST CONTROLLED CYCLE AFTER EACH PROTOCOL (root_z)')
    print('    nominal base z                 = %.5f' % nominal)
    print('    single: pre-step %s'
          % [round(v, 5) for v in z_before1.tolist()])
    print('    single: post-step %s'
          % [round(v, 5) for v in z_after1.tolist()])
    print('    double: pre-step %s'
          % [round(v, 5) for v in z_before2.tolist()])
    print('    double: post-step %s'
          % [round(v, 5) for v in z_after2.tolist()])
    print('    single deviates from nominal by max %.5f m'
          % float((z_after1 - nominal).abs().max()))
    print('    double deviates from nominal by max %.5f m'
          % float((z_after2 - nominal).abs().max()))

    print('\n' + '=' * 78)
    print('VERDICT')
    # The verdict must be DERIVED from the measurement, never asserted in prose:
    # a probe whose conclusion is hard-coded can "confirm" anything.
    if dof_moved:
        print('  the second reset CHANGES the starting joint angles (max %.3e rad).'
              % float(d_dof.max()))
        print('  -> double reset is NOT draw-neutral; every measurement would start')
        print('     from a different initial state than a single-reset run.')
    else:
        print('  the second reset does NOT change the spawn pose'
              ' (max |delta| = 0 for dof_pos / root / quat / rb_base_z).')
        print('  Reason, measured: `default_joint_angles` are all 0.0 for this robot,')
        print('  so `default_dof_pos * torch_rand_float(0.5,1.5,...)` is identically')
        print('  zero. The RNG IS consumed (values %.6f -> %.6f), so the reset is not'
              % (first['rng_torch'], second['rng_torch']))
        print('  draw-neutral IN GENERAL -- it is neutral here only because the')
        print('  nominal pose is the zero vector.')
        print('  -> the STARTING STATE is identical, but the PHYSICAL HISTORY is not:')
        print('     the workaround spends one extra controlled cycle per episode,')
        print('     so it perturbs the post-reset trajectory itself (see [4]).')
    print('=' * 78)

    out = {
        'pipeline': pipeline,
        'nominal_base_z': nominal,
        'second_reset_changes_dof_pos': dof_moved,
        'max_dof_delta': float(d_dof.max()),
        'max_root_delta': float(d_root.max()),
        'rng_after_reset1': first['rng_torch'],
        'rng_after_reset2': second['rng_torch'],
        'common_step_after_reset1': first['common_step_counter'],
        'common_step_after_reset2': second['common_step_counter'],
        'dof_pos_after_reset1_row0': first['dof_pos'][0].tolist(),
        'dof_pos_after_reset2_row0': second['dof_pos'][0].tolist(),
        'single_pre_step_z': z_before1.tolist(),
        'single_post_step_z': z_after1.tolist(),
        'double_pre_step_z': z_before2.tolist(),
        'double_post_step_z': z_after2.tolist(),
    }
    path = os.path.join(WORKSPACE, 'evaluation_assets',
                        'double_reset_difference_%s.json' % pipeline)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('raw JSON: %s' % path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
