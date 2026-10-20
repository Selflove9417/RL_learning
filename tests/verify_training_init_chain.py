"""Verify the REAL training initialisation chain on the GPU pipeline.

WHY THIS FILE EXISTS
--------------------
A previous report claimed "training never calls `env.reset()`, so training is
unaffected by the GPU first-reset artifact". That claim was WRONG:

    rsl_rl/rsl_rl/runners/on_policy_runner.py:94
        _, _ = self.env.reset()          # inside OnPolicyRunner.__init__

`learn()` indeed never calls reset, but `__init__` does, and `train.py` builds the
runner *before* calling `learn()`:

    legged_gym/legged_gym/scripts/train.py:126-133
        env, env_cfg = task_registry.make_env(...)
        ppo_runner, train_cfg = task_registry.make_alg_runner(...)   # -> __init__ -> env.reset()
        ppo_runner.learn(...)

So the artifact is squarely ON the training path and must be measured there, not
on a bespoke evaluation harness.

WHAT THIS SCRIPT DOES -- AND WHAT IT DELIBERATELY DOES NOT DO
-------------------------------------------------------------
It reproduces `train.py`'s exact construction order, then drives the real PPO
rollout loop (`runner.alg.act` -> `env.step`) for the first 25 steps with
num_envs=64 on the GPU pipeline, recording per-step for env0:

  * root z / root pitch / forward velocity
  * which bodies carry real contact (net contact force > 1 N)
  * feet (wheel) contact, and whether the robot is standing on wheels or on legs
  * the termination flags the trainer would actually see

It does NOT train: no `runner.learn()`, no `update()`, no weight change, no
checkpoint, no log directory (`log_root=None`). It is a read-only probe of the
initialisation chain.

Run from the repository root, after `source ./activate_jump.sh`:

    python tests/verify_training_init_chain.py --pipeline gpu --num_envs 64
"""

import argparse
import hashlib
import json
import math
import traceback
from datetime import datetime, timezone
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import isaacgym  # noqa: F401  (must precede torch)
from isaacgym import gymtorch  # noqa: F401
import torch

from legged_gym.envs import MyRobotObstacleJump  # noqa: F401
from legged_gym.utils import get_args, task_registry
from training_init_gate import evaluate, EXIT_CODES

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Bodies that must never touch the ground for a legitimate wheeled stance.
# From myrobot_config.py:159 -- these are the termination contacts.
LEG_CONTACT_NAMES = ['base_link', 'link_002', 'link_003', 'link_005', 'link_006']


def build(num_envs, pipeline, seed=11):
    """Mirror train.py's construction order exactly."""
    argv = [sys.argv[0], '--task', 'myrobot_obstacle_jump', '--headless',
            '--seed', str(seed), '--num_envs', str(num_envs),
            '--pipeline', pipeline]
    if pipeline == 'cpu':
        # On the CPU pipeline the env publishes its tensors on the CPU, while
        # --rl_device still defaults to cuda:0. That mixed pair is rejected by
        # pre-existing repo code (myrobot_dual_env.py writes policy actions
        # straight into CPU buffers), so the consistent CPU/CPU pair is used --
        # the same configuration tests/test_ppo_smoke.py documents.
        argv += ['--rl_device', 'cpu']
    sys.argv = argv
    args = get_args()
    cfg, _ = task_registry.get_cfgs(name=args.task)
    cfg.env.num_envs = num_envs
    cfg.seed = seed
    cfg.env.episode_length_s = 4.0
    cfg.trigger.mode = 'spatial'

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=cfg)
    # THIS is the call that contains the reset: OnPolicyRunner.__init__:94
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, log_root=None)
    return env, runner, train_cfg, args


def body_contacts(env, threshold=1.0):
    """Per-body names carrying a real normal force above `threshold` N."""
    names = []
    cf = env.contact_forces[0]            # (num_bodies, 3)
    for i in range(cf.shape[0]):
        if float(torch.norm(cf[i])) > threshold:
            names.append(i)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--out', help='new evidence file; existing files are never overwritten')
    options = parser.parse_args()
    pipeline, num_envs = options.pipeline, options.num_envs
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = options.out or os.path.join(
        WORKSPACE, 'evaluation_assets', 'phaseD2', 'init_%s_%s.json' % (pipeline, stamp))
    if os.path.exists(path):
        parser.error('refusing to overwrite existing evidence: %s' % path)
    evidence = {'pipeline': pipeline, 'num_envs': num_envs, 'seed': options.seed,
                'started_utc': stamp, 'frames': [], 'validation_errors': []}
    try:
        collect(evidence, options)
    except Exception as exc:
        traceback.print_exc()
        evidence['runtime_error'] = '%s: %s' % (type(exc).__name__, exc)
    evidence['verdict'] = evaluate(evidence)
    # Preserve non-finite failures as JSON nulls; validation_errors retains their location.
    def finite_json(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        return value
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'x') as fh:
        json.dump(finite_json(evidence), fh, allow_nan=False)
    print('raw JSON: %s' % path)
    print('ADMISSION %s: %s' % (evidence['verdict']['status'], evidence['verdict']['reasons']))
    return EXIT_CODES[evidence['verdict']['status']]


def collect(evidence, options):
    pipeline, num_envs = options.pipeline, options.num_envs
    print('=' * 78)
    print('TRAINING INITIALISATION CHAIN -- pipeline=%s num_envs=%d' % (pipeline, num_envs))
    print('reproduces train.py: make_env -> make_alg_runner (__init__ calls env.reset())')
    print('=' * 78)

    # Count actual calls without changing reset behavior or adding a reset.
    from legged_gym.envs.base.base_task import BaseTask
    original_reset = BaseTask.reset
    reset_calls = []
    def counted_reset(self):
        reset_calls.append(True)
        return original_reset(self)
    BaseTask.reset = counted_reset
    try:
        env, runner, train_cfg, args = build(num_envs, pipeline, options.seed)
    finally:
        BaseTask.reset = original_reset
    evidence['reset_calls'] = len(reset_calls)
    evidence['probe_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
    evidence['gate_sha256'] = hashlib.sha256(open(
        os.path.join(os.path.dirname(__file__), 'training_init_gate.py'), 'rb').read()).hexdigest()
    evidence['initial_state'] = {
        'root_states': env.root_states.tolist(),
        'wheel_normal_forces': env.contact_forces[:, env.feet_indices, 2].tolist(),
        'root_z_relative_to_peers': (env.root_states[:, 2] - env.root_states[1:, 2].median()).tolist(),
    }
    policy_before = {k: v.clone() for k, v in runner.alg.actor_critic.state_dict().items()}
    def optimizer_steps():
        return sum(float(v.get('step', 0)) for v in runner.alg.optimizer.state.values())
    evidence['optimizer_steps_before'] = optimizer_steps()
    print('env.device            = %s' % env.device)
    print('runner.device         = %s' % runner.device)
    print('num_envs / obs / act  = %d / %d / %d'
          % (env.num_envs, env.num_obs, env.num_actions))
    print('max_episode_length    = %d' % env.max_episode_length)

    # ---- state RIGHT AFTER the runner's env.reset() -----------------------
    z0 = float(env.root_states[0, 2])
    print('\n[after OnPolicyRunner.__init__ -> env.reset()]')
    print('  env0 root_z = %.6f (nominal base %.6f)'
          % (z0, float(env.base_init_state[2])))
    print('  env0 rb base z = %.6f' % float(env.rb_states[0, 0, 2]))

    # ---- the real PPO rollout path, first 25 steps ------------------------
    # runner.alg.act is the exact call the learning loop makes; it samples from
    # the policy and fills the transition buffer. `process_env_step` is included
    # so the reward/time_outs/actor_loss_mask wiring is exercised too.
    alg = runner.alg
    rows = []
    obs = env.get_observations()
    critic_obs = env.get_privileged_observations()
    critic_obs = critic_obs if critic_obs is not None else obs
    obs, critic_obs = obs.to(runner.device), critic_obs.to(runner.device)

    leg_indices = [int(i) for i in env.termination_contact_indices.tolist()]
    foot_indices = [int(i) for i in env.feet_indices.tolist()]

    # SCOPE: is the artifact env0-only or fleet-wide? Recorded for every env at
    # the first frame the trainer sees, because "1 of 64 envs" and "all 64 envs"
    # are completely different findings.
    z_after_reset_all = env.root_states[:, 2].clone()
    nominal_z = float(env.base_init_state[2])
    per_env_first_z = None
    per_env_first_done = torch.full((env.num_envs,), -1, device=env.device,
                                    dtype=torch.long)

    early_reset_step = None
    def tensor_check(name, tensor, shape, step):
        if tensor is None:
            evidence['validation_errors'].append('%s absent at %s' % (name, step))
            return
        if tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
            evidence['validation_errors'].append('%s invalid shape/nonfinite at %s: %s'
                                                 % (name, step, tuple(tensor.shape)))

    with torch.inference_mode():
        for step_i in range(25):
            tensor_check('input observations', obs, (num_envs, 35), step_i)
            tensor_check('input critic observations', critic_obs, (num_envs, 35), step_i)
            input_obs, input_critic = obs.clone(), critic_obs.clone()
            actions = alg.act(obs, critic_obs)
            tensor_check('actions', actions, (num_envs, 6), step_i)
            input_actions = actions.clone()
            obs, priv, rewards, dones, infos = env.step(actions)
            critic_obs = priv if priv is not None else obs
            obs, critic_obs = obs.to(runner.device), critic_obs.to(runner.device)
            rewards, dones = rewards.to(runner.device), dones.to(runner.device)
            for name, value, shape in (
                    ('observations', obs, (num_envs, 35)),
                    ('critic observations', critic_obs, (num_envs, 35)),
                    ('rewards', rewards, (num_envs,)),
                    ('dones', dones, (num_envs,)),
                    ('actor_loss_mask', infos.get('actor_loss_mask'), (num_envs,)),
                    ('time_outs', infos.get('time_outs'), (num_envs,)),
                    ('terminal_observation', infos.get('terminal_observation'), (num_envs, 35))):
                tensor_check(name, value, shape, step_i)
            alg.process_env_step(rewards, dones, infos)
            st = alg.storage
            for name in ('observations', 'privileged_observations', 'actions',
                         'rewards', 'values', 'actions_log_prob', 'actor_masks'):
                value = getattr(st, name)[step_i]
                if not bool(torch.isfinite(value).all()):
                    evidence['validation_errors'].append('stored %s nonfinite at %d' % (name, step_i))
            mask = infos.get('actor_loss_mask')
            if mask is not None:
                if not bool(((mask == 0) | (mask == 1)).all()):
                    evidence['validation_errors'].append('actor mask not binary at %d' % step_i)
                if not torch.equal(st.actor_masks[step_i].flatten(), mask.to(st.device).flatten()):
                    evidence['validation_errors'].append('actor mask storage mismatch at %d' % step_i)
            if not torch.equal(st.observations[step_i], input_obs):
                evidence['validation_errors'].append('input observation storage mismatch at %d' % step_i)
            if not torch.equal(st.actions[step_i], input_actions):
                evidence['validation_errors'].append('action storage mismatch at %d' % step_i)
            if not torch.equal(st.privileged_observations[step_i], input_critic):
                evidence['validation_errors'].append('critic observation storage mismatch at %d' % step_i)
            wheel_forces = env.contact_forces[:, foot_indices, 2]
            leg_loaded = (torch.norm(env.contact_forces[:, leg_indices, :], dim=-1) > 1.0).any(dim=1)
            wheels_loaded = (wheel_forces > 20.0)
            leg_only = leg_loaded & (~wheels_loaded.any(dim=1))
            supported = wheels_loaded.all(dim=1) & (~leg_loaded) & (~dones.bool())
            ids = lambda value: value.nonzero(as_tuple=False).flatten().tolist()
            terminal_ids = ids(dones.bool())
            snap = env.episode_snapshot
            terminal = [{
                'env_id': i, 'snapshot_valid': bool(snap['valid'][i]),
                'fell': bool(snap['fell'][i]), 'reason': snap['failure_reason'][i],
                'root_z_before_reset': float(snap['root_z'][i]),
                'roll_rad_before_reset': float(snap['roll_rad'][i]),
                'pitch_rad_before_reset': float(snap['pitch_rad'][i]),
                'terminal_observation': infos['terminal_observation'][i].tolist(),
            } for i in terminal_ids]
            evidence['frames'].append({
                'step': step_i, 'storage_step': st.step,
                'root_states': env.root_states.tolist(),
                'state_timing': 'post-step; root states are post-reset for done envs; see terminal',
                'observations': input_obs.tolist(),
                'critic_observations_equal_actor': bool(torch.equal(input_obs, input_critic)),
                'critic_observations': None if torch.equal(input_obs, input_critic) else input_critic.tolist(),
                'actions': input_actions.tolist(), 'rewards': rewards.tolist(),
                'stored_rewards': st.rewards[step_i].flatten().tolist(),
                'actor_loss_mask': None if mask is None else mask.tolist(),
                'stored_actor_masks': st.actor_masks[step_i].flatten().tolist(),
                'time_outs': infos['time_outs'].tolist(),
                'wheel_normal_forces': wheel_forces.tolist(),
                'leg_loaded': leg_loaded.tolist(),
                'done_env_ids': terminal_ids, 'leg_only_env_ids': ids(leg_only),
                'supported_env_ids': ids(supported), 'terminal': terminal,
            })

            if per_env_first_z is None:
                per_env_first_z = env.root_states[:, 2].clone()
            newly_done = (dones > 0) & (per_env_first_done < 0)
            per_env_first_done[newly_done] = step_i

            z = float(env.root_states[0, 2])
            pitch = float(env.root_states[0, 4])
            vfwd = float(env.root_states[0, 8])
            n_leg = int(torch.sum(
                torch.norm(env.contact_forces[0, leg_indices, :], dim=-1) > 1.0))
            n_foot = int(torch.sum(
                torch.norm(env.contact_forces[0, foot_indices, :], dim=-1) > 1.0))
            done0 = bool(dones[0] > 0)
            rows.append({
                'step': step_i,
                'root_z': round(z, 6),
                'quat_y': round(pitch, 6),
                'vfwd': round(vfwd, 6),
                'leg_contacts': n_leg,
                'foot_contacts': n_foot,
                'done': done0,
                'reset_buf': bool(env.reset_buf[0] > 0),
                'ep_len': int(env.episode_length_buf[0]),
            })
            if done0 and early_reset_step is None:
                early_reset_step = step_i
            print('  step %2d  z=%.5f  qy=%+.5f  v=%+.4f  legC=%d  footC=%d%s'
                  % (step_i, z, pitch, vfwd, n_leg, n_foot,
                     '   <-- DONE' if done0 else ''))

    evidence['policy_weights_unchanged'] = all(
        torch.equal(value, policy_before[name])
        for name, value in runner.alg.actor_critic.state_dict().items())
    evidence['optimizer_steps_after'] = optimizer_steps()

    # ---- verdict ---------------------------------------------------------
    first = rows[0]
    print('\n' + '=' * 78)
    print('ENV0 FIRST-STEP CONTACT / ATTITUDE')
    print('  root_z after 1 PPO step = %.6f' % first['root_z'])
    print('  leg contacts = %d   foot(wheel) contacts = %d'
          % (first['leg_contacts'], first['foot_contacts']))
    print('  standing on wheels = %s' % (first['foot_contacts'] > 0
                                         and first['leg_contacts'] == 0))
    print('\nTERMINATION')
    print('  first done step = %s' % (early_reset_step if early_reset_step is not None
                                      else 'none in 25 steps'))
    n_done = sum(1 for r in rows if r['done'])
    print('  total done steps in 25 = %d' % n_done)
    print('  env0 episode lengths observed: %s'
          % sorted({r['ep_len'] for r in rows}))

    # ---- SCOPE: how many of the 64 envs are affected ---------------------
    z_off = (z_after_reset_all - nominal_z).abs()
    n_off = int((z_off > 1e-4).sum())
    first_z_off = (per_env_first_z - nominal_z).abs()
    n_first_off = int((first_z_off > 1e-4).sum())
    n_done_any = int((per_env_first_done >= 0).sum())
    print('\nSCOPE ACROSS ALL %d ENVS' % env.num_envs)
    print('  envs with |root_z - %.3f| > 1e-4 right after runner.__init__ reset: %d'
          % (nominal_z, n_off))
    if n_off:
        idx = (z_off > 1e-4).nonzero(as_tuple=False).flatten().tolist()
        print('    affected env ids: %s' % idx[:16])
        print('    their z values  : %s'
              % [round(float(z_after_reset_all[i]), 5) for i in idx[:16]])
    print('  envs whose FIRST trainer frame is off-nominal: %d' % n_first_off)
    print('  envs that terminated within 25 steps: %d (ids=%s)'
          % (n_done_any,
             (per_env_first_done >= 0).nonzero(as_tuple=False).flatten().tolist()))
    print('=' * 78)

    out = {
        'pipeline': pipeline,
        'num_envs': num_envs,
        'env_device': str(env.device),
        'runner_device': str(runner.device),
        'root_z_after_runner_init_reset': round(z0, 6),
        'nominal_base_z': round(nominal_z, 6),
        'n_envs_off_nominal_after_init_reset': n_off,
        'n_envs_off_nominal_first_frame': n_first_off,
        'n_envs_terminated_within_25': n_done_any,
        'envs_terminated_ids': (per_env_first_done >= 0).nonzero(
            as_tuple=False).flatten().tolist(),
        'rows': rows,
        'first_done_step': early_reset_step,
        'standing_on_wheels_first_step': (first['foot_contacts'] > 0
                                          and first['leg_contacts'] == 0),
    }
    evidence['legacy_diagnostics'] = out


if __name__ == '__main__':
    raise SystemExit(main())
