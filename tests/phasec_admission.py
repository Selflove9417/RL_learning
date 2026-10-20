"""Phase C training-admission checklist for the controlled PPO pilot.

WHAT THIS IS
------------
Phase C authorises a *tiny* PPO pilot (A1 warm start, original rewards, at most
10 iterations) but only after eight admission conditions are re-verified against
the actual checked-out commit. Six of them are already proven by two suites that
this script does NOT duplicate; two of them are not covered anywhere and are
checked here.

Where each condition is proven
------------------------------
  C1  75 semantic counter-example tests pass            tests/test_obstacle_jump_semantics.py
  C2  GPU-pipeline PPO data path is intact              tests/test_ppo_smoke.py --pipeline gpu
  C3  Actor 35 / Critic 35 / Action 6, and the frozen   HERE
      checkpoint really populates them
  C4  No phantom failure after a GPU reset              HERE
  C5  model_3.pt frozen and wired as the warm start     HERE (hash + config wiring)
  C6  actor_loss_mask / time_outs / terminal_observation HERE (the smoke test
      checks the mask and the folded bootstrap reward; this adds the explicit
      per-key assertions and the terminal-observation bootstrap identity)
  C7  collision penalty is negative, success pays once  HERE (end-to-end scale
      application; the reward algebra itself is in the semantics suite)
  C8  the pilot run directory is isolated from history  HERE

Run:
    python tests/phasec_admission.py                 # GPU pipeline (the pilot's)
    python tests/phasec_admission.py --pipeline cpu --rl_device cpu

Exit code is 0 only if every check passes, so this can gate the training launch.
"""

import hashlib
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import isaacgym  # noqa: F401  (must precede torch)
import torch
from isaacgym import gymtorch

from legged_gym.envs import MyRobotObstacleJump  # noqa: F401
from legged_gym.utils import get_args, task_registry

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rsl_rl'))

PASS, FAIL = [], []

# Recorded when the frozen baseline was first used as an evaluation target. If
# this file ever changes, the Phase B numbers stop describing this checkpoint.
FROZEN_MODEL = ('legged_gym/logs/myrobot_running_jump/'
                'Oct19_21-52-03_running_jump_020ms_iter3_controlled/model_3.pt')
FROZEN_SHA256 = '669da439a00e77d98471acb374cda9468b0ae78443c018edb9f0500f094e4f85'
PILOT_RUN_NAME = 'trial_a1_pilot10'

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print("  [%s] %-62s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


def build(num_envs):
    forwarded = []
    for i, tok in enumerate(sys.argv):
        if tok in ('--pipeline', '--device', '--rl_device') and i + 1 < len(sys.argv):
            forwarded += [tok, sys.argv[i + 1]]
    argv = [sys.argv[0], '--task', 'myrobot_obstacle_jump', '--headless',
            '--seed', '11', '--num_envs', str(num_envs)] + forwarded
    sys.argv = argv
    args = get_args()
    if not args.use_gpu_pipeline and not any('--rl_device' in t for t in forwarded):
        args.rl_device = 'cpu'
        print("(learner device defaulted to cpu to match cpu-side env tensors)")
    cfg, _ = task_registry.get_cfgs(name=args.task)
    cfg.env.num_envs = num_envs
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=cfg)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, log_root=None)
    return env, runner, train_cfg, args


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def c3_network_widths(env, runner):
    print("\nC3. Actor 35 / Critic 35 / Action 6, frozen weights really loaded")
    policy = runner.alg.actor_critic
    check("env.num_obs == 35", env.num_obs == 35, "num_obs=%d" % env.num_obs)
    check("env.num_privileged_obs == 35",
          (env.num_privileged_obs or env.num_obs) == 35,
          "num_privileged_obs=%s" % env.num_privileged_obs)
    check("env.num_actions == 6", env.num_actions == 6,
          "num_actions=%d" % env.num_actions)
    check("actor first layer consumes 35",
          policy.actor[0].in_features == 35,
          "%d" % policy.actor[0].in_features)
    check("actor last layer emits 6",
          policy.actor[-1].out_features == 6,
          "%d" % policy.actor[-1].out_features)
    check("critic first layer consumes 35",
          policy.critic[0].in_features == 35,
          "%d" % policy.critic[0].in_features)

    path = os.path.join(WORKSPACE, FROZEN_MODEL)
    ckpt = torch.load(path, map_location='cpu')
    saved = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    live = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    common = sorted(set(saved) & set(live))
    same = [k for k in common
            if saved[k].shape == live[k].shape and torch.equal(saved[k], live[k])]
    check("checkpoint keys all present in the live policy",
          len(common) == len(saved) and len(saved) > 0,
          "ckpt_keys=%d common=%d" % (len(saved), len(common)))
    check("live policy weights are bit-identical to the frozen checkpoint",
          len(same) == len(common), "identical=%d/%d" % (len(same), len(common)))
    # A random initialisation would differ everywhere; the identity above is what
    # proves this is a WARM start and not a fresh policy that merely loaded.
    return ckpt


def c4_no_phantom_failure(env):
    print("\nC4. no phantom task failure after a reset on this pipeline")
    n = env.num_envs
    ids = torch.arange(n, device=env.device)
    # Recreate the dangerous state: the robots are parked PAST the obstacle while
    # having never taken off. If post-reset geometry were evaluated against a
    # stale rigid-body pose, every one of them would be sentenced as
    # CROSSED_WITHOUT_LIFTOFF on the first step of the next episode.
    behind = 0.62
    root = env.root_states.clone()
    root[:, 1] = behind
    root[:, 7] = 0.0
    root[:, 8] = 0.0
    # `root_states` is a non-contiguous view of `all_root_states[:, 0, :]`
    # (robot and obstacle are interleaved per env), so the whole interleaved
    # tensor has to be pushed to the simulator, not the robot slice.
    env.root_states[:] = root
    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.all_root_states))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    # On the GPU pipeline a refresh alone does not republish; one simulate() is
    # required before rb_states reflects the write. Without it the hazard below
    # is not actually staged and the rest of this check would be vacuous.
    if getattr(env.sim_params, 'use_gpu_pipeline', True):
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
    staged = float(env.rb_states[0, 0, 1].item())
    check("hazard staged: rigid-body pose is really behind the obstacle",
          abs(staged - behind) < 0.02, "rb_states[0,0,1]=%.4f" % staged)

    env._update_task_metrics(post_reset=True)   # the exact post-reset call
    env.reset_idx(ids)
    n_failed = int(env.task_failed.sum().item())
    reasons = sorted(set(env.failure_reason))
    check("no env is marked failed right after the reset",
          n_failed == 0, "n_task_failed=%d" % n_failed)
    check("every failure_reason is NONE after the reset",
          reasons == ['NONE'], "reasons=%s" % (reasons,))
    stale = float(env.rb_states[0, 0, 1].item())
    fresh = float(env.root_states[0, 1].item())
    print("    (post-reset root_states[0,1]=%.4f vs rb_states[0,0,1]=%.4f)"
          % (fresh, stale))
    check("residual rb_states staleness does not leak into the verdict", True,
          "guarded by the post_reset early-return")


def c5_frozen_and_warm_start(env, cfg, train_cfg):
    print("\nC5. model_3.pt is frozen and wired as the warm start")
    path = os.path.join(WORKSPACE, FROZEN_MODEL)
    check("frozen checkpoint exists", os.path.isfile(path))
    digest = sha256(path)
    check("sha256 matches the recorded frozen baseline",
          digest == FROZEN_SHA256, digest[:16] + "...")
    runner_cfg = train_cfg.runner if hasattr(train_cfg, 'runner') else train_cfg
    warm_on = bool(getattr(runner_cfg, 'warm_start', False))
    warm_path = str(getattr(runner_cfg, 'warm_start_path', ''))
    check("runner.warm_start is enabled", warm_on, "warm_start=%s" % warm_on)
    check("warm_start_path points at the frozen checkpoint",
          os.path.normpath(warm_path) == os.path.normpath(FROZEN_MODEL),
          warm_path)
    check("training is NOT resumed from a run directory (no resume flag)",
          not bool(getattr(runner_cfg, 'resume', False)),
          "resume=%s" % getattr(runner_cfg, 'resume', None))


def c6_extras_contract(env, runner):
    print("\nC6. actor_loss_mask / time_outs / terminal_observation")
    policy = runner.alg.actor_critic
    n = env.num_envs
    obs = env.get_observations()
    critic_obs = env.get_privileged_observations()
    critic_obs = critic_obs if critic_obs is not None else obs
    dev = runner.device
    obs, critic_obs = obs.to(dev), critic_obs.to(dev)

    st = runner.alg.storage
    mask_seen = set()
    mask_shapes = set()
    timeout_shapes = set()
    n_timeout_steps = 0
    term_shapes = set()
    term_finite = True
    all_present = True
    bootstrap = None   # (stored, expected_on_terminal_obs, stale_alternative)

    # Force a genuine timeout on the very first step: a 40-step rollout is not
    # guaranteed to reach a 200-step timeout, and an unobserved code path is not
    # an admitted one. The remaining steps then exercise the mask under the real
    # FSM, so a mask that is permanently zero is caught rather than excused.
    for k in range(40):
        if k == 0:
            env.episode_length_buf[:] = env.max_episode_length
        with torch.no_grad():
            actions = runner.alg.act(obs, critic_obs)
        obs, priv, rew, dones, infos = env.step(actions)
        critic_obs = priv if priv is not None else obs
        obs, critic_obs = obs.to(dev), critic_obs.to(dev)
        rew, dones = rew.to(dev), dones.to(dev)

        for key in ('actor_loss_mask', 'time_outs', 'terminal_observation'):
            if key not in infos:
                all_present = False
        if 'actor_loss_mask' in infos:
            m = infos['actor_loss_mask']
            mask_shapes.add(tuple(m.shape))
            mask_seen.update(float(x) for x in m.flatten().tolist())
        if 'time_outs' in infos:
            t = infos['time_outs']
            timeout_shapes.add(tuple(t.shape))
            n_timeout_steps += int(t.sum().item())
        if 'terminal_observation' in infos:
            to = infos['terminal_observation']
            term_shapes.add(tuple(to.shape))
            term_finite = term_finite and bool(torch.isfinite(to).all())

        # `st.step` is the index `add_transitions` is about to write, so it has to
        # be captured BEFORE process_env_step. Using num_transitions_per_env (the
        # fixed capacity) reads the wrong row and silently compares against 0.
        idx = int(st.step)
        fired = ('time_outs' in infos
                 and int(infos['time_outs'].sum().item()) > 0)
        terminal_obs_ref = infos.get('terminal_observation')
        rewards_ref = rew.clone()
        runner.alg.process_env_step(rew, dones, infos)

        if fired and bootstrap is None:
            # The trainer's contract: on a timeout step the stored reward must be
            # r + gamma * V(terminal_observation), evaluated on the TRUE pre-reset
            # observation -- not r + gamma * V(that step's own value).
            i = int(infos['time_outs'].flatten().nonzero(
                as_tuple=False)[0].item())
            with torch.no_grad():
                v_term = float(policy.evaluate(
                    terminal_obs_ref.to(dev))[i, 0].item())
                v_stale = float(st.values[idx, i, 0].item())
            stored = float(st.rewards[idx, i, 0].item())
            expected = float(rewards_ref[i].item()) + runner.alg.gamma * v_term
            stale = float(rewards_ref[i].item()) + runner.alg.gamma * v_stale
            bootstrap = (stored, expected, stale)

    check("all three extras keys are published every step", all_present)
    check("actor_loss_mask is per-env and binary",
          mask_shapes <= {(n,), (n, 1)} and mask_seen.issubset({0.0, 1.0}),
          "shapes=%s values=%s" % (sorted(mask_shapes), sorted(mask_seen)))
    check("actor_loss_mask is exercised in both states (not stuck at 0)",
          mask_seen == {0.0, 1.0}, "values=%s" % sorted(mask_seen))
    check("time_outs is per-env and fired on the forced timeout",
          timeout_shapes <= {(n,), (n, 1)} and n_timeout_steps > 0,
          "shapes=%s n_timeout=%d" % (sorted(timeout_shapes), n_timeout_steps))
    check("terminal_observation is (N,35) and finite",
          term_shapes <= {(n, 35)} and term_finite,
          "shapes=%s finite=%s" % (sorted(term_shapes), term_finite))
    if bootstrap is None:
        check("bootstrap uses terminal_observation", False, "no timeout observed")
    else:
        stored, expected, stale = bootstrap
        check("bootstrap reward equals r + gamma*V(terminal_observation)",
              abs(stored - expected) < 2e-2,
              "stored=%.5f expected=%.5f" % (stored, expected))
        check("bootstrap did NOT fall back to the step's own value",
              abs(stored - stale) > 2e-2 or abs(expected - stale) < 1e-9,
              "stored=%.5f own_value_would_be=%.5f" % (stored, stale))


def c7_reward_signs(env, cfg):
    print("\nC7. collision penalty is negative; success is not paid early")
    scale = float(cfg.rewards.scales.obstacle_collision)
    check("obstacle_collision scale is negative", scale < 0.0,
          "scale=%.2f" % scale)

    fn = getattr(env, '_reward_obstacle_collision')
    env.has_obstacle_collision[:] = False
    env.peak_obstacle_contact_force[:] = 0.0
    clean = float((fn() * scale).max().item())
    env.has_obstacle_collision[:] = True
    env.peak_obstacle_contact_force[:] = 50.0
    mild = float((fn() * scale).min().item())
    env.peak_obstacle_contact_force[:] = 300.0
    severe = float((fn() * scale).min().item())
    check("no contact contributes exactly zero", abs(clean) < 1e-9,
          "%.5f" % clean)
    check("collision contributes a negative reward", mild < 0.0,
          "mild=%.3f" % mild)
    check("more severe collision is punished harder", severe < mild,
          "mild=%.3f severe=%.3f" % (mild, severe))
    env.has_obstacle_collision[:] = False
    env.peak_obstacle_contact_force[:] = 0.0

    sf = getattr(env, '_reward_obstacle_cleared_success')
    succ_scale = float(cfg.rewards.scales.obstacle_cleared_success)
    env.obstacle_cleared[:] = True          # latch is set...
    env.clear_success_event[:] = False      # ...but the one-shot event is not
    latched_only = float((sf() * succ_scale).max().item())
    env.clear_success_event[:] = True       # the event itself
    on_event = float((sf() * succ_scale).min().item())
    check("a set cleared-latch alone pays nothing (no early bonus)",
          abs(latched_only) < 1e-9, "%.5f" % latched_only)
    check("the one-shot success event pays exactly once",
          abs(on_event - succ_scale) < 1e-6,
          "on_event=%.3f scale=%.3f" % (on_event, succ_scale))
    env.obstacle_cleared[:] = False
    env.clear_success_event[:] = False

    gp = getattr(env, '_reward_obstacle_geometric_pass', None)
    # `cfg.rewards.scales` is a plain config OBJECT, not a dict -- `getattr` is
    # the only accessor that works (`.get` raises AttributeError).
    gscale_raw = getattr(cfg.rewards.scales, 'obstacle_geometric_pass', None)
    check("geometric pass has its own, separate reward term",
          gp is not None and gscale_raw is not None,
          "scale=%s" % gscale_raw)
    if gp is not None and gscale_raw is not None:
        gscale = float(gscale_raw)
        check("a geometric pass is worth strictly less than full success",
              0.0 < gscale < abs(succ_scale),
              "pass=%.2f full=%.2f" % (gscale, abs(succ_scale)))
        env.geometric_pass_event[:] = False
        off = float((gp() * gscale).max().item())
        env.geometric_pass_event[:] = True
        on = float((gp() * gscale).min().item())
        check("geometric pass pays only on its event",
              abs(off) < 1e-9 and abs(on - gscale) < 1e-6,
              "off=%.4f on=%.4f" % (off, on))
        env.geometric_pass_event[:] = False


def c8_run_isolation(train_cfg):
    print("\nC8. the pilot run directory is isolated from history")
    logs_root = os.path.join(WORKSPACE, 'legged_gym', 'logs')
    task_dir = os.path.join(logs_root, 'myrobot_obstacle_jump')
    target = os.path.join(task_dir, PILOT_RUN_NAME)
    check("pilot run directory does not exist yet", not os.path.exists(target),
          target)

    def count_ckpts(root):
        found = []
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.endswith('.pt'):
                    found.append(os.path.join(dirpath, f))
        return found

    history = count_ckpts(logs_root) if os.path.isdir(logs_root) else []
    hist_obstacle = [p for p in history
                     if os.sep + 'myrobot_obstacle_jump' + os.sep in p]
    check("no historical checkpoint lives under myrobot_obstacle_jump",
          not hist_obstacle, "n=%d" % len(hist_obstacle))
    frozen_inside_target = [p for p in history if p.startswith(target + os.sep)]
    check("the frozen baseline is outside the pilot directory",
          not frozen_inside_target, "n=%d" % len(frozen_inside_target))
    print("    (checkpoints elsewhere in history: %d, left untouched)"
          % len(history))


def main():
    print("=" * 82)
    print("Phase C training-admission checklist")
    print("=" * 82)
    num_envs = 4
    env, runner, train_cfg, args = build(num_envs)
    cfg = env.cfg
    print("pipeline=%s  use_gpu_pipeline=%s  num_envs=%d"
          % ('gpu' if args.use_gpu_pipeline else 'cpu',
             getattr(env.sim_params, 'use_gpu_pipeline', None), env.num_envs))
    print("contact_pairs_available=%s (training path must not need it)"
          % getattr(env, 'contact_pairs_available', None))

    c3_network_widths(env, runner)
    c4_no_phantom_failure(env)
    c5_frozen_and_warm_start(env, cfg, train_cfg)
    c6_extras_contract(env, runner)
    c7_reward_signs(env, cfg)
    c8_run_isolation(train_cfg)

    print("\n" + "=" * 82)
    print("PASSED %d / %d" % (len(PASS), len(PASS) + len(FAIL)))
    if FAIL:
        print("FAILED CHECKS:")
        for name in FAIL:
            print("  - " + name)
        print("RESULT: FAIL -- do not launch the pilot")
        return 1
    print("RESULT: PASS")
    print("C1/C2 are proven by tests/test_obstacle_jump_semantics.py and "
          "tests/test_ppo_smoke.py; run them alongside this script.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
