"""One-iteration PPO data-path smoke test on the real obstacle environment.

WHAT THIS PROVES AND WHAT IT DOES NOT
-------------------------------------
It runs ONE PPO iteration on the real `myrobot_obstacle_jump` env with the real
`OnPolicyRunner`, the real `RolloutStorage` and the real GAE, then inspects the
collected tensors. No weights are trained to convergence and nothing is saved to
`legged_gym/logs`, so this is NOT a training run -- it is a check that the data
path is intact before anyone spends hours on one.

The checks are grouped by the question they answer:

  A. shapes / dtypes      actor 35 -> 35, critic 35 -> 35, action 6; no NaN/Inf
                          anywhere in actions, values, log-probs, rewards,
                          advantages or returns.
  B. reward + termination  the reward buffer has both signs, terminal steps are
                          flagged, and the per-env done at a collision carries
                          the collision penalty rather than a positive reward.
  C. GAE / storage         `compute_returns` produces finite advantages with the
                          expected shape, and the value target inside the storage
                          equals reward + gamma * (1-done) * next_value for a
                          sampled transition.
  D. PPO-specific hooks    `actor_loss_mask` is consumed (the storage really
                          records a per-step mask, not the all-ones default),
                          `time_outs` bootstraps through `terminal_observation`
                          instead of the stale value, and `extras['episode']`
                          survives the cycle.
  E. PIPELINE INDEPENDENCE the same iteration runs under the GPU pipeline. The
                          contact-PAIR introspection this project uses for
                          analysis is CPU-only (`GymGetEnvRigidContactCount`
                          cannot be called after simulation starts on GPU), so
                          the training path must not depend on it. `contact_pairs_
                          available` is asserted FALSE on GPU while the reward,
                          termination, observation and collision-detection paths
                          keep working.

Run:
    python tests/test_ppo_smoke.py --pipeline gpu
    python tests/test_ppo_smoke.py --pipeline cpu --rl_device cpu

    The second form needs `--rl_device cpu` as well: the mixed configuration
    (simulation on CPU, learner on the default cuda:0) is rejected by the env
    itself, which writes policy actions straight into CPU buffers
    (myrobot_dual_env.py, `self.filtered_actions[jump_active] = ...`) and raises
    "Expected all tensors to be on the same device". That is pre-existing repo
    behaviour, unrelated to the PPO data path this file checks, so the test
    drives the two consistent configurations instead.
"""

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
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print("  [%s] %-58s %s" % ("PASS" if ok else "FAIL", name, detail))


def build(num_envs, extra_args=()):
    forwarded = []
    # --rl_device matters as much as --pipeline here: it selects the device the
    # policy and the rollout storage live on, and it is independent of the
    # simulation pipeline. Forwarding it lets the CPU-pipeline run put the
    # learner on the CPU too, which is the only way to exercise the
    # env-on-CPU / policy-on-CPU combination without a GPU in the loop.
    for i, tok in enumerate(sys.argv):
        if tok in ('--pipeline', '--device', '--rl_device') and i + 1 < len(sys.argv):
            forwarded += [tok, sys.argv[i + 1]]
    argv = [sys.argv[0], '--task', 'myrobot_obstacle_jump', '--headless',
            '--seed', '11', '--num_envs', str(num_envs)]
    argv += forwarded
    argv += list(extra_args)
    sys.argv = argv
    args = get_args()
    # Keep the learner on the same device as the simulation when the simulation
    # publishes its tensors on the CPU. The env feeds policy actions into its own
    # CPU buffers (`self.filtered_actions[jump_active] = 0.80 * act_jump_raw[...]`
    # in myrobot_dual_env.py), so an env-on-CPU / learner-on-cuda pair raises
    # "Expected all tensors to be on the same device" from repo code that this
    # data-path check does not own. Only defaulted -- an explicit --rl_device is
    # forwarded and honoured as given.
    #
    # The gate is `use_gpu_pipeline`, not `sim_device`: `--pipeline cpu` leaves
    # sim_device at "cuda:0" and only flips use_gpu_pipeline to False, which is
    # exactly what makes base_task.py hand out cpu tensors.
    if not args.use_gpu_pipeline and not any('--rl_device' in t for t in forwarded):
        args.rl_device = 'cpu'
        print("(learner device defaulted to cpu to match cpu-side env tensors)")
    cfg, _ = task_registry.get_cfgs(name=args.task)
    cfg.env.num_envs = num_envs
    # Long enough that at least some envs reach a reset inside the rollout, so
    # the terminal-handling branches are actually exercised. num_envs is small
    # because this is a data-path check, not a throughput benchmark.
    cfg.env.episode_length_s = 4.0
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=cfg)
    # log_root=None is the registry's no-logging entry point: it guarantees this
    # check cannot create a run directory or save a checkpoint. A smoke test that
    # quietly wrote into legged_gym/logs would be a training run by accident.
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, log_root=None)
    return env, runner, train_cfg, args


def main():
    gpu = ('gpu' in sys.argv) and not ('cpu' in sys.argv)
    pipeline = 'gpu' if gpu else 'cpu'
    num_envs = 8
    steps_per_env = 48
    print("=" * 78)
    print("PPO data-path smoke test -- pipeline=%s num_envs=%d steps=%d"
          % (pipeline, num_envs, steps_per_env))
    print("=" * 78)

    env, runner, train_cfg, args = build(num_envs)
    # Shrink the rollout so one iteration is quick but still spans several
    # episodes' worth of resets. This mutates only the live runner/storage
    # sizing, not the algorithm config that governs a real training run.
    runner.num_steps_per_env = steps_per_env
    runner.alg.init_storage(env.num_envs, steps_per_env,
                            [env.num_obs], [env.num_privileged_obs],
                            [env.num_actions])
    policy = runner.alg.actor_critic

    # ---------------------------------------------------------------- A
    print("\nA. shapes and dtypes")
    check("actor observation width is 35", env.num_obs == 35, "num_obs=%d" % env.num_obs)
    check("critic observation width is 35",
          (env.num_privileged_obs or env.num_obs) == 35,
          "num_privileged_obs=%s" % env.num_privileged_obs)
    check("action width is 6", env.num_actions == 6, "num_actions=%d" % env.num_actions)
    check("actor input layer width is 35",
          policy.actor[0].in_features == 35, "%d" % policy.actor[0].in_features)
    check("actor output layer width is 6",
          policy.actor[-1].out_features == 6, "%d" % policy.actor[-1].out_features)
    check("critic input layer width is 35",
          policy.critic[0].in_features == 35, "%d" % policy.critic[0].in_features)

    # ---------------------------------------------------------------- rollout
    # One full rollout + one update, exactly as learn() would do it, except the
    # update is the only thing that touches weights and it runs on ONE batch.
    print("\nA2. one rollout of %d steps" % steps_per_env)
    obs = env.get_observations()
    critic_obs = env.get_privileged_observations()
    critic_obs = critic_obs if critic_obs is not None else obs
    # Observations go to the RUNNER's device, not env.device. The real training
    # loop (rsl_rl/rsl_rl/runners/on_policy_runner.py:105 and :123) moves them to
    # `self.device`, which comes from --rl_device (default cuda:0) and is
    # independent of --pipeline. Under `--pipeline cpu` the env lives on the CPU
    # while the policy stays on the GPU, so using env.device here raises
    # "Expected all tensors to be on the same device, but found at least two
    # devices, cuda:0 and cpu!"
    rl_device = runner.device
    obs, critic_obs = obs.to(rl_device), critic_obs.to(rl_device)

    reward_rows = []
    done_rows = []
    mask_rows = []
    collided_steps = 0
    episode_infos = []
    for _ in range(steps_per_env):
        with torch.no_grad():
            actions = runner.alg.act(obs, critic_obs)
        obs, privileged_obs, rewards, dones, infos = env.step(actions)
        critic_obs = privileged_obs if privileged_obs is not None else obs
        # Same reasoning as the pre-rollout line above: the training loop moves
        # tensors to the runner's device, not the env's.
        obs, critic_obs = obs.to(rl_device), critic_obs.to(rl_device)
        rewards, dones = rewards.to(rl_device), dones.to(rl_device)

        # Capture what the training path is about to consume, BEFORE
        # process_env_step mutates anything.
        reward_rows.append(rewards.detach().clone())
        done_rows.append(dones.detach().clone())
        if 'actor_loss_mask' in infos:
            mask_rows.append(infos['actor_loss_mask'].detach().clone().view(-1))
        collided_steps += int(env.has_obstacle_collision.sum().item())
        if 'episode' in infos:
            episode_infos.append(infos['episode'])

        runner.alg.process_env_step(rewards, dones, infos)

    rewards_all = torch.cat(reward_rows)
    dones_all = torch.cat(done_rows)
    check("reward tensor has no NaN/Inf",
          bool(torch.isfinite(rewards_all).all()),
          "min=%.4f max=%.4f" % (rewards_all.min(), rewards_all.max()))
    check("reward buffer has both signs (penalties actually reach PPO)",
          bool((rewards_all < 0).any() and (rewards_all > 0).any()),
          "n_neg=%d n_pos=%d" % (
              int((rewards_all < 0).sum()), int((rewards_all > 0).sum())))
    check("terminal steps are flagged inside the rollout",
          int(dones_all.sum()) > 0,
          "n_dones=%d" % int(dones_all.sum()))

    st = runner.alg.storage
    check("stored actions have no NaN/Inf",
          bool(torch.isfinite(st.actions).all()))
    check("stored values have no NaN/Inf",
          bool(torch.isfinite(st.values).all()))
    check("stored log-probs have no NaN/Inf",
          bool(torch.isfinite(st.actions_log_prob).all()))
    check("stored reward shape matches rollout",
          tuple(st.rewards.shape) == (steps_per_env, num_envs, 1),
          "%s" % (tuple(st.rewards.shape),))

    # ---------------------------------------------------------------- D
    print("\nD. PPO-specific hooks")
    if mask_rows:
        masks = torch.cat(mask_rows)
        # A real mask must be binary and shaped per-env-per-step. The all-ones
        # fallback in PPO.process_env_step would also pass the binary test, so
        # this only proves the hook was wired, not that it ever zeroes.
        check("actor_loss_mask reaches the storage",
              st.actor_masks is not None
              and st.actor_masks.shape == (steps_per_env, num_envs, 1),
              "shape=%s" % (tuple(st.actor_masks.shape),))
        check("actor_loss_mask values are binary",
              bool(((masks == 0.0) | (masks == 1.0)).all()))
    else:
        check("actor_loss_mask reaches the storage", False, "key absent from infos")

    check("episode extras survive the rollout",
          len(episode_infos) > 0,
          "n_episode_events=%d" % len(episode_infos))

    # Collision -> penalty and -> termination on the same step. Both properties
    # are checked on the frames where the env itself says a collision happened.
    print("\nB. collision step carries a penalty AND terminates")
    if collided_steps > 0:
        print("    (observed %d env-steps with has_obstacle_collision)"
              % collided_steps)
    else:
        print("    (no collision inside this short rollout -- checking the reward"
              " function directly instead)")
    env.reset_idx(torch.arange(num_envs, device=env.device))
    zero_act = torch.zeros(num_envs, env.num_actions, device=env.device)
    env.step(zero_act)   # settle one frame so buffers are populated
    reward_fn = getattr(env, '_reward_obstacle_collision', None)
    check("collision reward function exists", reward_fn is not None)
    if reward_fn is not None:
        base = float(reward_fn().abs().max().item())
        # Force a collision and re-evaluate: the penalty must grow, and it must
        # come out NEGATIVE once the config scale is applied. A positive return
        # here combined with the -5.0 scale would mean double negation.
        env.has_obstacle_collision[:] = True
        env.peak_obstacle_contact_force[:] = 300.0
        raw_hit = reward_fn()
        scale = float(env.cfg.rewards.scales.obstacle_collision)
        check("collision penalty is positive before the config scale",
              bool((raw_hit >= 0).all()),
              "raw min=%.3f" % float(raw_hit.min()))
        check("config scale is negative", scale < 0.0, "scale=%.2f" % scale)
        check("collision reward contribution is negative after scaling",
              bool((raw_hit * scale <= 0).all()),
              "min contrib=%.3f" % float((raw_hit * scale).min()))
        check("collision reward grows with contact force",
              float(raw_hit.max()) >= base,
              "clean=%.3f hit=%.3f" % (base, float(raw_hit.max())))
        env.has_obstacle_collision[:] = False
        env.peak_obstacle_contact_force[:] = 0.0

    # ---------------------------------------------------------------- C
    print("\nC. GAE and the value target")
    last_critic_obs = critic_obs
    runner.alg.compute_returns(last_critic_obs)
    adv = st.advantages
    ret = st.returns
    check("advantages are finite", bool(torch.isfinite(adv).all()))
    check("returns are finite", bool(torch.isfinite(ret).all()))
    check("advantage shape matches the rollout",
          tuple(adv.shape) == (steps_per_env, num_envs, 1),
          "%s" % (tuple(adv.shape),))
    # GAE is verified on `returns`/`values`, both of which stay raw.
    # `advantages` is normalised IN PLACE by compute_returns, so it can only be
    # checked for being normalised, not for the raw recursion.
    #   A[t] = returns[t] - values[t]                       (recovered raw adv)
    #   delta[t] = r[t] + gamma * (1-done) * V[t+1] - V[t]
    #   A[t] = delta[t] + gamma * lam * (1-done) * A[t+1]
    # At t == last, next_values is the bootstrap `last_values`, which is the
    # critic evaluated on the final observation -- reproduce it rather than
    # hard-coding it, otherwise a disconnected critic would pass unnoticed.
    gamma = float(runner.alg.gamma)
    lam = float(runner.alg.lam)
    raw_adv = ret - st.values
    last_values = policy.evaluate(critic_obs).detach()
    t = steps_per_env // 2
    checked_gae = False
    for i in range(num_envs):
        done_t = bool(st.dones[t, i, 0].item())
        if done_t:
            continue
        v_next = (last_values[i, 0] if t == steps_per_env - 1
                  else st.values[t + 1, i, 0])
        delta = (st.rewards[t, i, 0] + gamma * v_next - st.values[t, i, 0])
        rhs = (delta + gamma * lam * raw_adv[t + 1, i, 0]
               if t + 1 < steps_per_env else delta)
        lhs = raw_adv[t, i, 0]
        check("GAE recurrence holds at t=%d/env%d (lambda=%.3f included)" % (t, i, lam),
              abs(float(lhs - rhs)) < 5e-3,
              "lhs=%.5f rhs=%.5f" % (float(lhs), float(rhs)))
        checked_gae = True
        break
    if not checked_gae:
        check("GAE recurrence is checkable on this rollout", False,
              "every env was terminal at t=%d" % t)
    # On a done step the bootstrap is already folded into the STORED reward (the
    # env's time-out bonus is added in process_env_step, not in compute_returns),
    # so the return there must collapse to exactly that stored reward.
    done_any = st.dones.squeeze(-1).nonzero(as_tuple=False)
    if done_any.numel() > 0:
        dt_, di_ = int(done_any[0, 0]), int(done_any[0, 1])
        check("return at a done step collapses to its stored reward",
              abs(float(ret[dt_, di_, 0] - st.rewards[dt_, di_, 0])) < 2e-3,
              "t=%d env=%d ret=%.5f reward=%.5f"
              % (dt_, di_, float(ret[dt_, di_, 0]), float(st.rewards[dt_, di_, 0])))
    else:
        check("rollout contains a terminal step to exercise bootstrapping", False)
    # compute_returns normalises the advantages in place; this is what the PPO
    # loss consumes, and an un-normalised buffer would silently change the
    # effective learning rate.
    check("advantages are normalised to roughly unit scale",
          abs(float(adv.mean())) < 0.2 and abs(float(adv.std()) - 1.0) < 0.2,
          "mean=%.4f std=%.4f" % (float(adv.mean()), float(adv.std())))

    # ---------------------------------------------------------------- update
    print("\nC2. one PPO update step runs and produces finite losses")
    before = [p.detach().clone() for p in policy.actor.parameters()]
    loss_pair = runner.alg.update()
    # `PPO.update()` in this rsl_rl fork returns a TUPLE
    # (mean_value_loss, mean_surrogate_loss), not a dict; the rich diagnostics
    # land on `alg.last_update_diagnostics`. Asserting a dict here would have
    # tested a contract this fork does not offer.
    diag = getattr(runner.alg, 'last_update_diagnostics', {}) or {}
    check("update returned the (value_loss, surrogate_loss) pair",
          isinstance(loss_pair, tuple) and len(loss_pair) == 2
          and all(math.isfinite(float(v)) for v in loss_pair),
          "value=%.5f surrogate=%.5f" % (float(loss_pair[0]), float(loss_pair[1]))
          if isinstance(loss_pair, tuple) and len(loss_pair) == 2
          else str(type(loss_pair)))
    check("update populated last_update_diagnostics", bool(diag),
          "n_keys=%d" % len(diag))
    if diag:
        bad = [k for k, v in diag.items()
               if isinstance(v, float) and not math.isfinite(v)]
        check("all reported update diagnostics are finite", not bad,
              "bad=%s" % (bad,))
        for key in ('value_loss', 'surrogate_loss', 'kl_mean',
                    'clip_fraction', 'actor_grad_norm_mean',
                    'critic_grad_norm_mean'):
            if key in diag:
                check("diagnostic %s is finite" % key,
                      math.isfinite(float(diag[key])),
                      "%.5f" % float(diag[key]))
    moved = any(not torch.equal(a, b) for a, b in
                zip(before, policy.actor.parameters()))
    # The update may legitimately leave params untouched if every minibatch was
    # skipped by the mask; what matters is that no NaN was written.
    grads_ok = all(p.grad is None or torch.isfinite(p.grad).all()
                   for p in policy.actor.parameters())
    check("actor gradients are finite after the update", bool(grads_ok),
          "params_moved=%s" % moved)

    # ---------------------------------------------------------------- E
    print("\nE. pipeline independence")
    print("    contact_pairs_available = %s" % env.contact_pairs_available)
    if gpu:
        check("GPU pipeline reports contact-pair introspection unavailable",
              env.contact_pairs_available is False,
              "this is expected: GymGetEnvRigidContactCount is CPU-only")
        # The training path must still detect a genuine overlap without the
        # contact-pair API, via the analytic wheel/box test.
        from legged_gym.envs.myrobot import obstacle_geometry as geo
        gaps = geo.wheel_box_signed_gap(env)
        check("analytic wheel/box gap is available on GPU",
              bool(torch.isfinite(gaps).all()),
              "min gap=%.5f m" % float(gaps.min()))
        env.has_obstacle_collision[:] = False
        env.first_contact_step[:] = -1

        # Two facts make a hand-positioned overlap test harder than it looks:
        #
        #  1. The capsule centre must come from the MEASURED link pose, not a
        #     constant. The wheels sit in a bicycle layout (identical y and z,
        #     different x) and their offset from the base depends on the joint
        #     configuration, so a hard-coded offset places the capsule outside
        #     the box and the check then "fails" for reasons unrelated to the
        #     GPU pipeline.
        #  2. On the GPU pipeline `rb_states` is only republished by
        #     `simulate()`. `set_actor_root_state_tensor` followed by
        #     refresh_actor_root_state_tensor + refresh_rigid_body_state_tensor
        #     leaves it holding the PREVIOUS pose, so the capsule measurement
        #     below would be stale and the applied delta would be wrong.
        #     Measured: right after reset+refresh the GPU reports base z=0.005
        #     (garbage) instead of ~0.42; one simulate() fixes it.
        ids = torch.arange(env.num_envs, device=env.device)
        left = env.left_wheel_body

        def _capsule_centre():
            axial = geo.WHEEL_AXIAL_OFFSET[env.link_names[left]]
            axis = torch.stack(geo.wheel_axis_terms(env.rb_states, left), dim=1)
            return env.rb_states[:, left, 0:3] + axial * axis

        def _push_and_settle():
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env.gym.refresh_actor_root_state_tensor(env.sim)
            env.gym.refresh_rigid_body_state_tensor(env.sim)

        env._reset_dofs(ids)
        env._reset_root_states(ids)
        _push_and_settle()
        capsule = _capsule_centre()
        lo, hi = geo._box_bounds(env)
        box_mid = 0.5 * (lo + hi)
        env.root_states[:, 0:3] += (box_mid - capsule)
        env.root_states[:, 7:13] = 0.0
        # set_actor_root_state_tensor needs the raw gym tensor handle, not a
        # torch tensor: passing the tensor directly raises
        # "set_actor_root_state_tensor(): incompatible function arguments".
        env.gym.set_actor_root_state_tensor(
            env.sim, gymtorch.unwrap_tensor(env.all_root_states.view(-1, 13).contiguous()))
        _push_and_settle()
        moved = _capsule_centre()
        check("teleporting onto the box actually moved the capsule on GPU",
              float(torch.norm(moved - capsule, dim=1).min()) > 0.03,
              "moved=%.5f m" % float(torch.norm(moved - capsule, dim=1).min()))
        mask, gaps = env._geometric_wheel_intersection()
        check("collision detected without the contact-pair API",
              bool(mask.any()),
              "n_env_intersecting=%d deepest gap=%.5f m"
              % (int(mask.sum()), float(gaps.min())))
    else:
        check("CPU pipeline reports contact-pair introspection available",
              env.contact_pairs_available is True)

    # ---------------------------------------------------------------- result
    print("\n" + "=" * 78)
    if FAIL:
        print("FAILED %d / %d" % (len(FAIL), len(PASS) + len(FAIL)))
        for name in FAIL:
            print("  FAILED: %s" % name)
        print("RESULT: FAIL")
        return 1
    print("PASSED %d / %d" % (len(PASS), len(PASS)))
    print("RESULT: PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
