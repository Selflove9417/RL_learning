"""Counter-example tests for the obstacle-jump task semantics.

Every test here encodes a SPECIFIC way the task could silently mislead PPO, and
asserts the property, not an implementation detail. They are deliberately
independent of the policy: no checkpoint is loaded and no training happens.

Run from the repository root, after `source ./activate_jump.sh`:

    python tests/test_obstacle_jump_semantics.py --pipeline cpu

Checks, grouped by the four defects reported against the previous code:

  A. Collision penalty SIGN -- a harder collision must score WORSE.
     The historical bug multiplied an already-negative function by a negative
     cfg scale, so the robot was PAID for ramming the fence.

  B. Success reward is an EDGE EVENT -- the latch alone must pay nothing.

  C. Metric TIMING -- termination and reward must see the CURRENT step's
     obstacle metrics, not the previous step's.

  D. Termination -- genuine collisions and ploughed crossings must terminate;
     a clean jump over the box must not.

  E. Per-environment obstacle state -- one env repositioning its box must not
     move any other env's box, and reset must restore the configured offset.

  F. Geometry discipline -- the trigger envelope must be measured from the real
     wheel/leg front, not `base_y + wheel_radius`, and must ignore geometry that
     passes above the box top.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import isaacgym  # noqa: F401  (must precede torch)
from isaacgym import gymtorch
import torch

from legged_gym.envs import MyRobotObstacleJump  # noqa: F401
from legged_gym.envs.myrobot import obstacle_geometry as geo
from legged_gym.utils import get_args, task_registry

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print("  [%s] %-58s %s" % ("PASS" if ok else "FAIL", name, detail))


def build(extra_args=(), num_envs=1):
    # Forward any --pipeline / --device flags the user passed, so the suite runs
    # on the same pipeline the caller selected (contact PAIRS require the CPU
    # pipeline; leaving it to the default silently disables that evidence path).
    forwarded = []
    for i, tok in enumerate(sys.argv):
        if tok == '--pipeline' and i + 1 < len(sys.argv):
            forwarded += ['--pipeline', sys.argv[i + 1]]
        elif tok == '--device' and i + 1 < len(sys.argv):
            forwarded += ['--device', sys.argv[i + 1]]
    argv = [sys.argv[0], '--task', 'myrobot_obstacle_jump', '--headless',
            '--seed', '11', '--num_envs', str(num_envs)]
    argv += forwarded
    argv += list(extra_args)
    sys.argv = argv
    args = get_args()
    cfg, _ = task_registry.get_cfgs(name=args.task)
    cfg.env.num_envs = num_envs
    cfg.env.episode_length_s = 4.0
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=cfg)
    return env


# ---------------------------------------------------------------------------
# A. Collision penalty sign
# ---------------------------------------------------------------------------

def test_collision_penalty_sign(env):
    print("\nA. collision penalty sign")
    scale = getattr(env.cfg.rewards.scales, 'obstacle_collision')

    # The reward function must return a NON-NEGATIVE severity; the cfg scale owns
    # the sign. Positive return + negative scale = penalty.
    env.has_obstacle_collision[:] = False
    env.peak_obstacle_contact_force[:] = 0.0
    clean = float(env._reward_obstacle_collision()[0]) * scale

    env.has_obstacle_collision[:] = True
    env.peak_obstacle_contact_force[:] = 100.0
    soft = float(env._reward_obstacle_collision()[0]) * scale

    env.peak_obstacle_contact_force[:] = 400.0
    hard = float(env._reward_obstacle_collision()[0]) * scale

    check("no collision contributes exactly 0", clean == 0.0, "clean=%.3f" % clean)
    check("collision contributes a penalty", soft < 0.0, "soft=%.3f" % soft)
    check("harder collision scores strictly worse", hard < soft,
          "soft=%.3f hard=%.3f" % (soft, hard))
    # Monotonicity across a sweep, so a plateau cannot hide a sign flip.
    vals = []
    for force in (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0):
        env.peak_obstacle_contact_force[:] = force
        vals.append(float(env._reward_obstacle_collision()[0]) * scale)
    non_increasing = all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))
    check("penalty is monotone non-increasing in severity", non_increasing,
          " ".join("%.2f" % v for v in vals))

    env.has_obstacle_collision[:] = False
    env.peak_obstacle_contact_force[:] = 0.0


# ---------------------------------------------------------------------------
# B. Success reward is an edge event
# ---------------------------------------------------------------------------

def test_success_is_edge_event(env):
    print("\nB. success reward is a one-shot edge event")
    env.clear_success_event[:] = False
    env.obstacle_cleared[:] = False
    zero = float(env._reward_obstacle_cleared_success()[0])

    # Latched flag alone (what the buggy version paid for) must be worth nothing.
    env.obstacle_cleared[:] = True
    latched_only = float(env._reward_obstacle_cleared_success()[0])

    # The edge, on the step it happens.
    env.clear_success_event[:] = True
    on_edge = float(env._reward_obstacle_cleared_success()[0])

    check("no success -> 0", zero == 0.0, "%.3f" % zero)
    check("latched flag alone -> 0 (no重复发放)", latched_only == 0.0,
          "%.3f" % latched_only)
    check("edge event -> positive reward", on_edge > 0.0, "%.3f" % on_edge)

    # Two consecutive calls while still latched: the second must not pay again.
    env.clear_success_event[:] = False
    env._update_task_metrics()          # clears the one-shot events
    second = float(env._reward_obstacle_cleared_success()[0])
    check("reward not repeated on the next step", second == 0.0, "%.3f" % second)

    env.obstacle_cleared[:] = False


# ---------------------------------------------------------------------------
# C. Metric timing: hooks run before reward/termination
# ---------------------------------------------------------------------------

def test_metric_timing(env):
    print("\nC. obstacle metrics are refreshed before reward and termination")
    import inspect

    src = inspect.getsource(type(env).__mro__[1].step)   # MyRobotRunningJump.step
    i_hook = src.find('_update_task_metrics')
    i_term = src.find('_task_failure_mask')
    i_reward = src.find('compute_reward()')
    check("hook precedes termination", 0 <= i_hook < i_term,
          "hook@%d term@%d" % (i_hook, i_term))
    check("hook precedes reward", 0 <= i_hook < i_reward,
          "hook@%d reward@%d" % (i_hook, i_reward))
    check("termination consults the task failure mask", i_term > 0)


# ---------------------------------------------------------------------------
# D. Termination semantics
# ---------------------------------------------------------------------------

def test_termination(env):
    print("\nD. termination semantics")
    ids = torch.arange(env.num_envs, device=env.device)

    # A genuine collision must terminate.
    env.has_obstacle_collision[:] = True
    env.task_failed[:] = False
    mask = env._task_failure_mask()
    check("collision enters the termination mask", bool(mask[0].item()))

    # Ploughing through the box without ever leaving the ground must terminate.
    # The criterion is the rear envelope being past the box rear face: proximity
    # alone must NOT trip it (see the seed-15 false positive below).
    env.has_obstacle_collision[:] = False
    env.robot_rear_margin[:] = 0.02      # whole envelope past the far face
    env.has_taken_off[:] = False
    env.task_failed[:] = False
    env._update_task_outcome()
    check("ploughing without liftoff is a failure",
          bool(env.task_failed[0].item()),
          "reason=%s" % env.failure_reason[0])
    check("failure mask follows task_failed",
          bool(env._task_failure_mask()[0].item()))

    # REGRESSION (seed 15): being merely adjacent to the box with wheels down is
    # NOT a failure. The real analytic wheel/box gap there was +0.43 mm.
    env.has_obstacle_collision[:] = False
    env.task_failed[:] = False
    env.robot_rear_margin[:] = -0.02     # rear envelope still before the far face
    env.has_taken_off[:] = False
    env.failure_reason[0] = "NONE"
    env._update_task_outcome()
    check("proximity with wheels down is not a failure",
          not bool(env.task_failed[0].item()),
          "reason=%s" % env.failure_reason[0])

    # A clean jump passing ABOVE the box must NOT be a failure: the wheels are
    # over the box in Y but well above its top plane.
    env.has_obstacle_collision[:] = False
    env.task_failed[:] = False
    env.robot_rear_margin[:] = -0.02
    env.has_taken_off[:] = True
    env.obstacle_cleared[:] = True
    env.failure_reason[0] = "NONE"          # clear the stale label from above
    env._update_task_outcome()
    check("clean clearance is not a failure",
          not bool(env.task_failed[0].item()),
          "reason=%s" % env.failure_reason[0])

    # Episode bookkeeping must be zeroed on reset regardless of extras['episode']
    # having been created by a super() call.
    for key in env.episode_sums:
        env.episode_sums[key][:] = 12.34
    env.extras.pop('episode', None)
    env.reset_idx(ids)
    residual = max(float(env.episode_sums[k].abs().max().item())
                   for k in env.episode_sums)
    check("episode_sums zeroed on reset", residual < 1e-9, "max=%.3g" % residual)

    env.task_failed[:] = False
    env.has_taken_off[:] = False
    env.obstacle_cleared[:] = False
    env.robot_rear_margin[:] = -1.0


# ---------------------------------------------------------------------------
# E. Per-environment obstacle state
# ---------------------------------------------------------------------------

def test_per_env_obstacle_state(env):
    print("\nE. per-environment obstacle coordinates")
    ids = torch.arange(env.num_envs, device=env.device)
    env.reset()
    env.reset_idx(ids)                      # deterministic obstacle placement

    before = env.obstacle_py_t.clone()
    before_world = env.obstacle_root_states[:, 1].clone()

    # Reposition only env 0, exactly as continuous mode does.
    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    mask[0] = True
    env._reposition_obstacles(mask)

    moved = float(env.obstacle_py_t[0].item()) - float(before[0].item())
    others_unchanged = bool(torch.allclose(env.obstacle_py_t[1:], before[1:]))
    check("repositioning moves the target env", abs(moved) > 1e-3,
          "delta=%.4f" % moved)
    if env.num_envs > 1:
        check("repositioning leaves other envs untouched", others_unchanged,
              "others=%s" % env.obstacle_py_t[1:].tolist())
        world_ok = bool(torch.allclose(env.obstacle_root_states[1:, 1],
                                       before_world[1:]))
        check("other envs' obstacle actors unmoved", world_ok)

    # Reset must restore the configured offset for the reset envs only.
    cfg_py = float(getattr(env.cfg.obstacle, 'pos_y', 0.300))
    reset_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    env.reset_idx(reset_ids)
    check("reset restores configured obstacle offset",
          abs(float(env.obstacle_py_t[0].item()) - cfg_py) < 1e-6,
          "got=%.4f want=%.4f" % (float(env.obstacle_py_t[0].item()), cfg_py))

    # Relative distance must be measured against the same env's obstacle.
    front, rear, top = geo.obstacle_y_interval(env)
    expect_front = env.env_origins[:, 1] + env.obstacle_py_t - 0.5 * env.obstacle_w
    check("obstacle interval follows each env's own coordinates",
          bool(torch.allclose(front, expect_front, atol=1e-5)))

    # The box must actually be solid: dropping a wheel onto it produces force.
    # (One Foundation per process, so reuse this env rather than building again.)
    from isaacgym import gymtorch
    env._reset_dofs(ids)
    env._reset_root_states(ids)
    env.root_states[0, 1] = float(env.obstacle_py_t[0]) - 0.114
    env.root_states[0, 2] = 0.330
    env.root_states[0, 7:13] = 0.0
    flat = torch.cat([env.root_states, env.obstacle_root_states],
                     dim=1).contiguous().view(-1)
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(flat))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    peak = 0.0
    for _ in range(60):
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.refresh_net_contact_force_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        peak = max(peak, float(torch.norm(env.obstacle_contact_forces[0]).item()))
    check("static obstacle participates in physics", peak > 100.0,
          "peak=%.0f N" % peak)

    env.reset()
    env.reset_idx(ids)


# ---------------------------------------------------------------------------
# F. Geometry discipline
# ---------------------------------------------------------------------------

def test_geometry(env):
    print("\nF. real geometric envelope for triggering")
    env.reset()
    obs_front, obs_rear, obs_top = geo.obstacle_y_interval(env)
    envl = geo.robot_forward_interval(env, z_ceiling=obs_top)

    historic = env.root_states[:, 1] + env.wheel_radius
    real = envl['front_y']
    under = (real - historic).min()
    check("real front is ahead of base_y + wheel_radius", float(under) > 0.05,
          "min under-estimate=%.4f m" % float(under))

    # Nothing that passes above the box top may count as front geometry.
    min_y, max_y = geo.link_y_extents(env)
    unl = geo.robot_forward_interval(env)
    check("z ceiling suppresses geometry that flies over the box",
          float(unl['front_y'].min()) > float(real.min()),
          "no-ceiling=%.4f with-ceiling=%.4f" % (
              float(unl['front_y'].min()), float(real.min())))

    # The analytic wheel interval must lie inside the robot envelope.
    wheel_iv = envl['exact_wheels']
    wmin = torch.minimum(wheel_iv[:, 0], wheel_iv[:, 2])
    wmax = torch.maximum(wheel_iv[:, 1], wheel_iv[:, 3])
    check("wheel interval is contained in the robot envelope",
          bool((wmax <= real + 1e-6).all() and (wmin >= envl['rear_y'] - 1e-6).all()))

    # At nominal stance the robot must NOT already be inside the trigger window
    # from reset -- that was the symptom of the un-ceilinged envelope.
    triggered_at_reset = float((obs_front - real).min()) <= env.spatial_trigger_dist
    check("nominal stance is not already in the trigger window",
          not triggered_at_reset,
          "dist=%.4f window=%.3f" % (float((obs_front - real).min()),
                                     env.spatial_trigger_dist))

    # REGRESSION: clearance above the box top must be measured only over the
    # box footprint. `wheel_bottom_z - box_top_z` compared two quantities taken
    # at different places and reported -0.0477 m ("inside the obstacle") for a
    # robot merely standing next to it, on a frame where the analytic gap was
    # +0.0979 m.
    gaps = geo.wheel_box_signed_gap(env)
    clr = geo.wheel_clearance_above_box_top(env)
    check("clearance is footprint-limited, not wheel-bottom minus box-top",
          float(clr.min()) > 0.0,
          "gap=%.4f clr=%+.4f (old formula would say %+.4f)"
          % (float(gaps.min()), float(clr.min()),
             float(geo.wheel_bottom_zs(env).min() - obs_top.min())))

    # And the two independently derived quantities must never contradict:
    # a negative analytic gap implies the footprint clearance is below the top.
    env._reset_dofs(torch.tensor([0], device=env.device))
    env._reset_root_states(torch.tensor([0], device=env.device))
    env.root_states[0, 1] = 0.130          # right at the box front face
    env.root_states[0, 2] = 0.400
    env.root_states[0, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.all_root_states.view(-1, 13).contiguous()))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    gap_at = float(geo.wheel_box_signed_gap(env)[0].min())
    clr_at = float(geo.wheel_clearance_above_box_top(env)[0].min())
    check("analytic gap and footprint clearance agree on intersection",
          (gap_at < 0.0) == (clr_at < 0.0),
          "gap=%+.5f clr=%+.5f" % (gap_at, clr_at))
    env._reset_root_states(torch.arange(env.num_envs, device=env.device))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)


def main():
    print("=" * 78)
    print("obstacle-jump task semantics: counter-example tests")
    print("=" * 78)
    # One Foundation per process: build a single multi-env scene and reuse it.
    env = build(num_envs=3)
    env.reset()
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    test_collision_penalty_sign(env)
    test_success_is_edge_event(env)
    test_metric_timing(env)
    test_termination(env)
    test_geometry(env)
    test_per_env_obstacle_state(env)

    print("\n" + "=" * 78)
    print("PASSED %d / %d" % (len(PASS), len(PASS) + len(FAIL)))
    if FAIL:
        for name in FAIL:
            print("  FAILED: %s" % name)
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
