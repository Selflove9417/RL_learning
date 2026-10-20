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

  E. TIER-2 mission-success gate -- a passage is not a mission. Crossing the box
     and then colliding, or crossing it and never recovering a sustained cruise,
     must NOT pay the full success reward.

  F. Contact point frame -- the obstacle may arrive in either slot of the contact
     pair, and the reported point must be the same world point either way.

  G. Per-environment obstacle state -- one env repositioning its box must not
     move any other env's box, and reset must restore the configured offset.

  H. Geometry counter-examples -- the envelope must stay conservative under
     arbitrary attitude, near the box EDGES, and must not be fooled into
     claiming the non-wheel links are exact.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import isaacgym  # noqa: F401  (must precede torch)
from isaacgym import gymtorch
import numpy as np
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
# E. TIER-2 mission-success gate
# ---------------------------------------------------------------------------

def _arm_success(env):
    """Put every env into the state where TIER-2 success is otherwise earned."""
    env.success_armed[:] = True
    env.obstacle_cleared[:] = True
    env.has_obstacle_collision[:] = False
    env.task_failed[:] = False
    env.takeoff_step[:] = 40
    env.touchdown_step[:] = 60
    env.final_success_paid[:] = False
    env.has_taken_off[:] = True
    env.has_touched_down[:] = True


def _recovered(env, ok=True):
    """Force `_recovery_succeeded()` to a known value for all envs."""
    if ok:
        env.recovery_outcome = ["NORMAL_HANDOFF"] * env.num_envs
        env.recovery_best_hold_steps[:] = env.RECOVERY_HOLD_STEPS + 5
        env.recovery_regression_steps[:] = 0
        env.recovery_grounded_steps[:] = 20
    else:
        env.recovery_outcome = ["TIMEOUT_HANDOFF"] * env.num_envs
        env.recovery_best_hold_steps[:] = 0
        env.recovery_regression_steps[:] = 0
        env.recovery_grounded_steps[:] = 0


def test_mission_success_gate(env):
    print("\nE. TIER-2 mission success requires the whole story")
    ids = torch.arange(env.num_envs, device=env.device)

    # Baseline: everything satisfied -> success must fire.
    env.reset_idx(ids)
    _arm_success(env)
    _recovered(env, True)
    check("fully satisfied mission is a success",
          bool(env._mission_success_mask()[0].item()))

    # COUNTER-EXAMPLE 1 (spec item 1.5): PASS FIRST, COLLIDE LATER.
    # Seed 15 did exactly this -- cleared at step 118, hit the box at 120 -- and
    # the Phase A event still paid the full success reward, because it only asked
    # whether `obstacle_cleared` had ever latched.
    env.reset_idx(ids)
    _arm_success(env)
    _recovered(env, True)
    env.has_obstacle_collision[:] = True
    env.success_armed[:] = False          # what _update_task_metrics latches
    check("pass-then-collide is NOT a success",
          not bool(env._mission_success_mask()[0].item()),
          "collision revokes the arm even with clearance latched")

    # The collision must also survive a re-arm attempt: `request_teleop_jump` /
    # cycle restarts must not resurrect a poisoned episode.
    env.success_armed[:] = True
    check("re-arming cannot resurrect a collided episode",
          not bool(env._mission_success_mask()[0].item()),
          "the ~has_obstacle_collision clause is checked again here")

    # COUNTER-EXAMPLE 2 (spec item 1.5): PASSED, BUT NEVER RECOVERED CRUISE.
    # The robot crossed the box cleanly and landed, but the post-landing speed
    # window never reached the sustained 70% cruise, so the mission is unfinished.
    env.reset_idx(ids)
    _arm_success(env)
    _recovered(env, False)
    check("pass without sustained recovery is NOT a success",
          not bool(env._mission_success_mask()[0].item()),
          "outcome=%s" % env.recovery_outcome[0])

    # TIMEOUT_HANDOFF / RECOVER_ABORT must never be laundered into success even
    # if the hold counter happens to look good -- this is spec item 2's explicit
    # "must not impersonate success" clause.
    for bad in ("TIMEOUT_HANDOFF", "RECOVER_ABORT"):
        env.reset_idx(ids)
        _arm_success(env)
        _recovered(env, True)
        env.recovery_outcome = [bad] * env.num_envs
        check("%s is not success" % bad,
              not bool(env._recovery_succeeded()[0].item()))

    # A pass whose landing never latched (episode ended in the air) is not a
    # success either: `touchdown_step < 0` means "we never saw it come down".
    env.reset_idx(ids)
    _arm_success(env)
    _recovered(env, True)
    env.touchdown_step[:] = -1
    check("pass without a recorded touchdown is NOT a success",
          not bool(env._mission_success_mask()[0].item()))

    # And a geometric plough-through disqualifies it even if all else lines up.
    env.reset_idx(ids)
    _arm_success(env)
    _recovered(env, True)
    env.task_failed[:] = True
    check("plough-through is NOT a success",
          not bool(env._mission_success_mask()[0].item()))

    # At most one payment per episode.
    env.reset_idx(ids)
    _arm_success(env)
    _recovered(env, True)
    first = bool(env._mission_success_mask()[0].item())
    env.final_success_paid[:] = True
    second = bool(env._mission_success_mask()[0].item())
    check("mission success pays at most once", first and not second)

    # TIER 1 (geometric passage) must still be reachable WITHOUT tier 2: crossing
    # the box is real progress and deserves its own small reward, but it must not
    # be the same event as the full success.
    env.reset_idx(ids)
    env.obstacle_cleared[:] = False
    env.geometric_pass_event[:] = False
    env.success_armed[:] = False
    env.geometric_pass_event[:] = True
    tier1 = float(env._reward_obstacle_geometric_pass()[0])
    tier2 = float(env._reward_obstacle_cleared_success()[0])
    check("tier-1 passage pays on its own event, not the success event",
          tier1 > 0.0 and tier2 == 0.0,
          "tier1=%.3f tier2=%.3f" % (tier1, tier2))
    check("tier-1 reward is smaller than the full success reward",
          tier1 < float(env.cfg.rewards.scales.obstacle_cleared_success) * 0.0 + 5.0,
          "tier1=%.3f vs full=5.0 nominal" % tier1)

    env.reset_idx(ids)


# ---------------------------------------------------------------------------
# F. Contact point frame (obstacle may be body0 OR body1)
# ---------------------------------------------------------------------------

class _FakeContact(dict):
    """Minimal stand-in for one `RigidContact` record.

    Isaac Gym exposes no constructible `gymapi.RigidContact`, so the reversed-pair
    case cannot be produced by the physics engine on demand. A dict with the same
    field access pattern is enough: the production code only ever does
    `contact['body0']`, `contact['localPos0']['x']` etc.
    """

    def __init__(self, body0, body1, local0, local1):
        super(_FakeContact, self).__init__()
        self['body0'] = body0
        self['body1'] = body1
        for slot, vec in ((0, local0), (1, local1)):
            self['localPos%d' % slot] = {'x': float(vec[0]),
                                         'y': float(vec[1]),
                                         'z': float(vec[2])}


def test_contact_point_frame(env):
    print("\nF. contact point is taken from the obstacle's own slot")
    from legged_gym.envs.myrobot.myrobot_obstacle_jump_env import (
        contact_point_world)
    obs_idx = env.num_bodies
    robot_body = 3
    # Box centred at (0, 0.30, 0.025) with identity rotation, half extents
    # (0.5, 0.02, 0.025). A point on its front face, in BOX-local coordinates.
    obs_root = torch.tensor([0.0, 0.30, 0.025, 0.0, 0.0, 0.0, 1.0])
    box_local = [0.0, -0.02, 0.0]           # front face, centre
    # A deliberately DIFFERENT point in the robot body's frame, so a wrong slot
    # choice produces a visibly different answer instead of a coincidence.
    robot_local = [0.35, 0.0, 0.0]

    # Order A: PhysX puts the obstacle in body0.
    ca = _FakeContact(obs_idx, robot_body, box_local, robot_local)
    pa, frame_a = contact_point_world(ca, obs_root, True, obs_idx,
                                      device=torch.device('cpu'))
    # Order B: same physical point, but the pair is reported the other way round.
    cb = _FakeContact(robot_body, obs_idx, robot_local, box_local)
    pb, frame_b = contact_point_world(cb, obs_root, False, obs_idx,
                                      device=torch.device('cpu'))

    check("reversed pair order yields the same world point",
          pa is not None and pb is not None
          and float(abs(pa - pb).max()) < 1e-6,
          "body0-obstacle=%s body1-obstacle=%s" % (
              np.round(pa, 5).tolist() if pa is not None else None,
              np.round(pb, 5).tolist() if pb is not None else None))
    check("point lands on the obstacle front face, not the robot",
          pa is not None and abs(float(pa[1]) - 0.28) < 1e-6,
          "y=%.5f (front face 0.280, robot-frame would be 0.30)" % (
              float(pa[1]) if pa is not None else float('nan')))
    check("the slot actually used is reported",
          frame_a == 'obstacle_local0' and frame_b == 'obstacle_local1',
          "%s / %s" % (frame_a, frame_b))

    # A caller that builds the flag wrongly must be REJECTED, not silently given
    # the robot-frame point.
    mism, why = contact_point_world(ca, obs_root, False, obs_idx,
                                    device=torch.device('cpu'))
    check("a wrong obstacle_is_body0 flag is rejected",
          mism is None and why == 'flag_mismatch', "why=%s" % why)
    # And a pair that does not contain the obstacle at all is rejected too.
    nn, why2 = contact_point_world(_FakeContact(1, 2, robot_local, robot_local),
                                   obs_root, False, obs_idx,
                                   device=torch.device('cpu'))
    check("a pair without the obstacle is rejected",
          nn is None and why2 == 'obstacle_not_in_pair', "why=%s" % why2)

    # Rotated obstacle: the local offset must be rotated with it, i.e. a local
    # +Y offset must come out along world -X after a +90 deg yaw. Catches code
    # that adds localPos to the box centre without rotating.
    theta = math.pi / 2.0
    # Pose layout is [x, y, z, qx, qy, qz, qw]: the quaternion is (x, y, z, w).
    # A yaw about world Z therefore lives in qz/qw. An earlier version put the
    # cosine in qx and the sine in qw, which is a 90 deg rotation about X.
    quat_yaw = torch.tensor([0.0, 0.30, 0.025,
                             0.0, 0.0, math.sin(theta / 2.0),
                             math.cos(theta / 2.0)])
    prot, _ = contact_point_world(_FakeContact(obs_idx, robot_body,
                                               [0.0, 0.02, 0.0], robot_local),
                                  quat_yaw, True, obs_idx,
                                  device=torch.device('cpu'))
    # local (0, +0.02, 0) under +90 deg yaw about Z -> world (-0.02, 0, 0)
    check("obstacle rotation is applied to the local point",
          prot is not None and abs(float(prot[0]) + 0.02) < 1e-5,
          "world=%s expected x=-0.02" % (
              np.round(prot, 5).tolist() if prot is not None else None))


# ---------------------------------------------------------------------------
# G. Geometry counter-examples (attitude / box edges / non-wheel links)
# ---------------------------------------------------------------------------

def test_geometry_counter_examples(env):
    """Adversarial cases the envelope must survive, not just nominal stance.

    Spec item 3 asks for three specific families of counter-example:
      (a) ATTITUDE change -- pitching/rolling the robot must not make the
          envelope under-report, because the trigger geometry is evaluated from
          body poses and a mesh-extent bug only shows up when the rotation sends
          a non-extreme vertex to the extreme position.
      (b) OBSTACLE EDGE proximity -- the wheel/box gap must behave at the box
          CORNERS and EDGES, where the nearest point of the box is a vertex
          rather than a face, and must never depend on the box being approached
          face-on.
      (c) NON-WHEEL link contact -- for the mesh links there is no closed form,
          so the code must never report them as an exact penetration depth. The
          test asserts the honest labelling rather than a number.
    """
    print("\nH. geometry counter-examples: attitude, box edges, mesh links")
    ids = torch.arange(env.num_envs, device=env.device)
    env.reset()
    env.reset_idx(ids)

    urdf_path = env.cfg.asset.file.format(
        LEGGED_GYM_ROOT_DIR=__import__('legged_gym').LEGGED_GYM_ROOT_DIR)
    hull = geo.urdf_link_vertex_cloud(urdf_path)
    full = geo.urdf_link_vertex_cloud(urdf_path, reduce_to_hull=False)

    # ---------------------------------------------------------------- (a)
    # ATTITUDE: the hull reduction must reproduce the FULL cloud's world-Y
    # extent for every rotation, not just a few axis-aligned ones. This is
    # exactly the property the old 96-direction reduction violated (up to
    # 8.896 mm under-report on base_link).
    rng = np.random.RandomState(20240)
    worst = {}
    n_frames = 400
    for name, hull_pts in hull.items():
        pts_full = full[name]
        h = torch.from_numpy(hull_pts).double()
        f = torch.from_numpy(pts_full).double()
        under = -1e9
        for _ in range(n_frames):
            q = rng.randn(4)
            q /= np.linalg.norm(q)
            R = geo.quat_to_matrix(torch.from_numpy(q).double().view(1, 4))[0]
            # world Y = <R p, e_y> = row 1 of R dotted with p
            row = R[1]
            exact_max = float((f @ row).max())
            hull_max = float((h @ row).max())
            exact_min = float((f @ row).min())
            hull_min = float((h @ row).min())
            under = max(under, exact_max - hull_max, exact_min - hull_min)
        worst[name] = under
    worst_under = max(worst.values())
    check("hull vertices reproduce the full cloud's Y extent under rotation",
          worst_under < 1e-6,
          "worst under-report=%.3e m over %d random attitudes x %d links"
          % (worst_under, n_frames, len(hull)))
    # And the recorded cost of the OLD reduction must still be reproducible, so a
    # future "optimisation" cannot silently reintroduce it.
    base_pts = full['base_link'].astype(np.float64)
    old = geo._fibonacci_directions(96)
    old_pts = []
    for d in old:
        old_pts.append(base_pts[int(np.argmax(base_pts @ d))])
    old_pts = np.unique(np.array(old_pts), axis=0)
    rng2 = np.random.RandomState(7)
    old_under = 0.0
    for _ in range(n_frames):
        q = rng2.randn(4)
        q /= np.linalg.norm(q)
        R = geo.quat_to_matrix(torch.from_numpy(q).double().view(1, 4))[0]
        row = R[1].numpy()
        exact_max = float((base_pts @ row).max())
        reduced_max = float((old_pts @ row).max())
        old_under = max(old_under, exact_max - reduced_max)
    check("the superseded 96-direction reduction under-reports (regression anchor)",
          old_under > 1e-3,
          "measured=%.4f m vs recorded %.6f m"
          % (old_under, geo.REDUCTION_MEASURED_UNDERESTIMATE_M['base_link']))

    # A pitched robot: the trigger envelope is computed from `rb_states`, so an
    # attitude change about the LATERAL axis must move the world-Y extent the way
    # the hull says it does.  (Rotating about the forward axis would leave the Y
    # extent untouched and prove nothing -- an earlier version of this test did
    # exactly that and passed vacuously.)
    env._reset_dofs(ids)
    env._reset_root_states(ids)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    base_front_upright = float(geo.robot_forward_interval(env)['front_y'].max())

    # Nose-down pitch by 10 deg about the lateral (world X) axis.  Quaternion
    # layout in this codebase is (x, y, z, w).
    theta = math.radians(10.0)
    q = torch.tensor([math.sin(theta / 2.0), 0.0, 0.0, math.cos(theta / 2.0)],
                     device=env.device)
    env.root_states[:, 3:7] = q
    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.all_root_states.view(-1, 13).contiguous()))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    pitched = geo.robot_forward_interval(env)
    check("envelope is finite under a pitched attitude",
          bool(torch.isfinite(pitched['front_y']).all()),
          "front=%.4f upright=%.4f" % (float(pitched['front_y'].max()),
                                       base_front_upright))
    check("pitching the base changes the measured front envelope",
          abs(float(pitched['front_y'].max()) - base_front_upright) > 1e-4,
          "pitched=%.4f upright=%.4f"
          % (float(pitched['front_y'].max()), base_front_upright))
    # The z-ceiling clip must stay a SUBSET of the unclipped envelope: clipping
    # can only remove points, never invent a longer robot.
    obs_front, obs_rear, obs_top = geo.obstacle_y_interval(env)
    clipped = geo.robot_forward_interval(env, z_ceiling=obs_top)['front_y']
    unclipped = geo.robot_forward_interval(env)['front_y']
    check("z-ceiling envelope never exceeds the unclipped envelope",
          bool((clipped <= unclipped + 1e-6).all()),
          "clipped=%.4f unclipped=%.4f"
          % (float(clipped.max()), float(unclipped.max())))
    env._reset_dofs(ids)
    env._reset_root_states(ids)

    # ---------------------------------------------------------------- (b)
    # BOX EDGES AND CORNERS. Place the wheel on the box's vertical SIDE EDGES
    # (both x ends) and on its TOP face, at 5-6 mm clear and 4-5 mm penetrating,
    # and check the analytic gap against two independent references.
    lo, hi = geo._box_bounds(env)
    # The obstacle is a 1.0 m long, 4 cm thick, 5 cm tall WALL, so its two x
    # ENDS are real corners a wheel can miss. The wheel capsule's axis is
    # LATERAL, so a wheel can only meet such a corner with its axis END --
    # exactly the case where an implementation that sampled only the capsule
    # CENTRE would be off by the half length (44 mm).
    lo_np = [float(lo[0, i]) for i in range(3)]
    hi_np = [float(hi[0, i]) for i in range(3)]
    box_front = lo_np[1]
    box_top = hi_np[2]
    half_len = 0.5 * geo.WHEEL_LENGTH
    wheel_r = geo.WHEEL_RADIUS
    left_body = env.left_wheel_body
    # URDF cylinder-origin offset along the link's local X. It is a documented
    # asset constant (obstacle_geometry.WHEEL_AXIAL_OFFSET), not a test invention,
    # and _place_capsule verifies the placement against production's own
    # wheel_axis_terms below.
    axial = geo.WHEEL_AXIAL_OFFSET[env.link_names[left_body]]
    # Put the axis at box mid height so the nearest box feature is the vertical
    # CORNER, not the top face -- otherwise the corner is masked by the face.
    mid_z = 0.5 * box_top
    pose = {}

    def _place_capsule(capsule_xy, capsule_z):
        """Put the left wheel capsule's AXIS MIDPOINT at a world position.

        Every body offset is MEASURED from the freshly reset state instead of
        hard-coded. An earlier version of this test hard-coded only the lateral
        offset and passed the capsule-centre target straight into the BASE height
        slot, which left the wheel 0.33 m too high; four assertions then failed
        for reasons that had nothing to do with the code under test.
        """
        env._reset_dofs(ids)
        env._reset_root_states(ids)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        body_off = env.rb_states[:, left_body, 0:3] - env.rb_states[:, 0, 0:3]
        # At the identity attitude used here the link's local X is world +X, so
        # the axial offset shifts the capsule centre in x only.
        env.root_states[:, 0] = capsule_xy[0] - (body_off[:, 0] + axial)
        env.root_states[:, 1] = capsule_xy[1] - body_off[:, 1]
        env.root_states[:, 2] = capsule_z - body_off[:, 2]
        env.root_states[:, 3:7] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            device=env.device, dtype=env.root_states.dtype)
        env.root_states[:, 7:13] = 0.0
        env.gym.set_actor_root_state_tensor(
            env.sim,
            gymtorch.unwrap_tensor(env.all_root_states.view(-1, 13).contiguous()))
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        # Read the capsule centre back out the way PRODUCTION does it, so the
        # reference functions are guaranteed to describe the pose being tested.
        ax, ay, az = geo.wheel_axis_terms(env.rb_states, left_body)
        centre = env.rb_states[:, left_body, 0:3] + axial * torch.stack(
            [ax, ay, az], dim=1)
        pose['centre'] = [float(v) for v in centre[0].tolist()]
        gaps = geo.wheel_box_signed_gap(env)
        # The wheels sit in a BICYCLE layout: identical y and z, different x.
        # `wheel_box_signed_gap` returns (N, 2) and collapses to the min for
        # collision purposes, so the TRAILING wheel can dominate the number even
        # when this helper has aimed the LEADING one. Every assertion below is
        # about the leading (left) capsule, which is the one this helper places,
        # so read column 0 -- and keep the pair minimum for the sanity check.
        pose['both_min'] = float(gaps[0].min())
        pose['left_gap'] = float(gaps[0][0])
        return pose['left_gap']

    def _ref_axis_sweep(capsule_xy, capsule_z, n=4001):
        """Same CONSTRUCTION as production, 60x finer sampling.

        Shares the method with `wheel_box_signed_gap` on purpose: it checks the
        implementation (axis direction, radius, axial offset, sampling) while
        being immune to the 65-sample resolution. It says nothing about whether
        the method is right -- `_ref_surface_cloud` covers that.
        """
        xs = capsule_xy[0] + np.linspace(-half_len, half_len, n)
        dx = np.maximum(np.maximum(lo_np[0] - xs, 0.0), xs - hi_np[0])
        dy = max(lo_np[1] - capsule_xy[1], 0.0, capsule_xy[1] - hi_np[1])
        dz = max(lo_np[2] - capsule_z, 0.0, capsule_z - hi_np[2])
        return float(np.sqrt(dx * dx + dy * dy + dz * dz).min()) - wheel_r

    def _ref_surface_cloud(capsule_xy, capsule_z, n_x=25, n_az=96, n_el=49):
        """METHOD-INDEPENDENT reference: dense points ON the capsule surface,
        each measured to the box with a signed point-to-AABB distance.

        Cloud spacing is 4-5 mm, so the deepest penetration it can find may be
        up to ~2.5 mm shallower than the truth. It is therefore used for SIGN
        and for magnitudes on SEPARATED configurations (accurate well under a
        millimetre there), and never reported as a penetration depth.
        """
        SX, AZ, EL = np.meshgrid(np.linspace(-half_len, half_len, n_x),
                                 np.linspace(0.0, 2.0 * np.pi, n_az, endpoint=False),
                                 np.linspace(-np.pi / 2.0, np.pi / 2.0, n_el),
                                 indexing='ij')
        px = capsule_xy[0] + SX + wheel_r * np.cos(EL) * np.sin(AZ)
        py = capsule_xy[1] + wheel_r * np.cos(EL) * np.cos(AZ)
        pz = capsule_z + wheel_r * np.sin(EL)
        dx = np.maximum(np.maximum(lo_np[0] - px, 0.0), px - hi_np[0])
        dy = np.maximum(np.maximum(lo_np[1] - py, 0.0), py - hi_np[1])
        dz = np.maximum(np.maximum(lo_np[2] - pz, 0.0), pz - hi_np[2])
        outside = np.sqrt(dx * dx + dy * dy + dz * dz)
        inside_depth = np.minimum.reduce(
            [px - lo_np[0], hi_np[0] - px, py - lo_np[1],
             hi_np[1] - py, pz - lo_np[2], hi_np[2] - pz])
        signed = np.where(outside > 0.0, outside, -inside_depth)
        return float(signed.min())

    # Placement self-check: the helper must actually deliver the requested pose,
    # otherwise every number below is measuring the wrong geometry.
    want = (0.0, box_front - wheel_r - 0.006, mid_z)
    _place_capsule(want[0:2], want[2])
    check("placement helper delivers the requested capsule centre",
          max(abs(pose['centre'][i] - want[i]) for i in range(3)) < 1e-4,
          "got=%s want=%s" % (np.round(pose['centre'], 5).tolist(),
                              [round(v, 5) for v in want]))

    # The two wheel bodies are laid out BICYCLE-style: same y, same z, different
    # x (link_004 is the leading one at x=+0.353, link_007 the trailing one at
    # x=-0.011 in the nominal stance). That is why `wheel_box_signed_gap`
    # returning min over both wheels is NOT a per-wheel measure -- the trailing
    # wheel can dominate while this helper is steering the leading one. Record
    # the fact here so the assertions below query the right column on purpose.
    env._reset_dofs(ids)
    env._reset_root_states(ids)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    off_l = (env.rb_states[0, env.left_wheel_body, 0:3]
             - env.rb_states[0, 0, 0:3]).tolist()
    off_r = (env.rb_states[0, env.right_wheel_body, 0:3]
             - env.rb_states[0, 0, 0:3]).tolist()
    check("wheels are laid out in a bicycle arrangement (same y and z)",
          abs(off_l[1] - off_r[1]) < 1e-5 and abs(off_l[2] - off_r[2]) < 1e-5
          and abs(off_l[0] - off_r[0]) > 0.1,
          "lead=%s trail=%s" % (np.round(off_l, 4).tolist(),
                                np.round(off_r, 4).tolist()))

    # --- corners: axis END `beyond` past the box end, so the vertical corner
    # edge is the nearest box feature (a centre-only implementation would answer
    # 0.0275 m here instead of 0.006 m).
    beyond = 0.020
    for sign, label in ((-1.0, 'left'), (1.0, 'right')):
        end_x = lo_np[0] if sign < 0 else hi_np[0]
        centre_x = end_x + sign * (half_len + beyond)
        dy_clear = math.sqrt((wheel_r + 0.006) ** 2 - beyond ** 2)
        g_clear = _place_capsule((centre_x, box_front - dy_clear), mid_z)
        ref_a = _ref_axis_sweep((centre_x, box_front - dy_clear), mid_z)
        ref_s = _ref_surface_cloud((centre_x, box_front - dy_clear), mid_z)
        check("axis end clears the box's %s corner by 6 mm" % label,
              abs(g_clear - 0.006) < 0.9e-3, "gap=%+.5f m" % g_clear)
        check("%s-corner clearance matches a 4001-point axis sweep" % label,
              abs(g_clear - ref_a) < 0.9e-3,
              "analytic=%+.5f sweep=%+.5f" % (g_clear, ref_a))
        check("%s-corner clearance matches a surface-cloud reference" % label,
              abs(g_clear - ref_s) < 1.2e-3,
              "analytic=%+.5f cloud=%+.5f" % (g_clear, ref_s))

        dy_hit = math.sqrt((wheel_r - 0.004) ** 2 - beyond ** 2)
        g_hit = _place_capsule((centre_x, box_front - dy_hit), mid_z)
        ref_a_hit = _ref_axis_sweep((centre_x, box_front - dy_hit), mid_z)
        check("axis end overlaps the box's %s corner by 4 mm" % label,
              abs(g_hit + 0.004) < 0.9e-3, "gap=%+.5f m" % g_hit)
        check("%s-corner overlap matches a 4001-point axis sweep" % label,
              abs(g_hit - ref_a_hit) < 0.9e-3,
              "analytic=%+.5f sweep=%+.5f" % (g_hit, ref_a_hit))
        # The surface-cloud reference under-reports CORNER penetration (the
        # deepest surface point sits near two faces at once, so its inside depth
        # is smaller than the translation depth), so it is only asked for the
        # sign here. The top-face case below is where it is quantitative.
        ref_s_hit = _ref_surface_cloud((centre_x, box_front - dy_hit), mid_z)
        check("%s-corner overlap is confirmed negative by the surface cloud"
              % label, ref_s_hit < 0.0, "cloud=%+.5f" % ref_s_hit)

    # --- box TOP face, approached from above: the degenerate edge-on case.
    top_cy = 0.5 * (lo_np[1] + hi_np[1])
    for label, dz_off, expect in (("above the box top by 5 mm", 0.005, 1.0),
                                  ("dipping 5 mm into the box top", -0.005, -1.0)):
        z = box_top + wheel_r + dz_off
        g = _place_capsule((0.0, top_cy), z)
        ref_a = _ref_axis_sweep((0.0, top_cy), z)
        ref_s = _ref_surface_cloud((0.0, top_cy), z)
        check("wheel %s is reported %s" % (
                  label, "separated" if expect > 0 else "overlapping"),
              abs(g - expect * abs(dz_off)) < 0.9e-3, "gap=%+.5f m" % g)
        check("top-face %s matches a 4001-point axis sweep" % label,
              abs(g - ref_a) < 0.9e-3, "analytic=%+.5f sweep=%+.5f" % (g, ref_a))
        check("top-face %s matches the surface-cloud reference" % label,
              abs(g - ref_s) < 1.2e-3, "analytic=%+.5f cloud=%+.5f" % (g, ref_s))

    # Sampling-resolution honesty: the reported gap magnitude must not be
    # presented as better than h/2 = 0.691 mm. Assert the bound the docstring
    # claims is actually the one used.
    h = geo.WHEEL_LENGTH / 64.0
    check("declared sampling error bound matches the documented formula",
          abs(h / 2.0 - 0.6914e-3) < 5e-5,
          "h/2=%.4f mm" % (h / 2.0 * 1e3))

    # ---------------------------------------------------------------- (c)
    # NON-WHEEL LINKS: no closed form exists, so the module must never present
    # them as exact. Assert the labelling, and assert the envelope is an OUTER
    # bound of the raw mesh (i.e. never smaller than the true vertex extent).
    src = open(geo.__file__).read()
    check("mesh-link envelope is not advertised as exact",
          'EXACT SUPPORT FUNCTION OVER A CONSERVATIVE OUTER BOUND' in src,
          "header keeps the two families separate")
    check("mesh links are documented as an outer bound, not a penetration depth",
          'never used as a penetration depth' in src)

    env._reset_dofs(ids)
    env._reset_root_states(ids)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    # For every body, the envelope from `link_y_extents` must CONTAIN the Y range
    # of the raw (non-hull) cloud transformed by the same body pose. This is the
    # outer-bound property, checked against data the production path never sees.
    rb = env.rb_states
    R = geo.quat_to_matrix(rb[:, :, 3:7].reshape(-1, 4)).view(
        rb.shape[0], rb.shape[1], 3, 3)
    min_y, max_y = geo.link_y_extents(env)
    violations = 0
    worst_viol = 0.0
    for idx, name in enumerate(env.link_names):
        if name not in full:
            continue
        pts = torch.from_numpy(full[name]).to(rb.device).double()
        # pts is (K,3) and R[idx] is (3,3), so rotate with a MATRIX-vector
        # product: world = R p + t. (An earlier version passed R[idx] where the
        # einsum subscripts wanted a vector and crashed the whole suite.)
        world = torch.einsum('ij,kj->ki', R[0, idx].double(), pts) + \
            rb[0, idx, 0:3].double()
        raw_lo = float(world[:, 1].min())
        raw_hi = float(world[:, 1].max())
        if raw_lo < float(min_y[0, idx]) - 1e-6:
            violations += 1
            worst_viol = max(worst_viol, float(min_y[0, idx]) - raw_lo)
        if raw_hi > float(max_y[0, idx]) + 1e-6:
            violations += 1
            worst_viol = max(worst_viol, raw_hi - float(max_y[0, idx]))
    check("per-link envelope is an outer bound of the raw mesh cloud",
          violations == 0,
          "violations=%d worst=%.3e m across %d links"
          % (violations, worst_viol, len(full)))

    env.reset()
    env.reset_idx(ids)


# ---------------------------------------------------------------------------
# I. Per-environment obstacle state
# ---------------------------------------------------------------------------
def test_per_env_obstacle_state(env):
    print("\nG. per-environment obstacle coordinates")
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
    test_mission_success_gate(env)
    test_contact_point_frame(env)
    test_geometry(env)
    test_geometry_counter_examples(env)
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
