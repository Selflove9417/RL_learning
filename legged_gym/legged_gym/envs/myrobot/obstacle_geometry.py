# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# URDF-derived collision envelopes for the MyRobot obstacle-clearance task.
#
# IMPORTANT EPISTEMIC CONTRACT
# ---------------------------
# Two families of quantities are kept strictly separate and must never be
# conflated in logs, rewards or reports:
#
#   1. EXACT (closed form, no approximation):
#        * wheel collision volume -- URDF declares an analytic cylinder
#          (radius 0.070 m, length 0.0885 m, axis along the link-local X axis
#          with a link-local X offset of +0.03825001 m for `link_004` and
#          +0.01925 m for `link_007`).  PhysX, however, substitutes a capsule
#          when `asset.replace_cylinder_with_capsule` is True (the legged_gym
#          default), so the *collision* volume is a capsule while the URDF
#          document says cylinder.  Both closed forms are provided and the
#          caller selects the one matching the asset option actually used.
#        * obstacle box -- analytic box, axis-aligned (spawned with identity
#          quaternion), so its Y interval is exact.
#
#   2. EXACT SUPPORT FUNCTION OVER A CONSERVATIVE OUTER BOUND:
#        * every non-wheel link is described in the URDF by triangle meshes.
#          Isaac Gym exposes no per-shape AABB or shape-pair query, so the
#          envelope of these links is taken from the CONVEX HULL VERTICES of
#          the URDF collision vertex cloud (see `urdf_link_vertex_cloud`).
#          That set is exact for support-function queries: the world-Y extent
#          of a rotated cloud is max over points of <R p, e_y>, a linear
#          functional of p, and a linear functional attains its maximum at a
#          vertex of the convex hull.  So no extent is ever under-reported.
#
#          It is an OUTER bound of the true (non-convex) mesh union, and it is
#          never used as a penetration depth -- only as a trigger / containment
#          margin.
#
# A previous version used the 96-direction support-point reduction and called
# it a conservative outer bound.  MEASUREMENT PROVES THAT WAS FALSE: over 4000
# random link rotations the reduced set UNDER-estimated the true world-Y extent
# by up to 8.896 mm on `base_link` (4.505 mm link_005, 4.308 mm link_002,
# 1.730 mm link_003, 1.166 mm link_006; 480/4000 frames under-reported for
# base_link).  The 96 directions sample the sphere, so any direction between
# two samples can be missed.  The hull vertex set costs 443 points total and
# removes the error entirely.  `REDUCTION_MEASURED_UNDERESTIMATE_M` keeps those
# numbers for regression tests.
#
# Nothing in this module may be described as "measured penetration depth"
# unless it comes from a real rigid-body contact pair.

import os
import math
import struct
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Small linear-algebra helpers
# ---------------------------------------------------------------------------

def rpy_to_matrix(rpy):
    """URDF fixed-axis RPY (R = Rz @ Ry @ Rx) -> 3x3 numpy rotation."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def quat_to_matrix(q):
    """(N, 4) quaternion (x, y, z, w) -> (N, 3, 3) rotation matrix.

    Matches isaacgym.torch_utils.quat_rotate for unit quaternions.
    """
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.empty(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)
    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[:, 1, 2] = 2.0 * (yz - wx)
    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


# ---------------------------------------------------------------------------
# URDF parsing -> link-local vertex clouds
# ---------------------------------------------------------------------------

def _stl_vertices(path):
    """Return the (M, 3) vertex array of a binary STL file."""
    with open(path, 'rb') as handle:
        head = handle.read(84)
        if len(head) < 84:
            raise RuntimeError('truncated STL: ' + path)
        n_tri = struct.unpack('<I', head[80:84])[0]
        if os.path.getsize(path) != 84 + 50 * n_tri:
            raise RuntimeError('non-binary STL unsupported: ' + path)
        raw = handle.read(50 * n_tri)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_tri, 50)
    tri = arr[:, 12:48].copy().view('<f4').reshape(n_tri, 3, 3)
    return tri.reshape(-1, 3).astype(np.float64)


def _primitive_vertices(geom):
    """Corner/rim samples for the analytic primitives supported by URDF."""
    tag = geom.tag
    if tag == 'box':
        size = np.array([float(v) for v in geom.get('size').split()])
        lo, hi = -0.5 * size, 0.5 * size
        return _box_corners(lo, hi)
    if tag == 'cylinder':
        # URDF convention: the cylinder axis is the local Z axis.
        r = float(geom.get('radius'))
        half = float(geom.get('length')) * 0.5
        ang = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
        ring = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros_like(ang)], axis=1)
        return np.concatenate([ring + np.array([0.0, 0.0, -half]),
                               ring + np.array([0.0, 0.0, half])], axis=0)
    if tag == 'sphere':
        r = float(geom.get('radius'))
        return _sphere_samples(r)
    if tag == 'capsule':
        r = float(geom.get('radius'))
        half = float(geom.get('length')) * 0.5
        sph = _sphere_samples(r)
        return np.concatenate([sph + np.array([0.0, 0.0, half]),
                               sph + np.array([0.0, 0.0, -half])], axis=0)
    raise RuntimeError('unsupported URDF geometry: ' + tag)


def _box_corners(lo, hi):
    return np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                     [lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]],
                     [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                     [lo[0], hi[1], hi[2]], [hi[0], hi[1], hi[2]]])


def _sphere_samples(r):
    pts = []
    for theta in np.linspace(0.0, math.pi, 7):
        for phi in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False):
            pts.append([r * math.sin(theta) * math.cos(phi),
                        r * math.sin(theta) * math.sin(phi),
                        r * math.cos(theta)])
    return np.array(pts)


def _fibonacci_directions(count):
    """Well-distributed unit directions for support-point reduction."""
    idx = np.arange(count, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / count)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * idx
    dirs = np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)
    # make sure the six world axes are present, they dominate the
    # forward/vertical margins we care about
    axes = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
    return np.concatenate([axes, dirs], axis=0)


@lru_cache(maxsize=4)
def urdf_link_vertex_cloud(urdf_path, directions=None, reduce_to_hull=True):
    """Parse a URDF and return {link_name: (K, 3) points in link frame}.

    With `reduce_to_hull=True` (the production path) each link's combined vertex
    cloud is reduced to the vertices of its CONVEX HULL.  With
    `reduce_to_hull=False` the raw union of collision-mesh vertices is returned
    -- that set is the ground truth the hull is supposed to reproduce, and it is
    exposed so `tests/test_obstacle_jump_semantics.py` can verify the reduction
    under random rotations instead of trusting the argument below.

    Why the hull and not a support-point set
    ----------------------------------------
    The support-function argument only works if the query directions are the
    ones used for the reduction.  `_fibonacci_directions(96)` gives 102 sampled
    directions; a world-Y query after an arbitrary link rotation asks for a
    direction that in general lies BETWEEN two samples, so the reduced set can
    miss the true extremum.  Measured over 4000 random rotations this under-
    estimated the world-Y extent by up to 8.896 mm on `base_link` (see
    `REDUCTION_MEASURED_UNDERESTIMATE_M`).  Those reduced points were therefore
    NOT a conservative bound, contrary to what the previous docstring claimed.

    The hull vertex set has no such gap: for any rotation R, the world-Y extent
    is max_p <R p, e_y>, a linear functional of p, and a linear functional over
    a polytope attains its maximum at a hull vertex.  So the reduced set always
    reproduces the full-cloud extent exactly (verified to <1e-9 m).

    Cost: 443 points total across the 7 links (base_link 105, link_003 89,
    link_006 89, link_004/007 48 each as analytic primitives).  Non-hull input
    points are dropped because they cannot be the extremum of any linear
    functional.

    `directions` is accepted and ignored for backwards compatibility with
    callers that still pass the old 96.
    """
    try:
        from scipy.spatial import ConvexHull
    except ImportError:                       # pragma: no cover - scipy is present
        ConvexHull = None

    mesh_root = os.path.dirname(os.path.abspath(urdf_path))
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    out = {}
    for link in root.findall('link'):
        name = link.get('name')
        clouds = []
        for coll in link.findall('collision'):
            origin = coll.find('origin')
            xyz = np.zeros(3)
            rot = np.eye(3)
            if origin is not None:
                if origin.get('xyz'):
                    xyz = np.array([float(v) for v in origin.get('xyz').split()])
                if origin.get('rpy'):
                    rot = rpy_to_matrix([float(v) for v in origin.get('rpy').split()])
            geom_node = coll.find('geometry')
            if geom_node is None or len(geom_node) == 0:
                continue
            geom = geom_node[0]
            if geom.tag == 'mesh':
                scale = np.ones(3)
                if geom.get('scale'):
                    scale = np.array([float(v) for v in geom.get('scale').split()])
                path = os.path.join(mesh_root, geom.get('filename'))
                verts = _stl_vertices(path) * scale
            else:
                verts = _primitive_vertices(geom)
            clouds.append(verts @ rot.T + xyz)
        if not clouds:
            continue
        full = np.concatenate(clouds, axis=0)
        if reduce_to_hull and ConvexHull is not None and full.shape[0] > 4:
            full = full[np.unique(ConvexHull(full).vertices)]
        out[name] = full.astype(np.float32)
    return out


def urdf_link_collision_aabb(urdf_path):
    """Conservative link-local AABB per link, from the URDF collision shapes."""
    clouds = urdf_link_vertex_cloud(urdf_path)
    return {name: (pts.min(axis=0), pts.max(axis=0)) for name, pts in clouds.items()}


# ---------------------------------------------------------------------------
# Wheel (analytic cylinder / capsule) envelope
# ---------------------------------------------------------------------------
# URDF: <cylinder length="0.0885" radius="0.070"> with origin xyz="0.03825001 0 0"
# for link_004 and xyz="0.01925 0 0" for link_007, rpy="0 1.57079633 0" so the
# cylinder axis is the link-local X axis.
WHEEL_RADIUS = 0.070
WHEEL_LENGTH = 0.0885
WHEEL_AXIAL_OFFSET = {'link_004': 0.03825001, 'link_007': 0.01925}

# Cost of the obsolete reduction, kept as a regression anchor.
#
# The previous implementation reduced each link's vertex cloud to the arg-max
# vertex for each of 102 sampled directions and called the result an outer
# bound.  Over 4000 uniform-random link rotations, comparing that reduced set
# against the FULL URDF vertex cloud gave these maximum UNDER-estimates of the
# world-Y extent (metres):
#
#     base_link  0.008896    link_005  0.004505    link_002  0.004308
#     link_003   0.001730    link_006  0.001166    wheels    0.0
#
# (base_link under-reported in 480 of 4000 frames, link_003 in 781, link_006 in
# 772).  The convex-hull reduction used now reproduces the full-cloud extent to
# better than 1e-9 m for every rotation.  `tests/test_obstacle_jump_semantics.py`
# re-derives both numbers so this cannot silently regress.
REDUCTION_MEASURED_UNDERESTIMATE_M = {
    'base_link': 0.008896,
    'link_002': 0.004308,
    'link_003': 0.001730,
    'link_005': 0.004505,
    'link_006': 0.001166,
}


def wheel_axis_terms(rb_states, body_index):
    """World-frame axis components of a wheel's local X axis.

    Returns (ax, ay, az) tensors of shape (N,) -- the world components of the
    link-local X unit axis.
    """
    quat = rb_states[:, body_index, 3:7]
    R = quat_to_matrix(quat)
    return R[:, 0, 0], R[:, 1, 0], R[:, 2, 0]


def wheel_forward_interval(rb_states, body_index, axial_offset,
                           radius=WHEEL_RADIUS, length=WHEEL_LENGTH,
                           as_capsule=True):
    """EXACT world Y interval [min_y, max_y] of one wheel collision volume.

    Cylinder  : half extent along +Y = (L/2)|ay| + R sqrt(1 - ay^2)
    Capsule   : half extent along +Y = (L/2)|ay| + R
    The axial local-X offset shifts the volume centre by offset * ay.
    """
    ay = wheel_axis_terms(rb_states, body_index)[1]
    shift = axial_offset * ay
    if as_capsule:
        half = 0.5 * length * torch.abs(ay) + radius
    else:
        half = 0.5 * length * torch.abs(ay) + radius * torch.sqrt(
            torch.clamp(1.0 - ay * ay, min=0.0))
    centre_y = rb_states[:, body_index, 1] + shift
    return centre_y - half, centre_y + half


def wheel_bottom_z(rb_states, body_index, axial_offset,
                   radius=WHEEL_RADIUS, length=WHEEL_LENGTH, as_capsule=True):
    """EXACT world Z of the lowest point of one wheel collision volume."""
    az = wheel_axis_terms(rb_states, body_index)[2]
    shift = axial_offset * az
    if as_capsule:
        half = 0.5 * length * torch.abs(az) + radius
    else:
        half = 0.5 * length * torch.abs(az) + radius * torch.sqrt(
            torch.clamp(1.0 - az * az, min=0.0))
    return rb_states[:, body_index, 2] + shift - half


def wheels_used_as_capsules(env):
    """Whether PhysX substitutes capsules for the URDF cylinders."""
    return bool(getattr(env.cfg.asset, 'replace_cylinder_with_capsule', True))


def wheel_forward_intervals(env):
    """(N, 4) tensor: [left_min, left_max, right_min, right_max] exact."""
    rb = env.rb_states
    cap = wheels_used_as_capsules(env)
    lmin, lmax = wheel_forward_interval(
        rb, env.left_wheel_body, WHEEL_AXIAL_OFFSET['link_004'], as_capsule=cap)
    rmin, rmax = wheel_forward_interval(
        rb, env.right_wheel_body, WHEEL_AXIAL_OFFSET['link_007'], as_capsule=cap)
    return torch.stack([lmin, lmax, rmin, rmax], dim=1)


def wheel_bottom_zs(env):
    """(N, 2) tensor: [left_bottom_z, right_bottom_z] exact, clamped at 0."""
    rb = env.rb_states
    cap = wheels_used_as_capsules(env)
    left = wheel_bottom_z(rb, env.left_wheel_body,
                          WHEEL_AXIAL_OFFSET['link_004'], as_capsule=cap)
    right = wheel_bottom_z(rb, env.right_wheel_body,
                           WHEEL_AXIAL_OFFSET['link_007'], as_capsule=cap)
    return torch.stack([torch.clamp(left, min=0.0), torch.clamp(right, min=0.0)], dim=1)


# ---------------------------------------------------------------------------
# Whole-robot envelope (support points -> conservative world bound)
# ---------------------------------------------------------------------------

def register_link_points(env, urdf_path):
    """Cache the per-body hull-vertex cloud of the robot actor on the env.

    Sets:
        env.link_point_cloud : (num_bodies, K, 3) float tensor on env.device
        env.link_point_valid: (num_bodies,) bool tensor -- body has a cloud
        env.link_point_mask : (num_bodies, K) bool tensor -- point i is real
        env.link_names       : list[str]

    ZERO PADDING BUG (fixed here)
    -----------------------------
    Link clouds have different sizes (36..105 points), so they are padded into
    one (B, K, 3) tensor.  The padding was `torch.zeros`, and `link_y_extents`
    only masked by BODY validity, so the padded zeros were live points: they
    transform to a world point at `rb_states[b, 0:3]` (the body origin).  For
    `base_link` that is a real point inside the hull, so the Y extent usually
    survived by luck -- but the padding is a phantom collider that becomes real
    the moment a body origin sits outside its own hull (or once a different
    link's cloud is shorter in a way that lets a body-origin point win the
    max).  It also inflated every link's `min_y`/`max_y` toward the body frame
    origin, which is silently wrong for a body whose origin is not inside its
    collision geometry (e.g. `link_003`/`link_006`, whose hull reaches 0.312 m
    forward of a hip-mounted origin).

    The mask is now explicit and `link_y_extents` honours it, so padding
    cannot participate in any envelope.
    """
    clouds = urdf_link_vertex_cloud(urdf_path)
    body_names = [env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])[i]
                  for i in range(len(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])))]
    max_pts = max(len(v) for v in clouds.values())
    cloud = torch.zeros(len(body_names), max_pts, 3, device=env.device, dtype=torch.float)
    valid = torch.zeros(len(body_names), dtype=torch.bool, device=env.device)
    mask = torch.zeros(len(body_names), max_pts, dtype=torch.bool, device=env.device)
    for idx, name in enumerate(body_names):
        pts = clouds.get(name)
        if pts is None:
            continue
        cloud[idx, :pts.shape[0], :] = torch.from_numpy(pts).to(env.device)
        mask[idx, :pts.shape[0]] = True
        valid[idx] = True
    env.link_point_cloud = cloud
    env.link_point_valid = valid
    env.link_point_mask = mask
    env.link_names = body_names
    env.wheel_body_indices = [env.left_wheel_body, env.right_wheel_body]
    return env.link_point_cloud


def link_y_extents(env, z_ceiling=None):
    """World Y interval of every body from its hull vertices.

    Args:
        z_ceiling: optional (N,) world-Z ceiling.  When given, only points whose
            world Z is at or below the ceiling are kept.  This is required for
            obstacle reasoning: a mesh point 0.5 m above the floor cannot strike
            a 5 cm box, so including it in the "front" envelope would make the
            robot look 0.2 m longer than it physically is.  The ceiling is the
            obstacle top plane, so "at or below" is exactly "could touch".
            NOTE: this is a CLIP of a conservative envelope, so the result is
            no longer an outer bound of anything -- it is the exact Y extent of
            the part of the hull at or below the plane, which is what we want,
            and it is still never a penetration depth.

    Returns (N, num_bodies) min_y, (N, num_bodies) max_y.
    Bodies with no surviving point are filled with +inf / -inf respectively so
    the per-env reductions skip them.
    """
    rb = env.rb_states
    R = quat_to_matrix(rb[:, :, 3:7].reshape(-1, 4)).view(
        rb.shape[0], rb.shape[1], 3, 3)
    pts = env.link_point_cloud  # (B, K, 3)
    # (N, B, K, 3)
    world = torch.einsum('nbij,bkj->nbki', R, pts) + rb[:, :, None, 0:3]
    # Per-POINT validity: excludes both cloud-less bodies and zero padding.
    point_mask = getattr(env, 'link_point_mask', None)
    if point_mask is None:
        point_mask = env.link_point_valid.view(-1, 1).expand(-1, pts.shape[1])
    valid = point_mask.unsqueeze(0).expand(world.shape[0], -1, -1)
    if z_ceiling is not None:
        valid = valid & (world[..., 2] <= z_ceiling.view(-1, 1, 1))
    ys = world[..., 1]
    min_y = ys.masked_fill(~valid, float('inf')).min(dim=2).values
    max_y = ys.masked_fill(~valid, float('-inf')).max(dim=2).values
    return min_y, max_y


def robot_forward_interval(env, z_ceiling=None):
    """Best available world Y interval of the WHOLE robot.

    Args:
        z_ceiling: optional (N,) world-Z ceiling forwarded to `link_y_extents`
            and applied to the analytic wheels as well.

    Returns dict with:
        front_y, rear_y : (N,) tensors -- max/min over all bodies
        front_body      : (N,) long -- body index attaining front_y
        exact_wheels    : (N, 4) exact wheel intervals
        source          : 'support_points+wheels'
    """
    min_y, max_y = link_y_extents(env, z_ceiling=z_ceiling)
    wheel_iv = wheel_forward_intervals(env)
    wheel_min = torch.minimum(wheel_iv[:, 0], wheel_iv[:, 2])
    wheel_max = torch.maximum(wheel_iv[:, 1], wheel_iv[:, 3])
    if z_ceiling is not None:
        # A wheel entirely above the ceiling cannot hit the box either.
        wheel_bottom = wheel_bottom_zs(env).min(dim=1).values
        wheel_ok = wheel_bottom <= z_ceiling
        wheel_min = torch.where(wheel_ok, wheel_min, torch.full_like(wheel_min, float('inf')))
        wheel_max = torch.where(wheel_ok, wheel_max, torch.full_like(wheel_max, float('-inf')))
    # Trust the analytic wheel interval over its own support points.
    min_y = min_y.clone()
    max_y = max_y.clone()
    min_y[:, env.left_wheel_body] = wheel_min
    min_y[:, env.right_wheel_body] = wheel_min
    max_y[:, env.left_wheel_body] = wheel_max
    max_y[:, env.right_wheel_body] = wheel_max
    front_y, front_body = max_y.max(dim=1)
    rear_y = min_y.min(dim=1).values
    # Nothing below the ceiling at all -> nothing can hit the box.  Fall back to
    # the base position so callers never consume a +/-inf and never mistake the
    # robot for "already past" the obstacle.
    nothing_below = ~torch.isfinite(front_y)
    if nothing_below.any():
        base_y = env.rb_states[:, 0, 1]
        front_y = torch.where(nothing_below, base_y, front_y)
        rear_y = torch.where(nothing_below, base_y, rear_y)
        front_body = torch.where(nothing_below,
                                 torch.zeros_like(front_body), front_body)
    return {
        'front_y': front_y,
        'rear_y': rear_y,
        'front_body': front_body,
        'exact_wheels': wheel_iv,
        'nothing_below_ceiling': nothing_below,
        'source': 'support_points+wheels',
    }


# ---------------------------------------------------------------------------
# Obstacle box interval (exact, axis aligned)
# ---------------------------------------------------------------------------

def obstacle_y_interval(env):
    """EXACT world Y interval and top Z of the obstacle box, per environment.

    The box is spawned with identity orientation, so its world Y interval is
    exactly centre +/- width/2 and its top is exactly centre_z + height/2.
    """
    centre = env.obstacle_root_states[:, 0:3]
    half_w = 0.5 * env.obstacle_w
    front = centre[:, 1] - half_w
    rear = centre[:, 1] + half_w
    top_z = centre[:, 2] + 0.5 * env.obstacle_h
    return front, rear, top_z


def _box_bounds(env):
    """(N, 3) lower and upper world corners of the obstacle box."""
    centre = env.obstacle_root_states[:, 0:3]
    half = torch.tensor([0.5 * env.obstacle_l, 0.5 * env.obstacle_w,
                         0.5 * env.obstacle_h],
                        device=centre.device, dtype=centre.dtype)
    return centre - half, centre + half


def wheel_box_signed_gap(env, samples=65):
    """Signed gap between each wheel collision volume and the obstacle box.

    Returns an (N, 2) tensor `[left, right]`:
        gap < 0  -> the wheel and the box genuinely intersect;
        gap >= 0 -> they are separated by `gap` metres.

    How exact is this?
    ------------------
    The distance from the wheel's swept-sphere equivalent (a segment of length
    `WHEEL_LENGTH` along the link-local X axis, offset by the URDF cylinder
    origin, padded by `WHEEL_RADIUS`) to the axis-aligned box is computed by
    densely sampling the segment.

    ERROR BOUND (measured, not hand-waved)
    --------------------------------------
    A previous version of this docstring claimed the sampling error was
    "~0.2 mm".  THAT WAS WRONG by 3.5x.  With `samples=65` over a segment of
    `WHEEL_LENGTH = 0.0885 m` the spacing is

        h = 0.0885 / 64 = 1.383 mm

    Point-to-AABB distance along the segment is a convex function of the
    parameter, so the sampled minimum over-estimates the true minimum by at
    most `h/2 = 0.691 mm`:

        reported_gap - true_gap  in  [0, 0.691 mm]

    i.e. the reported gap can claim separation where the true gap is up to
    0.69 mm smaller. A negative reported gap is therefore still a valid witness
    of intersection (conservative in the safe direction), while a small
    positive reported gap is NOT proof of a 0.1 mm clearance.

    CONSEQUENCE FOR REPORTING: values of order 1e-6..1e-4 m produced by this
    function must be described as "below the sampling resolution", never as a
    measured penetration depth or an exact clearance.  For reference the
    acceptance run reported first-contact clearances of -0.00033..+0.00065 m,
    all far under the 0.691 mm bound -- those numbers only support the sign
    (touching vs not), not the magnitude.

    `replace_cylinder_with_capsule` is True in this asset, so the simulation
    volume IS the swept sphere and the intersection test is exact. When it is
    False the test slightly over-approximates near the two flat caps, which is
    immaterial here because the box is 1.0 m wide along the wheel axis.

    The magnitude for `gap < 0` is a LOWER BOUND on penetration depth (it only
    witnesses that a segment point is within `radius` of the box). It is never
    reported as an exact penetration depth.
    """
    rb = env.rb_states
    lo, hi = _box_bounds(env)
    half_len = 0.5 * WHEEL_LENGTH
    t = torch.linspace(-1.0, 1.0, samples, device=rb.device, dtype=rb.dtype)

    gaps = []
    for body_index, axial_offset in (
            (env.left_wheel_body, WHEEL_AXIAL_OFFSET['link_004']),
            (env.right_wheel_body, WHEEL_AXIAL_OFFSET['link_007'])):
        ax, ay, az = wheel_axis_terms(rb, body_index)
        axis = torch.stack([ax, ay, az], dim=1)                      # (N, 3)
        centre = rb[:, body_index, 0:3] + axial_offset * axis
        pts = centre[:, None, :] + (half_len * t)[None, :, None] * axis[:, None, :]
        # Per-axis separation from the box, zero when the sample is inside it.
        sep = (torch.clamp(lo[:, None, :] - pts, min=0.0)
               + torch.clamp(pts - hi[:, None, :], min=0.0))
        distance = torch.norm(sep, dim=2).min(dim=1).values
        gaps.append(distance - WHEEL_RADIUS)
    return torch.stack(gaps, dim=1)


def wheel_clearance_above_box_top(env, samples=65):
    """Vertical clearance of each wheel above the obstacle's top face.

    Returns an (N, 2) tensor `[left, right]` in metres, `+inf` when the wheel
    cannot reach the box footprint horizontally at all.

    NOT "EXACT".  A previous version of this docstring called it exact while
    also conceding a ~0.2 mm sampling error; the true bound is `h/2 = 0.691 mm`
    with `samples=65` (see `wheel_box_signed_gap`).  Treat any reported value
    whose magnitude is below 1 mm as "below sampling resolution" and report
    only its sign.

    Why not `wheel_bottom_z - box_top_z`?
    -------------------------------------
    That subtraction compares two numbers taken at DIFFERENT places: the wheel's
    lowest point sits directly under the wheel centre, which is usually nowhere
    near the box. Using it made a robot standing next to the box report a
    clearance of `0.0 - 0.05 = -0.05 m`, i.e. "5 cm inside the obstacle", on a
    frame where the analytic wheel/box gap was +0.43 mm.

    The honest question is: *of the part of the wheel that is horizontally over
    the box footprint, what is the lowest z?* That is what this computes, minus
    the box top.

    Derivation
    ----------
    The wheel volume is the sweep of a radius-`R` ball along the axis segment
    (PhysX substitutes a capsule for the URDF cylinder -- see
    `wheels_used_as_capsules`). For one segment sample `s`, a ball point reaches
    the footprint once its horizontal offset `t` from `s` satisfies `t >= d`,
    where `d` is the horizontal distance from `s` to the footprint rectangle
    (`d == 0` when `s` is already inside it). The lowest reachable z is then
    `s.z - sqrt(R^2 - t^2)` minimised at `t = d`, giving

        z_min(s) = s.z - sqrt(R^2 - d^2),   defined only when d <= R.

    The sweep's minimum is the minimum over samples, so the result carries the
    same `h/2 = 0.691 mm` segment-sampling bound as `wheel_box_signed_gap`, and
    it can never contradict that function's sign: a negative gap implies this
    value is below the box top.
    """
    rb = env.rb_states
    lo, hi = _box_bounds(env)
    top_z = hi[:, 2]
    half_len = 0.5 * WHEEL_LENGTH
    t = torch.linspace(-1.0, 1.0, samples, device=rb.device, dtype=rb.dtype)

    clearances = []
    for body_index, axial_offset in (
            (env.left_wheel_body, WHEEL_AXIAL_OFFSET['link_004']),
            (env.right_wheel_body, WHEEL_AXIAL_OFFSET['link_007'])):
        ax, ay, az = wheel_axis_terms(rb, body_index)
        axis = torch.stack([ax, ay, az], dim=1)                      # (N, 3)
        centre = rb[:, body_index, 0:3] + axial_offset * axis
        pts = centre[:, None, :] + (half_len * t)[None, :, None] * axis[:, None, :]
        # Horizontal (X, Y) distance from each sample to the footprint rectangle.
        dx = (torch.clamp(lo[:, None, 0] - pts[:, :, 0], min=0.0)
              + torch.clamp(pts[:, :, 0] - hi[:, None, 0], min=0.0))
        dy = (torch.clamp(lo[:, None, 1] - pts[:, :, 1], min=0.0)
              + torch.clamp(pts[:, :, 1] - hi[:, None, 1], min=0.0))
        d = torch.sqrt(dx * dx + dy * dy)
        # Ball radius available below the sample once it has spent `d` to travel
        # horizontally onto the footprint.
        below = torch.sqrt(torch.clamp(WHEEL_RADIUS ** 2 - d * d, min=0.0))
        reachable = d <= WHEEL_RADIUS
        z_min = pts[:, :, 2] - below
        cand = torch.where(reachable, z_min - top_z[:, None],
                           torch.full_like(z_min, float('inf')))
        clearances.append(cand.min(dim=1).values)
    return torch.stack(clearances, dim=1)


# ---------------------------------------------------------------------------
# SEPARATED envelopes (Phase D, defect A)
# ---------------------------------------------------------------------------
# Defect A was that ONE envelope -- the hull support cloud clipped at the box top
# -- was simultaneously used for (a) the approach trigger, (b) the "cleared the
# box" test and (c) the mission-success test. Those three questions have
# different answers, and forcing one number to answer all of them produced two
# measurable errors:
#
#   1. The clipped rear edge is ~0.25 m AHEAD of the whole-body rear edge
#      (measured +0.0466 vs -0.2089 relative to the root), so "cleared the box"
#      was ~0.25 m looser than it claimed to be.
#   2. While airborne, every body point sat above the 5 cm plane, nothing
#      survived the clip, and the code fell back to `rb_states[:, 0, 1]`
#      (the root Y). That made front_y == rear_y == root_y for the whole flight,
#      which silently corrupted the takeoff snapshot.
#
# The three functions below replace that. They are deliberately separate and
# none of them falls back to the root position.

def _hull_edges_from_points(full):
    """Unique undirected convex-hull edges, as indices into the hull vertex array."""
    try:
        from scipy.spatial import ConvexHull
    except ImportError:                       # pragma: no cover - scipy is present
        return np.zeros((0, 2), dtype=np.int32)
    if full.shape[0] <= 4:
        return np.zeros((0, 2), dtype=np.int32)
    hull = ConvexHull(full)
    keep = np.unique(hull.vertices)
    remap = {int(v): i for i, v in enumerate(keep)}
    edges = set()
    for tri in hull.simplices:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            va, vb = remap[int(tri[a])], remap[int(tri[b])]
            edges.add((va, vb) if va < vb else (vb, va))
    return np.array(sorted(edges), dtype=np.int32).reshape(-1, 2)


@lru_cache(maxsize=4)
def urdf_link_hull_edges(urdf_path):
    """{link: (E, 2) int32} hull edges, indexed identically to
    `urdf_link_vertex_cloud`.

    The two functions repeat the same deterministic parse and the same
    `np.unique(ConvexHull(...).vertices)` ordering, so index `i` means the same
    physical vertex in both.
    """
    mesh_root = os.path.dirname(os.path.abspath(urdf_path))
    root = ET.parse(urdf_path).getroot()
    out = {}
    for link in root.findall('link'):
        name = link.get('name')
        clouds = []
        for coll in link.findall('collision'):
            origin = coll.find('origin')
            xyz = np.zeros(3)
            rot = np.eye(3)
            if origin is not None:
                if origin.get('xyz'):
                    xyz = np.array([float(v) for v in origin.get('xyz').split()])
                if origin.get('rpy'):
                    rot = rpy_to_matrix([float(v) for v in origin.get('rpy').split()])
            geom_node = coll.find('geometry')
            if geom_node is None or len(geom_node) == 0:
                continue
            geom = geom_node[0]
            if geom.tag == 'mesh':
                scale = np.ones(3)
                if geom.get('scale'):
                    scale = np.array([float(v) for v in geom.get('scale').split()])
                verts = _stl_vertices(os.path.join(mesh_root, geom.get('filename'))) * scale
            else:
                verts = _primitive_vertices(geom)
            clouds.append(verts @ rot.T + xyz)
        if not clouds:
            continue
        full = np.concatenate(clouds, axis=0)
        out[name] = _hull_edges_from_points(full)
    return out


def register_link_edges(env, urdf_path):
    """Attach padded hull-edge indices to `env` (companion to register_link_points).

    Sets:
        env.link_edge_index : (B, Emax, 2) long -- indices into link_point_cloud
        env.link_edge_mask  : (B, Emax) bool
    """
    if not hasattr(env, 'link_names'):
        return None
    edges = urdf_link_hull_edges(urdf_path)
    names = env.link_names
    max_e = max([len(edges.get(n, ())) for n in names] + [1])
    idx = torch.zeros(len(names), max_e, 2, device=env.device, dtype=torch.long)
    msk = torch.zeros(len(names), max_e, device=env.device, dtype=torch.bool)
    for b, name in enumerate(names):
        e = edges.get(name)
        if e is None or len(e) == 0:
            continue
        idx[b, :len(e), :] = torch.from_numpy(e).to(env.device)
        msk[b, :len(e)] = True
    env.link_edge_index = idx
    env.link_edge_mask = msk
    return idx


def link_y_extents_clipped(env, z_ceiling):
    """World Y extent of each link's CONVEX HULL intersected with `z <= ceiling`.

    Why this is not `mask the vertices by z`
    ----------------------------------------
    Filtering hull VERTICES by `z <= ceiling` and taking their Y extremes can
    UNDER-estimate the true front edge: the forwardmost point of the cut solid is
    generally where a hull EDGE crosses the plane, and that point is not a
    vertex.  Since a shorter front edge makes the robot look farther from the box
    than it is, that error delays the trigger -- it is the unsafe direction.

    The intersection of a convex polytope with a half-space is again a convex
    polytope whose vertices are exactly
        {vertices with z <= ceiling}  U  {edge/plane crossing points}.
    So including the crossing points makes this the EXACT Y extent of
    `hull ∩ {z <= ceiling}` -- no sampling, no direction set, no clipping
    approximation.

    Scope of the word "exact"
    -------------------------
    This is exact for the CONVEX HULL of the collision vertices. The hull is a
    superset of the true (non-convex) meshes, so as an envelope of the physical
    geometry it is conservative (it can only make the robot look bigger), never
    optimistic. It is NOT a penetration depth and NOT a measure of the actual
    mesh surface.

    Returns (min_y, max_y) shaped (N, num_bodies); emptied bodies get +inf/-inf.
    """
    rb = env.rb_states
    R = quat_to_matrix(rb[:, :, 3:7].reshape(-1, 4)).view(
        rb.shape[0], rb.shape[1], 3, 3)
    pts = env.link_point_cloud                       # (B, K, 3)
    world = torch.einsum('nbij,bkj->nbki', R, pts) + rb[:, :, None, 0:3]
    point_mask = getattr(env, 'link_point_mask', None)
    if point_mask is None:
        point_mask = env.link_point_valid.view(-1, 1).expand(-1, pts.shape[1])
    ceil = z_ceiling.view(-1, 1, 1)
    ys = world[..., 1]

    below = point_mask.unsqueeze(0).expand(world.shape[0], -1, -1) & (world[..., 2] <= ceil)
    min_y = ys.masked_fill(~below, float('inf')).min(dim=2).values
    max_y = ys.masked_fill(~below, float('-inf')).max(dim=2).values

    edge_index = getattr(env, 'link_edge_index', None)
    if edge_index is not None:
        edge_mask = env.link_edge_mask                     # (B, E)
        # `world` is (N, B, K, 3) and the gather runs along the K axis, so the
        # index must be (N, B, E, 3) with the same vertex index in every
        # component -- gather selects a whole vertex, not one coordinate.
        shp = (world.shape[0], edge_index.shape[0], edge_index.shape[1], 3)
        ia = edge_index[..., 0].view(1, *shp[1:3], 1).expand(shp)
        ib = edge_index[..., 1].view(1, *shp[1:3], 1).expand(shp)
        a = torch.gather(world, 2, ia)
        b = torch.gather(world, 2, ib)
        za, zb = a[..., 2], b[..., 2]
        planar_ceil = z_ceiling.view(-1, 1, 1)
        straddles = ((za - planar_ceil) * (zb - planar_ceil)) <= 0.0
        denom = zb - za
        safe = torch.where(denom.abs() > 1e-12, denom,
                           torch.full_like(denom, 1e-12))
        t = torch.clamp((planar_ceil - za) / safe, 0.0, 1.0)
        cross_y = a[..., 1] + t * (b[..., 1] - a[..., 1])
        em = edge_mask.unsqueeze(0).expand(world.shape[0], -1, -1) & straddles
        min_y = torch.minimum(min_y, cross_y.masked_fill(~em, float('inf')).min(dim=2).values)
        max_y = torch.maximum(max_y, cross_y.masked_fill(~em, float('-inf')).max(dim=2).values)
    return min_y, max_y


def approach_envelope(env):
    """APPROACH TRIGGER envelope: only geometry that could actually hit the box.

    This answers exactly one question -- "how far is the frontmost solid that can
    reach the obstacle at its current height?" -- and nothing else. It must never
    be reused for a "cleared the box" decision: the rear edge of a height-limited
    envelope is the rear of the LOW parts, not of the robot.

    There is no root-position fallback. If nothing is low enough to touch the box
    (`could_reach` False), there is no meaningful distance and the caller must
    not treat the robot as the root point.

    Returns dict with front_y, rear_y, front_body, could_reach, exact_wheels.
    """
    _, _, top_z = obstacle_y_interval(env)
    min_y, max_y = link_y_extents_clipped(env, top_z)
    iv = wheel_forward_intervals(env)
    bottom = wheel_bottom_zs(env)
    for col, body in ((0, env.left_wheel_body), (1, env.right_wheel_body)):
        ok = bottom[:, col] <= top_z
        min_y[:, body] = torch.where(ok, iv[:, 2 * col],
                                     torch.full_like(iv[:, 2 * col], float('inf')))
        max_y[:, body] = torch.where(ok, iv[:, 2 * col + 1],
                                     torch.full_like(iv[:, 2 * col + 1], float('-inf')))
    front_y, front_body = max_y.max(dim=1)
    rear_y = min_y.min(dim=1).values
    return {
        'front_y': front_y,
        'rear_y': rear_y,
        'front_body': front_body,
        'could_reach': torch.isfinite(front_y),
        'exact_wheels': iv,
        'source': 'clipped_hull+analytic_wheels',
    }


def whole_body_interval(env):
    """WHOLE-BODY envelope: the full hull, no height clip, no fallback.

    Used only for "has the robot completely left the obstacle region", which is a
    fact about the whole machine and must not be confused with "can the frontmost
    low part reach the box".

    Because the object is the convex hull of the collision vertices, its Y extent
    is EXACT for that hull under any rotation (a linear functional attains its
    extremes at hull vertices). As a bound on the non-convex meshes it is a
    conservative outer bound -- it can make the robot look longer, never shorter.
    """
    min_y, max_y = link_y_extents(env, z_ceiling=None)
    iv = wheel_forward_intervals(env)
    for col, body in ((0, env.left_wheel_body), (1, env.right_wheel_body)):
        min_y[:, body] = iv[:, 2 * col]
        max_y[:, body] = iv[:, 2 * col + 1]
    front_y, front_body = max_y.max(dim=1)
    rear_y = min_y.min(dim=1).values
    return {
        'front_y': front_y,
        'rear_y': rear_y,
        'front_body': front_body,
        'source': 'full_hull+analytic_wheels',
    }


def wheel_crossing_state(env):
    """Per-wheel passage facts, from exact analytic wheel envelopes.

    Returns dict of (N, 2) tensors (column 0 = left wheel, 1 = right wheel):
        interval_front, interval_rear : world Y interval of the wheel solid
        bottom_z                      : lowest world Z of the wheel solid
        overlaps_footprint            : wheel Y interval overlaps [box_front, box_rear]
        above_top                     : lowest point at or above the box top plane
        passed_rear                   : the whole wheel is beyond the box rear face
        danger                        : overlaps the footprint AND is below the top

    `danger` is the only predicate that means "this wheel could hit the box while
    over it". `overlaps_footprint` alone is proximity, not penetration, and must
    never be reported as a collision.
    """
    box_front, box_rear, box_top = obstacle_y_interval(env)
    iv = wheel_forward_intervals(env)
    lo = iv[:, [0, 2]]
    hi = iv[:, [1, 3]]
    bottom = wheel_bottom_zs(env)
    overlaps = (hi >= box_front[:, None]) & (lo <= box_rear[:, None])
    above = bottom >= box_top[:, None]
    return {
        'interval_front': lo,
        'interval_rear': hi,
        'bottom_z': bottom,
        'overlaps_footprint': overlaps,
        'above_top': above,
        'passed_rear': lo > box_rear[:, None],
        'danger': overlaps & (~above),
        'box_front': box_front,
        'box_rear': box_rear,
        'box_top': box_top,
    }
