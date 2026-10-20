"""What unit is `RigidContact.lambda`?  Decisive, self-contained experiment.

Build the smallest possible scene: a ground plane and one rigid box of KNOWN
mass, let it come to rest, then read the contacts. At rest the ONLY vertical
force is gravity, so the sum of the normal contact forces must equal m*g.

    lambda summed over pairs == m*g      -> lambda is a FORCE in N
    lambda summed over pairs == m*g*dt   -> lambda is an IMPULSE in N*s

The discriminating trick is to repeat the whole measurement at SEVERAL dt. A
force does not care about the timestep; an impulse is proportional to it. That
separates the two hypotheses far more convincingly than a single ratio, because
a single ratio could always be a coincidence of the chosen dt.

Result (measured, this machine, PhysX CPU):

    dt = 1/200  ->  lambda sum = 78.4796   m*g = 78.4800   ratio 0.999995
    dt = 1/100  ->  lambda sum = 78.4796   m*g = 78.4800   ratio 0.999995
    dt = 1/400  ->  lambda sum = 78.4796   m*g = 78.4800   ratio 0.999995

    lambda is INVARIANT to dt => it is a FORCE in Newtons.

An earlier version of the obstacle environment called it an impulse in N*s and
claimed a factor-1/dt = 200 unit error in the previous code. THAT WAS WRONG:
the pre-existing `force` naming was correct, and the "fix" introduced the very
error it claimed to remove. This script is the evidence, and it is why the field
is now named `force_n`.
"""
import os
import sys

import isaacgym  # noqa: F401
from isaacgym import gymapi, gymutil


def measure(dt):
    """Settle a box of known mass and return (lambda_sum, m*g, dt)."""
    # gymutil.parse_arguments() reads sys.argv itself and rejects unknown flags,
    # so our own --single/--dt arguments must be removed before it runs.
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        return _measure(dt)
    finally:
        sys.argv = saved_argv


def _measure(dt):
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments(description='lambda unit probe')
    args.headless = True
    args.use_gpu_pipeline = False
    args.physics_engine = gymapi.SIM_PHYSX

    sim_params = gymapi.SimParams()
    sim_params.dt = dt
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = False
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.rest_offset = 0.0

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    plane.static_friction = 1.0
    plane.dynamic_friction = 1.0
    gym.add_ground(sim, plane)

    opts = gymapi.AssetOptions()
    opts.fix_base_link = False
    opts.density = 1000.0
    opts.thickness = 0.001
    box = gym.create_box(sim, 0.2, 0.2, 0.2, opts)
    mass = (0.2 ** 3) * 1000.0            # gym exposes no asset-mass getter

    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.5)
    gym.create_actor(env, box, pose, 'box', 0, 0)
    gym.prepare_sim(sim)

    for _ in range(400):                  # fall and settle
        gym.simulate(sim)
        gym.fetch_results(sim, True)

    contacts = gym.get_env_rigid_contacts(env)
    lam = sum(float(c['lambda']) for c in contacts) if contacts is not None else 0.0
    return lam, mass * 9.81, dt


def main():
    # Isaac Gym allows exactly ONE Foundation per process, so each dt needs its
    # own process. Trying to loop over `measure()` in-process dies with
    #   FdFoundation.cpp (166) : Foundation object exists already.
    if len(sys.argv) > 2 and sys.argv[1] == '--single':
        dt = float(sys.argv[2])
        lam, weight, dt_used = measure(dt)
        print('SINGLE %.9f %.6f %.6f' % (dt_used, lam, weight))
        return 0

    print('lambda unit experiment: settle a known mass, compare to m*g')
    print('  expected outcome if lambda is a FORCE   : ratio -> 1.0 at every dt')
    print('  expected outcome if lambda is an IMPULSE: ratio -> dt at every dt\n')
    print('%10s %14s %14s %14s %14s' %
          ('dt', 'lambda sum', 'm*g [N]', 'lam/(m*g)', 'lam/(m*g*dt)'))
    rows = []
    import subprocess
    for dt in (1.0 / 200.0, 1.0 / 100.0, 1.0 / 400.0):
        proc = subprocess.run([sys.executable, os.path.abspath(__file__),
                               '--single', repr(dt)],
                              capture_output=True, text=True)
        line = next((ln for ln in proc.stdout.splitlines()
                     if ln.startswith('SINGLE ')), None)
        if line is None:
            print('subprocess for dt=%g produced no measurement' % dt)
            print(proc.stdout[-1500:], proc.stderr[-1500:])
            print('RESULT: FAIL')
            return 1
        _, dt_s, lam_s, w_s = line.split()
        dt_used, lam, weight = float(dt_s), float(lam_s), float(w_s)
        rows.append((dt_used, lam, weight))
        print('%10.6f %14.5f %14.5f %14.6f %14.3f'
              % (dt_used, lam, weight, lam / weight, lam / (weight * dt_used)))

    forces = [r[1] for r in rows]
    spread = (max(forces) - min(forces)) / (sum(forces) / len(forces))
    ratio_force = max(abs(r[1] / r[2] - 1.0) for r in rows)
    ratio_impulse = max(abs(r[1] / (r[2] * r[0]) - 1.0) for r in rows)

    print('\nspread of lambda across dt (relative) = %.3e' % spread)
    print('max |lambda/(m*g) - 1|                = %.3e' % ratio_force)
    print('max |lambda/(m*g*dt) - 1|             = %.3e' % ratio_impulse)

    ok = (spread < 0.01 and ratio_force < 0.01)
    print('\nVERDICT: lambda is %s' % ('a FORCE in N' if ok else 'NOT established as a force'))
    print('RESULT: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
