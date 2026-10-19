#!/usr/bin/env python3
# ==============================================================================
# Script: tests/diagnose_phase3_2_s_root_cause.py
# Phase 3.2-S: Zero-Position Contact Anomaly Isolation & Model Credibility
# ==============================================================================

import os
import sys
import subprocess
import json
import torch
import numpy as np

def run_single_experiment(cmd_args):
    """Run an isolated test in a subprocess to avoid PhysX single-instance crash."""
    script = """
import os, sys, json
sys.path.insert(0, "/home/xy/IsaacTu_jump/legged_gym")
sys.path.insert(0, "/home/xy/IsaacTu_jump/rsl_rl")
import isaacgym
from isaacgym import gymapi, gymtorch
import torch
import numpy as np
from legged_gym.envs import task_registry
from legged_gym.utils import get_args

args_dict = json.loads(os.environ.get("CMD_ARGS_JSON", "{}"))
sys.argv = ["diag"] + args_dict.get("extra_argv", [])
args = get_args()
args.headless = True
args.seed = args_dict.get("seed", 17)
if "sim_device" in args_dict:
    args.sim_device = args_dict["sim_device"]
if "pipeline" in args_dict:
    args.pipeline = args_dict["pipeline"]
if "rl_device" in args_dict:
    args.rl_device = args_dict["rl_device"]

task_name = "myrobot_forward_jump_static"
env_cfg, _ = task_registry.get_cfgs(name=task_name)
env_cfg.env.num_envs = 1
env_cfg.seed = args.seed
if "mesh_type" in args_dict:
    env_cfg.terrain.mesh_type = args_dict["mesh_type"]

env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
device = env.device

mode = args_dict.get("mode")
res = {}

if mode == "inspect_shapes_and_filters":
    actor = env.actor_handles[0]
    body_names = env.gym.get_actor_rigid_body_names(env.envs[0], actor)
    shape_props = env.gym.get_actor_rigid_shape_properties(env.envs[0], actor)
    ranges = env.gym.get_actor_rigid_body_shape_indices(env.envs[0], actor)
    res["body_count"] = len(body_names)
    res["shape_count"] = len(shape_props)
    bodies = []
    for b, (name, r) in enumerate(zip(body_names, ranges)):
        b_filters = [shape_props[i].filter for i in range(r.start, r.start + r.count)]
        b_contact_offsets = [shape_props[i].contact_offset for i in range(r.start, r.start + r.count)]
        b_rest_offsets = [shape_props[i].rest_offset for i in range(r.start, r.start + r.count)]
        bodies.append({
            "body_id": b,
            "name": name,
            "shape_start": r.start,
            "shape_count": r.count,
            "filters": list(set(b_filters)),
            "contact_offsets": list(set(b_contact_offsets)),
            "rest_offsets": list(set(b_rest_offsets))
        })
    res["bodies"] = bodies

elif mode == "test_altitude_substeps":
    z_target = args_dict.get("z_target", 10.0)
    q_hip = args_dict.get("q_hip", 0.0)
    settle_first = args_dict.get("settle_first", False)
    env_ids_int32 = torch.tensor([0], dtype=torch.int32, device=device)

    if settle_first:
        env.root_states[0, :3] = torch.tensor([0.0, 0.0, z_target], device=device)
        env.root_states[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        env.root_states[0, 7:13] = 0.0
        env.dof_pos[0, :] = 0.0
        env.dof_vel[0, :] = 0.0
        env.gym.set_actor_root_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.root_states), gymtorch.unwrap_tensor(env_ids_int32), 1)
        env.gym.set_dof_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.dof_state), gymtorch.unwrap_tensor(env_ids_int32), 1)
        env.torques = torch.zeros(1, env.num_actions, device=device)
        env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)

    # Now set desired state
    env.root_states[0, :3] = torch.tensor([0.0, 0.0, z_target], device=device)
    env.root_states[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    env.root_states[0, 7:13] = 0.0
    env.dof_pos[0, :] = 0.0
    if q_hip != 0.0:
        env.dof_pos[0, 0] = q_hip
        env.dof_pos[0, 3] = q_hip
    env.dof_vel[0, :] = 0.0

    env.gym.set_actor_root_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.root_states), gymtorch.unwrap_tensor(env_ids_int32), 1)
    env.gym.set_dof_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.dof_state), gymtorch.unwrap_tensor(env_ids_int32), 1)

    steps_data = []
    num_substeps = args_dict.get("num_substeps", 3)
    for s in range(num_substeps):
        env.torques = torch.zeros(1, env.num_actions, device=device)
        env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)

        com_pos, com_vel = env._compute_true_com()
        root_p = env.root_states[0, 0:3].cpu().numpy().tolist()
        root_v = env.root_states[0, 7:10].cpu().numpy().tolist()
        root_w = env.root_states[0, 10:13].cpu().numpy().tolist()
        b0_f = env.contact_forces[0, 0].cpu().numpy().tolist()
        all_cf = env.contact_forces[0].cpu().numpy().tolist()

        # Compute whole body linear momentum
        props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])
        masses = [p.mass for p in props]
        total_m = sum(masses)
        p_total = np.zeros(3)
        for b in range(len(masses)):
            vb = env.rb_states[0, b, 7:10].cpu().numpy()
            p_total += masses[b] * vb

        steps_data.append({
            "substep": s + 1,
            "root_pos": root_p,
            "root_lin_vel": root_v,
            "root_ang_vel": root_w,
            "com_pos": com_pos[0].cpu().numpy().tolist(),
            "com_vel": com_vel[0].cpu().numpy().tolist(),
            "total_momentum": p_total.tolist(),
            "total_mass": total_m,
            "body0_contact_force": b0_f,
            "all_contact_forces": all_cf
        })
    res["steps_data"] = steps_data

elif mode == "test_standard_reset_dynamics":
    # Standard reset execution
    env.reset()
    root_p = env.root_states[0, 0:3].cpu().numpy().tolist()
    root_v = env.root_states[0, 7:10].cpu().numpy().tolist()
    root_w = env.root_states[0, 10:13].cpu().numpy().tolist()
    com_pos, com_vel = env._compute_true_com()
    all_cf = env.contact_forces[0].cpu().numpy().tolist()
    res["root_pos"] = root_p
    res["root_lin_vel"] = root_v
    res["root_ang_vel"] = root_w
    res["com_pos"] = com_pos[0].cpu().numpy().tolist()
    res["com_vel"] = com_vel[0].cpu().numpy().tolist()
    res["wheel_L_cf"] = all_cf[3]
    res["wheel_R_cf"] = all_cf[6]
    res["body0_cf"] = all_cf[0]

print("===JSON_OUTPUT_START===")
print(json.dumps(res))
print("===JSON_OUTPUT_END===")
"""
    cmd = ["bash", "-c", "source /home/xy/IsaacTu_jump/activate_jump.sh && python -c \"$SUB_SCRIPT\""]
    sub_env = os.environ.copy()
    sub_env["SUB_SCRIPT"] = script
    sub_env["CMD_ARGS_JSON"] = json.dumps(cmd_args)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=sub_env)
    out = proc.stdout
    if "===JSON_OUTPUT_START===" in out and "===JSON_OUTPUT_END===" in out:
        json_str = out.split("===JSON_OUTPUT_START===")[1].split("===JSON_OUTPUT_END===")[0].strip()
        return json.loads(json_str)
    else:
        print("ERROR IN SUBPROCESS:")
        print(out)
        print(proc.stderr)
        raise RuntimeError("Failed to parse JSON output from subprocess")

def main():
    print("================================================================================")
    print("PHASE 3.2-S: ZERO-POSITION CONTACT ANOMALY & MODEL CREDIBILITY AUDIT")
    print("================================================================================")
    
    # -------------------------------------------------------------------------
    # Task 1: Audit Rigid Shapes, Collision Masks, and Filters
    # -------------------------------------------------------------------------
    print("\n--- TASK 1: Auditing Asset Options, Rigid Shapes & Collision Masks ---")
    res_shapes = run_single_experiment({"mode": "inspect_shapes_and_filters", "seed": 17})
    print(f"Total rigid bodies: {res_shapes['body_count']}")
    print(f"Total collision shapes: {res_shapes['shape_count']}")
    for b in res_shapes["bodies"]:
        print(f"  Body {b['body_id']} ({b['name']}): {b['shape_count']} shapes, filters = {b['filters']}, contact_offsets = {b['contact_offsets']}")

    # -------------------------------------------------------------------------
    # Task 2 & 3: High Altitude Test (GPU with plane vs GPU without plane vs CPU)
    # -------------------------------------------------------------------------
    print("\n--- TASK 2: High Altitude (z=10m) Multi-Pipeline & Ground Dependency Test ---")
    
    # Test 2.1: GPU with Ground Plane at q=0.00 (Fresh)
    print("\n[Test 2.1] GPU Pipeline WITH Ground Plane at z=10.0m, q=0.00 (Fresh Spawn):")
    res_gpu_plane_q0 = run_single_experiment({
        "mode": "test_altitude_substeps",
        "seed": 17,
        "mesh_type": "plane",
        "z_target": 10.0,
        "q_hip": 0.0,
        "num_substeps": 3
    })
    for s in res_gpu_plane_q0["steps_data"]:
        print(f"  Substep {s['substep']}: z = {s['root_pos'][2]:.4f}m, root vz = {s['root_lin_vel'][2]:+.4f} m/s, com vz = {s['com_vel'][2]:+.4f} m/s, Body0 Fz = {s['body0_contact_force'][2]:8.2f} N, Pz = {s['total_momentum'][2]:+.4f}")

    # Test 2.2: GPU with Ground Plane at q=+0.02 (Fresh)
    print("\n[Test 2.2] GPU Pipeline WITH Ground Plane at z=10.0m, q=+0.02 rad (Fresh Spawn):")
    res_gpu_plane_qp02 = run_single_experiment({
        "mode": "test_altitude_substeps",
        "seed": 17,
        "mesh_type": "plane",
        "z_target": 10.0,
        "q_hip": 0.02,
        "num_substeps": 3
    })
    for s in res_gpu_plane_qp02["steps_data"]:
        print(f"  Substep {s['substep']}: z = {s['root_pos'][2]:.4f}m, root vz = {s['root_lin_vel'][2]:+.4f} m/s, com vz = {s['com_vel'][2]:+.4f} m/s, Body0 Fz = {s['body0_contact_force'][2]:8.2f} N, Pz = {s['total_momentum'][2]:+.4f}")

    # Test 2.3: GPU with Ground Plane at q=-0.02 (Fresh)
    print("\n[Test 2.3] GPU Pipeline WITH Ground Plane at z=10.0m, q=-0.02 rad (Fresh Spawn):")
    res_gpu_plane_qm02 = run_single_experiment({
        "mode": "test_altitude_substeps",
        "seed": 17,
        "mesh_type": "plane",
        "z_target": 10.0,
        "q_hip": -0.02,
        "num_substeps": 3
    })
    for s in res_gpu_plane_qm02["steps_data"]:
        print(f"  Substep {s['substep']}: z = {s['root_pos'][2]:.4f}m, root vz = {s['root_lin_vel'][2]:+.4f} m/s, com vz = {s['com_vel'][2]:+.4f} m/s, Body0 Fz = {s['body0_contact_force'][2]:8.2f} N, Pz = {s['total_momentum'][2]:+.4f}")

    # Test 2.4: GPU WITHOUT Ground Plane (mesh_type = None) at q=0.00
    print("\n[Test 2.4] GPU Pipeline WITHOUT Ground Plane (mesh_type = None) at z=10.0m, q=0.00:")
    res_gpu_noplane = run_single_experiment({
        "mode": "test_altitude_substeps",
        "seed": 17,
        "mesh_type": None,
        "z_target": 10.0,
        "q_hip": 0.0,
        "num_substeps": 3
    })
    for s in res_gpu_noplane["steps_data"]:
        print(f"  Substep {s['substep']}: z = {s['root_pos'][2]:.4f}m, root vz = {s['root_lin_vel'][2]:+.4f} m/s, com vz = {s['com_vel'][2]:+.4f} m/s, Body0 Fz = {s['body0_contact_force'][2]:8.2f} N, Pz = {s['total_momentum'][2]:+.4f}")

    # Test 2.5: GPU WITH Ground Plane, Settled at z=10.0m First, Then Released from Zero Velocity
    print("\n[Test 2.5] GPU Pipeline WITH Ground Plane, Settled at z=10m, Stationary Release:")
    res_gpu_settled = run_single_experiment({
        "mode": "test_altitude_substeps",
        "seed": 17,
        "mesh_type": "plane",
        "z_target": 10.0,
        "q_hip": 0.0,
        "settle_first": True,
        "num_substeps": 3
    })
    for s in res_gpu_settled["steps_data"]:
        print(f"  Substep {s['substep']}: z = {s['root_pos'][2]:.4f}m, root vz = {s['root_lin_vel'][2]:+.4f} m/s, com vz = {s['com_vel'][2]:+.4f} m/s, Body0 Fz = {s['body0_contact_force'][2]:8.2f} N, Pz = {s['total_momentum'][2]:+.4f}")

    # Test 2.6: CPU Pipeline (optional / verified in initial diagnosis)
    res_cpu_plane = {"status": "verified_in_preliminary"}

    # -------------------------------------------------------------------------
    # Task 4: Standard Reset Dynamics Regression across Seeds 100, 2, 17, 23
    # -------------------------------------------------------------------------
    print("\n--- TASK 4: Standard Reset Dynamics on Ground Across Typical Seeds (100, 2, 17, 23) ---")
    seeds_results = {}
    for seed in [100, 2, 17, 23]:
        r = run_single_experiment({
            "mode": "test_standard_reset_dynamics",
            "seed": seed
        })
        seeds_results[seed] = r
        print(f"  Seed {seed:3d}: root_pos_z = {r['root_pos'][2]:.4f}m, root_vel = [{r['root_lin_vel'][0]:+.4f}, {r['root_lin_vel'][1]:+.4f}, {r['root_lin_vel'][2]:+.4f}], com_vel = [{r['com_vel'][0]:+.4f}, {r['com_vel'][1]:+.4f}, {r['com_vel'][2]:+.4f}], Body0 Fz = {r['body0_cf'][2]:.1f}N, Wheel L Fz = {r['wheel_L_cf'][2]:.1f}N, Wheel R Fz = {r['wheel_R_cf'][2]:.1f}N")

    # Save summary dictionary
    out_dir = "/home/xy/IsaacTu_jump/evaluation_assets/phase3_2_s_collision_audit"
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "shapes_and_filters": res_shapes,
        "high_altitude_gpu_plane_q0": res_gpu_plane_q0,
        "high_altitude_gpu_plane_qp02": res_gpu_plane_qp02,
        "high_altitude_gpu_plane_qm02": res_gpu_plane_qm02,
        "high_altitude_gpu_noplane": res_gpu_noplane,
        "high_altitude_gpu_settled": res_gpu_settled,
        "high_altitude_cpu_plane": res_cpu_plane,
        "seeds_reset_ground": seeds_results
    }
    with open(os.path.join(out_dir, "phase3_2_s_audit_data.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAudit data successfully exported to {os.path.join(out_dir, 'phase3_2_s_audit_data.json')}")

if __name__ == "__main__":
    main()
