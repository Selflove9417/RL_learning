#!/usr/bin/env python3
import os
import sys

workspace_root = "/home/xy/IsaacTu_jump"
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse
import torch
import numpy as np

from legged_gym.envs import task_registry
from legged_gym.utils import get_args

def trace_substeps():
    task_name = "myrobot_forward_jump_static"
    sys.argv = ["substep_trace", "--task", task_name, "--num_envs", "1"]
    args = get_args()
    args.headless = True
    args.seed = 17

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.normalization.clip_actions = 100.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.num_envs = 1
    env_cfg.seed = 17
    env_cfg.noise.add_noise = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # Let's inspect decimation and substep execution
    # In LeggedRobot:
    # step() does:
    # for _ in range(self.cfg.control.decimation):
    #     self.torques = self._compute_torques(self.actions)
    #     self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
    #     self.gym.simulate(self.sim)
    #     self.gym.fetch_results(self.sim, True)
    #     self.gym.refresh_dof_state_tensor(self.sim)
    print("=== SEED 17: SUBSTEP-BY-SUBSTEP TRACE INSIDE RESET STEP ===")
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    # We will manually step each substep of the first 0.02s step!
    # env.cfg.control.decimation is 4
    for substep in range(1, 5):
        # compute torques
        env.torques = env._compute_torques(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)

        t = substep * 0.005
        z_root = env.root_states[0, 2].item()
        vz_root = env.root_states[0, 9].item()
        vy_root = env.root_states[0, 8].item()
        pos_L = env.rb_states[0, env.left_wheel_body, 2].item() - env.wheel_radius
        pos_R = env.rb_states[0, env.right_wheel_body, 2].item() - env.wheel_radius
        fn_L = env.contact_forces[0, env.left_wheel_body, 2].item()
        fn_R = env.contact_forces[0, env.right_wheel_body, 2].item()
        fy_L = env.contact_forces[0, env.left_wheel_body, 1].item()
        fy_R = env.contact_forces[0, env.right_wheel_body, 1].item()
        print(f"Substep {substep} (t={t:.3f}s): z_root={z_root:.5f}, vy={vy_root:+.4f}, vz={vz_root:+.4f}, clr_L={pos_L*1000:+.2f}mm, clr_R={pos_R*1000:+.2f}mm, fn=({fn_L:.1f}, {fn_R:.1f})N, fy=({fy_L:.1f}, {fy_R:.1f})N")

if __name__ == "__main__":
    trace_substeps()
