#!/usr/bin/env python3
"""
MyRobot Jump Stage 8: 连续周期跳跃 3D GUI 可视化脚本
======================================================
展示 Stage 8 已训练 PPO 模型 (model_40.pt) 的原生高频连续起跳能力。
双轮共同净空达到 20cm 级别，空中姿态与落地弹跳自循环。
"""

import os
import sys
import isaacgym
from isaacgym import gymapi, gymtorch
import torch
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "legged_gym"))
sys.path.insert(0, os.path.join(workspace_root, "rsl_rl"))

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from rsl_rl.modules import ActorCritic

MODEL_PATH = '/home/xy/IsaacTu_jump/evaluation_assets/frozen_stage8/model_40.pt'

def run_stage8_periodic_viewer():
    sys.argv = ['view', '--task', 'myrobot_jump']
    args = get_args()
    args.headless = False
    args.seed = 1

    env_cfg, train_cfg = task_registry.get_cfgs(name='myrobot_jump')
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 3600.0

    env, _ = task_registry.make_env(name='myrobot_jump', args=args, env_cfg=env_cfg)

    ckpt = torch.load(MODEL_PATH, map_location=env.device)
    ac = ActorCritic(35, 35, 6, [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    ac.load_state_dict(ckpt['model_state_dict'])
    ac.eval()

    obs = env.reset()
    cam_offset = np.array([-1.8, -1.8, 0.8])

    print(">>> Stage 8 原生连续跳跃 3D GUI 窗口已就绪，按 ESC 退出...")

    while not env.gym.query_viewer_has_closed(env.viewer):
        with torch.no_grad():
            actions = ac.act_inference(obs)
        obs, _, _, _, _ = env.step(actions)

        # 摄像机跟随
        robot_pos = env.root_states[0, 0:3].cpu().numpy()
        cam_target = gymapi.Vec3(robot_pos[0], robot_pos[1], robot_pos[2] + 0.1)
        cam_eye = gymapi.Vec3(robot_pos[0] + cam_offset[0], robot_pos[1] + cam_offset[1], robot_pos[2] + cam_offset[2])
        env.gym.viewer_camera_look_at(env.viewer, None, cam_eye, cam_target)

        env.render(sync_frame_time=True)

if __name__ == '__main__':
    run_stage8_periodic_viewer()
