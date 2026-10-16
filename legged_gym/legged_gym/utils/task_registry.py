# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
from datetime import datetime
from typing import Tuple, Type
import torch
import numpy as np

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TaskRegistry():
    """任务注册表类，用于管理所有强化学习任务的环境类、环境配置和算法训练配置。"""
    def __init__(self):
        self.task_classes = {}  # 存放任务环境类字典，如 {"anymal_c_rough": Anymal, ...}
        self.env_cfgs = {}      # 存放环境配置字典
        self.train_cfgs = {}    # 存放训练/算法配置字典
    
    def register(self, name: str, task_class: Type[VecEnv], env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        """向注册表中注册一个新任务。

        参数:
            name (str): 任务名称（如 "a1", "myrobot"）
            task_class (Type[VecEnv]): 任务对应的环境类（继承自 VecEnv）
            env_cfg (LeggedRobotCfg): 环境配置对象
            train_cfg (LeggedRobotCfgPPO): 算法训练配置对象
        """
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> Type[VecEnv]:
        """根据任务名称获取对应的环境类。"""
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        """根据任务名称获取环境配置和训练配置，并同步随机种子。"""
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # 将训练配置中的随机种子同步复制到环境配置中
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """根据已注册的任务名称或传入的配置对象创建仿真环境实例。

        参数:
            name (string): 已注册的任务名称。
            args (Args, 可选): Isaac Gym 命令行解析参数。若为 None 则自动调用 get_args()。默认为 None。
            env_cfg (Dict/LeggedRobotCfg, 可选): 用于覆盖注册配置的环境配置对象。默认为 None。

        异常:
            ValueError: 如果未找到名称为 'name' 的已注册任务。

        返回:
            VecEnv: 创建好的向量化仿真环境实例。
            LeggedRobotCfg: 最终使用的环境配置对象。
        """
        # 若未传入 args，则从命令行获取参数
        if args is None:
            args = get_args()
        # 检查是否注册了对应名称的任务环境
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # 加载已注册的配置
            env_cfg, _ = self.get_cfgs(name)
        # 如果命令行指定了参数，用其覆盖环境配置中的对应项
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # 解析仿真参数（先转为字典）
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """创建训练算法运行器 (Runner)，可基于注册名称或传入的配置对象。

        参数:
            env (VecEnv): 用于训练的环境实例。
            name (string, 可选): 已注册的任务名称。若为 None 则必须提供 train_cfg。默认为 None。
            args (Args, 可选): Isaac Gym 命令行参数。若为 None 则自动调用 get_args()。默认为 None。
            train_cfg (Dict/LeggedRobotCfgPPO, 可选): 训练算法配置。若为 None 则通过 'name' 读取对应配置。
            log_root (str, 可选): TensorBoard 与模型保存的根目录。设为 None 时禁用日志记录（例如评估/测试阶段）。
                                默认值 "default" 为 <LEGGED_GYM_ROOT_DIR>/logs/<experiment_name>。

        异常:
            ValueError: 当 'name' 和 'train_cfg' 均未提供时抛出异常。
            Warning: 若同时提供了 'name' 和 'train_cfg'，将忽略 'name' 的配置。

        返回:
            OnPolicyRunner: 创建完成的算法运行器。
            LeggedRobotCfgPPO: 最终使用的训练算法配置。
        """
        # 若未传入 args，则从命令行获取参数
        if args is None:
            args = get_args()
        # 若传入了自定义配置则优先使用，否则根据任务名称加载配置
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # 加载已注册的训练配置
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # 用命令行参数覆盖算法配置（如果有指定）
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        # 确定日志与模型保存目录路径
        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        train_cfg_dict = class_to_dict(train_cfg)
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
        
        # 检查是否需要恢复之前的训练检查点
        resume = train_cfg.runner.resume
        if resume:
            # 加载之前训练的模型检查点（包括优化器状态）
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        elif getattr(train_cfg.runner, 'warm_start', False):
            # 热启动（Warm start）：加载模型网络权重，并可配置是否恢复优化器状态，迭代步数重置为 0
            if getattr(train_cfg.runner, 'warm_start_path', None) and os.path.isfile(train_cfg.runner.warm_start_path):
                warm_start_path = train_cfg.runner.warm_start_path
            else:
                warm_start_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            load_optimizer = getattr(train_cfg.runner, 'warm_start_optimizer', False)
            print(f"Warm starting model weights from: {warm_start_path} (load_optimizer={load_optimizer})")
            runner.load(warm_start_path, load_optimizer=load_optimizer)
            runner.current_learning_iteration = 0

            # 严格对齐算法层与优化器 param_groups 学习率
            target_lr = train_cfg_dict['algorithm']['learning_rate']
            runner.alg.learning_rate = target_lr
            for param_group in runner.alg.optimizer.param_groups:
                param_group['lr'] = target_lr
        return runner, train_cfg

# 创建全局单例任务注册表
task_registry = TaskRegistry()