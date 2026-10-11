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

import distutils.version
import json
import numpy as np
import os
import shutil
import subprocess
from datetime import datetime

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import class_to_dict, get_args, task_registry
import torch


def save_run_snapshot(log_dir, env_cfg, train_cfg, args):
    """保存本次运行的完整上下文信息（配置、代码与版本状态），以便后续精准复现。"""
    if log_dir is None:
        return
    os.makedirs(log_dir, exist_ok=True)
    snapshot = {
        "created_at": datetime.now().isoformat(),
        "env_cfg": class_to_dict(env_cfg),
        "train_cfg": class_to_dict(train_cfg),
        "arguments": vars(args),
    }
    with open(os.path.join(log_dir, "training_snapshot.json"), "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, ensure_ascii=False, default=str)

    # 记录当前的 git 状态和 git diff 差异
    for filename, command in (
        ("git_status.txt", ["git", "status", "--short"]),
        ("git_diff.patch", ["git", "diff", "--no-ext-diff"]),
    ):
        result = subprocess.run(
            command,
            cwd=LEGGED_GYM_ROOT_DIR,
            check=False,
            capture_output=True,
            text=True,
        )
        with open(os.path.join(log_dir, filename), "w", encoding="utf-8") as handle:
            handle.write(result.stdout)

    # 自定义机器人的实现文件若未被 Git 追踪，普通 git diff 无法保留；
    # 因此将训练时的关键源代码直接复制一份留存快照。
    source_dir = os.path.join(log_dir, "source_snapshot")
    os.makedirs(source_dir, exist_ok=True)
    for relative_path in (
        "legged_gym/envs/myrobot/myrobot_env.py",
        "legged_gym/envs/myrobot/myrobot_config.py",
        "legged_gym/scripts/train.py",
    ):
        shutil.copy2(
            os.path.join(LEGGED_GYM_ROOT_DIR, relative_path),
            os.path.join(source_dir, os.path.basename(relative_path)),
        )
    runner_source = os.path.abspath(os.path.join(
        LEGGED_GYM_ROOT_DIR, "..", "rsl_rl", "rsl_rl", "runners", "on_policy_runner.py"
    ))
    if os.path.isfile(runner_source):
        shutil.copy2(runner_source, os.path.join(source_dir, "on_policy_runner.py"))

def train(args):
    # 根据任务名称和命令行参数创建强化学习环境及对应配置
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # 创建训练算法运行器 (OnPolicyRunner/PPO) 及算法配置
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    # 保存本次实验运行的代码与配置快照
    save_run_snapshot(ppo_runner.log_dir, env_cfg, train_cfg, args)
    # 启动 PPO 策略训练迭代
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
