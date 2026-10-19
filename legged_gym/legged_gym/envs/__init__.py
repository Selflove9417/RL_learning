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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO


import os

# 全局任务注册表实例
from legged_gym.utils.task_registry import task_registry

# 注册 ANYmal C 崎岖地形任务
task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
# 注册 ANYmal C 平坦地形任务
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
# 注册 ANYmal B 任务
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
# 注册 Unitree A1 四足机器人任务
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
# 注册 Cassie 双足机器人任务
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )

# 导入 PyApple 环境与配置
from legged_gym.envs.pyapple.pyapple_env import PyApple
from legged_gym.envs.pyapple.pyapple_config import PyAppleCfg, PyAppleCfgPPO

# 注册 PyApple 机器人任务
task_registry.register("pyapple", PyApple, PyAppleCfg(), PyAppleCfgPPO())

# 导入自定义机器人 MyRobot 环境与配置
from legged_gym.envs.myrobot.myrobot_env import MyRobot
from legged_gym.envs.myrobot.myrobot_config import MyRobotCfg, MyRobotCfgPPO

# 注册自定义机器人 MyRobot 任务
task_registry.register("myrobot", MyRobot, MyRobotCfg(), MyRobotCfgPPO())

# 导入与注册 20cm 跳跃专用环境与配置
from legged_gym.envs.myrobot.myrobot_jump_env import MyRobotJump
from legged_gym.envs.myrobot.myrobot_jump_config import MyRobotJumpCfg, MyRobotJumpCfgPPO

task_registry.register("myrobot_jump", MyRobotJump, MyRobotJumpCfg(), MyRobotJumpCfgPPO())

# 导入与注册 双策略连续跳跃与巡航正式集成环境
from legged_gym.envs.myrobot.myrobot_dual_env import MyRobotDualEnv
from legged_gym.envs.myrobot.myrobot_dual_config import MyRobotDualCfg, MyRobotDualCfgPPO

task_registry.register("myrobot_dual", MyRobotDualEnv, MyRobotDualCfg(), MyRobotDualCfgPPO())

# 导入与注册 动态行进中跑跳任务 (Dynamic Running Jump)
from legged_gym.envs.myrobot.myrobot_running_jump_env import MyRobotRunningJump
from legged_gym.envs.myrobot.myrobot_running_jump_config import (
    MyRobotRunningJumpCfg, MyRobotRunningJumpCfgPPO,
    MyRobotRunningJump035Cfg, MyRobotRunningJump035CfgPPO,
    MyRobotRunningJump050Cfg, MyRobotRunningJump050CfgPPO
)

task_registry.register("myrobot_running_jump", MyRobotRunningJump, MyRobotRunningJumpCfg(), MyRobotRunningJumpCfgPPO())
task_registry.register("myrobot_running_jump_035", MyRobotRunningJump, MyRobotRunningJump035Cfg(), MyRobotRunningJump035CfgPPO())
task_registry.register("myrobot_running_jump_050", MyRobotRunningJump, MyRobotRunningJump050Cfg(), MyRobotRunningJump050CfgPPO())

# 导入与注册 真实碰撞物理障碍跨越任务 (Physical Obstacle Clearance Jump)
from legged_gym.envs.myrobot.myrobot_obstacle_jump_env import MyRobotObstacleJump
from legged_gym.envs.myrobot.myrobot_obstacle_jump_config import (
    MyRobotObstacleJumpCfg, MyRobotObstacleJumpCfgPPO,
    MyRobotObstacleJump020Cfg, MyRobotObstacleJump020CfgPPO
)

task_registry.register("myrobot_obstacle_jump", MyRobotObstacleJump, MyRobotObstacleJumpCfg(), MyRobotObstacleJumpCfgPPO())
task_registry.register("myrobot_obstacle_jump_020", MyRobotObstacleJump, MyRobotObstacleJump020Cfg(), MyRobotObstacleJump020CfgPPO())




