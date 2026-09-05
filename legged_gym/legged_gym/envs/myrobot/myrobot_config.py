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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class MyRobotCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2048
        num_actions = 6
        num_observations = 30
        num_privileged_obs = None

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0
        heading_command = False
        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [-0.8, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.40]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            'link_002_joint': 0.0,
            'link_003_joint': 0.0,
            'link_004_joint': 0.0,
            'link_005_joint': 0.0,
            'link_006_joint': 0.0,
            'link_007_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        # 柔顺 PD (Kp=40, Kd=2.0)
        stiffness = {
            'link_002_joint': 40.0,
            'link_003_joint': 40.0,
            'link_005_joint': 40.0,
            'link_006_joint': 40.0,
            'link_004_joint': 0.0,
            'link_007_joint': 0.0,
        }
        damping = {
            'link_002_joint': 2.0,
            'link_003_joint': 2.0,
            'link_005_joint': 2.0,
            'link_006_joint': 2.0,
            'link_004_joint': 0.0,
            'link_007_joint': 0.0,
        }
        action_scale = 0.25
        decimation = 4

        # 黄金轮速参数: wheel_action_scale=15.0 (对应 1.05 m/s 巡航极速，杜绝零速抖动)
        # wheel_damping=1.0 (柔顺平稳，杜绝死区颤振)
        # 轮端直接转矩控制与有界弱轮速阻尼 (阶段 19)
        wheel_torque_scale = 7.0     # 直接转矩尺度: action [-1, 1] 对应 [-7.0, 7.0] Nm 前馈
        wheel_torque_curriculum = False  # 停用阶段 18 的速度环-转矩混合课程，始终使用转矩前馈
        wheel_torque_curriculum_steps = 2400

        # 有界弱轮速阻尼: τ_damp = clip(Kd * ω, -2.0, 2.0)
        wheel_velocity_damping = 0.3          # 固定 0.3 Nm/(rad/s)
        wheel_velocity_damping_start = 0.3
        wheel_velocity_damping_limit = 2.0    # 阻尼力矩截断上限 [Nm]
        wheel_velocity_damping_curriculum = False  # 9950 已达最终阻尼，关闭课程固定 0.3
        wheel_velocity_damping_curriculum_steps = 1200

        # 有界偏航率闭环差动反馈 (阶段 22)
        # Δτ_yaw = clip(Kyaw * (yaw_cmd - yaw_actual), -limit, limit)
        wheel_yaw_damping = 1.5               # 偏航率闭环差动增益 [Nm / (rad/s)]
        wheel_yaw_damping_start = 1.5         # 阶段 22 已升温完毕，保持 1.5
        wheel_yaw_torque_limit = 1.5          # 差动力矩截断上限 [Nm] (占 7Nm 电机峰值 21%)
        wheel_yaw_damping_curriculum = False  # 阶段 22 已升温完毕，阶段 23 固定在 1.5 冻结
        wheel_yaw_damping_curriculum_steps = 1200  # 50 PPO iterations * 24 steps

        # 阶段 24: 非转向轮速差闭环抑制 (电子差速/防滑纠偏)
        # Δτ_omega = clip(K_omega * (omega_L - omega_R), -limit, limit)
        wheel_diff_damping = 0.10             # 轮速差闭环差动增益 [Nm / (rad/s)]
        wheel_diff_damping_start = 0.10        # 阶段 24 已升温完毕，阶段 25 保持 0.10 冻结
        wheel_diff_torque_limit = 1.5         # 轮速差闭环差动力矩上限 [Nm]
        wheel_diff_damping_curriculum = False  # 阶段 24 已升温完毕，阶段 25 固定关闭课程
        wheel_diff_damping_curriculum_steps = 1200  # 50 PPO iterations * 24 steps

        # 轮速对称奖励归一化常量 (独立命名，不与旧目标轮速混用)
        wheel_speed_normalization = 15.0

        # 迁移期旧控制器参数保留兼容
        wheel_action_scale = 15.0
        wheel_damping = 1.2
        wheel_damping_curriculum = False

        wheel_action_limit = 1.0     # 轮端固定有效动作范围
        wheel_action_limit_curriculum = False  # 关闭上一轮动作限幅课程
        hip_scale_reduction = 0.7    # 释放髋关节主动保平代偿行程 (从 0.5 提至 0.7，赋能 75Nm 髋电机主动托平箱体)
        knee_scale_reduction = 0.5   # 软化膝关节大角度抽动，平抑动态下蹲起伏
        leg_target_limit_margin = 0.02  # 腿部位置目标与 URDF 硬限位保持的安全裕量 [rad]

        # 阶段 28.1: 接触感知共模与差动同步降额 (CACM-TD 2.0)
        enable_cacm_td = True
        cacm_derate_diff = True  # 阶段 28.1: 同步对差动项 tau_d 实施接触降额
        fn_ref = 75.0            # 额定接地法向力门限 (N)
        fn_off = 15.0            # 失载断开法向力门限 (N)
        alpha_lpf_drop = 0.35    # 载荷跌落响应系数 (约 40ms 快速响应)
        alpha_lpf_rise = 0.15    # 载荷恢复滞后系数 (约 100ms 迟滞保护)

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 0.05
            ang_vel = 0.02
            lin_vel = 0.02
            gravity = 0.01

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/myrobot/myrobot_urdf/robot.urdf'
        name = "myrobot"
        foot_name = "link_004"
        terminate_after_contacts_on = ["base_link", "link_002", "link_003", "link_005", "link_006"]
        penalize_contacts_on = ["base_link", "link_002", "link_003", "link_005", "link_006"]
        self_collisions = 1
        flip_visual_attachments = False
        collapse_fixed_joints = True

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [2.0, 2.0, 1.0]
        lookat = [0.0, 0.0, 0.4]

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.40
        only_positive_rewards = True
        tracking_sigma = 0.18    # 强化速度跟踪刚度 (杜绝妥协性失速下滑)

        # 阶段 23 课程已完成，阶段 24/25 冻结在 3.0
        tracking_lin_vel_start = 3.0
        tracking_lin_vel_curriculum = False
        tracking_lin_vel_curriculum_steps = 1200  # 50 PPO iterations * 24 steps
        # 阶段 27: 差动优先力矩分配 (在低层 _compute_torques 中优先分配纠偏力矩)

        class scales(LeggedRobotCfg.rewards.scales):
            # 主任务奖励
            tracking_lin_vel = 3.0   # 阶段 23 目标权重 (从 2.0 提至 3.0，打破零速负死锁)
            tracking_ang_vel = 0.8
            wheel_contact = 1.0

            # 机身姿态与对称性
            orientation = -2.5       # L1 + L2 箱体水平刚性保平约束 (强制髋关节主动代偿，箱体保持绝对水平)
            pitch_rate = -0.02       # 软化俯仰角速度阻尼 (降低 5 倍，避免阻碍正常加减速物理俯仰，根除抽搐)
            base_height = -15.0      # L1 死区锁高 (死区 1.5cm 允许前倾巡航几何沉降，越界严厉惩罚)
            leg_symmetry = -3.0      # 保持左右腿镜像对称
            roll = -2.0
            stand_nice = 0.0         # 释放 CAD 零位束缚，允许策略微后仰寻优静态平衡点
            stand_still = -5.0       # 静止时强化 L1 线性漂移惩罚 (强力按死原地晃荡)
            stand_still_wheel = -0.15 # 静止时强化车轮转速锁定 (3x 刹车力矩，杜绝前后倒腾)

            # 动作平滑与消振约束
            action_rate = -0.015     # 适度平滑一阶变率，解除过度阻尼
            action_smoothness = -0.008 # 适度二阶平滑，打破黏滑弛豫振荡，允许关节细微连续微调
            dof_acc = -1.5e-6        # 关节加速度惩罚 (平抑阶跃冲击)
            leg_vel = -0.0002        # 适度腿部关节角速度阻尼 (释放髋关节抵消驱动轮反向力矩)
            lin_vel_z = -2.0         # 强力惩罚机身垂向颠簸起伏 (消除法向力脉动)
            ang_vel_xy = -0.1
            # 环境内已按 wheel_speed_normalization 归一化，避免原始 rad^2/s^2 数值淹没主任务奖励。
            wheel_symmetry = -1.0
            # 非转向工况轮动作对称惩罚 (禁止双轮反向对推撕裂)
            wheel_action_symmetry = -1.0

            # 阶段 27: 撤销节流软裕度奖励，转由执行器低层差动优先分配器硬性保障
            wheel_throttle_margin = 0.0

            # 轮端专用约束。禁止通过提高全关节 torques 惩罚来处理车轮饱和，
            # 否则会错误惩罚承担重力的髋膝关节。
            wheel_effort = -0.05
            wheel_saturation = -0.10
            wheel_action_excess = -0.05

            torques = -0.00002
            dof_vel = 0.0
            collision = -2.0
            dof_pos_limits = -5.0

            feet_air_time = 0.0
            feet_stumble = 0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 0.5


class MyRobotCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.e-3     # 从头训练使用标准学习率
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 3000
        save_interval = 25
        run_name = ''
        experiment_name = 'myrobot'
