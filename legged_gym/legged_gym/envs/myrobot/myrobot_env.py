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

import math
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs.base.legged_robot import LeggedRobot

class MyRobot(LeggedRobot):
    """
    双足轮足机器人环境类 (对齐 TITA 控制架构):
    - 腿部 4 个转动关节采用 PD 位置控制
    - 轮子 2 个连续转动关节采用 TITA 式前馈+速度阻尼控制
    - 动作平滑在 step() 中一次性完成 (非 decimation 循环内)
    """

    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props, env_id)
        # 精准定位连续旋转轮关节：通过 self.dof_names 匹配，避免依赖尚未初始化的 self.left_wheel_idx
        wheel_names = ["link_004_joint", "link_007_joint"]
        for i, name in enumerate(self.dof_names):
            if name in wheel_names:
                props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
                props["stiffness"][i] = 0.0
                props["damping"][i] = 0.0
        return props

    def _init_buffers(self):
        # 关节索引必须在 super()._init_buffers() 之前初始化
        # 因为 super 内部会调用 _get_noise_scale_vec 需要这些索引
        self.left_hip_idx = self.dof_names.index("link_002_joint")
        self.left_knee_idx = self.dof_names.index("link_003_joint")
        self.right_hip_idx = self.dof_names.index("link_005_joint")
        self.right_knee_idx = self.dof_names.index("link_006_joint")
        self.left_wheel_idx = self.dof_names.index("link_004_joint")
        self.right_wheel_idx = self.dof_names.index("link_007_joint")

        self.wheel_indices = torch.tensor(
            [self.left_wheel_idx, self.right_wheel_idx],
            device=self.device, dtype=torch.long
        )
        self.leg_indices = torch.tensor(
            [self.left_hip_idx, self.left_knee_idx, self.right_hip_idx, self.right_knee_idx],
            device=self.device, dtype=torch.long
        )

        super()._init_buffers()

        # 确保 feet_indices 准确包含双轮
        wheel_body_names = ["link_004", "link_007"]
        self.feet_indices = torch.tensor(
            [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], b_name) for b_name in wheel_body_names],
            device=self.device, dtype=torch.long
        )
        self.feet_air_time = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)

        # 平滑动作缓冲区 (包含一阶与二阶历史动作)
        self.filtered_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # 平滑后、执行器限幅前的策略请求。轮端实际动作与请求动作必须分开，
        # 才能既保证物理安全，又持续惩罚策略在无效范围内输出。
        self.requested_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_filtered_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_filtered_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # 仅腿部使用的位置目标；供诊断日志读取实际送入 PD 的目标值。
        self.position_targets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # 保存施加 URDF 力矩限幅前的执行器请求，用于区分正常比例控制与硬饱和。
        self.unclipped_torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # 记录差动优先分配器是否将任一轮推至分配边界 (±7.0 Nm)
        self.wheel_alloc_boundary_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.wheel_diff_clipped = self.wheel_alloc_boundary_hit  # 兼容旧代码引用

        # 阶段 28.1: 接触感知共模与差动同步降额 (CACM-TD 2.0) 参数与低通滤波载荷比缓冲区
        self.enable_cacm_td = getattr(self.cfg.control, 'enable_cacm_td', True)
        self.cacm_derate_diff = getattr(self.cfg.control, 'cacm_derate_diff', True)
        self.fn_ref = getattr(self.cfg.control, 'fn_ref', 75.0)
        self.fn_off = getattr(self.cfg.control, 'fn_off', 15.0)
        self.alpha_lpf_drop = getattr(self.cfg.control, 'alpha_lpf_drop', 0.35)
        self.alpha_lpf_rise = getattr(self.cfg.control, 'alpha_lpf_rise', 0.15)
        self.alpha_contact = torch.ones(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self._init_state_machine_buffers()

        # 阶段 5.1 / 14: 动态高度指令与目标缓冲区初始化
        self.nominal_height = 0.39774
        self.height_cmd_buf = torch.full((self.num_envs,), self.nominal_height, device=self.device, dtype=torch.float)
        self.target_height_buf = torch.full((self.num_envs,), self.nominal_height, device=self.device, dtype=torch.float)

        # 轮速控制参数 (速度闭环迁移期、直接转矩控制与有界弱转速阻尼)
        self.wheel_torque_scale = getattr(self.cfg.control, 'wheel_torque_scale', 7.0)
        self.wheel_torque_curriculum = getattr(self.cfg.control, 'wheel_torque_curriculum', False)
        self.wheel_torque_curriculum_steps = max(
            1, getattr(self.cfg.control, 'wheel_torque_curriculum_steps', 2400)
        )
        self.wheel_torque_alpha = 0.0 if self.wheel_torque_curriculum else 1.0

        # 有界弱轮速阻尼 (阶段 19)
        self.wheel_velocity_damping = getattr(self.cfg.control, 'wheel_velocity_damping', 0.3)
        self.wheel_velocity_damping_start = getattr(self.cfg.control, 'wheel_velocity_damping_start', 0.0)
        self.wheel_velocity_damping_limit = getattr(self.cfg.control, 'wheel_velocity_damping_limit', 2.0)
        self.wheel_velocity_damping_curriculum = getattr(self.cfg.control, 'wheel_velocity_damping_curriculum', False)
        self.wheel_velocity_damping_curriculum_steps = max(
            1, getattr(self.cfg.control, 'wheel_velocity_damping_curriculum_steps', 1200)
        )
        self.current_wheel_velocity_damping = (
            self.wheel_velocity_damping_start if self.wheel_velocity_damping_curriculum else self.wheel_velocity_damping
        )
        self.wheel_speed_normalization = getattr(
            self.cfg.control, 'wheel_speed_normalization', getattr(self.cfg.control, 'wheel_action_scale', 15.0)
        )

        # 有界偏航率闭环差动反馈 (阶段 22)
        self.wheel_yaw_damping = getattr(self.cfg.control, 'wheel_yaw_damping', 1.5)
        self.wheel_yaw_damping_start = getattr(self.cfg.control, 'wheel_yaw_damping_start', 0.0)
        self.wheel_yaw_torque_limit = getattr(self.cfg.control, 'wheel_yaw_torque_limit', 1.5)
        self.wheel_yaw_damping_curriculum = getattr(self.cfg.control, 'wheel_yaw_damping_curriculum', False)
        self.wheel_yaw_damping_curriculum_steps = max(
            1, getattr(self.cfg.control, 'wheel_yaw_damping_curriculum_steps', 1200)
        )
        self.current_wheel_yaw_damping = (
            self.wheel_yaw_damping_start if self.wheel_yaw_damping_curriculum else self.wheel_yaw_damping
        )

        # 阶段 24: 非转向轮速差闭环抑制 (电子差速/防滑纠偏)
        self.wheel_diff_damping = getattr(self.cfg.control, 'wheel_diff_damping', 0.10)
        self.wheel_diff_damping_start = getattr(self.cfg.control, 'wheel_diff_damping_start', 0.0)
        self.wheel_diff_torque_limit = getattr(self.cfg.control, 'wheel_diff_torque_limit', 1.5)
        self.wheel_diff_damping_curriculum = getattr(self.cfg.control, 'wheel_diff_damping_curriculum', False)
        self.wheel_diff_damping_curriculum_steps = max(
            1, getattr(self.cfg.control, 'wheel_diff_damping_curriculum_steps', 1200)
        )
        self.current_wheel_diff_damping = (
            self.wheel_diff_damping_start if self.wheel_diff_damping_curriculum else self.wheel_diff_damping
        )

        # 迁移期旧控制器参数 (用于混合过渡)
        self.wheel_action_scale = getattr(self.cfg.control, 'wheel_action_scale', 15.0)
        self.wheel_damping = getattr(self.cfg.control, 'wheel_damping', 1.2)
        self.wheel_damping_start = getattr(self.cfg.control, 'wheel_damping_start', self.wheel_damping)
        self.wheel_damping_curriculum = getattr(self.cfg.control, 'wheel_damping_curriculum', False)
        self.wheel_damping_curriculum_steps = max(
            1, getattr(self.cfg.control, 'wheel_damping_curriculum_steps', 1)
        )
        self.current_wheel_damping = (
            self.wheel_damping_start if self.wheel_damping_curriculum else self.wheel_damping
        )
        self.wheel_action_limit = getattr(self.cfg.control, 'wheel_action_limit', 1.0)
        self.wheel_action_limit_start = getattr(self.cfg.control, 'wheel_action_limit_start', self.wheel_action_limit)
        self.wheel_action_limit_curriculum = getattr(self.cfg.control, 'wheel_action_limit_curriculum', False)
        self.wheel_action_limit_curriculum_steps = max(
            1, getattr(self.cfg.control, 'wheel_action_limit_curriculum_steps', 1)
        )
        self.current_wheel_action_limit = (
            self.wheel_action_limit_start if self.wheel_action_limit_curriculum else self.wheel_action_limit
        )
        self.hip_scale_reduction = getattr(self.cfg.control, 'hip_scale_reduction', 0.5)
        self.knee_scale_reduction = getattr(self.cfg.control, 'knee_scale_reduction', 0.5)
        self.leg_target_limit_margin = getattr(self.cfg.control, 'leg_target_limit_margin', 0.02)

        # 阶段 23: 线速度跟踪课程 (2.0 -> 3.0，阶段 24/25 冻结在 3.0)
        self.tracking_lin_vel_start = getattr(self.cfg.rewards, 'tracking_lin_vel_start', 3.0)
        self.tracking_lin_vel_target = getattr(self.cfg.rewards.scales, 'tracking_lin_vel', 3.0)
        self.tracking_lin_vel_curriculum = getattr(self.cfg.rewards, 'tracking_lin_vel_curriculum', False)
        self.tracking_lin_vel_curriculum_steps = max(
            1, getattr(self.cfg.rewards, 'tracking_lin_vel_curriculum_steps', 1200)
        )
        if self.tracking_lin_vel_curriculum:
            self.reward_scales['tracking_lin_vel'] = self.tracking_lin_vel_start
        else:
            self.reward_scales['tracking_lin_vel'] = self.tracking_lin_vel_target

    def _init_state_machine_buffers(self):
        """
        阶段 4.3 六模态动态指令马尔可夫状态机初始化
        0: Stand (驻车自平衡)
        1: Inplace_Turn (原地差动转向)
        2: Fwd_Straight (前进直行巡航)
        3: Bwd_Straight (后退直行巡航)
        4: Fwd_Turn (前进复合转向)
        5: Bwd_Turn (后退复合转向)
        """
        self.command_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.task_type_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.yaw_direction = torch.where(
            torch.rand(self.num_envs, device=self.device) < 0.5,
            torch.tensor(1.0, device=self.device),
            torch.tensor(-1.0, device=self.device),
        )
        self._is_resetting = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # 6x6 状态转移概率矩阵 P(S_{t+1} | S_t)
        # Phase 14 Exp 3: 强化向模式 0 (Stand 驻车自平衡) 的转移概率，高频覆盖移动、转向后制动停车场景
        self.transition_matrix = torch.tensor([
            [0.35, 0.15, 0.15, 0.15, 0.10, 0.10],  # 0: Stand
            [0.30, 0.25, 0.10, 0.10, 0.15, 0.10],  # 1: Inplace_Turn -> 30% to Stand
            [0.30, 0.05, 0.25, 0.10, 0.25, 0.05],  # 2: Fwd_Straight -> 30% to Stand
            [0.30, 0.05, 0.10, 0.25, 0.05, 0.25],  # 3: Bwd_Straight -> 30% to Stand
            [0.30, 0.05, 0.20, 0.05, 0.25, 0.15],  # 4: Fwd_Turn -> 30% to Stand
            [0.30, 0.05, 0.05, 0.20, 0.15, 0.25],  # 5: Bwd_Turn -> 30% to Stand
        ], device=self.device, dtype=torch.float)

        # 初始平稳分布 (用于环境重置采样，35% 初始驻车)
        self.init_mode_distribution = torch.tensor(
            [0.35, 0.10, 0.15, 0.15, 0.13, 0.12],
            device=self.device, dtype=torch.float
        )

    def compute_leg_ik(self, H_target):
        """
        阶段 5.1 单腿矢状面解析逆运动学 (闭式矢量几何解)
        严格保证轮心前后位置 y_wheel 不动 (y_wheel = 0.11362779 m)。
        输入: H_target (标量或 (num_envs,) Tensor)
        输出: q1 (hip_joint), q2 (knee_joint) 弧度
        """
        dy1, dz1 = -0.29348091, -0.06220095
        dy2, dz2 = 0.2821087, -0.19553796
        y_hip, z_hip = 0.125, -0.070
        R_wheel = 0.070
        y_wheel_target = 0.11362779

        L1 = 0.30000000
        L2 = 0.34324976
        delta_beta = 2.326640924565626

        dY = y_wheel_target - y_hip
        dZ = (R_wheel - H_target) - z_hip
        D_sq = dY**2 + dZ**2

        cos_phi = torch.clamp((D_sq - L1**2 - L2**2) / (2.0 * L1 * L2), -1.0, 1.0)
        phi = torch.acos(cos_phi)

        pi = math.pi
        q2 = (phi - delta_beta + pi) % (2.0 * pi) - pi
        wx = dy1 + torch.cos(q2) * dy2 - torch.sin(q2) * dz2
        wz = dz1 + torch.sin(q2) * dy2 + torch.cos(q2) * dz2

        theta_w = torch.atan2(wz, wx)
        if isinstance(H_target, torch.Tensor):
            theta_des = torch.atan2(dZ, torch.full_like(dZ, dY))
        else:
            theta_des = math.atan2(dZ, dY)

        q1 = (theta_des - theta_w + pi) % (2.0 * pi) - pi
        return q1, q2

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if not hasattr(self, 'command_mode'):
            self._init_state_machine_buffers()
        if not hasattr(self, 'height_cmd_buf'):
            self.nominal_height = 0.39774
            self.height_cmd_buf = torch.full((self.num_envs,), self.nominal_height, device=self.device, dtype=torch.float)
        if not hasattr(self, 'target_height_buf'):
            self.target_height_buf = torch.full((self.num_envs,), self.nominal_height, device=self.device, dtype=torch.float)
        self.height_cmd_buf[env_ids] = self.nominal_height
        self.target_height_buf[env_ids] = self.nominal_height
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        self._is_resetting[env_ids] = True
        super().reset_idx(env_ids)
        self._is_resetting[env_ids] = False
        self.filtered_actions[env_ids] = 0.0
        self.requested_actions[env_ids] = 0.0
        self.last_filtered_actions[env_ids] = 0.0
        self.last_last_filtered_actions[env_ids] = 0.0
        self.position_targets[env_ids] = 0.0
        self.unclipped_torques[env_ids] = 0.0
        self.wheel_alloc_boundary_hit[env_ids] = False
        self.wheel_diff_clipped[env_ids] = False
        self.alpha_contact[env_ids] = 1.0

    def step(self, actions):
        """
        重写 step(): 在 decimation 循环之前一次性完成动作平滑。
        """
        # 保存前两步的滤波动作 (用于二阶 action_smoothness 奖励)
        self.last_last_filtered_actions[:] = self.last_filtered_actions
        self.last_filtered_actions[:] = self.filtered_actions

        # TITA 式动作平滑: 75% 当前策略请求 + 25% 上一步实际下发动作。
        # 轮端随后独立限幅，观测、奖励与执行器统一使用 filtered_actions。
        clip_actions = self.cfg.normalization.clip_actions
        clipped = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.requested_actions[:] = 0.75 * clipped + 0.25 * self.last_filtered_actions
        self.filtered_actions[:] = self.requested_actions
        if self.wheel_action_limit_curriculum:
            progress = min(1.0, self.common_step_counter / self.wheel_action_limit_curriculum_steps)
            self.current_wheel_action_limit = (
                self.wheel_action_limit_start
                + progress * (self.wheel_action_limit - self.wheel_action_limit_start)
            )
        else:
            self.current_wheel_action_limit = self.wheel_action_limit
        self.filtered_actions[:, self.wheel_indices] = torch.clamp(
            self.requested_actions[:, self.wheel_indices],
            -self.current_wheel_action_limit,
            self.current_wheel_action_limit,
        )

        if self.wheel_velocity_damping_curriculum:
            progress = min(1.0, self.common_step_counter / self.wheel_velocity_damping_curriculum_steps)
            self.current_wheel_velocity_damping = (
                self.wheel_velocity_damping_start
                + progress * (self.wheel_velocity_damping - self.wheel_velocity_damping_start)
            )
        else:
            self.current_wheel_velocity_damping = self.wheel_velocity_damping

        if self.wheel_torque_curriculum:
            self.wheel_torque_alpha = min(1.0, self.common_step_counter / self.wheel_torque_curriculum_steps)
        else:
            self.wheel_torque_alpha = 1.0

        if self.wheel_yaw_damping_curriculum:
            progress = min(1.0, self.common_step_counter / self.wheel_yaw_damping_curriculum_steps)
            self.current_wheel_yaw_damping = (
                self.wheel_yaw_damping_start
                + progress * (self.wheel_yaw_damping - self.wheel_yaw_damping_start)
            )
        else:
            self.current_wheel_yaw_damping = self.wheel_yaw_damping

        if self.tracking_lin_vel_curriculum:
            progress = min(1.0, self.common_step_counter / self.tracking_lin_vel_curriculum_steps)
            cur_scale = (
                self.tracking_lin_vel_start
                + progress * (self.tracking_lin_vel_target - self.tracking_lin_vel_start)
            )
            self.reward_scales['tracking_lin_vel'] = cur_scale * self.dt
        else:
            self.reward_scales['tracking_lin_vel'] = self.tracking_lin_vel_target * self.dt

        if self.wheel_diff_damping_curriculum:
            progress = min(1.0, self.common_step_counter / self.wheel_diff_damping_curriculum_steps)
            self.current_wheel_diff_damping = (
                self.wheel_diff_damping_start
                + progress * (self.wheel_diff_damping - self.wheel_diff_damping_start)
            )
        else:
            self.current_wheel_diff_damping = self.wheel_diff_damping

        # 阶段 14: 统一 4 cm/s (0.04 m/s) 机身高度平滑斜坡发生器
        if hasattr(self, 'target_height_buf'):
            elevation_speed = getattr(self.cfg.commands, 'elevation_speed', 0.04)
            dh = elevation_speed * self.dt
            diff_h = self.target_height_buf - self.height_cmd_buf
            self.height_cmd_buf += torch.clamp(diff_h, min=-dh, max=dh)
        else:
            self.target_height_buf = self.height_cmd_buf.clone()

        return super().step(actions)

    def _get_noise_scale_vec(self, cfg):
        """
        精确对齐 31 维观测向量的噪声分配
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.0  # commands
        noise_vec[12:18] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12 + self.left_wheel_idx] = 0.0
        noise_vec[12 + self.right_wheel_idx] = 0.0
        noise_vec[18:24] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[24:30] = 0.0  # actions
        noise_vec[30] = 0.0     # height command
        return noise_vec

    def compute_observations(self):
        """31 维观测: 车轮 dof_pos 置零，动作反馈使用 filtered_actions，追加归一化高度指令"""
        obs_dof_pos = (self.dof_pos - self.default_dof_pos).clone()
        obs_dof_pos[:, self.wheel_indices] = 0.0

        # 阶段 5.1: 归一化高度指令 (dim 31)
        # H_cmd 归一化: (H_cmd - 0.40) / 0.15
        norm_h = ((self.height_cmd_buf - 0.40) / 0.15).unsqueeze(-1)

        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            obs_dof_pos * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.filtered_actions,
            norm_h,
        ), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _compute_torques(self, actions):
        """
        计算各关节力矩 (滤波已在 step() 中完成):
        - 腿部 PD 位置控制 (髋关节应用 hip_scale_reduction 软化，消除机身俯仰抖动)
        - 轮子速度闭环 (高响应大轮速覆盖)
        """
        actions_scaled = self.filtered_actions * self.cfg.control.action_scale
        # 髋关节软化 (hip_scale_reduction = 0.7) 与膝关节缩放 (knee_scale_reduction = 0.5)，平抑机身前后俯仰抖动与动态自主深蹲起伏
        actions_scaled[:, self.left_hip_idx] *= self.hip_scale_reduction
        actions_scaled[:, self.right_hip_idx] *= self.hip_scale_reduction
        actions_scaled[:, self.left_knee_idx] *= self.knee_scale_reduction
        actions_scaled[:, self.right_knee_idx] *= self.knee_scale_reduction

        torques = torch.zeros_like(actions)

        # 阶段 5.1: 解析 IK 前馈 + PPO 残差控制
        q1_ik, q2_ik = self.compute_leg_ik(self.height_cmd_buf)
        ik_targets = {
            self.left_hip_idx: q1_ik,
            self.right_hip_idx: q1_ik,
            self.left_knee_idx: q2_ik,
            self.right_knee_idx: q2_ik,
        }

        # 腿部 PD 位置控制。策略可能输出超过关节可达范围的动作；将实际位置
        # 目标裁剪在 URDF 硬限位以内，避免长期力矩饱和和撞击机械限位。
        for idx in [self.left_hip_idx, self.left_knee_idx,
                    self.right_hip_idx, self.right_knee_idx]:
            target = actions_scaled[:, idx] + ik_targets[idx] + self.default_dof_pos[:, idx]
            lower = self.dof_pos_limits[idx, 0] + self.leg_target_limit_margin
            upper = self.dof_pos_limits[idx, 1] - self.leg_target_limit_margin
            target = torch.clamp(target, min=lower, max=upper)
            self.position_targets[:, idx] = target
            torques[:, idx] = (self.p_gains[idx] * (target - self.dof_pos[:, idx])
                             - self.d_gains[idx] * self.dof_vel[:, idx])

        # 轮端控制 (阶段 19: 直接前馈转矩 + 有界弱转速阻尼)
        # τ = 7.0 * action - clip(Kd * ω, -2.0, 2.0)
        for idx in [self.left_wheel_idx, self.right_wheel_idx]:
            act = self.filtered_actions[:, idx]
            tau_feedforward = self.wheel_torque_scale * act
            damping_raw = self.current_wheel_velocity_damping * self.dof_vel[:, idx]
            tau_damping = torch.clamp(
                damping_raw,
                -self.wheel_velocity_damping_limit,
                self.wheel_velocity_damping_limit,
            )
            tau_direct = tau_feedforward - tau_damping

            if self.wheel_torque_curriculum:
                tau_old = self.current_wheel_damping * (act * self.wheel_action_scale - self.dof_vel[:, idx])
                torques[:, idx] = (1.0 - self.wheel_torque_alpha) * tau_old + self.wheel_torque_alpha * tau_direct
            else:
                torques[:, idx] = tau_direct

        # 保存施加差动校正前的基准力矩 (前馈 + 单轮速度阻尼)
        base_tau_L = torques[:, self.left_wheel_idx].clone()
        base_tau_R = torques[:, self.right_wheel_idx].clone()

        # 阶段 22: 有界偏航率闭环差动转矩反馈
        # e_yaw = yaw_cmd - yaw_actual
        # Δτ_yaw = clip(Kyaw * e_yaw, -limit, limit)
        # 左轮 (+X) 前向推力诱发负偏航 (-Z), 右轮 (-X) 前向推力诱发正偏航 (+Z)
        yaw_rate_error = self.commands[:, 2] - self.base_ang_vel[:, 2]
        delta_tau_yaw = torch.clamp(
            self.current_wheel_yaw_damping * yaw_rate_error,
            -self.wheel_yaw_torque_limit,
            self.wheel_yaw_torque_limit,
        )

        # 阶段 24: 非转向工况下的轮速差闭环抑制 (电子差速/防滑纠偏)
        # e_omega = omega_L - omega_R
        # Δτ_omega = clip(K_omega * e_omega, -limit, limit)
        # 仅在非转向 (|yaw_cmd| < 0.1) 时生效：左轮飞转则对左轮减扭刹车、向右轮补扭
        is_straight_cmd = (torch.abs(self.commands[:, 2]) < 0.1).float()
        wheel_speed_error = self.dof_vel[:, self.left_wheel_idx] - self.dof_vel[:, self.right_wheel_idx]
        delta_tau_omega = torch.clamp(
            self.current_wheel_diff_damping * wheel_speed_error,
            -self.wheel_diff_torque_limit,
            self.wheel_diff_torque_limit,
        ) * is_straight_cmd

        # 阶段 27: 差动优先力矩分配 (Differential-Priority Torque Allocation)
        # tau_c: 共模推进力矩
        # tau_d: 期望差动力矩 (tau_R 增扭、tau_L 减扭)
        tau_c = 0.5 * (base_tau_L + base_tau_R)
        tau_d = 0.5 * (base_tau_R - base_tau_L) + delta_tau_yaw + delta_tau_omega

        # 阶段 28: 接触感知共模牵引降额 (Contact-Aware Common-Mode Traction Derating)
        if self.enable_cacm_td:
            # 1. 提取垂直接触法向力 F_N (world Z)
            fn_L = torch.clamp(self.contact_forces[:, self.feet_indices[0], 2], min=0.0)
            fn_R = torch.clamp(self.contact_forces[:, self.feet_indices[1], 2], min=0.0)

            # 2. 归一化接触载荷比 [0.0, 1.0]
            alpha_raw_L = torch.clamp((fn_L - self.fn_off) / (self.fn_ref - self.fn_off), 0.0, 1.0)
            alpha_raw_R = torch.clamp((fn_R - self.fn_off) / (self.fn_ref - self.fn_off), 0.0, 1.0)

            # 3. 非对称低通滤波与接触迟滞更新
            rate_L = torch.where(alpha_raw_L < self.alpha_contact[:, 0], self.alpha_lpf_drop, self.alpha_lpf_rise)
            rate_R = torch.where(alpha_raw_R < self.alpha_contact[:, 1], self.alpha_lpf_drop, self.alpha_lpf_rise)
            self.alpha_contact[:, 0] = (1.0 - rate_L) * self.alpha_contact[:, 0] + rate_L * alpha_raw_L
            self.alpha_contact[:, 1] = (1.0 - rate_R) * self.alpha_contact[:, 1] + rate_R * alpha_raw_R

            # 4. 直行时以较差一侧载荷决定两轮共同可用推进量
            gamma_common = torch.minimum(self.alpha_contact[:, 0], self.alpha_contact[:, 1])
            gamma_alloc = is_straight_cmd * gamma_common + (1.0 - is_straight_cmd) * 1.0

            # 5. 仅对前向推进力矩 (tau_c < 0) 进行降额，反向制动不削弱
            tau_c_prop = torch.clamp(tau_c, max=0.0)
            tau_c_brake = torch.clamp(tau_c, min=0.0)
            tau_c_derated = gamma_alloc * tau_c_prop + tau_c_brake

            # 6. 阶段 28.1: 同步对差动项实施接触降额，消灭失载时承载轮单侧推进力偶
            tau_d_derated = gamma_alloc * tau_d if self.cacm_derate_diff else tau_d
        else:
            tau_c_derated = tau_c
            tau_d_derated = tau_d

        # 电机物理极限与差动优先共模动态裁剪
        # 当需要差动纠偏时，动态收拢共模推进力矩上限，彻底杜绝单侧硬件截断
        tau_limit = self.torque_limits[self.left_wheel_idx]
        tau_d_clamped = torch.clamp(tau_d_derated, -tau_limit, tau_limit)
        tau_c_limit = torch.clamp(tau_limit - torch.abs(tau_d_clamped), min=0.0)
        tau_c_allocated = torch.clamp(tau_c_derated, -tau_c_limit, tau_c_limit)

        tau_out_L = tau_c_allocated - tau_d_clamped
        tau_out_R = tau_c_allocated + tau_d_clamped

        # 阶段 28.1: 双向对称限扭，杜绝失载轮反转空转
        if self.enable_cacm_td:
            is_fwd_cmd = (self.commands[:, 1] >= 0.0).float()
            is_rev_cmd = (self.commands[:, 1] < 0.0).float()

            # 前向巡航 (cmd_v >= 0) 时禁止正向力矩 (倒车反转)
            tau_upper_L = torch.where(is_fwd_cmd > 0.5, self.alpha_contact[:, 0] * tau_limit, tau_limit)
            tau_upper_R = torch.where(is_fwd_cmd > 0.5, self.alpha_contact[:, 1] * tau_limit, tau_limit)
            # 倒车巡航 (cmd_v < 0) 时禁止负向力矩 (正向前转)
            tau_lower_L = torch.where(is_rev_cmd > 0.5, -self.alpha_contact[:, 0] * tau_limit, -tau_limit)
            tau_lower_R = torch.where(is_rev_cmd > 0.5, -self.alpha_contact[:, 1] * tau_limit, -tau_limit)

            tau_out_L = torch.clamp(tau_out_L, min=tau_lower_L, max=tau_upper_L)
            tau_out_R = torch.clamp(tau_out_R, min=tau_lower_R, max=tau_upper_R)

        torques[:, self.left_wheel_idx] = tau_out_L
        torques[:, self.right_wheel_idx] = tau_out_R

        self.unclipped_torques[:] = torques
        wheel_limits = self.torque_limits[self.wheel_indices]

        # 阶段 28 语义修正: 分配边界命中判定 (两轮实际输出力矩天然位于 [-tau_limit, tau_limit] 内，
        # 当需要大差动纠偏时，某轮主动达到边界，另一轮对称退让以满足 2*tau_d)
        hit_L = torch.abs(torques[:, self.left_wheel_idx]) >= (wheel_limits[0] - 1.0e-3)
        hit_R = torch.abs(torques[:, self.right_wheel_idx]) >= (wheel_limits[1] - 1.0e-3)
        self.wheel_alloc_boundary_hit = hit_L | hit_R
        self.wheel_diff_clipped = self.wheel_alloc_boundary_hit  # 兼容旧代码引用
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def check_termination(self):
        """
        终止判定:
        1. 非轮子部位触地 -> 重置
        2. 严重倾斜 (>45°) -> 重置
        3. 机身过低 (<0.25m) -> 重置
        4. 超时
        """
        if len(self.termination_contact_indices) > 0:
            contact_reset = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        else:
            contact_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        tilt_reset = self.projected_gravity[:, 2] > -0.70
        height_reset = self.root_states[:, 2] < 0.25

        self.reset_buf = contact_reset | tilt_reset | height_reset
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def _resample_commands(self, env_ids):
        """
        阶段 4.3: 六模态动态指令马尔可夫状态机采样与偏航方向管理
        0: Stand (驻车自平衡)
        1: Inplace_Turn (原地偏航转向)
        2: Fwd_Straight (前进直行巡航)
        3: Bwd_Straight (后退直行巡航)
        4: Fwd_Turn (前进复合转向)
        5: Bwd_Turn (后退复合转向)
        """
        if len(env_ids) == 0:
            return

        if not hasattr(self, 'command_mode'):
            self._init_state_machine_buffers()

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # 区分重置环境与运行中步进切换环境
        reset_mask = self._is_resetting[env_ids]
        curr_modes = self.command_mode[env_ids]
        next_modes = torch.zeros_like(curr_modes)

        # 1. 采样 next_mode
        # 重置环境: 按平稳分布独立采样
        if reset_mask.any():
            reset_ids = env_ids[reset_mask]
            n_rst = len(reset_ids)
            rst_probs = self.init_mode_distribution.unsqueeze(0).expand(n_rst, -1)
            next_modes[reset_mask] = torch.multinomial(rst_probs, num_samples=1).squeeze(1)
            # 偏航方向随机初始化 50% 左转 / 50% 右转
            self.yaw_direction[reset_ids] = torch.where(
                torch.rand(n_rst, device=self.device) < 0.5,
                torch.tensor(1.0, device=self.device),
                torch.tensor(-1.0, device=self.device),
            )

        # 运行中切换环境: 按转移矩阵行概率采样
        step_mask = ~reset_mask
        if step_mask.any():
            step_ids = env_ids[step_mask]
            n_stp = len(step_ids)
            curr_step_modes = curr_modes[step_mask]
            step_probs = self.transition_matrix[curr_step_modes]
            step_next = torch.multinomial(step_probs, num_samples=1).squeeze(1)
            next_modes[step_mask] = step_next

            # 偏航方向动态反转与维护逻辑:
            # (1) 原地转向自转移 (1 -> 1): 50% 强制偏航符号反转 (Task B 原地左<->右), 50% 保持稳态同向
            mask_11 = (curr_step_modes == 1) & (step_next == 1)
            if mask_11.any():
                ids_11 = step_ids[mask_11]
                flip_mask = torch.rand(len(ids_11), device=self.device) < 0.5
                self.yaw_direction[ids_11[flip_mask]] *= -1.0

            # (2) 前进复合转向自转移 (4 -> 4): 50% 偏航符号反转 (Task B 前进 S 弯), 50% 保持同向稳态
            mask_44 = (curr_step_modes == 4) & (step_next == 4)
            if mask_44.any():
                ids_44 = step_ids[mask_44]
                flip_mask = torch.rand(len(ids_44), device=self.device) < 0.5
                self.yaw_direction[ids_44[flip_mask]] *= -1.0

            # (3) 后退复合转向自转移 (5 -> 5): 50% 偏航符号反转 (Task B 倒车 S 弯), 50% 保持同向稳态
            mask_55 = (curr_step_modes == 5) & (step_next == 5)
            if mask_55.any():
                ids_55 = step_ids[mask_55]
                flip_mask = torch.rand(len(ids_55), device=self.device) < 0.5
                self.yaw_direction[ids_55[flip_mask]] *= -1.0

            # (4) 从非转向状态 (0, 2, 3) 进入转向状态 (1, 4, 5): 重新随机分配 50% 左转 / 50% 右转
            mask_enter_turn = ((curr_step_modes == 0) | (curr_step_modes == 2) | (curr_step_modes == 3)) & \
                              ((step_next == 1) | (step_next == 4) | (step_next == 5))
            if mask_enter_turn.any():
                ids_et = step_ids[mask_enter_turn]
                self.yaw_direction[ids_et] = torch.where(
                    torch.rand(len(ids_et), device=self.device) < 0.5,
                    torch.tensor(1.0, device=self.device),
                    torch.tensor(-1.0, device=self.device),
                )

            # (5) 前后转向互换 (4 -> 5 或 5 -> 4): 70% 维持原转向方向 (纯粹纵向换向), 30% 转向同时反向
            mask_45 = ((curr_step_modes == 4) & (step_next == 5)) | ((curr_step_modes == 5) & (step_next == 4))
            if mask_45.any():
                ids_45 = step_ids[mask_45]
                flip_mask = torch.rand(len(ids_45), device=self.device) < 0.3
                self.yaw_direction[ids_45[flip_mask]] *= -1.0

        # 更新状态机当前状态
        self.command_mode[env_ids] = next_modes

        # 2. 生成具体线速度与偏航角速度指令
        # lateral velocity (x) 严格置零 (仅训练纵向与偏航差动)
        self.commands[env_ids, 0] = 0.0
        if self.commands.shape[1] > 3:
            self.commands[env_ids, 3] = 0.0

        n = len(env_ids)
        # 纵向速度幅值采样: [0.10, 0.30] m/s
        v_rand = 0.10 + 0.20 * torch.rand(n, device=self.device)
        # 偏航角速度幅值采样: [0.20, 0.50] rad/s (避开死区，覆盖 0.35 标称工作点)
        w_rand = 0.20 + 0.30 * torch.rand(n, device=self.device)
        yaw_dirs = self.yaw_direction[env_ids]

        # 模式 0: Stand (v_y = 0.0, w_z = 0.0)
        m0 = (next_modes == 0)
        if m0.any():
            ids0 = env_ids[m0]
            self.commands[ids0, 1] = 0.0
            self.commands[ids0, 2] = 0.0

        # 模式 1: Inplace_Turn (v_y = 0.0, w_z = yaw_dir * w_rand)
        m1 = (next_modes == 1)
        if m1.any():
            ids1 = env_ids[m1]
            self.commands[ids1, 1] = 0.0
            self.commands[ids1, 2] = (yaw_dirs * w_rand)[m1]

        # 模式 2: Fwd_Straight (v_y = +v_rand, w_z = 0.0)
        m2 = (next_modes == 2)
        if m2.any():
            ids2 = env_ids[m2]
            self.commands[ids2, 1] = v_rand[m2]
            self.commands[ids2, 2] = 0.0

        # 模式 3: Bwd_Straight (v_y = -v_rand, w_z = 0.0)
        m3 = (next_modes == 3)
        if m3.any():
            ids3 = env_ids[m3]
            self.commands[ids3, 1] = -v_rand[m3]
            self.commands[ids3, 2] = 0.0

        # 模式 4: Fwd_Turn (v_y = +v_rand, w_z = yaw_dir * w_rand)
        m4 = (next_modes == 4)
        if m4.any():
            ids4 = env_ids[m4]
            self.commands[ids4, 1] = v_rand[m4]
            self.commands[ids4, 2] = (yaw_dirs * w_rand)[m4]

        # 模式 5: Bwd_Turn (v_y = -v_rand, w_z = yaw_dir * w_rand)
        m5 = (next_modes == 5)
        if m5.any():
            ids5 = env_ids[m5]
            self.commands[ids5, 1] = -v_rand[m5]
            self.commands[ids5, 2] = (yaw_dirs * w_rand)[m5]

        # Phase 14 Exp 3 针对性驻车微调样本分配:
        # 30% 原有高速巡航与倒车 (保持 v5 0.50 m/s 巡航与 -0.30 m/s 倒车防退化)
        # 35% 不同高度下零速驻车 (重点覆盖 0.30, 0.32, 0.48, 0.50 m)
        # 20% 移动、转向后制动并保持高度
        # 10% 原有移动变高
        # 5%  高度边界保持
        r_task = torch.rand(n, device=self.device)
        task_cruise_v5 = (r_task < 0.30)
        task_stand_h   = (r_task >= 0.30) & (r_task < 0.65)
        task_brake_hold = (r_task >= 0.65) & (r_task < 0.85)
        task_motion_h  = (r_task >= 0.85) & (r_task < 0.95)
        task_boundary_hold = (r_task >= 0.95)

        h_nom = torch.tensor(self.nominal_height, device=self.device)
        fwd_speed_max = getattr(self.cfg.commands, 'fwd_speed_max', 0.50)
        bwd_speed_max = getattr(self.cfg.commands, 'bwd_speed_max', 0.35)
        min_h = 0.30
        max_h = 0.50

        # 1. 任务 1 (30%): 原有高速巡航与倒车 (保持 Champion v5 0.50 m/s 巡航与倒车)
        if task_cruise_v5.any():
            ids_m = env_ids[task_cruise_v5]
            n_m = len(ids_m)
            self.target_height_buf[ids_m] = self.nominal_height
            self.task_type_buf[ids_m] = 4

            r_submode = torch.rand(n_m, device=self.device)
            is_fwd_straight = (r_submode < 0.60)
            is_bwd_straight = (r_submode >= 0.60) & (r_submode < 0.80)
            is_turn_motion = (r_submode >= 0.80)

            y_d = self.yaw_direction[ids_m]

            # 前向直行高速巡航: 60% 定焦 [0.47, 0.50], 20% [0.35, 0.45], 20% [0.20, 0.35]
            if is_fwd_straight.any():
                ids_fwd = ids_m[is_fwd_straight]
                n_fwd = len(ids_fwd)
                r_fwd_split = torch.rand(n_fwd, device=self.device)
                v_fwd_low = 0.20 + 0.15 * torch.rand(n_fwd, device=self.device)
                v_fwd_mid = 0.35 + 0.10 * torch.rand(n_fwd, device=self.device)
                v_fwd_target = 0.47 + 0.03 * torch.rand(n_fwd, device=self.device)
                v_fwd = torch.where(
                    r_fwd_split < 0.20,
                    v_fwd_low,
                    torch.where(r_fwd_split < 0.40, v_fwd_mid, v_fwd_target)
                )
                self.commands[ids_fwd, 1] = v_fwd
                self.commands[ids_fwd, 2] = 0.0
                self.command_mode[ids_fwd] = 2 # Fwd Straight

            # 后退直行巡航: 覆盖 [-bwd_speed_max, -0.15]
            if is_bwd_straight.any():
                ids_bwd = ids_m[is_bwd_straight]
                n_bwd = len(ids_bwd)
                v_bwd = -(0.15 + (bwd_speed_max - 0.15) * torch.rand(n_bwd, device=self.device))
                self.commands[ids_bwd, 1] = v_bwd
                self.commands[ids_bwd, 2] = 0.0
                self.command_mode[ids_bwd] = 3 # Bwd Straight

            # 转向与复合运动 (模式 1, 4, 5)
            if is_turn_motion.any():
                ids_trn = ids_m[is_turn_motion]
                n_trn = len(ids_trn)
                sub_modes = torch.randint(0, 3, (n_trn,), device=self.device)
                w_t = 0.20 + 0.15 * torch.rand(n_trn, device=self.device)
                v_t = 0.10 + 0.15 * torch.rand(n_trn, device=self.device)
                y_t = y_d[is_turn_motion]

                m_in = (sub_modes == 0)
                if m_in.any():
                    self.commands[ids_trn[m_in], 1] = 0.0
                    self.commands[ids_trn[m_in], 2] = (y_t * w_t)[m_in]
                    self.command_mode[ids_trn[m_in]] = 1

                m_ft = (sub_modes == 1)
                if m_ft.any():
                    self.commands[ids_trn[m_ft], 1] = v_t[m_ft]
                    self.commands[ids_trn[m_ft], 2] = (y_t * w_t)[m_ft]
                    self.command_mode[ids_trn[m_ft]] = 4

                m_bt = (sub_modes == 2)
                if m_bt.any():
                    self.commands[ids_trn[m_bt], 1] = -v_t[m_bt]
                    self.commands[ids_trn[m_bt], 2] = (y_t * w_t)[m_bt]
                    self.command_mode[ids_trn[m_bt]] = 5

        # 2. 任务 2 (35%): 不同高度下零速驻车 (重点覆盖 0.30, 0.32, 0.48, 0.50 m)
        if task_stand_h.any():
            ids_sh = env_ids[task_stand_h]
            n_sh = len(ids_sh)
            r_sub_h = torch.rand(n_sh, device=self.device)

            # 25% 0.30m, 25% 0.32m, 25% 0.48m, 25% 0.50m
            h_030 = 0.300 + 0.005 * torch.rand(n_sh, device=self.device)
            h_032 = 0.320 + 0.005 * torch.rand(n_sh, device=self.device)
            h_048 = 0.480 - 0.005 * torch.rand(n_sh, device=self.device)
            h_050 = 0.500 - 0.005 * torch.rand(n_sh, device=self.device)
            h_sampled = torch.where(
                r_sub_h < 0.25, h_030,
                torch.where(r_sub_h < 0.50, h_032,
                torch.where(r_sub_h < 0.75, h_048, h_050))
            )

            # 应用与遥控一致的补偿规则
            delta_h = torch.clamp(0.08 * (h_sampled - 0.39774), min=-0.010, max=0.010)
            self.target_height_buf[ids_sh] = h_sampled + delta_h

            self.commands[ids_sh, 1] = 0.0
            self.commands[ids_sh, 2] = 0.0
            self.command_mode[ids_sh] = 0 # Stand (驻车自平衡)
            self.task_type_buf[ids_sh] = 3

        # 3. 任务 3 (20%): 移动、转向后制动并保持高度
        if task_brake_hold.any():
            ids_bh = env_ids[task_brake_hold]
            n_bh = len(ids_bh)
            r_sub_bh = torch.rand(n_bh, device=self.device)

            # 高度分布: 20% 0.30m, 25% 0.32m, 10% 0.398m, 25% 0.48m, 20% 0.50m
            h_030_b = 0.300 + 0.005 * torch.rand(n_bh, device=self.device)
            h_032_b = 0.320 + 0.005 * torch.rand(n_bh, device=self.device)
            h_nom_b = torch.full((n_bh,), self.nominal_height, device=self.device)
            h_048_b = 0.480 - 0.005 * torch.rand(n_bh, device=self.device)
            h_050_b = 0.500 - 0.005 * torch.rand(n_bh, device=self.device)
            h_bh = torch.where(
                r_sub_bh < 0.20, h_030_b,
                torch.where(r_sub_bh < 0.45, h_032_b,
                torch.where(r_sub_bh < 0.55, h_nom_b,
                torch.where(r_sub_bh < 0.80, h_048_b, h_050_b)))
            )

            delta_h_b = torch.clamp(0.08 * (h_bh - 0.39774), min=-0.010, max=0.010)
            self.target_height_buf[ids_bh] = h_bh + delta_h_b

            # 制动归零指令
            self.commands[ids_bh, 1] = 0.0
            self.commands[ids_bh, 2] = 0.0
            self.command_mode[ids_bh] = 0 # Stand (制动驻车)
            self.task_type_buf[ids_bh] = 0

        # 4. 任务 4 (10%): 原有移动变高 (直行变高 50% + 复合转向变高 50%)
        if task_motion_h.any():
            ids_mh = env_ids[task_motion_h]
            n_mh = len(ids_mh)
            is_turn_mh = (torch.rand(n_mh, device=self.device) < 0.50)

            # 高度分布: 0.32m ~ 0.48m
            h_low_mh = 0.320 + 0.020 * torch.rand(n_mh, device=self.device)
            h_high_mh = 0.480 - 0.020 * torch.rand(n_mh, device=self.device)
            h_mh = torch.where(torch.rand(n_mh, device=self.device) < 0.50, h_low_mh, h_high_mh)
            delta_h_mh = torch.clamp(0.08 * (h_mh - 0.39774), min=-0.010, max=0.010)
            self.target_height_buf[ids_mh] = h_mh + delta_h_mh

            # 4.1 直行变高
            is_straight_mh = ~is_turn_mh
            if is_straight_mh.any():
                ids_smh = ids_mh[is_straight_mh]
                n_smh = len(ids_smh)
                is_fwd_smh = (torch.rand(n_smh, device=self.device) < 0.50)
                v_fwd_smh = 0.15 + 0.15 * torch.rand(n_smh, device=self.device)
                v_bwd_smh = -(0.15 + 0.10 * torch.rand(n_smh, device=self.device))
                self.commands[ids_smh, 1] = torch.where(is_fwd_smh, v_fwd_smh, v_bwd_smh)
                self.commands[ids_smh, 2] = 0.0
                self.command_mode[ids_smh] = torch.where(is_fwd_smh, 2, 3)

            # 4.2 转向变高
            if is_turn_mh.any():
                ids_tmh = ids_mh[is_turn_mh]
                n_tmh = len(ids_tmh)
                v_tmh = 0.10 + 0.10 * torch.rand(n_tmh, device=self.device)
                w_tmh = 0.15 + 0.15 * torch.rand(n_tmh, device=self.device)
                w_dir_tmh = torch.where(torch.rand(n_tmh, device=self.device) < 0.5, 1.0, -1.0)
                is_fwd_tmh = (torch.rand(n_tmh, device=self.device) < 0.5)

                self.commands[ids_tmh, 1] = torch.where(is_fwd_tmh, v_tmh, -v_tmh)
                self.commands[ids_tmh, 2] = w_dir_tmh * w_tmh
                self.command_mode[ids_tmh] = torch.where(is_fwd_tmh, 4, 5)

            self.task_type_buf[ids_mh] = 1

        # 5. 任务 5 (5%): 高度边界保持 (两端持续保持与极值稳定性)
        if task_boundary_hold.any():
            ids_b = env_ids[task_boundary_hold]
            n_b = len(ids_b)
            # 50% 下界极值 0.30m, 50% 上界极值 0.50m
            is_lower_b = (torch.rand(n_b, device=self.device) < 0.50)
            target_h_b = torch.where(is_lower_b, torch.tensor(0.300, device=self.device), torch.tensor(0.500, device=self.device))
            delta_h_b = torch.clamp(0.08 * (target_h_b - 0.39774), min=-0.010, max=0.010)

            # 70% 静止平衡保持, 30% 缓速直行保持
            is_still_b = (torch.rand(n_b, device=self.device) < 0.70)
            v_slow_b = 0.08 * (2 * torch.rand(n_b, device=self.device) - 1.0) # [-0.08, 0.08] m/s

            self.commands[ids_b, 1] = torch.where(is_still_b, torch.zeros_like(v_slow_b), v_slow_b)
            self.commands[ids_b, 2] = 0.0
            self.target_height_buf[ids_b] = target_h_b + delta_h_b
            self.command_mode[ids_b] = torch.where(is_still_b, 0, torch.where(v_slow_b >= 0, 2, 3))
            self.task_type_buf[ids_b] = 5

    # ==================== 奖励函数 ====================

    def _reward_wheel_contact(self):
        """双轮同时着地奖励"""
        if len(self.feet_indices) > 0:
            contacts = self.contact_forces[:, self.feet_indices, 2] > 1.0
            return (torch.sum(contacts.float(), dim=-1) == len(self.feet_indices)).float()
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_leg_symmetry(self):
        """
        左右腿关节角镜像对称惩罚 (L1 + L2 混合惩罚):
        使用 L1 绝对值项消除小角度死区，强迫策略消除左右腿前后剪刀步代偿。
        """
        hip_abs = torch.abs(self.dof_pos[:, self.left_hip_idx] - self.dof_pos[:, self.right_hip_idx])
        knee_abs = torch.abs(self.dof_pos[:, self.left_knee_idx] - self.dof_pos[:, self.right_knee_idx])
        return hip_abs + knee_abs + 0.5 * (torch.square(hip_abs) + torch.square(knee_abs))

    def _reward_orientation(self):
        """
        机身前后俯仰保平惩罚 (带 2.5° 工程水平死区):
        在 [-2.5°, +2.5°] (0.044 rad) 工程水平平台内不扣分，为巡航释放必要的微小动力学前倾呼吸裕量，
        彻底消灭为保平而主动失速(0.40->0.10 m/s)以及临界点突发急救暴冲(Panic Kick)抽搐；
        超出 2.5° 后施加 L1 + L2 刚性约束，驱动髋关节主动做云台代偿保持箱体水平。
        """
        pitch_error = torch.clamp(torch.abs(self.projected_gravity[:, 1]) - 0.044, min=0.0)
        return pitch_error + 2.0 * torch.square(pitch_error)

    def _reward_pitch_rate(self):
        """
        矢状面俯仰角速度阻尼 (绕 X 轴旋转速度):
        抑制机身前后俯冲与回正时的剧烈角速度震荡，消除抽搐顿挫感。
        """
        return torch.square(self.base_ang_vel[:, 0])

    def _reward_roll(self):
        """惩罚机身横滚侧倾"""
        return torch.square(self.projected_gravity[:, 0])

    def _reward_stand_still(self):
        """静止时惩罚漂移 (升级为 L1 绝对值惩罚，消除 0.2m/s 死区)"""
        zero_cmd = (torch.norm(self.commands[:, :2], dim=1) < 0.1).float()
        return torch.abs(self.base_lin_vel[:, 1]) * zero_cmd

    def _reward_stand_still_wheel(self):
        """静止时惩罚车轮旋转 (锁定驱动轮，彻底切断前后摇摆振荡；非零偏航时不触发)"""
        zero_cmd = ((torch.norm(self.commands[:, :2], dim=1) < 0.1) & (torch.abs(self.commands[:, 2]) < 0.1)).float()
        wheel_vel_sq = torch.square(self.dof_vel[:, self.left_wheel_idx]) + torch.square(self.dof_vel[:, self.right_wheel_idx])
        return wheel_vel_sq * zero_cmd

    def _reward_wheel_symmetry(self):
        """直行时无量纲左右轮速同步惩罚"""
        straight_cmd = (torch.abs(self.commands[:, 2]) < 0.1).float()
        wheel_diff = torch.square(
            (self.dof_vel[:, self.left_wheel_idx] - self.dof_vel[:, self.right_wheel_idx])
            / self.wheel_speed_normalization
        )
        return wheel_diff * straight_cmd

    def _reward_wheel_action_symmetry(self):
        """非转向工况下惩罚左右轮动作差 (禁止双轮反向对推撕裂)"""
        straight_cmd = (torch.abs(self.commands[:, 2]) < 0.1).float()
        action_diff = 0.5 * (self.filtered_actions[:, self.left_wheel_idx] - self.filtered_actions[:, self.right_wheel_idx])
        return torch.square(action_diff) * straight_cmd

    def _reward_wheel_effort(self):
        """仅惩罚轮端能耗，并按各轮电机物理力矩限幅归一化"""
        wheel_torques = self.torques[:, self.wheel_indices]
        wheel_limits = self.torque_limits[self.wheel_indices].unsqueeze(0)
        return torch.mean(torch.square(wheel_torques / wheel_limits), dim=1)

    def _reward_wheel_saturation(self):
        """随预限幅轮端力矩连续增长、但保持有界的饱和惩罚"""
        requested = torch.abs(self.unclipped_torques[:, self.wheel_indices])
        wheel_limits = self.torque_limits[self.wheel_indices].unsqueeze(0)
        saturation = torch.clamp((requested / wheel_limits - 0.85) / 0.15, 0.0, 4.0)
        return torch.mean(saturation + 0.1 * torch.square(saturation), dim=1)

    def _reward_wheel_action_excess(self):
        """惩罚轮动作在执行器有效区间外的请求，促使策略均值回到 [-1, 1]"""
        excess = torch.clamp(
            torch.abs(self.requested_actions[:, self.wheel_indices]) - self.wheel_action_limit,
            min=0.0,
            max=2.0,
        )
        return torch.mean(torch.square(excess), dim=1)

    def _reward_action_rate(self):
        """一阶动作平滑惩罚"""
        return torch.sum(torch.square(self.filtered_actions - self.last_filtered_actions), dim=1)

    def _reward_action_smoothness(self):
        """二阶动作平滑惩罚 (加速度惩罚，彻底切断抽搐震颤)"""
        return torch.sum(torch.square(self.filtered_actions - 2 * self.last_filtered_actions + self.last_last_filtered_actions), dim=1)

    def _reward_leg_vel(self):
        """腿部关节速度惩罚 (不含轮子)"""
        leg_vel_sq = torch.zeros(self.num_envs, device=self.device)
        for idx in [self.left_hip_idx, self.left_knee_idx,
                    self.right_hip_idx, self.right_knee_idx]:
            leg_vel_sq += torch.square(self.dof_vel[:, idx])
        return leg_vel_sq

    def _reward_stand_nice(self):
        """静止时鼓励腿部保持标准站立角度"""
        return (torch.sum(torch.abs(self.dof_pos[:, self.leg_indices] - self.default_dof_pos[:, self.leg_indices]), dim=1)
                * (torch.norm(self.commands[:, :2], dim=1) < 0.1))

    def _reward_base_height(self):
        """
        惩罚机身高度偏离动态目标值 (带可配几何死区的 L1 锁高):
        在 [H_cmd - deadzone, H_cmd + deadzone]m 之间允许自然几何微沉降，不产生罚分；
        一旦突破死区，施加严厉 L1 梯度强力拉回。
        """
        deadzone = getattr(self.cfg.rewards, 'height_deadzone', 0.015)
        base_height = self.root_states[:, 2]
        height_error = torch.clamp(torch.abs(base_height - self.height_cmd_buf) - deadzone, min=0.0)
        return height_error

    def _reward_wheel_throttle_margin(self):
        """阶段 27 撤销节流软裕度惩罚，低层由差动优先力矩分配器硬保障"""
        return torch.zeros(self.num_envs, device=self.device)

