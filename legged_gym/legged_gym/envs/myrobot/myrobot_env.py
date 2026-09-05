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

import torch
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs.base.legged_robot import LeggedRobot

class MyRobot(LeggedRobot):
    """
    双足轮足机器人环境类 (对齐 TITA 控制架构):
    - 腿部 4 个转动关节采用 PD 位置控制
    - 轮子 2 个连续转动关节采用 TITA 式前馈+速度阻尼控制
    - 动作平滑在 step() 中一次性完成 (非 decimation 循环内)
    """

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



    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
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

        return super().step(actions)

    def _get_noise_scale_vec(self, cfg):
        """
        精确对齐 30 维观测向量的噪声分配
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
        return noise_vec

    def compute_observations(self):
        """30 维观测: 车轮 dof_pos 置零，动作反馈使用 filtered_actions"""
        obs_dof_pos = (self.dof_pos - self.default_dof_pos).clone()
        obs_dof_pos[:, self.wheel_indices] = 0.0

        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            obs_dof_pos * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.filtered_actions
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
        # 髋关节与膝关节幅度减半 (0.5x)，平抑机身前后俯仰抖动与动态自主深蹲起伏
        actions_scaled[:, self.left_hip_idx] *= self.hip_scale_reduction
        actions_scaled[:, self.right_hip_idx] *= self.hip_scale_reduction
        actions_scaled[:, self.left_knee_idx] *= self.knee_scale_reduction
        actions_scaled[:, self.right_knee_idx] *= self.knee_scale_reduction

        torques = torch.zeros_like(actions)

        # 腿部 PD 位置控制。策略可能输出超过关节可达范围的动作；将实际位置
        # 目标裁剪在 URDF 硬限位以内，避免长期力矩饱和和撞击机械限位。
        for idx in [self.left_hip_idx, self.left_knee_idx,
                    self.right_hip_idx, self.right_knee_idx]:
            target = actions_scaled[:, idx] + self.default_dof_pos[:, idx]
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
        按实机部署工况重构的结构化四模态指令重采样 (阶段 21):
        - 25% 驻车站立: v_y = 0, yaw = 0 (强化原地自平衡，彻底消灭驻车对推)
        - 50% 纯直行/倒车: v_y in [-0.8, -0.2] U [0.2, 1.0], yaw = 0 (双轮同步控速)
        - 15% 原地纯转向: v_y = 0, yaw in [-1.0, -0.1] U [0.1, 1.0] (保留转向灵敏度)
        - 10% 混合运动: v_y in [-0.8, 1.0], yaw in [-1.0, 1.0] (复合机动)
        """
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        r = torch.rand(n, device=self.device)

        # 预先将 lateral velocity (x) 与 heading 统一置零
        self.commands[env_ids, 0] = 0.0
        if self.commands.shape[1] > 3:
            self.commands[env_ids, 3] = 0.0

        # 1. 25% 驻车: v_y = 0, yaw = 0
        mask_stand = (r < 0.25)
        if mask_stand.any():
            ids_stand = env_ids[mask_stand]
            self.commands[ids_stand, 1] = 0.0
            self.commands[ids_stand, 2] = 0.0

        # 2. 50% 直行/倒车: v_y in [-0.8, -0.2] U [0.2, 1.0], yaw = 0
        mask_straight = (r >= 0.25) & (r < 0.75)
        if mask_straight.any():
            ids_str = env_ids[mask_straight]
            n_str = len(ids_str)
            u = torch.rand(n_str, device=self.device)
            # 分段映射避免落入 [-0.2, 0.2] 死区
            vy = torch.where(u < (0.6 / 1.4), -0.8 + u * 1.4, -0.8 + u * 1.4 + 0.4)
            self.commands[ids_str, 1] = vy
            self.commands[ids_str, 2] = 0.0

        # 3. 15% 原地纯转向: v_y = 0, yaw in [-1.0, -0.1] U [0.1, 1.0]
        mask_turn = (r >= 0.75) & (r < 0.90)
        if mask_turn.any():
            ids_turn = env_ids[mask_turn]
            n_turn = len(ids_turn)
            s = torch.where(torch.rand(n_turn, device=self.device) < 0.5, -1.0, 1.0)
            m = 0.1 + 0.9 * torch.rand(n_turn, device=self.device)
            self.commands[ids_turn, 1] = 0.0
            self.commands[ids_turn, 2] = s * m

        # 4. 10% 混合运动: 随机指令
        mask_mixed = (r >= 0.90)
        if mask_mixed.any():
            ids_mixed = env_ids[mask_mixed]
            n_mixed = len(ids_mixed)
            self.commands[ids_mixed, 1] = -0.8 + 1.8 * torch.rand(n_mixed, device=self.device)
            self.commands[ids_mixed, 2] = -1.0 + 2.0 * torch.rand(n_mixed, device=self.device)
            small_mask = torch.norm(self.commands[ids_mixed, :2], dim=1) < 0.2
            if small_mask.any():
                self.commands[ids_mixed[small_mask], 1] = 0.0

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
        """静止时惩罚车轮旋转 (锁定驱动轮，彻底切断前后摇摆振荡)"""
        zero_cmd = (torch.norm(self.commands[:, :2], dim=1) < 0.1).float()
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
        惩罚机身高度偏离目标值 (带 1.5cm 几何前倾死区的 L1 锁高):
        在 [0.385, 0.415]m 之间允许前倾巡航的自然几何微沉降，不产生罚分；
        一旦突破 1.5cm 死区(如自主下蹲起伏)，施加严厉 L1 梯度强力拉回，彻底兼顾定高与巡航。
        """
        base_height = self.root_states[:, 2]
        height_error = torch.clamp(torch.abs(base_height - self.cfg.rewards.base_height_target) - 0.015, min=0.0)
        return height_error

    def _reward_wheel_throttle_margin(self):
        """阶段 27 撤销节流软裕度惩罚，低层由差动优先力矩分配器硬保障"""
        return torch.zeros(self.num_envs, device=self.device)

