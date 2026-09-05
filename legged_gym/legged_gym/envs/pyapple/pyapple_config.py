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

class PyAppleCfg( LeggedRobotCfg ):


    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 8
        
        num_observations = 36

        # 如果你没有 privileged obs，就保持 None
        #num_privileged_obs = 36
        num_privileged_obs = None

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.1, 1.0] # min max [m/s]
            lin_vel_y = [0, 0]   # min max [m/s]
            #ang_vel_yaw = [-1, 1]    # min max [rad/s]
            #heading = [-3.14, 3.14]
            
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [1.0, 1.0, 0.3]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.15] # x,y,z [m]

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.50,   # [rad]
            'RL_hip_joint': 0.80,   # [rad]
            'FR_hip_joint': 0.50 ,  # [rad]
            'RR_hip_joint': 0.80,   # [rad]


            'FL_thigh_joint': 0.0,     # [rad]
            'RL_thigh_joint': 0.0,   # [rad]
            'FR_thigh_joint': 0.0,     # [rad]
            'RR_thigh_joint': 0.0,   # [rad]


        }

  
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False     # 
        
        
            
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
        

    class asset( LeggedRobotCfg.asset ):
        #file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/Pyappledog/urdf/Pyapple_urdf.urdf"
        name = "pyapple"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "hip"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        #Only for Pyapple
        flip_visual_attachments = False
        replace_cylinder_with_capsule = True
        
    

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            

    '''    
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.0002
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            dof_pos_limits = -10.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 0.5 # forces above this value are penalized
        '''  
            
            
    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8
            num_velocity_iterations = 3
            contact_offset = 0.002  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
   
   

class PyAppleCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'pyapple'

  
