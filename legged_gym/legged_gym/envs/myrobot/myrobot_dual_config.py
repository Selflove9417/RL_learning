# ==============================================================================
# MyRobot Dual-Policy Continuous Jumping & Balance Configuration
# ==============================================================================

from legged_gym.envs.myrobot.myrobot_jump_config import MyRobotJumpCfg, MyRobotJumpCfgPPO

class MyRobotDualCfg(MyRobotJumpCfg):
    class dual_policy:
        # Default frozen PPO model paths
        balance_model_path = "evaluation_assets/phase15/models/phase15_m130.pt"
        jump_model_path = "evaluation_assets/frozen_stage8/model_40.pt"
        
        # Teleop speed and height bounds
        min_vy = -0.30
        max_vy = 0.50
        min_yaw = -0.35
        max_yaw = 0.35
        nominal_height = 0.39774
        min_height = 0.340
        max_height = 0.520
        
        # FSM Transition thresholds
        stopping_speed_thresh = 0.06
        stopping_yaw_thresh = 0.12
        gate_stable_steps_required = 5
        prepare_timeout_s = 0.40
        cushion_min_time_s = 0.35
        cushion_min_height_m = 0.320
        settle_stable_steps_required = 5
        handoff_timeout_s = 1.50
        hard_abort_timeout_s = 2.00
        
        # Gain schedule
        kp_balance = 40.0
        kd_balance = 2.0
        kp_jump = 180.0
        kd_jump = 4.5
        gain_ramp_time_s = 0.08
        decay_ramp_time_s = 0.20

class MyRobotDualCfgPPO(MyRobotJumpCfgPPO):
    pass
