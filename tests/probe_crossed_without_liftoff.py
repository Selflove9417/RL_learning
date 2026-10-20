"""Diagnose the CROSSED_WITHOUT_LIFTOFF failures seen in the A1 pilot evaluation.

Reproduces the evaluator worker's setup exactly (cfg.seed, spatial trigger, 1 env,
episode_length_s=4.0) and logs the two operands of

    ploughed = (self.robot_rear_margin > 0.005) & (~self.has_taken_off)

so the exact step at which the episode is killed -- and which operand fired --
is visible instead of inferred.
"""
import os
import sys

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry
from rsl_rl.modules import ActorCritic

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = int(os.environ.get('SEED', '13'))
MODEL = os.environ.get(
    'MODEL',
    'legged_gym/logs/myrobot_obstacle_jump/Oct20_18-27-49_trial_a1_pilot10/model_10.pt')


def main():
    sys.argv = ['probe', '--task', 'myrobot_obstacle_jump', '--headless',
                '--num_envs', '1', '--pipeline', 'cpu', '--seed', str(SEED)]
    from legged_gym.utils import get_args
    args = get_args()
    cfg, _ = task_registry.get_cfgs(name='myrobot_obstacle_jump')
    cfg.env.num_envs = 1
    cfg.seed = SEED
    cfg.env.episode_length_s = 4.0
    cfg.trigger.mode = 'spatial'
    env, _ = task_registry.make_env(name='myrobot_obstacle_jump', args=args, env_cfg=cfg)
    env.reset()

    ac = ActorCritic(env.num_obs, env.num_privileged_obs, env.num_actions,
                     [512, 256, 128], [512, 256, 128], 'elu', 1.0).to(env.device)
    path = os.path.join(WORKSPACE, MODEL)
    ckpt = torch.load(path, map_location=env.device)
    ac.load_state_dict(ckpt['model_state_dict'])
    ac.eval()
    print('loaded', path, 'seed', SEED)

    orig = env._update_task_outcome

    def traced():
        rm = float(env.robot_rear_margin[0].item())
        tk = bool(env.has_taken_off[0].item())
        col = bool(env.has_obstacle_collision[0].item())
        tf = bool(env.task_failed[0].item())
        plough = (rm > 0.005) and (not tk)
        if not tf and (plough or col):
            rs = env.root_states
            rb = env.rb_states
            obs_rear = (env.obstacle_root_states[0, 1] + 0.5 * env.obstacle_w).item()
            print('     ctx ep_len=%s fsm=%s root_y=%+.4f rb_base_y=%+.4f obs_rear=%+.4f'
                  % (int(env.episode_length_buf[0].item()), int(env.fsm_state[0].item()),
                     rs[0, 1].item(), rb[0, 0, 1].item(), obs_rear))
            print('  >> LATCH step=%d rear_margin=%+.4f taken_off=%s collision=%s'
                  % (int(env.common_step_counter), rm, tk, col))
        return orig()

    env._update_task_outcome = traced

    obs = env.get_observations()
    print('%4s %4s %8s %9s %5s %5s %5s %6s %s'
          % ('loop', 'fsm', 'root_y', 'rear_marg', 'taken', 'touch', 'fail', 'ep_len', 'reason'))
    for i in range(200):
        with torch.no_grad():
            actions = ac.act_inference(obs)
        obs, _, _, dones, infos = env.step(actions)
        print('%4d %4d %8.4f %9.4f %5d %5d %5d %6d %s'
              % (i, env.fsm_state[0].item(), env.root_states[0, 1].item(),
                 env.robot_rear_margin[0].item(),
                 int(env.has_taken_off[0].item()), int(env.has_touched_down[0].item()),
                 int(env.task_failed[0].item()), int(env.episode_length_buf[0].item()),
                 env.failure_reason[0]))
        if dones[0].item():
            print('DONE loop=%d reason=%s' % (i, env.failure_reason[0]))
            break


if __name__ == '__main__':
    main()
