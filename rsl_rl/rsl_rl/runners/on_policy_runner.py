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

import time
import os
import re
import json
from collections import deque
import statistics

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        # Phase 10: On-policy Mirrored-View Augmentation (MDA)
        self.enable_mirrored_augmentation = (
            self.cfg.get("enable_mirrored_augmentation", False) or
            self.alg_cfg.get("enable_mirrored_augmentation", False)
        )
        self.mirror_ratio = self.cfg.get("mirror_ratio", self.alg_cfg.get("mirror_ratio", 0.5))
        if self.enable_mirrored_augmentation:
            from rsl_rl.env.mirrored_vec_env_wrapper import MirroredVecEnvWrapper
            self.env = MirroredVecEnvWrapper(self.env, mirror_ratio=self.mirror_ratio)
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.diagnostics_history = []

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if hasattr(self.alg, 'last_update_diagnostics') and self.alg.last_update_diagnostics is not None:
                diag_record = dict(self.alg.last_update_diagnostics)
                diag_record['iteration'] = it + 1
                diag_record['timesteps'] = self.tot_timesteps + self.num_steps_per_env * self.env.num_envs
                diag_record['mean_reward'] = float(statistics.mean(rewbuffer)) if len(rewbuffer) > 0 else 0.0
                diag_record['mean_episode_length'] = float(statistics.mean(lenbuffer)) if len(lenbuffer) > 0 else 0.0
                fixed_l_sym = self._eval_fixed_obs_symmetry()
                if fixed_l_sym is not None:
                    diag_record['fixed_obs_l_sym_global'] = fixed_l_sym['global_l_sym']
                    diag_record['fixed_obs_l_sym_left'] = fixed_l_sym['left_4cases_l_sym']
                    diag_record['fixed_obs_l_sym_right'] = fixed_l_sym['right_4cases_l_sym']
                    diag_record['fixed_obs_l_sym_inplace'] = fixed_l_sym['inplace_squat_l_sym']
                self.diagnostics_history.append(diag_record)
                if self.log_dir is not None:
                    diag_path = os.path.join(self.log_dir, 'diagnostics.json')
                    with open(diag_path, 'w', encoding='utf-8') as f:
                        json.dump(self.diagnostics_history, f, indent=2, ensure_ascii=False)
            if self.log_dir is not None:
                self.log(locals())
            if (it + 1) % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it + 1)), iteration=it + 1)
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)), iteration=self.current_learning_iteration)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if hasattr(self.alg, 'last_update_diagnostics') and self.alg.last_update_diagnostics is not None:
            diag = self.alg.last_update_diagnostics
            self.writer.add_scalar('Policy/kl_mean', diag['kl_mean'], locs['it'])
            self.writer.add_scalar('Policy/kl_p95', diag['kl_p95'], locs['it'])
            self.writer.add_scalar('Policy/kl_max', diag['kl_max'], locs['it'])
            self.writer.add_scalar('Loss/clip_fraction', diag['clip_fraction'], locs['it'])
            self.writer.add_scalar('Loss/entropy', diag['entropy'], locs['it'])
            self.writer.add_scalar('Loss/grad_norm', diag['grad_norm_mean'], locs['it'])
            self.writer.add_scalar('Train/updates_executed', diag['updates_executed'], locs['it'])
            self.writer.add_scalar('Train/updates_planned', diag['updates_planned'], locs['it'])
            self.writer.add_scalar('Train/early_stop_triggered', diag['early_stop_triggered'], locs['it'])
            self.writer.add_scalar('Policy/param_delta_l2', diag['actor_param_delta_l2'], locs['it'])
            self.writer.add_scalar('Policy/param_delta_rel', diag['actor_param_delta_rel'], locs['it'])
            self.writer.add_scalar('Policy/min_noise_std', diag['action_std_min'], locs['it'])
            self.writer.add_scalar('Policy/max_noise_std', diag['action_std_max'], locs['it'])
            if diag.get('enable_ref_kl', False):
                self.writer.add_scalar('Policy/ref_kl_mean', diag['ref_kl_mean'], locs['it'])
                self.writer.add_scalar('Loss/ref_kl_loss', diag['ref_kl_loss_mean'], locs['it'])
            if self.diagnostics_history and 'fixed_obs_l_sym_global' in self.diagnostics_history[-1]:
                d_last = self.diagnostics_history[-1]
                self.writer.add_scalar('Policy/fixed_obs_l_sym_global', d_last['fixed_obs_l_sym_global'], locs['it'])
                self.writer.add_scalar('Policy/fixed_obs_l_sym_left', d_last['fixed_obs_l_sym_left'], locs['it'])
                self.writer.add_scalar('Policy/fixed_obs_l_sym_right', d_last['fixed_obs_l_sym_right'], locs['it'])
                self.writer.add_scalar('Policy/fixed_obs_l_sym_inplace', d_last['fixed_obs_l_sym_inplace'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        diag_str = ""
        if hasattr(self.alg, 'last_update_diagnostics') and self.alg.last_update_diagnostics is not None:
            diag = self.alg.last_update_diagnostics
            stop_flag = " [EARLY STOPPED]" if diag['early_stop_triggered'] else ""
            diag_str = (
                f"""{'KL (mean / p95 / max):':>{pad}} {diag['kl_mean']:.5f} / {diag['kl_p95']:.5f} / {diag['kl_max']:.5f}\n"""
                f"""{'PPO clip fraction:':>{pad}} {diag['clip_fraction']:.4f}\n"""
                f"""{'Policy entropy:':>{pad}} {diag['entropy']:.4f}\n"""
                f"""{'Mean grad norm:':>{pad}} {diag['grad_norm_mean']:.4f}\n"""
                f"""{'Optimizer updates:':>{pad}} {diag['updates_executed']}/{diag['updates_planned']}{stop_flag}\n"""
                f"""{'Policy delta L2 (rel):':>{pad}} {diag['actor_param_delta_l2']:.5f} ({diag['actor_param_delta_rel']:.3%})\n"""
            )
            if self.diagnostics_history and 'fixed_obs_l_sym_global' in self.diagnostics_history[-1]:
                d_last = self.diagnostics_history[-1]
                diag_str += f"""{'Fixed L_sym (glb/left/right):':>{pad}} {d_last['fixed_obs_l_sym_global']:.5f} / {d_last['fixed_obs_l_sym_left']:.5f} / {d_last['fixed_obs_l_sym_right']:.5f}\n"""
            if diag.get('enable_ref_kl', False):
                diag_str += f"""{'Ref KL (mean / coef / loss):':>{pad}} {diag['ref_kl_mean']:.5f} / {diag['ref_kl_coef']:.4f} / {diag['ref_kl_loss_mean']:.5f}\n"""
            if diag['early_stop_reason']:
                diag_str += f"""{'Early stop reason:':>{pad}} {diag['early_stop_reason']}\n"""

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{diag_str}""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{diag_str}""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None, iteration=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration if iteration is None else iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # Older intermediate checkpoints were saved with a stale internal
        # iteration value. Prefer the explicit model_<iteration>.pt filename so
        # resumed runs retain monotonic checkpoint and TensorBoard step numbers.
        checkpoint_match = re.search(r'model_(\d+)\.pt$', os.path.basename(path))
        self.current_learning_iteration = (
            int(checkpoint_match.group(1)) if checkpoint_match else loaded_dict.get('iter', 0)
        )
        return loaded_dict.get('infos', None)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def _eval_fixed_obs_symmetry(self):
        """Phase 10: Evaluates actor mirror error L_sym on evaluation_assets/phase9b/fixed_obs.pt."""
        fixed_obs_path = "evaluation_assets/phase9b/fixed_obs.pt"
        if not os.path.exists(fixed_obs_path) or self.env.num_obs != 31:
            return None
        if not hasattr(self, "_fixed_obs_data"):
            data = torch.load(fixed_obs_path, map_location=self.device)
            self._fixed_obs_data = data["observations"].to(self.device)
            self._fixed_obs_slices = data.get("sc_slices", {})
            obs_indices = [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                15, 16, 17, 12, 13, 14,
                21, 22, 23, 18, 19, 20,
                27, 28, 29, 24, 25, 26,
                30
            ]
            obs_signs = [
                -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0
            ]
            self._fixed_obs_sym_indices = torch.tensor(obs_indices, dtype=torch.long, device=self.device)
            self._fixed_obs_sym_signs = torch.tensor(obs_signs, dtype=torch.float, device=self.device)
            self._fixed_act_sym_indices = torch.tensor([3, 4, 5, 0, 1, 2], dtype=torch.long, device=self.device)

        was_training = self.alg.actor_critic.training
        self.alg.actor_critic.eval()
        with torch.no_grad():
            obs = self._fixed_obs_data
            obs_mirr = obs[:, self._fixed_obs_sym_indices] * self._fixed_obs_sym_signs
            mu_orig = self.alg.actor_critic.actor(obs)
            mu_mirr = self.alg.actor_critic.actor(obs_mirr)
            mu_expected = mu_orig[:, self._fixed_act_sym_indices]
            diff_sq = torch.square(mu_mirr - mu_expected)
            global_l_sym = torch.mean(diff_sq).item()

            per_sc_l_sym = {}
            for sc, (start, end) in self._fixed_obs_slices.items():
                per_sc_l_sym[sc] = torch.mean(diff_sq[start:end]).item()

            left_cases = ["g1_fwd_left_squat", "g1_bwd_left_squat", "g4_exp_fwd_left_squat", "g4_exp_bwd_left_squat"]
            right_cases = ["g1_fwd_right_squat", "g1_bwd_right_squat", "g4_exp_fwd_right_squat", "g4_exp_bwd_right_squat"]
            left_l_sym = float(np.mean([per_sc_l_sym[c] for c in left_cases])) if all(c in per_sc_l_sym for c in left_cases) else 0.0
            right_l_sym = float(np.mean([per_sc_l_sym[c] for c in right_cases])) if all(c in per_sc_l_sym for c in right_cases) else 0.0
            inplace_l_sym = per_sc_l_sym.get("g3_inplace_squat", 0.0)

        if was_training:
            self.alg.actor_critic.train()

        return {
            "global_l_sym": global_l_sym,
            "left_4cases_l_sym": left_l_sym,
            "right_4cases_l_sym": right_l_sym,
            "inplace_squat_l_sym": inplace_l_sym,
        }
