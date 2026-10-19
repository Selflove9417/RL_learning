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

import copy
import hashlib
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, kl_divergence

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 enable_early_stopping=False,
                 early_stop_kl=None,
                 enable_ref_kl=False,
                 ref_kl_coef=0.0,
                 ref_policy_path=None,
                 enable_sym_reg=False,
                 sym_reg_coef=0.0,
                 enable_mirrored_augmentation=False,
                 mirror_ratio=0.5,
                 **kwargs,
                 ):

        self.device = device
        self.enable_mirrored_augmentation = enable_mirrored_augmentation
        self.mirror_ratio = mirror_ratio

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.enable_early_stopping = enable_early_stopping
        if early_stop_kl is not None:
            self.early_stop_kl = early_stop_kl
        elif self.desired_kl is not None:
            self.early_stop_kl = 2.0 * self.desired_kl
        else:
            self.early_stop_kl = 0.02
        self.last_update_diagnostics = None

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # Reference-Policy KL Regularization (Phase 6B-Pilot)
        self.enable_ref_kl = enable_ref_kl
        self.ref_kl_coef = ref_kl_coef
        self.ref_policy_path = ref_policy_path
        self.ref_policy_sha256 = None
        self.ref_actor = None
        self.ref_std = None

        if self.enable_ref_kl:
            self._init_reference_policy()

        # Actor Symmetry Regularization (Phase 9A)
        self.enable_sym_reg = enable_sym_reg
        self.sym_reg_coef = sym_reg_coef
        if self.enable_sym_reg:
            self._init_symmetry_operators()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def _init_reference_policy(self):
        """Loads and freezes the reference actor and std from the reference policy checkpoint.
        Ensures strict mathematical correctness, no gradient leakage, and zero RNG side effects."""
        if not self.ref_policy_path:
            raise ValueError("enable_ref_kl is True but ref_policy_path is not specified.")
        resolved_path = self.ref_policy_path
        if not os.path.isfile(resolved_path):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            candidate = os.path.join(repo_root, self.ref_policy_path)
            if os.path.isfile(candidate):
                resolved_path = candidate
            else:
                raise FileNotFoundError(f"Reference policy file not found at '{self.ref_policy_path}' or '{candidate}'.")
        self.ref_policy_path = resolved_path

        with open(self.ref_policy_path, "rb") as f:
            self.ref_policy_sha256 = hashlib.sha256(f.read()).hexdigest()

        # Load reference checkpoint without consuming RNG numbers
        loaded_dict = torch.load(self.ref_policy_path, map_location=self.device)
        model_state = loaded_dict.get('model_state_dict', loaded_dict)

        # Clone actor structure directly without re-initialization (consumes 0 random numbers)
        self.ref_actor = copy.deepcopy(self.actor_critic.actor)
        ref_actor_sd = {k.replace('actor.', ''): v for k, v in model_state.items() if k.startswith('actor.')}
        if not ref_actor_sd:
            raise KeyError(f"No 'actor.' parameters found in reference model at '{self.ref_policy_path}'.")
        self.ref_actor.load_state_dict(ref_actor_sd)
        for param in self.ref_actor.parameters():
            param.requires_grad = False
        self.ref_actor.eval()
        self.ref_actor.to(self.device)

        # Reference noise std (frozen)
        if 'std' in model_state:
            self.ref_std = model_state['std'].clone().detach().to(self.device)
        else:
            self.ref_std = self.actor_critic.std.clone().detach().to(self.device)
        self.ref_std.requires_grad = False

    def _init_symmetry_operators(self):
        """Initializes exact bilateral reflection tensors for 31-dim obs and 6-dim action."""
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
        self.obs_sym_indices = torch.tensor(obs_indices, dtype=torch.long, device=self.device)
        self.obs_sym_signs = torch.tensor(obs_signs, dtype=torch.float, device=self.device)
        self.act_sym_indices = torch.tensor([3, 4, 5, 0, 1, 2], dtype=torch.long, device=self.device)

    def _apply_obs_symmetry(self, obs):
        return obs[..., self.obs_sym_indices] * self.obs_sym_signs

    def _apply_act_symmetry(self, act):
        return act[..., self.act_sym_indices]

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            time_outs = infos['time_outs'].unsqueeze(1).to(self.device)
            if 'terminal_observation' in infos:
                # Target B: evaluate Critic on the TRUE pre-reset terminal observation s_{t+1}
                with torch.no_grad():
                    terminal_critic_obs = infos['terminal_observation']
                    terminal_values = self.actor_critic.evaluate(terminal_critic_obs.to(self.device)).detach()
                self.transition.rewards += self.gamma * torch.squeeze(terminal_values * time_outs, 1)
            else:
                self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_outs, 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        # Snapshot initial policy and value parameters to measure parameter change magnitude
        with torch.no_grad():
            initial_actor_params = torch.cat([p.detach().flatten() for p in self.actor_critic.actor.parameters()]).clone()
            initial_critic_params = torch.cat([p.detach().flatten() for p in self.actor_critic.critic.parameters()]).clone()

            # Record GAE advantage distribution
            adv_flat = self.storage.advantages.detach().flatten()
            adv_mean = adv_flat.mean().item()
            adv_std = adv_flat.std().item()
            adv_min = adv_flat.min().item()
            adv_max = adv_flat.max().item()

        num_updates_planned = self.num_learning_epochs * self.num_mini_batches
        num_updates_executed = 0
        early_stop_triggered = False
        early_stop_reason = None

        kl_all = []
        clip_fractions = []
        grad_norms = []
        actor_grad_norms = []
        critic_grad_norms = []
        actual_lrs = []
        entropy_list = []
        surr_losses_list = []
        val_losses_list = []
        ref_kl_all = []
        ref_kl_loss_all = []
        sym_loss_all = []
        sym_loss_scaled_all = []

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        try:
            for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                    self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                    actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                    value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                    mu_batch = self.actor_critic.action_mean
                    sigma_batch = self.actor_critic.action_std
                    entropy_batch = self.actor_critic.entropy

                    # Numerical stability check for action distribution
                    if torch.isnan(mu_batch).any() or torch.isinf(mu_batch).any() or \
                       torch.isnan(sigma_batch).any() or torch.isinf(sigma_batch).any() or (sigma_batch <= 0).any():
                        raise FloatingPointError("Numerical instability detected in action distribution (NaN/Inf or non-positive sigma).")

                    # KL divergence computation
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)
                        kl_all.append(kl.detach())

                        # Adaptive LR schedule
                        if self.desired_kl is not None and self.schedule == 'adaptive':
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                            
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = self.learning_rate

                        # KL Early Stopping
                        if self.enable_early_stopping and kl_mean > self.early_stop_kl:
                            early_stop_triggered = True
                            early_stop_reason = f"KL mean ({kl_mean.item():.6f}) exceeded threshold ({self.early_stop_kl:.6f}) at update {num_updates_executed + 1}/{num_updates_planned}"
                            break

                    # Surrogate loss
                    ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                    surrogate = -torch.squeeze(advantages_batch) * ratio
                    surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                    1.0 + self.clip_param)
                    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                    # Clip fraction diagnostics
                    with torch.no_grad():
                        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_param).float()).item()
                        clip_fractions.append(clip_fraction)

                    # Value function loss
                    if self.use_clipped_value_loss:
                        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                        self.clip_param)
                        value_losses = (value_batch - returns_batch).pow(2)
                        value_losses_clipped = (value_clipped - returns_batch).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = (returns_batch - value_batch).pow(2).mean()

                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                    # Reference-Policy KL regularization (Phase 6B-Pilot, beta > 0)
                    if self.enable_ref_kl and self.ref_actor is not None:
                        with torch.no_grad():
                            mu_ref = self.ref_actor(obs_batch)
                        ref_dist = Normal(mu_ref, self.ref_std)
                        curr_dist = Normal(mu_batch, sigma_batch)
                        ref_kl_batch = kl_divergence(ref_dist, curr_dist).sum(dim=-1).mean()
                        ref_kl_loss = self.ref_kl_coef * ref_kl_batch
                        loss = loss + ref_kl_loss
                        ref_kl_all.append(ref_kl_batch.item())
                        ref_kl_loss_all.append(ref_kl_loss.item())

                    # Actor Symmetry Regularization (Phase 9A, lambda_sym > 0)
                    if self.enable_sym_reg and self.sym_reg_coef > 0:
                        obs_batch_mirr = self._apply_obs_symmetry(obs_batch)
                        mu_mirr = self.actor_critic.actor(obs_batch_mirr)
                        mu_mirr_target = self._apply_act_symmetry(mu_batch)
                        sym_loss_batch = torch.mean(torch.sum(torch.square(mu_mirr - mu_mirr_target), dim=-1) / 6.0)
                        sym_loss = self.sym_reg_coef * sym_loss_batch
                        loss = loss + sym_loss
                        sym_loss_all.append(sym_loss_batch.item())
                        sym_loss_scaled_all.append(sym_loss.item())

                    if torch.isnan(loss) or torch.isinf(loss):
                        raise FloatingPointError("Loss is NaN or Inf.")

                    # Gradient step
                    actual_lr = self.optimizer.param_groups[0]['lr']
                    actual_lrs.append(actual_lr)

                    self.optimizer.zero_grad()
                    loss.backward()

                    # Measure separate unclipped actor and critic gradient norms
                    with torch.no_grad():
                        actor_params = [p for p in self.actor_critic.actor.parameters() if p.grad is not None]
                        critic_params = [p for p in self.actor_critic.critic.parameters() if p.grad is not None]
                        actor_gn = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in actor_params]), 2).item() if actor_params else 0.0
                        critic_gn = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in critic_params]), 2).item() if critic_params else 0.0
                        actor_grad_norms.append(actor_gn)
                        critic_grad_norms.append(critic_gn)

                    grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        raise FloatingPointError("Gradient norm is NaN or Inf.")
                    self.optimizer.step()

                    num_updates_executed += 1
                    grad_norms.append(grad_norm.item())
                    entropy_list.append(entropy_batch.mean().item())
                    surr_losses_list.append(surrogate_loss.item())
                    val_losses_list.append(value_loss.item())

        finally:
            self.storage.clear()

        # Compute summary diagnostics
        num_executed = max(1, num_updates_executed)
        mean_value_loss = float(np.mean(val_losses_list)) if val_losses_list else 0.0
        mean_surrogate_loss = float(np.mean(surr_losses_list)) if surr_losses_list else 0.0
        mean_entropy = float(np.mean(entropy_list)) if entropy_list else 0.0
        mean_clip_fraction = float(np.mean(clip_fractions)) if clip_fractions else 0.0
        mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        mean_actor_grad_norm = float(np.mean(actor_grad_norms)) if actor_grad_norms else 0.0
        mean_critic_grad_norm = float(np.mean(critic_grad_norms)) if critic_grad_norms else 0.0
        mean_actual_lr = float(np.mean(actual_lrs)) if actual_lrs else self.learning_rate
        mean_ref_kl = float(np.mean(ref_kl_all)) if ref_kl_all else 0.0
        mean_ref_kl_loss = float(np.mean(ref_kl_loss_all)) if ref_kl_loss_all else 0.0
        mean_sym_loss = float(np.mean(sym_loss_all)) if sym_loss_all else 0.0
        mean_sym_loss_scaled = float(np.mean(sym_loss_scaled_all)) if sym_loss_scaled_all else 0.0

        with torch.no_grad():
            current_actor_params = torch.cat([p.detach().flatten() for p in self.actor_critic.actor.parameters()])
            actor_param_delta_l2 = torch.norm(current_actor_params - initial_actor_params, p=2).item()
            initial_actor_norm = torch.norm(initial_actor_params, p=2).item()
            actor_param_delta_rel = actor_param_delta_l2 / (initial_actor_norm + 1e-8)

            current_critic_params = torch.cat([p.detach().flatten() for p in self.actor_critic.critic.parameters()])
            critic_param_delta_l2 = torch.norm(current_critic_params - initial_critic_params, p=2).item()
            initial_critic_norm = torch.norm(initial_critic_params, p=2).item()
            critic_param_delta_rel = critic_param_delta_l2 / (initial_critic_norm + 1e-8)

            action_std = self.actor_critic.std.detach()
            action_std_mean = action_std.mean().item()
            action_std_min = action_std.min().item()
            action_std_max = action_std.max().item()

            if kl_all:
                kl_cat = torch.cat(kl_all)
                kl_mean_final = kl_cat.mean().item()
                kl_median_final = torch.median(kl_cat.float()).item()
                kl_p95_final = torch.quantile(kl_cat.float(), 0.95).item()
                kl_max_final = kl_cat.max().item()
            else:
                kl_mean_final, kl_median_final, kl_p95_final, kl_max_final = 0.0, 0.0, 0.0, 0.0

        self.last_update_diagnostics = {
            "kl_mean": kl_mean_final,
            "kl_median": kl_median_final,
            "kl_p95": kl_p95_final,
            "kl_max": kl_max_final,
            "clip_fraction": mean_clip_fraction,
            "surrogate_loss": mean_surrogate_loss,
            "value_loss": mean_value_loss,
            "entropy": mean_entropy,
            "action_std_mean": action_std_mean,
            "action_std_min": action_std_min,
            "action_std_max": action_std_max,
            "learning_rate": self.learning_rate,
            "actual_lr_before_step_mean": mean_actual_lr,
            "actual_lr_step_list": actual_lrs,
            "grad_norm_mean": mean_grad_norm,
            "actor_grad_norm_mean": mean_actor_grad_norm,
            "critic_grad_norm_mean": mean_critic_grad_norm,
            "updates_executed": num_updates_executed,
            "updates_planned": num_updates_planned,
            "early_stop_triggered": 1 if early_stop_triggered else 0,
            "early_stop_reason": early_stop_reason,
            "actor_param_delta_l2": actor_param_delta_l2,
            "actor_param_delta_rel": actor_param_delta_rel,
            "critic_param_delta_l2": critic_param_delta_l2,
            "critic_param_delta_rel": critic_param_delta_rel,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "adv_min": adv_min,
            "adv_max": adv_max,
            "enable_ref_kl": self.enable_ref_kl,
            "ref_kl_coef": self.ref_kl_coef,
            "ref_kl_mean": mean_ref_kl,
            "ref_kl_loss_mean": mean_ref_kl_loss,
            "ref_policy_path": self.ref_policy_path,
            "ref_policy_sha256": self.ref_policy_sha256,
            "enable_sym_reg": self.enable_sym_reg,
            "sym_reg_coef": self.sym_reg_coef,
            "sym_loss_mean": mean_sym_loss,
            "sym_loss_scaled_mean": mean_sym_loss_scaled,
        }

        return mean_value_loss, mean_surrogate_loss
