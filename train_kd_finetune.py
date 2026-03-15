# train_kd_finetune.py
# Phase C — Core Research Contribution: HA-KD-SAC
# (Hierarchical Attention-Based Multi-Teacher Knowledge Distillation SAC)
#
# ═══════════════════════════════════════════════════════════════════════════
# RESEARCH CONTRIBUTION (Novel element for the paper)
# ═══════════════════════════════════════════════════════════════════════════
# Standard KD uses FIXED linear annealing for teacher weights α(t), β(t).
# This is a simple heuristic — it treats all states uniformly.
#
# HA-KD-SAC introduces a LEARNED Context-Aware Attention Module that outputs
# STATE-DEPENDENT weights α(s), β(s):
#
#   Student loss = L_SAC_standard
#                + γ(t) · [ α(s) · KL(student_steer   ‖ teacher1_steer)
#                         + β(s) · KL(student_throttle ‖ teacher2_throttle) ]
#                - λ_H · H(α(s), β(s))   [entropy regularization]
#
# Where:
#   α(s), β(s) = ContextAttentionModule(s)  ∈ [0, 1]  (learned via sigmoid)
#   γ(t)       = Global annealing multiplier (1→0), ensures student graduation
#   H(·)       = Binary entropy, prevents attention collapse to 0 or 1
#
# KEY INSIGHT: The attention module learns WHEN to trust each teacher:
#   - Near intersections/curves: α(s) ↑ (lateral teacher is more relevant)
#   - On straight highways:      β(s) ↑ (longitudinal teacher is more relevant)
#   - In novel situations:       both ↓ (student explores independently)
#
# This produces INTERPRETABLE attention heatmaps for the paper, showing
# the student's context-dependent reliance on each teacher.
#
# ─────────────────────────────────────────────────────────────────────────
# KL DIVERGENCE (Gaussian, analytical form)
# ─────────────────────────────────────────────────────────────────────────
#   KL( N(μ₁,σ₁) ‖ N(μ₂,σ₂) )
#     = log(σ₂/σ₁) + (σ₁² + (μ₁−μ₂)²) / (2σ₂²) − ½
#
# ─────────────────────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────────────────────
#   python train_kd_finetune.py
#
# OUTPUT:
#   models/kd_finetune/  — model checkpoints + attention module
#   reports/kd_finetune/ — CSV with standard metrics + KL + attention curves

import os
import csv
import random
from datetime import datetime
from collections import OrderedDict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.type_aliases import GymEnv

import config as cfg
from simulation.sumo_env import SumoEnv
from wrappers import FullControlWrapper
from callbacks import CurriculumLogCallback

# ─────────────────────────────────────────────────────────────────────────
# KD CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────
KD_ALPHA_START = cfg.KD_ALPHA_START
KD_BETA_START  = cfg.KD_BETA_START
KD_ANNEAL_FRAC = cfg.KD_ANNEAL_FRAC

KD_TIMESTEPS   = cfg.KD_TIMESTEPS

# HA-KD-SAC attention config
KD_USE_ATTENTION    = cfg.KD_USE_ATTENTION
KD_ATTENTION_HIDDEN = cfg.KD_ATTENTION_HIDDEN
KD_ATTENTION_LR     = cfg.KD_ATTENTION_LR
KD_ENTROPY_COEF     = cfg.KD_ENTROPY_COEF
KD_GLOBAL_ANNEAL    = cfg.KD_GLOBAL_ANNEAL

# DC-KD-SAC config
KD_USE_CRITIC       = getattr(cfg, "KD_USE_CRITIC", False)
KD_CRITIC_COEF      = getattr(cfg, "KD_CRITIC_COEF", 0.0)

# Directories
KD_MODEL_DIR = cfg.KD_MODEL_DIR
KD_LOG_DIR   = cfg.KD_LOG_DIR

for d in [
    KD_MODEL_DIR, KD_LOG_DIR,
    os.path.join(KD_MODEL_DIR, "best/"),
    os.path.join(KD_MODEL_DIR, "checkpoints/"),
]:
    os.makedirs(d, exist_ok=True)

# Teacher action dimension indices
STEER_DIM    = 0  # Teacher1 (lateral) → student dimension 0
THROTTLE_DIM = 1  # Teacher2 (longitudinal) → student dimension 1


# ═══════════════════════════════════════════════════════════════════════════
# ANALYTICAL KL DIVERGENCE (Gaussian)
# ═══════════════════════════════════════════════════════════════════════════
def gaussian_kl_div(
    mu_p: torch.Tensor, log_std_p: torch.Tensor,
    mu_q: torch.Tensor, log_std_q: torch.Tensor,
) -> torch.Tensor:
    """
    KL( N(μ_p, σ_p) ‖ N(μ_q, σ_q) ) — analytically computed.

    All inputs: shape [batch, 1]
    Returns:    shape [batch], mean KL per sample.

    Formula:
        KL = log(σ_q/σ_p) + (σ_p² + (μ_p−μ_q)²) / (2σ_q²) − 0.5

    Works in log-space to avoid numerical issues:
        log(σ_q/σ_p) = log_std_q − log_std_p
    """
    # Clamp log_stds to avoid extreme values
    log_std_p = torch.clamp(log_std_p, -10, 2)
    log_std_q = torch.clamp(log_std_q, -10, 2)

    var_p = torch.exp(2 * log_std_p)  # σ_p²
    var_q = torch.exp(2 * log_std_q)  # σ_q²

    kl = (log_std_q - log_std_p) + (var_p + (mu_p - mu_q).pow(2)) / (2 * var_q) - 0.5
    return kl.squeeze(-1)  # [batch]


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT-AWARE ATTENTION MODULE (Novel HA-KD-SAC Component)
# ═══════════════════════════════════════════════════════════════════════════
class ContextAttentionModule(nn.Module):
    """
    Learned attention module that outputs state-dependent KD weights.

    Given observation s, produces:
        α(s) ∈ [0, 1] — how much to trust the lateral teacher (steering)
        β(s) ∈ [0, 1] — how much to trust the longitudinal teacher (throttle)

    Architecture: obs(25) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(2) → Sigmoid

    Design choices:
      - Sigmoid (not softmax): α and β are INDEPENDENT because lateral and
        longitudinal are ORTHOGONAL skills. The student can trust both fully,
        neither, or any combination.
      - Positive bias initialization: sigmoid output starts near 0.88 (high
        trust in teachers), allowing the student to rely on teachers early.
      - Small network (25→64→32→2): avoids overfitting to KD signal,
        keeps overhead minimal vs the main SAC policy network.
    """

    def __init__(self, obs_dim: int = 25, hidden_dim: int = 64, init_bias: float = 2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        # Initialize final layer:
        # - weights ≈ 0 so initial output is bias-dominated (same for all inputs)
        # - bias = init_bias so sigmoid(init_bias) ≈ 0.88 (high initial trust)
        # During training, the hidden layers break symmetry and the output
        # becomes state-dependent.
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)  # Near-zero, not exactly zero
            self.net[-1].bias.fill_(init_bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: [batch, obs_dim] observation tensor
        Returns:
            alpha: [batch, 1] — lateral teacher attention weight
            beta:  [batch, 1] — longitudinal teacher attention weight
        """
        logits = self.net(obs)       # [batch, 2]
        weights = torch.sigmoid(logits)  # [batch, 2], each ∈ (0, 1)
        return weights[:, 0:1], weights[:, 1:2]

    @staticmethod
    def binary_entropy(
        alpha: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean binary entropy of attention weights.
        H(p) = -(p·log(p) + (1-p)·log(1-p))
        Maximized at p=0.5, minimized at p=0 or p=1.

        Returns: scalar — mean entropy across batch and both dimensions.
        """
        eps = 1e-7
        a = torch.clamp(alpha, eps, 1.0 - eps)
        b = torch.clamp(beta, eps, 1.0 - eps)

        h_a = -(a * torch.log(a) + (1.0 - a) * torch.log(1.0 - a))
        h_b = -(b * torch.log(b) + (1.0 - b) * torch.log(1.0 - b))
        return (h_a + h_b).mean()


# ═══════════════════════════════════════════════════════════════════════════
# KD-AWARE SAC (Subclasses stable-baselines3 SAC)
# ═══════════════════════════════════════════════════════════════════════════
class KDSAC(SAC):
    """
    HA-KD-SAC: Hierarchical Attention Knowledge Distillation SAC.

    Extends SB3's SAC by:
    1. Injecting KL divergence terms (from frozen teachers) into actor loss
    2. Using a LEARNED attention module for state-dependent KD weights
    3. Entropy regularization to prevent attention collapse
    4. Global annealing to ensure student eventually becomes autonomous

    When KD_USE_ATTENTION=False, falls back to original linear annealing
    for ablation study comparison.
    """

    def __init__(
        self,
        policy,
        env: GymEnv,
        teacher_lateral: SAC,
        teacher_longitudinal: SAC,
        kd_alpha: float = KD_ALPHA_START,
        kd_beta: float  = KD_BETA_START,
        kd_anneal_steps: int = KD_TIMESTEPS,
        **kwargs,
    ):
        super().__init__(policy, env, **kwargs)

        self.teacher_lateral = teacher_lateral
        self.teacher_longitudinal = teacher_longitudinal

        self.kd_alpha_start = kd_alpha
        self.kd_beta_start  = kd_beta
        self.kd_anneal_steps = max(kd_anneal_steps, 1)

        # Current scalar values (used for logging and fallback mode)
        self.kd_alpha = kd_alpha
        self.kd_beta  = kd_beta
        self.kd_global_mult = 1.0

        # ── Context-Aware Attention Module (HA-KD-SAC novel component) ──
        self.use_attention = KD_USE_ATTENTION
        self.use_global_anneal = KD_GLOBAL_ANNEAL

        # ── Dual-Level Critic KD (DC-KD-SAC novel component) ──
        self.use_critic_kd = KD_USE_CRITIC
        self.critic_kd_coef = KD_CRITIC_COEF

        if self.use_attention:
            obs_dim = env.observation_space.shape[0]
            self.attention_module = ContextAttentionModule(
                obs_dim=obs_dim,
                hidden_dim=KD_ATTENTION_HIDDEN,
            ).to(self.device)
            self.attention_optimizer = torch.optim.Adam(
                self.attention_module.parameters(),
                lr=KD_ATTENTION_LR,
            )
            self.attention_entropy_coef = KD_ENTROPY_COEF
            print(f"  [HA-KD-SAC] Attention module: {obs_dim}→{KD_ATTENTION_HIDDEN}"
                  f"→{KD_ATTENTION_HIDDEN//2}→2 (device={self.device})")
        else:
            self.attention_module = None
            self.attention_optimizer = None
            self.attention_entropy_coef = 0.0
            print("  [KD-SAC] Using fixed linear annealing (baseline mode)")

        # Metrics exposed to callbacks
        self.kd_metrics = {
            "alpha": kd_alpha,
            "beta": kd_beta,
            "kl_steer": 0.0,
            "kl_throttle": 0.0,
            "global_mult": 1.0,
            "attn_alpha_mean": 0.0,
            "attn_beta_mean": 0.0,
            "attn_entropy": 0.0,
            "critic_kd_loss": 0.0,
        }

        # Freeze teachers permanently
        self._freeze_teacher(self.teacher_lateral)
        self._freeze_teacher(self.teacher_longitudinal)

    @staticmethod
    def _freeze_teacher(model: SAC):
        """Freeze all parameters of a teacher model."""
        model.policy.set_training_mode(False)
        for param in model.policy.parameters():
            param.requires_grad_(False)

    def _update_kd_weights(self):
        """Update KD schedule: global multiplier + legacy scalar weights."""
        frac = min(self.num_timesteps / self.kd_anneal_steps, 1.0)
        # Global multiplier decays 1→0 (used by both modes)
        self.kd_global_mult = max(0.0, 1.0 - frac) if self.use_global_anneal else 1.0
        # Legacy scalar weights (used when attention is disabled, and for logging)
        self.kd_alpha = self.kd_alpha_start * (1.0 - frac)
        self.kd_beta  = self.kd_beta_start  * (1.0 - frac)

    @torch.no_grad()
    def _get_teacher_distribution(
        self, actor, obs_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Query a FROZEN teacher actor's Gaussian distribution parameters.
        Uses @torch.no_grad() for efficiency (teachers are frozen anyway).

        Returns:
            mu:       [batch, action_dim]
            log_std:  [batch, action_dim]
        """
        mu, log_std, _ = actor.get_action_dist_params(obs_tensor)
        return mu, log_std

    def _get_student_distribution(
        self, obs_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get student actor's Gaussian distribution parameters WITH gradients.

        IMPORTANT: This must NOT use @torch.no_grad() — the KD loss needs
        to backpropagate through the student's actor to actually guide it.
        (The original code had a bug where _get_teacher_distribution was
        used for the student, blocking gradient flow.)
        """
        mu, log_std, _ = self.actor.get_action_dist_params(obs_tensor)
        return mu, log_std

    def save_attention_module(self, path: str):
        """Save attention module state dict for later analysis/visualization."""
        if self.attention_module is not None:
            torch.save(self.attention_module.state_dict(), path)
            print(f"  [HA-KD-SAC] Attention module saved: {path}")

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Override SB3 SAC.train() to inject KD loss into actor update.

        HA-KD-SAC enhancements over original:
        1. Student distribution computed WITH gradients (bug fix)
        2. Attention module outputs per-sample α(s), β(s)
        3. Entropy regularization prevents attention collapse
        4. Attention optimizer stepped alongside actor optimizer
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        if self.attention_module is not None:
            self.attention_module.train()

        # Update KD annealing schedule
        self._update_kd_weights()

        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, critic_kd_vals = [], [], []
        kl_steers, kl_throttles = [], []
        attn_alpha_vals, attn_beta_vals, attn_entropy_vals = [], [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            # For n-step replay, discount factor is gamma**n_steps
            discounts = (
                replay_data.discounts
                if hasattr(replay_data, "discounts") and replay_data.discounts is not None
                else self.gamma
            )

            # We need to sample because `log_std` may have changed
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # ─── ENTROPY COEFFICIENT ──────────────────────────────────────
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # ─── CRITIC UPDATE ────────────────────────────────────────────
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = torch.cat(
                    self.critic_target(
                        replay_data.next_observations, next_actions
                    ), dim=1
                )
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())

            # ─── Dual-Level KD: Critic Knowledge Distillation ─────────────
            critic_kd_loss_val = 0.0
            critic_total_loss = critic_loss

            if self.kd_global_mult > 1e-6 and self.use_critic_kd:
                with torch.no_grad():
                    # Get teacher critic values for the actions in the buffer
                    # Note: actions in replay_data has shape [batch, 2].
                    # Teacher 1 expects [batch, 1] (steer)
                    # Teacher 2 expects [batch, 1] (throttle)
                    t1_actions = replay_data.actions[:, STEER_DIM:STEER_DIM+1]
                    t2_actions = replay_data.actions[:, THROTTLE_DIM:THROTTLE_DIM+1]
                    obs_tensor = replay_data.observations

                    # Target teacher values: use the minimum Q of the two critics
                    q_t1 = torch.cat(self.teacher_lateral.critic(obs_tensor, t1_actions), dim=1)
                    q_t1, _ = torch.min(q_t1, dim=1, keepdim=True)

                    q_t2 = torch.cat(self.teacher_longitudinal.critic(obs_tensor, t2_actions), dim=1)
                    q_t2, _ = torch.min(q_t2, dim=1, keepdim=True)

                    # Get attention weights (detached so Critic KD doesn't alter attention module here)
                    if self.use_attention and self.attention_module is not None:
                        a_s, b_s = self.attention_module(obs_tensor)
                        alpha_q = (self.kd_global_mult * a_s).detach()
                        beta_q  = (self.kd_global_mult * b_s).detach()
                    else:
                        alpha_q = self.kd_alpha
                        beta_q  = self.kd_beta

                critic_kd_component = 0.0
                for current_q in current_q_values:
                    if isinstance(alpha_q, torch.Tensor):
                        loss_t1 = (alpha_q * F.mse_loss(current_q, q_t1, reduction="none")).mean()
                        loss_t2 = (beta_q * F.mse_loss(current_q, q_t2, reduction="none")).mean()
                    else:
                        loss_t1 = alpha_q * F.mse_loss(current_q, q_t1)
                        loss_t2 = beta_q * F.mse_loss(current_q, q_t2)
                    critic_kd_component += loss_t1 + loss_t2

                critic_kd_component = self.critic_kd_coef * critic_kd_component
                critic_total_loss = critic_total_loss + critic_kd_component
                critic_kd_loss_val = critic_kd_component.item()
                
            critic_kd_vals.append(critic_kd_loss_val)

            self.critic.optimizer.zero_grad()
            critic_total_loss.backward()
            self.critic.optimizer.step()

            # ─── ACTOR UPDATE (with KD loss injection) ───────────────────
            q_values_pi = torch.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss_sac = (ent_coef * log_prob - min_qf_pi).mean()

            # ─── KL Divergence terms (KD contribution) ────────────────────
            kd_loss = torch.tensor(0.0, device=self.device)
            entropy_loss = torch.tensor(0.0, device=self.device)
            kl_steer_val = 0.0
            kl_throttle_val = 0.0
            step_attn_alpha = 0.0
            step_attn_beta = 0.0
            step_attn_entropy = 0.0

            obs_tensor = replay_data.observations.to(self.device)

            # Determine if KD should be computed this step
            should_compute_kd = self.kd_global_mult > 1e-6

            if should_compute_kd:
                # Student distribution — WITH gradients (critical fix!)
                student_mu, student_log_std = self._get_student_distribution(
                    obs_tensor
                )

                # ── Compute effective per-sample weights ──────────────────
                if self.use_attention and self.attention_module is not None:
                    # Attention module: learned α(s), β(s) ∈ [0, 1]
                    alpha_s, beta_s = self.attention_module(obs_tensor)
                    # [batch, 1] each

                    # Apply global annealing on top
                    alpha_eff = self.kd_global_mult * alpha_s
                    beta_eff  = self.kd_global_mult * beta_s

                    step_attn_alpha = alpha_s.mean().item()
                    step_attn_beta = beta_s.mean().item()

                    # Entropy regularization (maximize → subtract from loss)
                    if self.attention_entropy_coef > 0:
                        attn_ent = ContextAttentionModule.binary_entropy(
                            alpha_s, beta_s
                        )
                        # Negative because we MINIMIZE total loss but
                        # MAXIMIZE entropy (higher H = more exploration)
                        entropy_loss = -self.attention_entropy_coef * attn_ent
                        step_attn_entropy = attn_ent.item()
                else:
                    # Fallback: scalar weights (original linear annealing)
                    alpha_eff = self.kd_alpha  # scalar
                    beta_eff  = self.kd_beta   # scalar

                # ── Term 1: KL(student_steer ‖ teacher_lateral_steer) ─────
                # Check if lateral KD has any weight
                alpha_active = (
                    alpha_eff.mean().item() > 1e-6
                    if isinstance(alpha_eff, torch.Tensor)
                    else alpha_eff > 1e-6
                )
                if alpha_active:
                    t1_mu, t1_log_std = self._get_teacher_distribution(
                        self.teacher_lateral.actor, obs_tensor
                    )
                    kl_steer = gaussian_kl_div(
                        student_mu[:, STEER_DIM:STEER_DIM+1],
                        student_log_std[:, STEER_DIM:STEER_DIM+1],
                        t1_mu,
                        t1_log_std,
                    )  # [batch]

                    if isinstance(alpha_eff, torch.Tensor):
                        # Per-sample weighted mean
                        weighted_kl_steer = (
                            alpha_eff.squeeze(-1) * kl_steer
                        ).mean()
                    else:
                        weighted_kl_steer = alpha_eff * kl_steer.mean()

                    kd_loss = kd_loss + weighted_kl_steer
                    kl_steer_val = kl_steer.mean().item()

                # ── Term 2: KL(student_throttle ‖ teacher_longi_throttle) ─
                beta_active = (
                    beta_eff.mean().item() > 1e-6
                    if isinstance(beta_eff, torch.Tensor)
                    else beta_eff > 1e-6
                )
                if beta_active:
                    t2_mu, t2_log_std = self._get_teacher_distribution(
                        self.teacher_longitudinal.actor, obs_tensor
                    )
                    kl_throttle = gaussian_kl_div(
                        student_mu[:, THROTTLE_DIM:THROTTLE_DIM+1],
                        student_log_std[:, THROTTLE_DIM:THROTTLE_DIM+1],
                        t2_mu,
                        t2_log_std,
                    )  # [batch]

                    if isinstance(beta_eff, torch.Tensor):
                        weighted_kl_throttle = (
                            beta_eff.squeeze(-1) * kl_throttle
                        ).mean()
                    else:
                        weighted_kl_throttle = beta_eff * kl_throttle.mean()

                    kd_loss = kd_loss + weighted_kl_throttle
                    kl_throttle_val = kl_throttle.mean().item()

            # ── Total actor loss ──────────────────────────────────────────
            total_actor_loss = actor_loss_sac + kd_loss + entropy_loss
            actor_losses.append(total_actor_loss.item())

            # ── Optimize actor + attention module ─────────────────────────
            self.actor.optimizer.zero_grad()
            if self.attention_optimizer is not None:
                self.attention_optimizer.zero_grad()

            total_actor_loss.backward()

            self.actor.optimizer.step()
            if self.attention_optimizer is not None:
                self.attention_optimizer.step()

            # ── Soft update target networks ───────────────────────────────
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau
                )
                polyak_update(
                    self.batch_norm_stats,
                    self.batch_norm_stats_target,
                    1.0
                )

            kl_steers.append(kl_steer_val)
            kl_throttles.append(kl_throttle_val)
            attn_alpha_vals.append(step_attn_alpha)
            attn_beta_vals.append(step_attn_beta)
            attn_entropy_vals.append(step_attn_entropy)

        # SB3 increments _n_updates once after the loop
        self._n_updates += gradient_steps

        # Update exposed KD metrics
        self.kd_metrics.update({
            "alpha":           self.kd_alpha,
            "beta":            self.kd_beta,
            "kl_steer":        float(np.mean(kl_steers)) if kl_steers else 0.0,
            "kl_throttle":     float(np.mean(kl_throttles)) if kl_throttles else 0.0,
            "global_mult":     self.kd_global_mult,
            "attn_alpha_mean": float(np.mean(attn_alpha_vals)) if attn_alpha_vals else 0.0,
            "attn_beta_mean":  float(np.mean(attn_beta_vals)) if attn_beta_vals else 0.0,
            "attn_entropy":    float(np.mean(attn_entropy_vals)) if attn_entropy_vals else 0.0,
            "critic_kd_loss":  float(np.mean(critic_kd_vals)) if critic_kd_vals else 0.0,
        })

        # SB3 internal logger
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        self.logger.record("kd/alpha",          self.kd_alpha)
        self.logger.record("kd/beta",           self.kd_beta)
        self.logger.record("kd/global_mult",    self.kd_global_mult)
        self.logger.record("kd/kl_steer",       self.kd_metrics["kl_steer"])
        self.logger.record("kd/kl_throttle",    self.kd_metrics["kl_throttle"])
        self.logger.record("kd/attn_alpha",     self.kd_metrics["attn_alpha_mean"])
        self.logger.record("kd/attn_beta",      self.kd_metrics["attn_beta_mean"])
        self.logger.record("kd/attn_entropy",   self.kd_metrics["attn_entropy"])
        self.logger.record("kd/critic_loss",    self.kd_metrics["critic_kd_loss"])

        self.policy.set_training_mode(False)
        if self.attention_module is not None:
            self.attention_module.eval()


# ═══════════════════════════════════════════════════════════════════════════
# KD LOGGING CALLBACK
# ═══════════════════════════════════════════════════════════════════════════
class KDLogCallback(BaseCallback):
    """
    Logs KD-specific metrics to CSV alongside standard training stats.

    For HA-KD-SAC, additionally logs:
    - attn_alpha_mean, attn_beta_mean: mean attention weights per rollout
    - attn_entropy: attention entropy (higher = more balanced)
    - global_mult: global annealing multiplier

    These CSVs produce the key evidence plots for the paper:
    1. KL divergence curves showing student "graduation" from teachers
    2. Attention weight evolution showing context-dependent behavior
    """

    def __init__(self, log_dir: str, model: "KDSAC", rolling_window: int = 50):
        super().__init__(verbose=0)
        self._kd_model = model
        self.rolling_window = rolling_window

        self._kl_history = deque(maxlen=rolling_window)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"kd_metrics_{now}.csv")
        os.makedirs(log_dir, exist_ok=True)

        self._f = open(self.csv_path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow([
            "timestep", "alpha", "beta", "global_mult",
            "kl_steer", "kl_throttle", "kl_total",
            "rolling_avg_kl_total",
            "attn_alpha_mean", "attn_beta_mean", "attn_entropy", "critic_kd_loss"
        ])
        mode_str = "HA-KD-SAC" if model.use_attention else "KD-SAC"
        if getattr(model, "use_critic_kd", False):
            mode_str = "DC-" + mode_str
            
        print(f"  [{mode_str}] Logging KL metrics → {self.csv_path}")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        """Log after every rollout collection."""
        m = self._kd_model.kd_metrics
        kl_total = m["kl_steer"] + m["kl_throttle"]
        self._kl_history.append(kl_total)
        rolling = float(np.mean(self._kl_history))

        self._w.writerow([
            self.model.num_timesteps,
            f"{m['alpha']:.5f}",
            f"{m['beta']:.5f}",
            f"{m['global_mult']:.5f}",
            f"{m['kl_steer']:.5f}",
            f"{m['kl_throttle']:.5f}",
            f"{kl_total:.5f}",
            f"{rolling:.5f}",
            f"{m['attn_alpha_mean']:.5f}",
            f"{m['attn_beta_mean']:.5f}",
            f"{m['attn_entropy']:.5f}",
            f"{m['critic_kd_loss']:.5f}",
        ])
        self._f.flush()

    def _on_training_end(self):
        self._f.close()
        m = self._kd_model
        print(f"\n  [KD] Final global_mult={m.kd_global_mult:.4f}")
        if m.use_attention:
            print(f"  [KD] Final attn α_mean={m.kd_metrics['attn_alpha_mean']:.4f}, "
                  f"β_mean={m.kd_metrics['attn_beta_mean']:.4f}")
            print(f"  [KD] Final entropy={m.kd_metrics['attn_entropy']:.4f}")
        else:
            print(f"  [KD] Final α={m.kd_alpha:.4f}, β={m.kd_beta:.4f}")
        print(f"  [KD] Metrics CSV: {self.csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════
def find_model(model_dir: str, keyword: str) -> str:
    """Locate the best available model by priority: final > best > latest."""
    candidates = [
        os.path.join(model_dir, f"sac_{keyword}_final.zip"),
        os.path.join(model_dir, f"sac_{keyword}_final"),
        os.path.join(model_dir, "best", "best_model.zip"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    if os.path.isdir(model_dir):
        zips = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir) if f.endswith(".zip")
        ]
        if zips:
            return max(zips, key=os.path.getmtime)
    raise FileNotFoundError(f"No model found for '{keyword}' in {model_dir}")


def transfer_weights(student: KDSAC, teacher: SAC, teacher_name: str) -> int:
    """
    Initialize student weights from teacher where shapes match.
    Same logic as train_finetune.py — provides good starting point before KD.
    """
    new_params  = student.policy.state_dict()
    tchr_params = teacher.policy.state_dict()
    transferred = OrderedDict()
    count = 0

    for key in new_params:
        if key in tchr_params and tchr_params[key].shape == new_params[key].shape:
            transferred[key] = tchr_params[key].clone()
            count += 1
        else:
            transferred[key] = new_params[key]

    student.policy.load_state_dict(transferred)
    print(f"  [{teacher_name}] Transferred {count}/{len(new_params)} param tensors")
    return count


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    algo_name = "HA-KD-SAC" if KD_USE_ATTENTION else "KD-SAC"
    if KD_USE_CRITIC:
        algo_name = "DC-" + algo_name
        
    print("=" * 70)
    print(f"  PHASE C: MULTI-TEACHER KNOWLEDGE DISTILLATION ({algo_name})")
    print(f"  Timesteps: {KD_TIMESTEPS:,}")
    print(f"  α_start={KD_ALPHA_START}  β_start={KD_BETA_START}")
    
    if KD_USE_CRITIC:
        print(f"  Critic KD: ENABLED (DC-KD-SAC), coef={KD_CRITIC_COEF}")
    else:
        print(f"  Critic KD: DISABLED")
        
    if KD_USE_ATTENTION:
        print(f"  Attention: hidden={KD_ATTENTION_HIDDEN} lr={KD_ATTENTION_LR}"
              f" entropy_coef={KD_ENTROPY_COEF}")
        print(f"  Global anneal: {KD_GLOBAL_ANNEAL}")
    else:
        print(f"  Mode: Fixed linear annealing → 0")
    print("=" * 70)

    # --- Reproducibility ---
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    # --- Locate teacher models ---
    lateral_path      = find_model(cfg.STAGE1_MODEL_DIR, "lateral")
    longitudinal_path = find_model(cfg.STAGE2_MODEL_DIR, "longitudinal")
    print(f"\n  Teacher 1 (Lateral):      {lateral_path}")
    print(f"  Teacher 2 (Longitudinal): {longitudinal_path}")

    # --- Load frozen teachers ---
    print("\n  Loading teachers... ", end="", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_lateral      = SAC.load(lateral_path, device=device)
    teacher_longitudinal = SAC.load(longitudinal_path, device=device)
    print("✓")

    # --- Environments ---
    train_env = FullControlWrapper(
        SumoEnv(render=False, traffic_scale=cfg.TRAFFIC_SCALE)
    )
    eval_env = FullControlWrapper(
        SumoEnv(render=False, traffic_scale=cfg.TRAFFIC_SCALE)
    )

    # --- KD-SAC hyperparameters ---
    kd_sac_params = cfg.SAC_PARAMS.copy()
    kd_sac_params["learning_rate"]   = 1e-4
    kd_sac_params["learning_starts"] = 2_000
    kd_sac_params["batch_size"]      = 256

    # --- Build KD-SAC student ---
    print(f"\n  Building {algo_name} student model...")
    model = KDSAC(
        policy="MlpPolicy",
        env=train_env,
        teacher_lateral=teacher_lateral,
        teacher_longitudinal=teacher_longitudinal,
        kd_alpha=KD_ALPHA_START,
        kd_beta=KD_BETA_START,
        kd_anneal_steps=KD_TIMESTEPS,
        verbose=0,
        device="auto",
        **kd_sac_params,
    )

    # --- Weight initialization from best teacher (Stage 2) ---
    print("\n  Weight transfer from Stage 2 (Longitudinal) → Student...")
    n_from_lon = transfer_weights(model, teacher_longitudinal, "Stage2")

    # Also fill from Stage 1 where Stage 2 couldn't (mismatched action heads)
    print("  Weight transfer from Stage 1 (Lateral) → gaps...")
    lon_sd  = teacher_longitudinal.policy.state_dict()
    lat_sd  = teacher_lateral.policy.state_dict()
    cur_sd  = model.policy.state_dict()
    extra = 0
    for key in cur_sd:
        lon_ok = key in lon_sd and lon_sd[key].shape == cur_sd[key].shape
        lat_ok = key in lat_sd and lat_sd[key].shape == cur_sd[key].shape
        if not lon_ok and lat_ok:
            cur_sd[key] = lat_sd[key].clone()
            extra += 1
    if extra > 0:
        model.policy.load_state_dict(cur_sd)
        print(f"  [Stage1] Additional {extra} tensors from lateral teacher")

    total_transferred = n_from_lon + extra
    total_params = len(model.policy.state_dict())
    print(f"\n  Weight Transfer Summary: {total_transferred}/{total_params}"
          f" tensors initialized")
    print(f"  Action heads (randomly initialized):"
          f" {total_params - total_transferred}/{total_params}")

    # --- Callbacks ---
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(KD_MODEL_DIR, "best/"),
        log_path=KD_LOG_DIR,
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=os.path.join(KD_MODEL_DIR, "checkpoints/"),
        name_prefix="kd_sac",
    )

    episode_log_cb = CurriculumLogCallback(
        log_dir=KD_LOG_DIR,
        stage_name="KD_FINETUNE",
        target_timesteps=KD_TIMESTEPS,
        verbose=1,
    )

    kd_log_cb = KDLogCallback(
        log_dir=KD_LOG_DIR,
        model=model,
        rolling_window=50,
    )

    all_callbacks = CallbackList([eval_cb, ckpt_cb, episode_log_cb, kd_log_cb])

    # --- Train ---
    final_path = os.path.join(KD_MODEL_DIR, "kd_sac_final")

    print(f"\n{'─' * 70}")
    print(f"  Starting {algo_name} training:")
    if KD_USE_ATTENTION:
        print(f"    Mode: Context-Aware Attention (learned α(s), β(s))")
        print(f"    Global anneal: 1.0 → 0.0 over {KD_TIMESTEPS:,} steps")
    else:
        print(f"    α: {KD_ALPHA_START} → 0  (linear over {KD_TIMESTEPS:,} steps)")
        print(f"    β: {KD_BETA_START} → 0  (linear over {KD_TIMESTEPS:,} steps)")
    print(f"    Budget: {KD_TIMESTEPS:,} timesteps")
    print(f"    Output: {KD_MODEL_DIR}")
    print(f"{'─' * 70}\n")

    try:
        model.learn(
            total_timesteps=KD_TIMESTEPS,
            callback=all_callbacks,
        )
        model.save(final_path)
        print(f"\n[{algo_name}] Training complete! Model saved: {final_path}")
        print(f"  KD metrics CSV:  {KD_LOG_DIR}")

        # Save attention module separately for analysis/visualization
        if model.use_attention:
            attn_path = os.path.join(KD_MODEL_DIR, "attention_module.pt")
            model.save_attention_module(attn_path)

    except KeyboardInterrupt:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(KD_MODEL_DIR, f"kd_sac_interrupted_{ts}")
        model.save(path)
        print(f"\n[{algo_name}] Interrupted! Saved: {path}")
        if model.use_attention:
            attn_path = os.path.join(KD_MODEL_DIR, f"attention_interrupted_{ts}.pt")
            model.save_attention_module(attn_path)

    except Exception as e:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(KD_MODEL_DIR, f"kd_sac_CRASH_{ts}")
        model.save(path)
        print(f"\n[{algo_name}] CRASH! Emergency save: {path}")
        raise e

    finally:
        train_env.close()
        eval_env.close()
        del teacher_lateral, teacher_longitudinal


if __name__ == "__main__":
    main()
