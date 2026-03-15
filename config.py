# config.py
# Centralized configuration for the 3-Stage Curriculum Learning pipeline.
# All hyperparameters, reward weights, and directory paths are defined here.

import os
from datetime import datetime

# =============================================================================
# GLOBAL / ENVIRONMENT
# =============================================================================
# Read seed from ENV for multi-seed generalization experiments (Phase D)
SEED = int(os.environ.get("SUMO_RL_SEED", "42"))

# Simulation speed & layout factors

# =============================================================================
# SUMO & VEHICLE PHYSICS
# =============================================================================
# Resolve absolute path to the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MAP_CONFIG = [
    os.path.join(BASE_DIR, "maps/map_grid_tuned/run.sumocfg"),
]

# Vinfast VF9 Specs
MAX_SPEED   = 55.6    # m/s (~200 km/h)
MAX_ACCEL   = 4.15    # m/s^2
MAX_DECEL   = 6.0     # m/s^2
MAX_ELEC    = 120.0   # Wh/s  (normalization ceiling)
MAX_SLOPE   = 20.0    # degrees
MAX_DIST    = 100.0   # meters (sensor range)
TARGET_DIST = 35.0    # meters (safe following distance)
MAX_LAT_OFFSET = 1.6  # meters (half standard 3.2m lane width)

TRAFFIC_SCALE      = 0.8
DRIVER_IMPERFECTION = 0.5
DRIVER_IMPATIENCE   = 0.5
MAX_EPISODE_STEPS   = 500
SIM_STEPS_PER_ACTION = 2  # micro-steps per macro-action

# Lane-change threshold for steering command
# Lowered from 0.3 → 0.1 so agent can trigger lane changes more easily.
# With the old 0.3 threshold, the "dead zone" was too large and caused
# collisions when the agent hesitated near the boundary.
LC_THRESHOLD = 0.1

# Desired speed range for reward shaping (m/s)
# ~36-80 km/h is the "sweet spot" for urban driving
MIN_DESIRED_SPEED = 10.0   # ~36 km/h — below this is considered too slow
TARGET_SPEED_RATIO = 0.9   # Target 90% of speed limit

# =============================================================================
# HYPERPARAMETERS (Chosen Algorithm)
# =============================================================================
SAC_PARAMS = {
    "learning_rate":  3e-4,
    "buffer_size":    100_000,
    "batch_size":     256,
    "tau":            0.005,
    "gamma":          0.99,
    "ent_coef":       "auto",
    "learning_starts": 1_000,
    "train_freq":     1,
    "gradient_steps":  1,
    "seed":           SEED,
}


PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 1024,        # good balance for single env
    "batch_size": 256,      # divides 1024 cleanly
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,       # small exploration boost for continuous control
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": SEED,
}

TD3_PARAMS = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "buffer_size": 200_000,
    "batch_size": 128,
    "total_timesteps": 2_000_000,
    "step_per_epoch": 4096,
    "step_per_collect": 100,
    "episode_per_test": 10,
    "update_per_step": 2,
    "noise_sigma": 0.3,
    "policy_noise": 0.1,
    "noise_clip": 0.2,
    "update_actor_freq": 2,
    "seed": SEED,
    "load_checkpoint": None,
}

# =============================================================================
# TRAINING TIMESTEPS PER STAGE
# =============================================================================
STAGE1_TIMESTEPS = 150_000   # Lateral control
STAGE2_TIMESTEPS = 200_000   # Longitudinal / Energy
STAGE3_TIMESTEPS = 150_000   # Fine-tuning (Best Teacher Transfer)

# =============================================================================
# KNOWLEDGE DISTILLATION HYPERPARAMETERS (Phase C)
# =============================================================================
# KD-SAC: Student learns from two frozen teachers via KL divergence in actor loss
# loss_total = loss_SAC + α·KL(student_steer ‖ teacher_lateral)
#                       + β·KL(student_throttle ‖ teacher_longitudinal)
KD_ALPHA_START  = 1.0     # Initial weight for lateral (steering) KD term
KD_BETA_START   = 1.0     # Initial weight for longitudinal (throttle) KD term
KD_ANNEAL_FRAC  = 1.0     # Fraction of training over which α,β decay to 0
KD_TIMESTEPS    = STAGE3_TIMESTEPS  # Training budget for KD fine-tuning

# =============================================================================
# CONTEXT-AWARE ATTENTION KD (HA-KD-SAC — Novel Research Contribution)
# =============================================================================
# Instead of fixed linear annealing α(t)→0, a learned attention module outputs
# state-dependent weights α(s), β(s) per observation.
#   - When approaching intersection: α(s) ↑ (trust lateral teacher more)
#   - When on straight highway:      β(s) ↑ (trust longitudinal teacher more)
# A global annealing multiplier still decays to 0 so the student graduates.
KD_USE_ATTENTION    = True      # True = HA-KD-SAC (novel), False = linear annealing
KD_ATTENTION_HIDDEN = 64        # Hidden dimension for attention MLP
KD_ATTENTION_LR     = 3e-4      # Learning rate for attention module
KD_ENTROPY_COEF     = 0.01      # Entropy regularization (prevents collapse to 0 or 1)
KD_GLOBAL_ANNEAL    = True      # Apply global linear decay on top of attention

# =============================================================================
# CRITIC KNOWLEDGE DISTILLATION (DC-KD-SAC — Idea 1 Enhancement)
# =============================================================================
# In addition to matching policies (Actor KD), we can also distill the Value
# Function (Critic KD). This accelerates early convergence.
KD_USE_CRITIC   = False         # DISABLED: Teacher critics trained on 1D action space
                                # cause conflicting gradients with 2D student
KD_CRITIC_COEF  = 0.1           # (unused when KD_USE_CRITIC=False)


# =============================================================================
# REWARD WEIGHTS — Stage 1: Lateral Control
# =============================================================================
# Design principle: Agent should learn to stay in lane and change lanes safely,
# while maintaining at least a minimum forward speed.
# NOTE: Terminal rewards (collision, success, stuck) are handled by SumoEnv base class.
REWARD_STAGE1 = {
    "lane_center":   2.0,   # Reward for staying centered in lane
    "smooth_steer": -1.0,   # Penalty for jerky steering
    "safety_side":  -1.5,   # Penalty for being close to side vehicles
    "progress":      1.5,   # Strong reward for forward progress
    "speed_maintain": 1.0,  # Reward for maintaining good speed
    "too_slow":     -2.0,   # Penalty for going below MIN_DESIRED_SPEED
    # --- New fixes ---
    "front_danger": -3.0,   # Penalty when leader is very close AND agent isn't steering
    "lane_change":   0.5,   # Reward for successfully completing a lane change
}

# =============================================================================
# REWARD WEIGHTS — Stage 2: Longitudinal / Energy
# =============================================================================
# Design principle: Agent should learn ENERGY-EFFICIENT driving, NOT slow driving.
# Key fix: Energy penalty is now per-unit-distance (efficiency), not absolute.
# This means fast-and-efficient is BETTER than slow.
REWARD_STAGE2 = {
    "energy_efficiency": -0.8,  # Penalty based on Wh/m, not raw Wh/s
    "speed_target":       1.5,  # Strong reward for target speed
    "speed_maintain":     1.0,  # Reward for staying above minimum speed
    "too_slow":          -2.0,  # Explicit penalty for crawling
    "progress":           1.2,  # Strong forward progress reward
    "safety_leader":     -2.0,  # Penalty for following too close
    "tls_violation":     -3.0,  # Penalty for running red lights
    "smooth_accel":      -0.3,  # Penalty for throttle changes
}

# =============================================================================
# REWARD WEIGHTS — Stage 3: Fine-Tuning (KD Fusion) — UNIFIED OBJECTIVE
# =============================================================================
# Design principle: Aligned with sumo_env.py 's calculate_reward logic.
REWARD_STAGE3 = {
    "speed_target":        1.0,   # W_SPEED_TARGET
    "too_slow":           -1.5,   # W_TOO_SLOW
    "progress":            0.85,  # W_PROGRESS
    "energy_efficiency":  -0.15,  # W_ENERGY (Wh/m)
    "smooth_accel":       -0.05,  # W_COMFORT
    "safety_leader":      -1.0,   # W_SAFETY
    "tls_violation":      -50.0,  # W_RED_LIGHT
    "time":               -0.2,   # W_TIME
}

# =============================================================================
# ADAPTIVE CURRICULUM-AWARE REWARD SHAPING (ACARS — Idea 3 Enhancement)
# =============================================================================
# Automatically adjusts reward weights based on agent's current performance against
# desired targets. If an objective is mastered, its weight decays. If struggling,
# it amplifies.
USE_ACARS = False  # DISABLED: With short episodes (~22 steps), adaptive weights were
                   # unstable. Re-enable once episode length is consistently >100 steps.
ACARS_WINDOW = 20  # episodes to calculate moving average
ACARS_TARGETS = {
    "energy_efficiency": 0.3,   # Target: average efficiency penalty <= 0.3
    "speed_target": 0.8,        # Target: average speed target reward >= 0.8
    "safety_leader": 0.1,       # Target: average safety penalty <= 0.1 (closer to 0 is better)
    "progress": 0.6,            # Target: average progress reward >= 0.6
    "tls_violation": 0.05,      # Target: average TLS penalty <= 0.05
}

# =============================================================================
# TERMINAL REWARDS (applied by wrappers, NOT base env)
# =============================================================================
# Base env delegates terminal event signaling via info["terminal_event"].
# Each wrapper applies these rewards based on the event type.
TERMINAL_REWARDS = {
    "collision":      -200.0,
    "teleport":        -50.0,
    "vanished":         -5.0,    
    "stuck":           -50.0,
    "stuck_too_long": -400.0,
    "timeout":         -50.0,
    "goal":            200.0,
    "success":         200.0,
}

# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================
NOW_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Curriculum ---
BASE_REPORT_DIR = "./reports/curriculum/"
BASE_MODEL_DIR  = "./models/curriculum/"

STAGE1_MODEL_DIR  = os.path.join(BASE_MODEL_DIR, "stage1_lateral/")
STAGE2_MODEL_DIR  = os.path.join(BASE_MODEL_DIR, "stage2_longitudinal/")
STAGE3_MODEL_DIR  = os.path.join(BASE_MODEL_DIR, "stage3_finetune/")

STAGE1_LOG_DIR  = os.path.join(BASE_REPORT_DIR, "stage1_lateral/")
STAGE2_LOG_DIR  = os.path.join(BASE_REPORT_DIR, "stage2_longitudinal/")
STAGE3_LOG_DIR  = os.path.join(BASE_REPORT_DIR, "stage3_finetune/")

# --- Phase B: Baselines ---
FLAT_MODEL_DIR   = "./models/baselines/flat_ppo/"
FLAT_LOG_DIR     = "./reports/baselines/flat_ppo/"
RULE_LOG_DIR     = "./reports/baselines/rule_based/"
EVAL_OUTPUT_DIR  = "./reports/evaluation/"

# --- Phase C: Knowledge Distillation ---
KD_MODEL_DIR     = "./models/kd_finetune/"
KD_LOG_DIR       = "./reports/kd_finetune/"

# --- TD3 Config ---
TD3_MODEL_DIR    = "./models/tianshou_td3/"
TD3_LOG_DIR      = "./reports/tianshou_td3/"

# Create all directories
for d in [
    STAGE1_MODEL_DIR, STAGE2_MODEL_DIR, STAGE3_MODEL_DIR,
    STAGE1_LOG_DIR, STAGE2_LOG_DIR, STAGE3_LOG_DIR,
    FLAT_MODEL_DIR, FLAT_LOG_DIR, RULE_LOG_DIR, EVAL_OUTPUT_DIR,
    KD_MODEL_DIR, KD_LOG_DIR, TD3_MODEL_DIR, TD3_LOG_DIR,
]:
    os.makedirs(d, exist_ok=True)
