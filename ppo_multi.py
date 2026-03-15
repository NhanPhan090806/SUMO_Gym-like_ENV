import os
import csv
import torch
import numpy as np
import gymnasium as gym
from datetime import datetime
import multiprocessing as mp

# Tianshou imports
from tianshou.policy import PPOPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv # Thay đổi từ Dummy sang Subproc
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.trainer import OnpolicyTrainer

# Import the provided SUMO environment

from simulation.env_multi import SumoEnv, BASE_TRACI_PORT



# =============================================================================
# CONFIGURATION
# =============================================================================

MAP_CONFIGS = [
    "maps/map1/run.sumocfg"
]

LOG_DIR   = "reports/tianshou_ppo/"
MODEL_DIR = "models/tianshou_ppo/"
CSV_FILENAME = f"training_log_multi{datetime.now().strftime('%d%m%Y_%H%M%S')}.csv"
CSV_PATH  = os.path.join(LOG_DIR, CSV_FILENAME)

SEED = 42

CSV_HEADER = [
    "episode", "steps", "ep_reward",
    "avg_speed", "total_energy", "wiggle",
    "safety", "success", "reason"
]

# --- Parallel environments ---
# Each SumoEnv instance owns its own TraCI label + port (see env file), so
# all NUM_ENVS processes run independently without interfering with each other.
# DummyVectorEnv is kept (not SubprocVectorEnv) to avoid TraCI fork issues.
NUM_ENVS = 4

# --- Hyperparameters ---
LR               = 3e-4
GAMMA            = 0.99
GAE_LAMBDA       = 0.95
MAX_GRAD_NORM    = 0.5
VF_COEF          = 0.25
ENT_COEF         = 0.01
EPOCH            = 500
STEP_PER_COLLECT = 2048          # steps gathered per collection round
STEP_PER_EPOCH   = STEP_PER_COLLECT  # one collection round per epoch
REPEAT_PER_COLLECT = 10          # PPO update passes per collection
BATCH_SIZE       = 128
# Buffer must hold at least one full collection; give 2× headroom.
# VectorReplayBuffer splits BUFFER_SIZE into NUM_ENVS equal sub-buffers
# automatically — no manual per-env calculation needed.
BUFFER_SIZE      = STEP_PER_COLLECT * 2

# --- Reward-shaping coefficients (tune to your env's reward scale) ---
ENERGY_PENALTY_COEF  = 0.01   # penalise energy consumption each step
SAFETY_PENALTY_COEF  = 1.0    # penalise proximity / collision risk
WIGGLE_PENALTY_COEF  = 0.05   # penalise jerk for smoother, more efficient driving

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# SILENT LOGGER  (CSV logging is handled by MetricsWrapper instead)
# =============================================================================

class SilentLogger:
    """Satisfies Tianshou's logger interface without doing anything."""

    def write(self, *args, **kwargs):           pass
    def log_train_data(self, *args, **kwargs):  pass
    def log_test_data(self, *args, **kwargs):   pass
    def log_update_data(self, *args, **kwargs): pass
    def save_data(self, *args, **kwargs):       pass
    def restore_data(self):                     pass  # required by some Tianshou versions


# =============================================================================
# CSV HELPER
# =============================================================================

def init_csv_logging(filepath: str) -> None:
    """Create log file with header if it does not already exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
        print(f"[Logger] Created new log file : {filepath}")
    else:
        print(f"[Logger] Appending to existing: {filepath}")


# =============================================================================
# METRICS WRAPPER
# =============================================================================

class MetricsWrapper(gym.Wrapper):
    """
    Wraps SumoEnv to:
      1. Apply reward shaping for energy efficiency, safety, and smoothness.
      2. Accumulate per-episode statistics.
      3. Write one CSV row immediately when an episode ends.

    The episode counter is thread-safe for single-process (DummyVectorEnv) use
    and is initialised from the existing file so resuming a run is seamless.
    """

    def __init__(self, env: gym.Env, log_filepath: str) -> None:
        super().__init__(env)
        self.log_filepath = log_filepath
        self._reset_accumulators()

        # Initialise episode counter from existing file (resume-friendly)
        self.global_ep_cnt = 0
        if os.path.exists(self.log_filepath):
            try:
                with open(self.log_filepath, "r") as f:
                    self.global_ep_cnt = max(0, sum(1 for _ in f) - 1)  # subtract header
            except Exception:
                self.global_ep_cnt = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_accumulators(self) -> None:
        self.episode_base_reward  = 0.0
        self.episode_shaped_reward = 0.0
        self.episode_energy       = 0.0
        self.episode_speed_sum    = 0.0
        self.episode_jerk_sum     = 0.0
        self.episode_safety_sum   = 0.0
        self.episode_steps        = 0

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        self._reset_accumulators()
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # ---- Reward shaping ------------------------------------------------
        energy = info.get("real_energy", 0.0)
        safety = info.get("safety", 1.0)   # assume 1.0 = perfectly safe
        wiggle = info.get("wiggle", 0.0)

        # All penalties are non-positive so they can only reduce (never inflate) reward
        energy_penalty  = -ENERGY_PENALTY_COEF  * abs(energy)
        safety_penalty  = -SAFETY_PENALTY_COEF  * max(0.0, 1.0 - safety)  # 0 when safe
        wiggle_penalty  = -WIGGLE_PENALTY_COEF  * abs(wiggle)

        shaped_reward = reward + energy_penalty + safety_penalty + wiggle_penalty

        # ---- Accumulate stats ----------------------------------------------
        self.episode_base_reward   += reward
        self.episode_shaped_reward += shaped_reward
        self.episode_energy        += energy
        self.episode_speed_sum     += info.get("real_speed", 0.0)
        self.episode_jerk_sum      += abs(wiggle)
        self.episode_safety_sum    += safety
        self.episode_steps         += 1

        # ---- End-of-episode logging ----------------------------------------
        if terminated or truncated:
            self.global_ep_cnt += 1
            n = max(1, self.episode_steps)

            row = [
                self.global_ep_cnt,
                self.episode_steps,
                f"{self.episode_shaped_reward:.2f}",
                f"{self.episode_speed_sum  / n:.2f}",
                f"{self.episode_energy:.2f}",
                f"{self.episode_jerk_sum   / n:.4f}",
                f"{self.episode_safety_sum / n:.4f}",
                info.get("is_success", 0),
                info.get("reason", "unknown"),
            ]
            try:
                with open(self.log_filepath, "a", newline="") as f:
                    csv.writer(f).writerow(row)
            except Exception as e:
                print(f"[Logger] CSV write error: {e}")

        return obs, shaped_reward, terminated, truncated, info


# =============================================================================
# NETWORK UTILITIES
# =============================================================================

def _init_weights(modules) -> None:
    """Orthogonal weight init + zero bias — standard for PPO."""
    for m in modules:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)


def make_actor_critic(state_shape, action_shape):
    """
    Returns (actor, critic, optimiser) with SEPARATE backbone networks.
    Sharing a single Net between actor and critic causes gradient interference
    and is a critical bug — this function avoids that entirely.
    """
    # Two independent feature extractors
    actor_net = Net(state_shape, hidden_sizes=[256, 256], device=DEVICE)
    critic_net = Net(state_shape, hidden_sizes=[256, 256], device=DEVICE)

    actor  = ActorProb(actor_net,  action_shape, device=DEVICE, unbounded=True).to(DEVICE)
    critic = Critic(critic_net, device=DEVICE).to(DEVICE)

    _init_weights(list(actor.modules()))
    _init_weights(list(critic.modules()))

    # Parameters are disjoint so no set() deduplication needed
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=LR
    )
    return actor, critic, optim


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_ppo() -> None:

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Logging setup ─────────────────────────────────────────────────────────
    init_csv_logging(CSV_PATH)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Environment factory ───────────────────────────────────────────────────
    def make_env(rank):
        def _init():
            env = SumoEnv(
                rank = rank,
                render=False,          # set True for SUMO GUI (much slower)
                map_config=MAP_CONFIGS,
                test_mode=False,
            )
            return MetricsWrapper(env, CSV_PATH)
        return _init

    # DummyVectorEnv runs all envs sequentially in one process.
    # Each SumoEnv owns a unique TraCI label + port, so the NUM_ENVS SUMO
    # processes are fully independent and never cross-talk.
    # Khởi tạo 8 môi trường song song trên 8 tiến trình riêng biệt
    train_envs = SubprocVectorEnv([make_env(i) for i in range(NUM_ENVS)])
    test_envs  = SubprocVectorEnv([make_env(i + NUM_ENVS) for i in range(2)])

    train_envs.seed(SEED)
    test_envs.seed(SEED + 100)

    # ── Networks ──────────────────────────────────────────────────────────────
    state_shape  = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].shape

    actor, critic, optim = make_actor_critic(state_shape, action_shape)

    # ── PPO Policy ────────────────────────────────────────────────────────────
    #
    # FIX: dist_fn must be a factory that accepts (mu, sigma) and returns a
    # *batched* distribution.  Using Independent(..., 1) correctly sums
    # log-probs across action dimensions — critical for multi-dim actions.
    #
    def dist_fn(mu, sigma):
        return torch.distributions.Independent(
            torch.distributions.Normal(mu, sigma), 1
        )

    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,
        action_space=train_envs.action_space[0],
        discount_factor=GAMMA,
        gae_lambda=GAE_LAMBDA,
        max_grad_norm=MAX_GRAD_NORM,
        vf_coef=VF_COEF,
        ent_coef=ENT_COEF,
        action_scaling=True,
        action_bound_method="clip",
    )

    # ── Collectors ────────────────────────────────────────────────────────────
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(BUFFER_SIZE, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs)

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def save_best_fn(policy):
        ts   = datetime.now().strftime("%d%m%Y_%H%M%S")
        path = os.path.join(MODEL_DIR, f"best_policy_multi{ts}.pth")
        torch.save(policy.state_dict(), path)
        print(f"[Checkpoint] Best model saved → {path}")

    def save_checkpoint(tag: str) -> None:
        ts   = datetime.now().strftime("%d%m%Y_%H%M%S")
        path = os.path.join(MODEL_DIR, f"multi_{tag}_{ts}.pth")
        torch.save(policy.state_dict(), path)
        print(f"[Checkpoint] {tag} saved → {path}")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=EPOCH,
        step_per_epoch=STEP_PER_EPOCH,
        step_per_collect=STEP_PER_COLLECT,
        repeat_per_collect=REPEAT_PER_COLLECT,
        episode_per_test=10,
        batch_size=BATCH_SIZE,
        save_best_fn=save_best_fn,
        logger=SilentLogger(),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"[Training] Device      : {DEVICE}")
    print(f"[Training] Num envs    : {NUM_ENVS}  (TraCI ports {BASE_TRACI_PORT}–{BASE_TRACI_PORT + NUM_ENVS - 1})")
    print(f"[Training] Log file    : {CSV_PATH}")
    print(f"[Training] Reward shape: energy×{ENERGY_PENALTY_COEF}"
          f"  safety×{SAFETY_PENALTY_COEF}  wiggle×{WIGGLE_PENALTY_COEF}")
    print("Press Ctrl+C to stop and save.\n")

    try:
        for epoch, epoch_stat, info in trainer:
            print(
                f"Epoch {epoch:>4d} | "
                f"rew={epoch_stat['rew']:+8.2f} | "
                f"clip_loss={info.get('loss/clip', 0.0):.4f} | "
                f"vf_loss={info.get('loss/vf',   0.0):.4f} | "
                f"ent={info.get('loss/ent',  0.0):.4f}"
            )

    except KeyboardInterrupt:
        print("\n[Training] Interrupted by user — saving emergency checkpoint.")
        save_checkpoint("emergency_save")

    except Exception as e:
        print(f"\n[Training] Critical error: {e}")
        save_checkpoint("crash_save")
        raise

    finally:
        train_envs.close()
        test_envs.close()
        print("[Training] Environments closed.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # CỰC KỲ QUAN TRỌNG: Ép sử dụng 'spawn' để tránh deadlock trên WSL/Linux
    try:
        mp.set_start_method('spawn', force=True)
        print("[System] Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        pass
        
    train_ppo()