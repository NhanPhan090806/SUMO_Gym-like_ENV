#Đây là bản ban đầu
import os
import math
import time
import csv
import torch
import numpy as np
import gymnasium as gym
from datetime import datetime
from typing import Optional, List, Dict, Any

# Tianshou imports
from tianshou.policy import TD3Policy
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.trainer import OffpolicyTrainer
from tianshou.exploration import GaussianNoise
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
# Import the provided SUMO environment

from simulation.sumo_env import SumoEnv


from callbacks import TianshouLogCallback, TianshouDummyLogger
from wrappers import FullControlWrapper

# import the shared configuration
import config

# --- CONFIGURATION ---
MAP_CONFIGS = config.MAP_CONFIG[0] if isinstance(config.MAP_CONFIG, list) else config.MAP_CONFIG

LOG_DIR = config.TD3_LOG_DIR
MODEL_DIR = config.TD3_MODEL_DIR 
# Log file name (fixed per day to avoid too many files, or add timestamp)
CSV_FILENAME = f"training_log_{datetime.now().strftime('%d%m%Y_%H%M%S')}.csv"
CSV_PATH = os.path.join(LOG_DIR, CSV_FILENAME)
SEED = config.TD3_PARAMS.get("seed", config.SEED)
LOAD_CHECKPOINT = config.TD3_PARAMS.get("load_checkpoint", None)

# Header requested by user
CSV_HEADER = ["episode", "steps", "ep_reward", "avg_speed", "total_energy", "wiggle", "safety", "success", "reason"]

# Hyperparameters
LR = config.TD3_PARAMS["learning_rate"]
GAMMA = config.TD3_PARAMS["gamma"]

TOTAL_TIMESTEPS = config.TD3_PARAMS["total_timesteps"]
STEP_PER_EPOCH = config.TD3_PARAMS["step_per_epoch"]
BATCH_SIZE = config.TD3_PARAMS["batch_size"]
EPOCH = int(np.ceil(TOTAL_TIMESTEPS / STEP_PER_EPOCH))
BUFFER_SIZE = config.TD3_PARAMS["buffer_size"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- HELPER: INIT LOG FILE ---
def init_csv_logging(filepath):
    """Creates the file and writes header ONLY if file does not exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
        print(f"Created new log file: {filepath}")
    else:
        print(f"Appending to existing log file: {filepath}")

# --- CUSTOM WRAPPER FOR LOGGING ---
# Callbacks imported from callbacks.py


def save_checkpoint(path, policy, actor_optim, critic1_optim, critic2_optim, train_collector, epoch=None):
    checkpoint = {
        "policy": policy.state_dict(),
        "actor_optim": actor_optim.state_dict(),
        "critic1_optim": critic1_optim.state_dict(),
        "critic2_optim": critic2_optim.state_dict(),
        "replay_buffer": train_collector.buffer,
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "epoch": epoch
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved -> {path}")

def load_checkpoint(path, policy, actor_optim, critic1_optim, critic2_optim, train_collector, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    policy.load_state_dict(checkpoint["policy"])
    actor_optim.load_state_dict(checkpoint["actor_optim"])
    critic1_optim.load_state_dict(checkpoint["critic1_optim"])
    critic2_optim.load_state_dict(checkpoint["critic2_optim"])

    train_collector.buffer = checkpoint["replay_buffer"]

    # Restore numpy RNG
    if "numpy_rng_state" in checkpoint:
        try:
            np.random.set_state(checkpoint["numpy_rng_state"])
        except Exception:
            print("Warning: could not restore numpy RNG")

    # Restore torch RNG
    if "rng_state" in checkpoint:
        try:
            rng_state = checkpoint["rng_state"]
            if isinstance(rng_state, torch.Tensor):
                torch.set_rng_state(rng_state.cpu().to(torch.uint8))
        except Exception:
            print("Warning: could not restore torch RNG")

    # Restore CUDA RNG
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    start_epoch = checkpoint.get("epoch") or 0

    print(f"Checkpoint loaded from {path} (epoch {start_epoch})")

    return start_epoch


def update_exploration_noise(policy, epoch, max_epoch):
    start_sigma = 0.3
    end_sigma = 0.02

    progress = min(epoch / max_epoch, 1.0)
    sigma = end_sigma + (start_sigma - end_sigma) * (0.5 * (1 + np.cos(np.pi * progress)))

    # Replace exploration noise object
    policy._exploration_noise = GaussianNoise(sigma=sigma)

    return sigma

# --- MAIN TRAINING FUNCTION ---
def train_td3():
    # 1. Init Logging


    
    init_csv_logging(CSV_PATH)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2. Define Environment Factory
    def make_env():
        # Pass the map list to the env
        env = SumoEnv(
            render=False, # Set to True if you want to see GUI (slower)
            map_config=MAP_CONFIGS,
            test_mode=False,
            TRAFFIC_SCALE=config.TRAFFIC_SCALE
        )
        print(env.observation_space)
        # Wrap it with our stage 3 logic and custom logger
        env = FullControlWrapper(env)
        env = TianshouLogCallback(env, CSV_PATH)
        return env

    # 3. Vectorized Environment
    # DummyVectorEnv is used to avoid Multiprocessing issues with TraCI
    train_envs = DummyVectorEnv([make_env])
    test_envs = DummyVectorEnv([make_env])

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_envs.seed(SEED)
    test_envs.seed(SEED)

    # 4. Network Setup
    state_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].shape
    
    net_a = Net(state_shape, hidden_sizes=[256, 256], device=DEVICE, activation=torch.nn.ReLU)
    max_action = train_envs.action_space[0].high[0]

    actor = Actor(
        net_a,
        action_shape,
        max_action=max_action,
        device=DEVICE
    ).to(DEVICE)

    net_c1 = Net(state_shape, action_shape, hidden_sizes=[256, 256], concat=True, device=DEVICE, activation=torch.nn.ReLU)
    critic1 = Critic(net_c1, device=DEVICE).to(DEVICE)

    net_c2 = Net(state_shape, action_shape, hidden_sizes=[256, 256], concat=True, device=DEVICE, activation=torch.nn.ReLU)
    critic2 = Critic(net_c2, device=DEVICE).to(DEVICE)
    
    for m in list(actor.modules()) + list(critic1.modules()) + list(critic2.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    # Fixed: Removed duplicate parameters warning
    actor_optim = torch.optim.Adam(actor.parameters(), lr=LR)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=LR)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=LR)

    # 5. Policy
    policy = TD3Policy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=config.TD3_PARAMS["tau"],
        gamma=GAMMA,
        exploration_noise=GaussianNoise(sigma=config.TD3_PARAMS["noise_sigma"]),
        policy_noise=config.TD3_PARAMS["policy_noise"],
        update_actor_freq=config.TD3_PARAMS["update_actor_freq"],
        noise_clip=config.TD3_PARAMS["noise_clip"],
        action_scaling=True,
        action_bound_method="clip",
        action_space=train_envs.action_space[0]
    )

    # 6. Collectors
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(BUFFER_SIZE, len(train_envs))
    )


    start_epoch = 0

    if LOAD_CHECKPOINT is not None and os.path.exists(LOAD_CHECKPOINT):
        print(f"Loading checkpoint from {LOAD_CHECKPOINT}")
        start_epoch = load_checkpoint(
            LOAD_CHECKPOINT,
            policy,
            actor_optim,
            critic1_optim,
            critic2_optim,
            train_collector,
            DEVICE
        )



    print("Collecting random warmup data...")
    if LOAD_CHECKPOINT is None:
        print("Collecting random warmup data...")
        train_collector.collect(n_step=30000, random=True)
    else:
        print("Checkpoint loaded — skipping warmup.")
    test_collector = Collector(policy, test_envs)

    # 7. Define Save Hook
    def save_best_fn(policy):
        path = os.path.join(
            MODEL_DIR,
            f"best_checkpoint_{datetime.now().strftime('%d%m%Y_%H%M%S')}.pth"
        )

        save_checkpoint(
            path,
            policy,
            actor_optim,
            critic1_optim,
            critic2_optim,
            train_collector
        )

        print(f"Saved Best Model to {path}")

    # 8. Trainer
    trainer = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=EPOCH,
        step_per_epoch=STEP_PER_EPOCH,
        step_per_collect = config.TD3_PARAMS["step_per_collect"],
        episode_per_test=config.TD3_PARAMS["episode_per_test"],
        batch_size=BATCH_SIZE,
        update_per_step = config.TD3_PARAMS["update_per_step"],
        save_best_fn=save_best_fn,
        logger=TianshouDummyLogger() # Logging is handled by the Wrapper now
    )

    # 9. Safe Training Loop
    print(f"Starting Training on device: {DEVICE}")
    print(f"Logging episodes to: {CSV_PATH}")
    print("Press Ctrl+C to stop and save.")

    try:
        # We iterate to print progress to console, but CSV writing happens in Wrapper
        for epoch, epoch_stat, info in trainer:

            sigma = update_exploration_noise(policy, epoch+1, EPOCH)


            real_epoch = epoch + start_epoch


            print(
                f"Epoch {real_epoch}: Reward={epoch_stat['rew']:.2f}, "
                f"Loss={info.get('loss/actor',0.0):.4f}, "
                f"Noise={sigma:.3f}"
            )

            if epoch % 1 == 0:
                path = os.path.join(MODEL_DIR, f"periodic_checkpoint_{epoch%10}.pth")

                save_checkpoint(
                    path,
                    policy,
                    actor_optim,
                    critic1_optim,
                    critic2_optim,
                    train_collector,
                    epoch
                )

                latest_path = os.path.join(MODEL_DIR, "latest_checkpoint.pth")

                save_checkpoint(
                    latest_path,
                    policy,
                    actor_optim,
                    critic1_optim,
                    critic2_optim,
                    train_collector,
                    epoch
                )

    except KeyboardInterrupt:
        print("\n\n!!! User Interrupted Training (Ctrl+C) !!!")
        print("Saving emergency checkpoint...")
        
        save_path = os.path.join(MODEL_DIR, f"emergency_save_{datetime.now().strftime('%d%m%Y_%H%M%S')}.pth")
        save_checkpoint(
            save_path,
            policy,
            actor_optim,
            critic1_optim,
            critic2_optim,
            train_collector,
            epoch
        )
        print(f"Model saved to: {save_path}")
        
    except Exception as e:
        print(f"\n!!! Critical Error: {e} !!!")
        save_path = os.path.join(MODEL_DIR, f"crash_save_{datetime.now().strftime('%d%m%Y_%H%M%S')}.pth")
        save_checkpoint(
            save_path,
            policy,
            actor_optim,
            critic1_optim,
            critic2_optim,
            train_collector,
            epoch
        )
        raise e
        
    finally:
        train_envs.close()
        test_envs.close()
        print("Training Closed.")

if __name__ == "__main__":
    train_td3()