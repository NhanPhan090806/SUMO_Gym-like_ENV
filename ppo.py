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
from tianshou.policy import PPOPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.trainer import OnpolicyTrainer

# Import the provided SUMO environment

from simulation.sumo_env import SumoEnv


# --- ADD THIS CLASS AFTER IMPORTS ---
class SilentLogger:
    """A dummy logger that does nothing, satisfying Tianshou's requirements."""
    def __init__(self):
        pass

    def write(self, *args, **kwargs):
        pass

    def log_train_data(self, *args, **kwargs):
        pass

    def log_test_data(self, *args, **kwargs):
        pass

    def log_update_data(self, *args, **kwargs):
        pass

    def save_data(self, *args, **kwargs):
        pass


# --- CONFIGURATION ---
# UPDATE THESE PATHS TO MATCH YOUR ACTUAL FILES
MAP_CONFIGS = [
    "maps/map_grid_tuned/run.sumocfg"
]

LOG_DIR = "reports/tianshou_ppo/"
MODEL_DIR = "models/tianshou_ppo/" 
# Log file name (fixed per day to avoid too many files, or add timestamp)
CSV_FILENAME = f"training_log_{datetime.now().strftime('%d%m%Y_%H%M%S')}.csv"
CSV_PATH = os.path.join(LOG_DIR, CSV_FILENAME)
SEED = 69

# Header requested by user
CSV_HEADER = ["episode", "steps", "ep_reward", "avg_speed", "total_energy", "wiggle", "safety", "success", "reason", "route"]

# Hyperparameters
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
MAX_GRAD_NORM = 0.3
VF_COEF = 0.25
ENT_COEF = 0.02  # Tăng lên để khuyến khích agent thử nghiệm nhiều hành động hơn (exploration) thay vì hội tụ sớm

TOTAL_TIMESTEPS = 2000000
STEP_PER_EPOCH = 4096
REPEAT_PER_COLLECT = 10
BATCH_SIZE = 128
EPOCH = int(np.ceil(TOTAL_TIMESTEPS / STEP_PER_EPOCH))
BUFFER_SIZE = 8192
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
class MetricsWrapper(gym.Wrapper):
    """
    Wraps SumoEnv to calculate episode stats and write to CSV 
    IMMEDIATELY upon episode termination.
    """
    def __init__(self, env, log_filepath):
        super().__init__(env)
        self.log_filepath = log_filepath
        
        # Accumulators
        self.episode_reward = 0.0
        self.episode_energy = 0.0
        self.episode_speed_sum = 0.0
        self.episode_steps = 0
        self.episode_jerk_sum = 0.0
        self.episode_safety_sum = 0.0
        
        # Try to determine global episode count from file line count
        self.global_ep_cnt = 0
        if os.path.exists(self.log_filepath):
            try:
                with open(self.log_filepath, 'r') as f:
                    # Subtract 1 for header, ensure non-negative
                    self.global_ep_cnt = max(0, sum(1 for _ in f) - 1)
            except:
                self.global_ep_cnt = 0

    def reset(self, **kwargs):
        # Reset accumulators
        self.episode_reward = 0.0
        self.episode_energy = 0.0
        self.episode_speed_sum = 0.0
        self.episode_steps = 0
        self.episode_jerk_sum = 0.0
        self.episode_safety_sum = 0.0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Accumulate per-step data
        self.episode_reward += reward
        self.episode_energy += info.get("real_energy", 0.0)
        # Assuming info['real_speed'] is the instant speed or average provided by Env
        # We sum it up here to calculate our own average over the episode length
        self.episode_speed_sum += info.get("real_speed", 0.0) 
        self.episode_jerk_sum += info.get("wiggle", 0.0)
        self.episode_safety_sum += info.get("safety", 0.0)
        self.episode_steps += 1
        
        # LOGGING TRIGGER
        if terminated or truncated:
            self.global_ep_cnt += 1
            
            # Calculate Averages
            avg_speed = self.episode_speed_sum / max(1, self.episode_steps)
            avg_safety = self.episode_safety_sum / max(1, self.episode_steps)
            avg_jerk = self.episode_jerk_sum / max(1, self.episode_steps)
            
            success = info.get("is_success", 0)
            reason = info.get("reason", "unknown")
            route_str = info.get("route", "") if reason in ["stuck_too_long", "timeout", "teleport"] else ""
            
            # Prepare Row
            row = [
                self.global_ep_cnt,
                self.episode_steps,
                f"{self.episode_reward:.2f}",
                f"{avg_speed:.2f}",
                f"{self.episode_energy:.2f}",
                f"{avg_jerk:.4f}",
                f"{avg_safety:.4f}",
                success,
                reason,
                route_str
            ]
            
            # Write to CSV immediately (Append mode)
            try:
                with open(self.log_filepath, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            except Exception as e:
                print(f"Logging Error: {e}")
            
        return obs, reward, terminated, truncated, info

# --- MAIN TRAINING FUNCTION ---
def train_ppo():
    # 1. Init Logging
    init_csv_logging(CSV_PATH)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2. Define Environment Factory
    def make_env():
        # Pass the map list to the env
        env = SumoEnv(
            render=False, # Set to True if you want to see GUI (slower)
            map_config=MAP_CONFIGS,
            test_mode=False
        )
        # Wrap it with our logger
        env = MetricsWrapper(env, CSV_PATH)
        return env

    # 3. Vectorized Environment
    # DummyVectorEnv is used to avoid Multiprocessing issues with TraCI
    train_envs = DummyVectorEnv([make_env for _ in range(1)])
    test_envs = DummyVectorEnv([make_env for _ in range(1)])

    # 4. Network Setup
    state_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].shape
    
    net = Net(state_shape, hidden_sizes=[256, 256], device=DEVICE)
    actor = ActorProb(net, action_shape, device=DEVICE, unbounded=True).to(DEVICE)
    critic = Critic(net, device=DEVICE).to(DEVICE)
    
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    # Fixed: Removed duplicate parameters warning
    optim = torch.optim.Adam(set(list(actor.parameters()) + list(critic.parameters())), lr=LR)

    # 5. Policy
    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist_fn=torch.distributions.Normal,
        action_space=train_envs.action_space[0],
        discount_factor=GAMMA,
        gae_lambda=GAE_LAMBDA,
        max_grad_norm=MAX_GRAD_NORM,
        vf_coef=VF_COEF,
        ent_coef=ENT_COEF,
        action_scaling=True,
        action_bound_method="clip"
    )

    # 6. Collectors
    train_collector = Collector(
        policy, 
        train_envs, 
        VectorReplayBuffer(BUFFER_SIZE, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs)

    # 7. Define Save Hook
    def save_best_fn(policy):
       
        path = os.path.join(MODEL_DIR, f"best_policy_{datetime.now().strftime('%d%m%Y_%H%M%S')}.pth")
        torch.save(policy.state_dict(), path)
        print(f"Saved Best Model to {path}")

    # 8. Trainer
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=EPOCH,
        step_per_epoch=STEP_PER_EPOCH,
        repeat_per_collect=REPEAT_PER_COLLECT,
        episode_per_test=10,
        batch_size=BATCH_SIZE,
        step_per_collect=2048,
        save_best_fn=save_best_fn,
        logger=SilentLogger() # Logging is handled by the Wrapper now
    )

    # 9. Safe Training Loop
    print(f"Starting Training on device: {DEVICE}")
    print(f"Logging episodes to: {CSV_PATH}")
    print("Press Ctrl+C to stop and save.")

    try:
        # We iterate to print progress to console, but CSV writing happens in Wrapper
        for epoch, epoch_stat, info in trainer:
            print(f"Epoch {epoch}: Reward={epoch_stat['rew']:.2f}, Loss={info.get('loss/clip', 0.0):.4f}")

    except KeyboardInterrupt:
        print("\n\n!!! User Interrupted Training (Ctrl+C) !!!")
        print("Saving emergency checkpoint...")
        
        save_path = os.path.join(MODEL_DIR, f"emergency_save_{datetime.now().strftime('%d%m%Y_%H%M%S')}.pth")
        torch.save(policy.state_dict(), save_path)
        print(f"Model saved to: {save_path}")
        
    except Exception as e:
        print(f"\n!!! Critical Error: {e} !!!")
        save_path = os.path.join(MODEL_DIR, f"crash_save_{datetime.now().strftime('%d%m%Y_%H%M%S')}.pth")
        torch.save(policy.state_dict(), save_path)
        raise e
        
    finally:
        train_envs.close()
        test_envs.close()
        print("Training Closed.")

if __name__ == "__main__":
    train_ppo()