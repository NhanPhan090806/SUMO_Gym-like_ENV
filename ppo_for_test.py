#Đây là bản để chạy test model
import traci
import os
import csv
import sys
import torch
import argparse
import numpy as np
import time
import gymnasium as gym
from datetime import datetime
from pathlib import Path

# Tianshou imports
from tianshou.policy import PPOPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.trainer import OnpolicyTrainer

# Import the provided SUMO environment
try:
    from simulation.continuous_sumo_env import SumoEnv
except ImportError:
    try:
        from continuous_sumo_env import SumoEnv
    except ImportError:
        # Fallback giả định nếu không tìm thấy file môi trường
        print("Warning: Could not import SumoEnv. Make sure the file exists.")
        SumoEnv = None


# =============================================================================
# CONFIGURATION
# =============================================================================

MAP_CONFIGS = [
    "maps/map1/run.sumocfg"
]

LOG_DIR   = "reports/tianshou_ppo/"
MODEL_DIR = "models/tianshou_ppo/"
CSV_FILENAME = f"training_log_{datetime.now().strftime('%d%m%Y_%H%M%S')}.csv"
CSV_PATH  = os.path.join(LOG_DIR, CSV_FILENAME)

SEED = 42

CSV_HEADER = [
    "episode", "steps", "ep_reward",
    "avg_speed", "total_energy", "wiggle",
    "safety", "success", "reason"
]

# --- Hyperparameters ---
LR               = 3e-4
GAMMA            = 0.99
GAE_LAMBDA       = 0.95
MAX_GRAD_NORM    = 0.5
VF_COEF          = 0.25
ENT_COEF         = 0.01
EPOCH            = 500
STEP_PER_COLLECT = 2048          
STEP_PER_EPOCH   = STEP_PER_COLLECT  
REPEAT_PER_COLLECT = 10          
BATCH_SIZE       = 64
BUFFER_SIZE      = STEP_PER_COLLECT * 2

# Network Architecture (Must be consistent between Train and Test)
HIDDEN_SIZES     = [256, 256]

# --- Reward-shaping coefficients ---
ENERGY_PENALTY_COEF  = 0.01   
SAFETY_PENALTY_COEF  = 1.0    
WIGGLE_PENALTY_COEF  = 0.05   

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# SILENT LOGGER
# =============================================================================

class SilentLogger:
    """Satisfies Tianshou's logger interface without doing anything."""
    def write(self, *args, **kwargs):           pass
    def log_train_data(self, *args, **kwargs):  pass
    def log_test_data(self, *args, **kwargs):   pass
    def log_update_data(self, *args, **kwargs): pass
    def save_data(self, *args, **kwargs):       pass
    def restore_data(self):                     pass


# =============================================================================
# CSV HELPER & METRICS WRAPPER
# =============================================================================

def init_csv_logging(filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
        print(f"[Logger] Created new log file : {filepath}")
    else:
        print(f"[Logger] Appending to existing: {filepath}")

class MetricsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, log_filepath: str) -> None:
        super().__init__(env)
        self.log_filepath = log_filepath
        self._reset_accumulators()
        self.global_ep_cnt = 0
        if os.path.exists(self.log_filepath):
            try:
                with open(self.log_filepath, "r") as f:
                    self.global_ep_cnt = max(0, sum(1 for _ in f) - 1)
            except Exception:
                self.global_ep_cnt = 0

    def _reset_accumulators(self) -> None:
        self.episode_base_reward  = 0.0
        self.episode_shaped_reward = 0.0
        self.episode_energy       = 0.0
        self.episode_speed_sum    = 0.0
        self.episode_jerk_sum     = 0.0
        self.episode_safety_sum   = 0.0
        self.episode_steps        = 0

    def reset(self, **kwargs):
        self._reset_accumulators()
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        energy = info.get("real_energy", 0.0)
        safety = info.get("safety", 1.0)
        wiggle = info.get("wiggle", 0.0)

        energy_penalty  = -ENERGY_PENALTY_COEF  * abs(energy)
        safety_penalty  = -SAFETY_PENALTY_COEF  * max(0.0, 1.0 - safety)
        wiggle_penalty  = -WIGGLE_PENALTY_COEF  * abs(wiggle)

        shaped_reward = reward + energy_penalty + safety_penalty + wiggle_penalty

        self.episode_base_reward   += reward
        self.episode_shaped_reward += shaped_reward
        self.episode_energy        += energy
        self.episode_speed_sum     += info.get("real_speed", 0.0)
        self.episode_jerk_sum      += abs(wiggle)
        self.episode_safety_sum    += safety
        self.episode_steps         += 1

        if terminated or truncated:
            self.global_ep_cnt += 1
            n = max(1, self.episode_steps)
            row = [
                self.global_ep_cnt, self.episode_steps,
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
            except Exception:
                pass
        return obs, shaped_reward, terminated, truncated, info


# =============================================================================
# NETWORK UTILITIES
# =============================================================================

def _init_weights(modules) -> None:
    for m in modules:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

def make_actor_critic(state_shape, action_shape):
    # Use HIDDEN_SIZES constant to ensure consistency
    actor_net = Net(state_shape, hidden_sizes=HIDDEN_SIZES, device=DEVICE)
    critic_net = Net(state_shape, hidden_sizes=HIDDEN_SIZES, device=DEVICE)

    actor  = ActorProb(actor_net,  action_shape, device=DEVICE, unbounded=True).to(DEVICE)
    critic = Critic(critic_net, device=DEVICE).to(DEVICE)

    _init_weights(list(actor.modules()))
    _init_weights(list(critic.modules()))

    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=LR
    )
    return actor, critic, optim

def dist_fn(mu, sigma):
    return torch.distributions.Independent(
        torch.distributions.Normal(mu, sigma), 1
    )


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_ppo() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    init_csv_logging(CSV_PATH)
    os.makedirs(MODEL_DIR, exist_ok=True)

    def make_env():
        env = SumoEnv(render=True, map_config=MAP_CONFIGS, test_mode=False)
        return MetricsWrapper(env, CSV_PATH)

    train_envs = DummyVectorEnv([make_env for _ in range(1)])
    test_envs  = DummyVectorEnv([make_env for _ in range(1)])
    train_envs.seed(SEED)
    test_envs.seed(SEED + 100)

    state_shape  = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].shape

    actor, critic, optim = make_actor_critic(state_shape, action_shape)

    policy = PPOPolicy(
        actor=actor, critic=critic, optim=optim, dist_fn=dist_fn,
        action_space=train_envs.action_space[0],
        discount_factor=GAMMA, gae_lambda=GAE_LAMBDA,
        max_grad_norm=MAX_GRAD_NORM, vf_coef=VF_COEF, ent_coef=ENT_COEF,
        action_scaling=True, action_bound_method="clip",
    )

    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(BUFFER_SIZE, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs)

    def save_best_fn(policy):
        ts   = datetime.now().strftime("%d%m%Y_%H%M%S")
        path = os.path.join(MODEL_DIR, f"best_policy_{ts}.pth")
        torch.save(policy.state_dict(), path)
        print(f"[Checkpoint] Best model saved → {path}")

    def save_checkpoint(tag: str) -> None:
        ts   = datetime.now().strftime("%d%m%Y_%H%M%S")
        path = os.path.join(MODEL_DIR, f"{tag}_{ts}.pth")
        torch.save(policy.state_dict(), path)
        print(f"[Checkpoint] {tag} saved → {path}")

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

    print(f"[Training] Device: {DEVICE} | Log: {CSV_PATH}")
    print("Press Ctrl+C to stop and save.\n")

    try:
        for epoch, epoch_stat, info in trainer:
            print(f"Epoch {epoch:>4d} | rew={epoch_stat['rew']:+8.2f} | loss={info.get('loss/clip',0):.4f}")
    except KeyboardInterrupt:
        print("\n[Training] Interrupted — saving emergency checkpoint.")
        save_checkpoint("emergency_save")
    except Exception as e:
        print(f"\n[Training] Error: {e}")
        save_checkpoint("crash_save")
        raise
    finally:
        train_envs.close()
        test_envs.close()


# =============================================================================
# TESTING FUNCTIONS (NEWLY ADDED)
# =============================================================================

def find_latest_checkpoint(model_dir: str) -> str | None:
    d = Path(model_dir)
    if not d.exists(): return None
    all_ckpts  = sorted(d.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    best_ckpts = [p for p in all_ckpts if "best_policy" in p.name]
    chosen     = (best_ckpts or all_ckpts)
    return str(chosen[0]) if chosen else None

def get_deterministic_action(policy, obs):
    """Get action without exploration noise (for testing)."""
    obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        (mu, _), _ = policy.actor(obs_t, state=None)
    return np.clip(mu.squeeze(0).cpu().numpy(), -1.0, 1.0)

def test_ppo(args) -> None:
    # 1. Tìm hoặc xác định model
    ckpt = args.model or find_latest_checkpoint(MODEL_DIR)
    if ckpt is None:
        print(f"[Error] No .pth checkpoint found in {MODEL_DIR}")
        return
    
    print(f"\n[Test] Mode: Evaluation")
    print(f"[Test] Loading model: {ckpt}")
    
    # 2. Khởi tạo môi trường để lấy shape
    env = SumoEnv(render=args.render, map_config=MAP_CONFIGS, test_mode=False)
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    
    # 3. Rebuild policy & Load weight
    actor_net = Net(obs_shape, hidden_sizes=HIDDEN_SIZES, device=DEVICE)
    critic_net = Net(obs_shape, hidden_sizes=HIDDEN_SIZES, device=DEVICE)
    actor = ActorProb(actor_net, action_shape, device=DEVICE, unbounded=True).to(DEVICE)
    critic = Critic(critic_net, device=DEVICE).to(DEVICE)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LR)
    
    policy = PPOPolicy(
        actor=actor, critic=critic, optim=optim, dist_fn=dist_fn,
        action_space=env.action_space, discount_factor=GAMMA,
        gae_lambda=GAE_LAMBDA, max_grad_norm=MAX_GRAD_NORM,
        vf_coef=VF_COEF, ent_coef=ENT_COEF, action_scaling=True,
    )
    
    policy.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    policy.eval()
    print("[Test] Model loaded successfully.\n")

    # 4. Chạy vòng lặp test
    try:
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset()

            # --- ĐOẠN CODE BẠN MUỐN THÊM VÀO ---
            # Thiết lập hành vi cho các xe khác (Traffic)
            imperfection = 0.5  # (sigma) Độ không hoàn hảo của tài xế (0=hoàn hảo, 1=hay mắc lỗi)
            impatience   = 0.5  # Độ mất kiên nhẫn của tài xế (0=bình tĩnh, 1=vội vã)
            
            try:
                # Lấy danh sách tất cả các loại xe đang có trong SUMO
                existing_types = traci.vehicletype.getIDList()
                for v_type in existing_types:
                    traci.vehicletype.setParameter(v_type, "sigma", str(imperfection))
                    traci.vehicletype.setParameter(v_type, "impatience", str(impatience))
            except Exception as e:
                # Bắt lỗi nếu traci chưa sẵn sàng (đề phòng)
                print(f"[Warning] Không thể set param cho xe: {e}")
            # -------------------------------------

            done = False
            total_rew = 0
            steps = 0
            
            while not done:
                # Chọn hành động deterministic
                action = get_deterministic_action(policy, obs)
                obs, rew, term, trunc, info = env.step(action)
                done = term or trunc
                total_rew += rew
                steps += 1
                
                if args.verbose:
                    spd = info.get('real_speed', 0) * 3.6
                    print(f"  Ep {ep} Step {steps}: Spd={spd:.1f} km/h Rew={rew:.2f}")

            success = "SUCCESS" if info.get('is_success', 0) else "FAIL"
            print(f"Episode {ep}/{args.episodes} | {success} | Reward: {total_rew:.2f} | Steps: {steps} | Reason: {info.get('reason')}")
            
    except KeyboardInterrupt:
        print("\n[Test] Stopped by user.")
    finally:
        env.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Train/Test Script")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Choose execution mode: 'train' or 'test'")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .pth model (only for test mode)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to test (default: 5)")
    parser.add_argument("--no-render", dest="render", action="store_false",
                        help="Disable GUI during test")
    parser.add_argument("--verbose", action="store_true",
                        help="Print step details")
    parser.set_defaults(render=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == "train":
        train_ppo()
    else:
        test_ppo(args)