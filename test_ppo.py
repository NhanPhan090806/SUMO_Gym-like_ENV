# Lệnh chạy python test_ppo.py --model models/tianshou_ppo/best_policy_09032026_033552.pth

"""
test_ppo.py — Evaluate a trained PPO model (Tianshou) in SumoEnv.

Usage:
    python test_ppo.py --model models/tianshou_ppo/best_policy_XXXXXXXX.pth
    python test_ppo.py --model models/tianshou_ppo/best_policy_XXXXXXXX.pth --no-gui
    python test_ppo.py --model models/tianshou_ppo/best_policy_XXXXXXXX.pth --episodes 50 --delay 50

The script:
  • Loads the saved policy weights (.pth)
  • Runs N episodes (default 50) with random traffic (test_mode=False)
  • Opens the SUMO GUI by default so you can watch (pass --no-gui to skip)
  • Logs every episode to  reports/test/ppo/<model_name>/results_<timestamp>.csv
  • Prints a summary table at the end
"""

import os
import sys
import csv
import argparse
import torch
import numpy as np
from datetime import datetime

# ── Tianshou ──────────────────────────────────────────────────────────────────
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.policy import PPOPolicy

# ── Local env ─────────────────────────────────────────────────────────────────
from simulation.sumo_env import SumoEnv
#dang test env cu

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG  (mirrors ppo.py – must stay in sync)
# ═══════════════════════════════════════════════════════════════════════════════
MAP_CONFIGS = [
    "maps/map_grid_tuned/run.sumocfg",
    #"maps/map2/run.sumocfg"
    #"maps/map3/run.sumocfg"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Network hyper-params (must match training)
HIDDEN_SIZES = [256, 256]

# PPO hyper-params (must match training – only relevant for policy init)
LR            = 3e-4
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
MAX_GRAD_NORM = 0.3
VF_COEF       = 0.25
ENT_COEF      = 0.01

# CSV columns
CSV_HEADER = [
    "episode", "steps", "ep_reward",
    "avg_speed", "total_energy",
    "avg_wiggle", "avg_safety",
    "success", "reason"
]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_policy(obs_shape, act_shape):
    """Reconstruct the exact same network + policy used during training."""
    net    = Net(obs_shape, hidden_sizes=HIDDEN_SIZES, device=DEVICE)
    actor  = ActorProb(net, act_shape, device=DEVICE, unbounded=True).to(DEVICE)
    critic = Critic(net, device=DEVICE).to(DEVICE)

    optim  = torch.optim.Adam(
        set(list(actor.parameters()) + list(critic.parameters())), lr=LR
    )

    policy = PPOPolicy(
        actor, critic, optim,
        dist_fn=torch.distributions.Normal,
        discount_factor=GAMMA,
        gae_lambda=GAE_LAMBDA,
        max_grad_norm=MAX_GRAD_NORM,
        vf_coef=VF_COEF,
        ent_coef=ENT_COEF,
        action_scaling=True,
        action_bound_method="clip",
    )
    return policy


def load_policy(model_path: str):
    """Load weights from a .pth file into a freshly built policy."""
    # We need obs/act shapes → build a temporary env just to read them
    print("Building temporary env to read observation / action shapes …")
    tmp_env = SumoEnv(render=False, map_config=MAP_CONFIGS, test_mode=False)
    obs_shape = tmp_env.observation_space.shape
    act_shape = tmp_env.action_space.shape
    tmp_env.close()
    print(f"  obs_shape={obs_shape}  act_shape={act_shape}")

    policy = build_policy(obs_shape, act_shape)

    state_dict = torch.load(model_path, map_location=DEVICE)
    policy.load_state_dict(state_dict)
    policy.eval()                          # inference mode
    print(f"Loaded weights from: {model_path}")
    return policy, obs_shape, act_shape


def init_csv(filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)
    print(f"Results will be saved to: {filepath}")


def append_csv(filepath: str, row: list):
    with open(filepath, "a", newline="") as f:
        csv.writer(f).writerow(row)


def select_action(policy, obs: np.ndarray) -> np.ndarray:
    """
    Run the PPO actor in deterministic mode (mean of the Gaussian).
    Returns a numpy action array.
    """
    policy.eval()
    with torch.no_grad():
        obs_t  = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        # ActorProb returns (loc, scale); we use the mean for evaluation
        logits, _  = policy.actor(obs_t)      # logits = (mu, sigma)
        mu         = logits[0]                  # deterministic: take the mean
        action     = mu.squeeze(0).cpu().numpy()

    # Clip to [-1, 1] just in case
    action = np.clip(action, -1.0, 1.0)
    return action


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TEST LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_test(args):
    model_path  = args.model
    n_episodes  = args.episodes
    render      = not args.no_gui
    delay       = args.delay

    # ── Output path ────────────────────────────────────────────────────────────
    model_name  = os.path.splitext(os.path.basename(model_path))[0]
    report_dir  = os.path.join("reports", "test", "ppo", model_name)
    timestamp   = datetime.now().strftime("%d%m%Y_%H%M%S")
    csv_path    = os.path.join(report_dir, f"results_{timestamp}.csv")

    init_csv(csv_path)

    # ── Load policy ────────────────────────────────────────────────────────────
    policy, obs_shape, act_shape = load_policy(model_path)

    # ── Build test environment ─────────────────────────────────────────────────
    env = SumoEnv(
        render      = render,
        map_config  = MAP_CONFIGS,
        test_mode   = False,   # random traffic
        delay       = delay,
    )

    # ── Tracking ───────────────────────────────────────────────────────────────
    all_rewards    = []
    all_speeds     = []
    all_energies   = []
    success_count  = 0

    print(f"\n{'='*60}")
    print(f"  Model   : {model_name}")
    print(f"  Episodes: {n_episodes}")
    print(f"  GUI     : {'ON' if render else 'OFF'}")
    print(f"  Device  : {DEVICE}")
    print(f"{'='*60}\n")

    for ep in range(1, n_episodes + 1):
        obs, _info = env.reset()

        ep_reward      = 0.0
        ep_energy      = 0.0
        ep_speed_sum   = 0.0
        ep_wiggle_sum  = 0.0
        ep_safety_sum  = 0.0
        ep_steps       = 0
        done           = False

        while not done:
            action = select_action(policy, obs)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward     += reward
            ep_energy     += info.get("real_energy", 0.0)
            ep_speed_sum  += info.get("real_speed",  0.0)
            ep_wiggle_sum += info.get("wiggle",       0.0)
            ep_safety_sum += info.get("safety",       0.0)
            ep_steps      += 1
            done = terminated or truncated

        # ── Per-episode stats ──────────────────────────────────────────────────
        avg_speed  = ep_speed_sum  / max(1, ep_steps)
        avg_wiggle = ep_wiggle_sum / max(1, ep_steps)
        avg_safety = ep_safety_sum / max(1, ep_steps)
        success    = info.get("is_success", 0)
        reason     = info.get("reason", "unknown")

        if success:
            success_count += 1

        all_rewards.append(ep_reward)
        all_speeds.append(avg_speed)
        all_energies.append(ep_energy)

        # ── CSV row ───────────────────────────────────────────────────────────
        row = [
            ep, ep_steps,
            f"{ep_reward:.2f}",
            f"{avg_speed:.2f}",
            f"{ep_energy:.2f}",
            f"{avg_wiggle:.4f}",
            f"{avg_safety:.4f}",
            success,
            reason,
        ]
        append_csv(csv_path, row)

        # ── Console ──────────────────────────────────────────────────────────
        status_icon = "✓" if success else "✗"
        print(
            f"[{ep:>3}/{n_episodes}] {status_icon}  "
            f"steps={ep_steps:>4}  reward={ep_reward:>8.2f}  "
            f"speed={avg_speed:>5.2f} m/s  "
            f"energy={ep_energy:>8.2f}  "
            f"reason={reason}"
        )

    env.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Success rate : {success_count}/{n_episodes}  ({100*success_count/n_episodes:.1f} %)")
    print(f"  Avg reward   : {np.mean(all_rewards):.2f}  ± {np.std(all_rewards):.2f}")
    print(f"  Avg speed    : {np.mean(all_speeds):.2f} m/s")
    print(f"  Avg energy   : {np.mean(all_energies):.2f}")
    print(f"  Results CSV  : {csv_path}")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a trained Tianshou-PPO policy in SumoEnv (random traffic)."
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the saved .pth policy file  "
             "(e.g. models/tianshou_ppo/best_policy_01012025_120000.pth)"
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of test episodes (default: 50)"
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Disable the SUMO GUI (headless mode)"
    )
    parser.add_argument(
        "--delay", type=int, default=0,
        help="SUMO GUI animation delay in ms (default: 0, use e.g. 50 to slow down)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_test(args)