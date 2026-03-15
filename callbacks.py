# callbacks.py
# Unified callback for research logging across all 3 stages.
# Features:
#   - CSV logging (for research papers)
#   - Rich terminal output with rolling averages
#   - Soft-stop mechanism (waits for episode boundary)

import os
import csv
from datetime import datetime
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumLogCallback(BaseCallback):
    """
    Logs episode-level telemetry to CSV and prints detailed live stats.
    Rolling averages are computed over the last N episodes for stability.
    """

    def __init__(
        self,
        log_dir: str,
        stage_name: str,
        target_timesteps: int,
        print_freq: int = 1,        # Print every N steps (1 = every step)
        rolling_window: int = 20,    # Rolling average over last N episodes
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.stage_name = stage_name
        self.target_timesteps = target_timesteps
        self.print_freq = print_freq
        self.rolling_window = rolling_window
        self.soft_stop_armed = False

        # --- Episode accumulators ---
        self.ep_len = 0
        self.ep_reward = 0.0
        self.ep_speed_sum = 0.0
        self.ep_energy_sum = 0.0
        self.ep_wiggle_sum = 0.0
        self.ep_safety_sum = 0.0
        self.total_episodes = 0
        self.total_successes = 0

        # --- Rolling history (for averages) ---
        self.reward_history = deque(maxlen=rolling_window)
        self.speed_history = deque(maxlen=rolling_window)
        self.energy_history = deque(maxlen=rolling_window)
        self.length_history = deque(maxlen=rolling_window)
        self.wiggle_history = deque(maxlen=rolling_window)
        self.safety_history = deque(maxlen=rolling_window)
        self.success_history = deque(maxlen=rolling_window)

        # --- Best tracking ---
        self.best_reward = float("-inf")
        self.worst_reward = float("inf")

        # --- Timing ---
        self.train_start_time = datetime.now()

        # --- CSV setup ---
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"{stage_name}_{now}.csv")
        os.makedirs(log_dir, exist_ok=True)
        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow([
            "episode", "timestep", "steps", "reward",
            "avg_speed", "total_energy", "avg_wiggle", "avg_safety",
            "success", "reason",
            "rolling_avg_reward", "rolling_avg_speed",
            "rolling_avg_energy", "rolling_success_rate",
        ])

        # Print header
        print("\n" + "=" * 80)
        print(f"  [{stage_name}] Training Started")
        print(f"  Target: {target_timesteps:,} timesteps | Rolling window: {rolling_window} eps")
        print("=" * 80)

    def _on_step(self) -> bool:
        self.ep_len += 1

        info = self.locals["infos"][0]
        reward = float(self.locals["rewards"][0])
        done = self.locals["dones"][0]
        action = self.locals["actions"][0]
        obs = self.locals.get("new_obs", [None])[0]

        self.ep_reward += reward
        self.ep_speed_sum += info.get("real_speed", 0.0)
        self.ep_energy_sum += info.get("real_energy", 0.0)
        self.ep_wiggle_sum += info.get("wiggle", 0.0)
        self.ep_safety_sum += info.get("safety", 0.0)

        # --- Step-level heartbeat ---
        if self.verbose >= 2 and self.num_timesteps % self.print_freq == 0:
            speed = obs[0] if obs is not None else 0.0
            elec = obs[2] if obs is not None else 0.0
            act_str = ", ".join(f"{a:+.2f}" for a in action)

            print(
                f"  [{self.stage_name}] "
                f"t={self.num_timesteps:>7,} | "
                f"ep_step={self.ep_len:>4} | "
                f"act=[{act_str}] | "
                f"rew={reward:>+7.3f} | "
                f"spd={speed:>5.2f} | "
                f"elec={elec:>+6.3f}"
            )

        # --- Arm soft stop ---
        if self.num_timesteps >= self.target_timesteps:
            self.soft_stop_armed = True

        # --- Episode end ---
        if done:
            self.total_episodes += 1
            avg_speed = self.ep_speed_sum / max(1, self.ep_len)
            avg_wiggle = self.ep_wiggle_sum / max(1, self.ep_len)
            avg_safety = self.ep_safety_sum / max(1, self.ep_len)
            reason = info.get("success_reason", "unknown")
            success = 1 if reason == "success" else 0

            if success:
                self.total_successes += 1

            # Update best/worst
            if self.ep_reward > self.best_reward:
                self.best_reward = self.ep_reward
            if self.ep_reward < self.worst_reward:
                self.worst_reward = self.ep_reward

            # Push to rolling history
            self.reward_history.append(self.ep_reward)
            self.speed_history.append(avg_speed)
            self.energy_history.append(self.ep_energy_sum)
            self.length_history.append(self.ep_len)
            self.wiggle_history.append(avg_wiggle)
            self.safety_history.append(avg_safety)
            self.success_history.append(float(success))

            # Compute rolling averages
            roll_reward = np.mean(self.reward_history)
            roll_speed = np.mean(self.speed_history)
            roll_energy = np.mean(self.energy_history)
            roll_length = np.mean(self.length_history)
            roll_wiggle = np.mean(self.wiggle_history)
            roll_safety = np.mean(self.safety_history)
            roll_success = np.mean(self.success_history) * 100  # percentage

            # Elapsed time
            elapsed = (datetime.now() - self.train_start_time).total_seconds()
            elapsed_str = self._format_time(elapsed)
            steps_per_sec = self.num_timesteps / max(1, elapsed)

            # ETA
            remaining = self.target_timesteps - self.num_timesteps
            eta_sec = remaining / max(1, steps_per_sec)
            eta_str = self._format_time(eta_sec)

            progress = (self.num_timesteps / self.target_timesteps) * 100

            # --- Write to CSV ---
            self._writer.writerow([
                self.total_episodes,
                self.num_timesteps,
                self.ep_len,
                f"{self.ep_reward:.3f}",
                f"{avg_speed:.3f}",
                f"{self.ep_energy_sum:.3f}",
                f"{avg_wiggle:.4f}",
                f"{avg_safety:.4f}",
                success,
                reason,
                f"{roll_reward:.3f}",
                f"{roll_speed:.3f}",
                f"{roll_energy:.3f}",
                f"{roll_success:.1f}",
            ])
            self._csv_file.flush()

            # --- Console output ---
            if self.verbose >= 1:
                # Episode summary line
                status_icon = "✅" if success else "❌"
                print(
                    f"\n{'─' * 80}\n"
                    f"  {status_icon} [{self.stage_name}] "
                    f"EP {self.total_episodes:>4} DONE  |  "
                    f"Reason: {reason:<12} | "
                    f"Steps: {self.ep_len:>4}\n"
                    f"{'─' * 80}"
                )

                # This episode stats
                print(
                    f"  📊 This Episode:\n"
                    f"     Reward:   {self.ep_reward:>+10.2f}  |  "
                    f"Avg Speed:  {avg_speed:>6.2f} m/s  |  "
                    f"Energy Used: {self.ep_energy_sum:>8.1f} Wh\n"
                    f"     Wiggle:   {avg_wiggle:>10.4f}  |  "
                    f"Safety Pen: {avg_safety:>6.4f}   |  "
                    f"Best Ever:   {self.best_reward:>+8.2f}"
                )

                # Output ACARS weights if active (Idea 3)
                acars_w = info.get("acars_weights", None)
                if acars_w:
                    fmt_w = " | ".join(f"{k[:4]}:{v:.2f}" for k, v in list(acars_w.items())[:5])
                    print(f"  ⚖️  ACARS Weights:  {fmt_w}")

                # Rolling averages
                n = len(self.reward_history)
                print(
                    f"  📈 Rolling Avg (last {n} eps):\n"
                    f"     Avg Reward:  {roll_reward:>+8.2f}  |  "
                    f"Avg Speed:  {roll_speed:>6.2f} m/s  |  "
                    f"Avg Energy:  {roll_energy:>8.1f} Wh\n"
                    f"     Avg Steps:   {roll_length:>8.1f}  |  "
                    f"Avg Wiggle: {roll_wiggle:>6.4f}   |  "
                    f"Success Rate: {roll_success:>5.1f}%"
                )

                # Progress bar
                bar_width = 30
                filled = int(bar_width * progress / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                print(
                    f"  ⏱️  Progress: [{bar}] {progress:>5.1f}%  |  "
                    f"t={self.num_timesteps:>8,}/{self.target_timesteps:,}\n"
                    f"     Elapsed: {elapsed_str}  |  "
                    f"ETA: {eta_str}  |  "
                    f"Speed: {steps_per_sec:>.0f} steps/s"
                )
                print(f"{'─' * 80}\n")

            # Reset episode accumulators
            self.ep_len = 0
            self.ep_reward = 0.0
            self.ep_speed_sum = 0.0
            self.ep_energy_sum = 0.0
            self.ep_wiggle_sum = 0.0
            self.ep_safety_sum = 0.0

            # Soft stop: exit at episode boundary
            if self.soft_stop_armed:
                print(
                    f"\n🛑 [{self.stage_name}] SOFT STOP @ "
                    f"t={self.num_timesteps:,} | "
                    f"Total Episodes: {self.total_episodes} | "
                    f"Success Rate: {roll_success:.1f}%\n"
                )
                return False

        return True

    def _on_training_end(self):
        """Print final summary and close CSV."""
        elapsed = (datetime.now() - self.train_start_time).total_seconds()

        if self.total_episodes > 0:
            final_success = (self.total_successes / self.total_episodes) * 100
            final_avg_reward = np.mean(self.reward_history) if self.reward_history else 0
        else:
            final_success = 0.0
            final_avg_reward = 0.0

        print("\n" + "=" * 80)
        print(f"  [{self.stage_name}] TRAINING COMPLETE")
        print(f"  {'─' * 60}")
        print(f"  Total Timesteps:  {self.num_timesteps:>10,}")
        print(f"  Total Episodes:   {self.total_episodes:>10}")
        print(f"  Total Successes:  {self.total_successes:>10}")
        print(f"  Success Rate:     {final_success:>9.1f}%")
        print(f"  {'─' * 60}")
        print(f"  Best Reward:      {self.best_reward:>+10.2f}")
        print(f"  Worst Reward:     {self.worst_reward:>+10.2f}")
        print(f"  Final Avg Reward: {final_avg_reward:>+10.2f}")
        print(f"  {'─' * 60}")
        print(f"  Total Time:       {self._format_time(elapsed)}")
        print(f"  CSV Log:          {self.csv_path}")
        print("=" * 80 + "\n")

        self._csv_file.close()

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable HH:MM:SS string."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        elif m > 0:
            return f"{m}m {s:02d}s"
        else:
            return f"{s}s"

# =============================================================================
# TIANSHOU-SPECIFIC CALLBACKS / LOGGERS
# =============================================================================

import gymnasium as gym

class TianshouDummyLogger:
    """A dummy logger that satisfies Tianshou's requirements, used when we log manually."""
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

class TianshouLogCallback(gym.Wrapper):
    """
    Wraps an Env for Tianshou to calculate episode stats and write to CSV
    IMMEDIATELY upon episode termination. 
    Distinct from CurriculumLogCallback (which is SB3 BaseCallback).
    """
    def __init__(self, env, log_filepath):
        super().__init__(env)
        self.log_filepath = log_filepath
        
        self.episode_reward = 0.0
        self.episode_energy = 0.0
        self.episode_speed_sum = 0.0
        self.episode_steps = 0
        self.episode_jerk_sum = 0.0
        self.episode_safety_sum = 0.0
        
        self.global_ep_cnt = 0
        if os.path.exists(self.log_filepath):
            try:
                with open(self.log_filepath, 'r') as f:
                    self.global_ep_cnt = max(0, sum(1 for _ in f) - 1)
            except:
                self.global_ep_cnt = 0

    def reset(self, **kwargs):
        self.episode_reward = 0.0
        self.episode_energy = 0.0
        self.episode_speed_sum = 0.0
        self.episode_steps = 0
        self.episode_jerk_sum = 0.0
        self.episode_safety_sum = 0.0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        self.episode_reward += reward
        self.episode_energy += info.get("real_energy", 0.0)
        self.episode_speed_sum += info.get("real_speed", 0.0) 
        self.episode_jerk_sum += info.get("wiggle", 0.0)
        self.episode_safety_sum += info.get("safety", 0.0)
        self.episode_steps += 1
        
        if terminated or truncated:
            self.global_ep_cnt += 1
            
            avg_speed = self.episode_speed_sum / max(1, self.episode_steps)
            avg_safety = self.episode_safety_sum / max(1, self.episode_steps)
            avg_jerk = self.episode_jerk_sum / max(1, self.episode_steps)
            success = info.get("is_success", 0)
            reason = info.get("reason", "unknown")
            
            row = [
                self.global_ep_cnt,
                self.episode_steps,
                f"{self.episode_reward:.2f}",
                f"{avg_speed:.2f}",
                f"{self.episode_energy:.2f}",
                f"{avg_jerk:.4f}",
                f"{avg_safety:.4f}",
                success,
                reason
            ]
            
            os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)
            if not os.path.exists(self.log_filepath):
                with open(self.log_filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["episode", "steps", "ep_reward", "avg_speed", "total_energy", "wiggle", "safety", "success", "reason"])

            try:
                with open(self.log_filepath, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            except Exception as e:
                print(f"Logging Error: {e}")
            
        return obs, reward, terminated, truncated, info
