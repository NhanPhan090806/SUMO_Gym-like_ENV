# wrappers.py
# Stage-specific Gymnasium wrappers that override Action Space and Reward
# for the 3-Stage Curriculum Learning pipeline.
#
# KEY DESIGN PRINCIPLES:
# 1. Base env ALWAYS returns reward=0.0. ALL rewards (step + terminal) are
#    computed here using cfg.TERMINAL_REWARDS and stage-specific weights.
# 2. Energy is measured as EFFICIENCY (Wh/m), not absolute consumption.
#    This prevents "speed collapse" where agent goes slow to avoid energy penalty.
# 3. Minimum-speed penalties enforce reasonable driving speed.

import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium import spaces
from stable_baselines3 import SAC

import config as cfg

# =============================================================================
# OBS INDEX CONSTANTS — 25-dim observation (updated for heading_cos)
# =============================================================================
IDX_VELOCITY    = 0
IDX_ACCEL       = 1
IDX_ELEC        = 2
IDX_NORM_LANE   = 3
IDX_SLOPE       = 4
IDX_LAT_OFFSET  = 5
IDX_LAT_SPEED   = 6   # Lateral speed (useful for lane-change dynamics)
IDX_HEADING_SIN = 7   # sin(heading) — continuous angular signal
IDX_HEADING_COS = 8   # cos(heading) — removes sin ambiguity (NEW)
# 13-20: surroundings (LF_d, LF_v, LB_d, LB_v, RF_d, RF_v, RB_d, RB_v)
IDX_CAN_LEFT    = 11
IDX_CAN_RIGHT   = 12

# Surroundings distance indices (even indices within the surroundings block)
IDX_SURR_LF_D = 13
IDX_SURR_LB_D = 15
IDX_SURR_RF_D = 17
IDX_SURR_RB_D = 19

# (Leftover indices for compatibility if needed, though replaced mostly above)
IDX_L_DIST      = 6
IDX_L_REL_SPEED = 7
IDX_SPEED_LIMIT = 8
IDX_TLS_DIST    = 9
IDX_TLS_STATE   = 10


# =============================================================================
# SHARED REWARD HELPERS
# =============================================================================
def compute_speed_rewards(obs: np.ndarray) -> tuple[float, float, float]:
    """
    Compute speed-related reward components shared across stages.

    Returns:
        r_speed_target: Bell-curve reward for driving near target speed
        r_speed_maintain: Reward for staying above minimum desired speed
        r_too_slow: Penalty for going below minimum desired speed
    """
    velocity = obs[IDX_VELOCITY] * cfg.MAX_SPEED  # denormalize to m/s
    speed_limit = obs[IDX_SPEED_LIMIT] * cfg.MAX_SPEED  # denormalize

    # Target speed: Increased from 70% to use new config target (default 90% or 100%)
    target_speed = max(speed_limit * cfg.TARGET_SPEED_RATIO, cfg.MIN_DESIRED_SPEED)

    # R_speed_target: Bell-curve peaking at target speed
    # Uses relative speed difference for a substantially clearer gradient signal to punish stopping
    speed_error = (velocity - target_speed) / max(target_speed, 1.0)
    r_speed_target = np.exp(-3.0 * speed_error ** 2)

    # R_speed_maintain: Linear reward for being above minimum speed
    if velocity >= cfg.MIN_DESIRED_SPEED:
        r_speed_maintain = min(velocity / cfg.MIN_DESIRED_SPEED, 2.0) - 1.0  # [0, 1]
    else:
        r_speed_maintain = 0.0

    # R_too_slow: Penalty that increases as speed drops below minimum
    if velocity < cfg.MIN_DESIRED_SPEED:
        # Normalized penalty: 2.0 when stopped, 0.0 at MIN_DESIRED_SPEED (much stronger signal)
        r_too_slow = 2.0 * (1.0 - (velocity / cfg.MIN_DESIRED_SPEED))
    else:
        r_too_slow = 0.0

    return r_speed_target, r_speed_maintain, r_too_slow


def compute_energy_efficiency(obs: np.ndarray) -> float:
    """
    Compute energy efficiency metric (Wh per meter traveled).

    This is the KEY FIX for speed collapse: by measuring Wh/m instead of
    raw Wh/s, the agent is NOT rewarded for going slow. Going slow actually
    INCREASES Wh/m because you consume idle energy for more time per meter.

    Returns:
        r_efficiency: Normalized efficiency penalty [0, 1], higher = worse
    """
    elec = abs(obs[IDX_ELEC]) * cfg.MAX_ELEC  # denormalize to Wh/s
    velocity = obs[IDX_VELOCITY] * cfg.MAX_SPEED  # denormalize to m/s

    if velocity > 0.5:  # avoid division by near-zero
        # Wh per meter: energy_rate / speed
        wh_per_meter = elec / velocity
        # Normalize: typical EV uses 0.1-0.3 Wh/m, cap at 1.0
        r_efficiency = np.clip(wh_per_meter / 0.5, 0.0, 1.0)
    else:
        # Vehicle nearly stopped: maximum inefficiency penalty
        r_efficiency = 1.0

    return r_efficiency


def _apply_terminal_reward(info: dict) -> float:
    """Apply terminal reward from config based on terminal_event in info."""
    terminal = info.get("terminal_event")
    if terminal:
        return cfg.TERMINAL_REWARDS.get(terminal, 0.0)
    return 0.0


# =============================================================================
# STAGE 1 WRAPPER — Lateral Control (Steering Only)
# =============================================================================
class LateralControlWrapper(gym.Wrapper):
    """
    Stage 1: Agent learns lane-keeping & lane-changing.
    - Action space reduced to 1D: [steering]
    - Throttle is fixed at a constant cruise speed
    - Reward focuses on lane center alignment, smooth steering, AND speed

    SAFETY NET: Emergency brake override activates when the leader is
    dangerously close AND no lane change is possible (single-lane road).
    This is NOT part of what the agent learns — it is a hard-coded safety
    floor that prevents guaranteed collisions during exploration, which would
    otherwise produce uninformative -50 reward signals.
    setSpeedMode(0) disables ALL SUMO built-in safety, so without this
    override the vehicle would simply drive into the car in front.
    """

    FIXED_THROTTLE = 0.3    # ~30% acceleration, gentle cruising
    # Emergency brake thresholds (normalized units, same as obs space)
    EMRG_DIST_FULL  = 0.08  # < 8% sensor range → full stop (brake hard)
    EMRG_DIST_SOFT  = 0.20  # < 20% sensor range → proportional slow-down

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.prev_steer = 0.0
        self.prev_norm_lane = 0.0  # track lane changes
        self.W = cfg.REWARD_STAGE1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_steer = 0.0
        self.prev_norm_lane = obs[IDX_NORM_LANE]
        self._last_obs = obs  # seed emergency brake with initial obs
        return obs, info

    def step(self, action):
        steer = float(action[0])

        # --- Emergency Brake Safety Override ---
        # setSpeedMode(0) disables ALL SUMO safety (no auto-braking).
        # In a single-lane road the agent CANNOT rẽ sang làn khác, so a
        # fixed throttle guarantees a collision. This override is a hard-coded
        # safety floor — the agent still controls steering freely.
        #
        # We peek at the last known obs (from reset or previous step) to decide.
        peek_obs = getattr(self, "_last_obs", None)
        throttle = self.FIXED_THROTTLE
        if peek_obs is not None:
            leader_dist  = peek_obs[IDX_L_DIST]        # normalized [0, 1]
            can_left     = peek_obs[IDX_CAN_LEFT]       # 1.0 if lane exists
            can_right    = peek_obs[IDX_CAN_RIGHT]      # 1.0 if lane exists
            no_escape    = (can_left < 0.5) and (can_right < 0.5)

            if leader_dist < self.EMRG_DIST_FULL and no_escape:
                # Full emergency stop — brake hard (negative throttle)
                throttle = -1.0
            elif leader_dist < self.EMRG_DIST_SOFT and no_escape:
                # Proportional slow-down: throttle scales from FIXED → 0
                # as leader_dist shrinks from EMRG_DIST_SOFT → EMRG_DIST_FULL
                ratio = (leader_dist - self.EMRG_DIST_FULL) / (
                    self.EMRG_DIST_SOFT - self.EMRG_DIST_FULL
                )  # 1.0 when far, 0.0 when at full-stop threshold
                throttle = self.FIXED_THROTTLE * ratio

        full_action = np.array([steer, throttle], dtype=np.float32)

        obs, base_reward, terminated, truncated, info = self.env.step(full_action)

        # --- Stage 1 Reward ---
        # Start with terminal rewards (base env always returns 0.0)
        reward = _apply_terminal_reward(info)

        if not terminated or info.get("is_success", 0):
            # R_center: Reward for staying centered in lane (Gaussian-like)
            lat_offset = obs[IDX_LAT_OFFSET] * cfg.MAX_LAT_OFFSET  # denormalize to meters
            r_center = np.exp(-2.0 * lat_offset ** 2)  # peaks at 0 offset

            # R_smooth: Penalty for jerky steering
            steer_delta = abs(steer - self.prev_steer)
            r_smooth = steer_delta

            # R_safety_side: Penalty if side vehicles are too close
            side_dists = [obs[IDX_SURR_LF_D], obs[IDX_SURR_LB_D],
                          obs[IDX_SURR_RF_D], obs[IDX_SURR_RB_D]]
            min_side_dist = min(side_dists)
            r_side = max(0.0, 1.0 - min_side_dist) if min_side_dist < 0.3 else 0.0

            # R_progress: Forward progress. Scaled so max expected speed (~15m/s) creates better gradient spread.
            r_progress = obs[IDX_VELOCITY] * (cfg.MAX_SPEED / 15.0)

            # Speed rewards — prevent slowing down even in Stage 1
            r_speed_target, r_speed_maintain, r_too_slow = compute_speed_rewards(obs)

            # R_front_danger: Penalize when leader is dangerously close AND agent
            # is not steering (steer within dead zone). This forces the agent to
            # commit to a lane change early instead of waiting until collision.
            leader_dist = obs[IDX_L_DIST]  # normalized [0, 1]
            in_dead_zone = abs(steer) <= cfg.LC_THRESHOLD
            r_front_danger = 0.0
            if leader_dist < 0.15 and in_dead_zone:
                # Scale: closer leader + more centered steer = larger penalty
                danger_level = (0.15 - leader_dist) / 0.15          # [0, 1]
                dead_zone_depth = 1.0 - abs(steer) / cfg.LC_THRESHOLD  # [0, 1]
                r_front_danger = danger_level * dead_zone_depth

            # R_lane_change: Reward agent for successfully completing a lane change.
            # Detected by comparing normalized lane index to previous step.
            # This gives explicit positive signal for what was only implicitly rewarded
            # (avoiding collision) before.
            current_norm_lane = obs[IDX_NORM_LANE]
            r_lane_change = 1.0 if abs(current_norm_lane - self.prev_norm_lane) > 0.05 else 0.0

            reward += (
                self.W["lane_center"]    * r_center +
                self.W["smooth_steer"]   * r_smooth +
                self.W["safety_side"]    * r_side +
                self.W["progress"]       * r_progress +
                self.W["speed_maintain"] * r_speed_maintain +
                self.W["too_slow"]       * r_too_slow +
                self.W["front_danger"]   * r_front_danger +
                self.W["lane_change"]    * r_lane_change
            )

        self.prev_steer = steer
        self.prev_norm_lane = obs[IDX_NORM_LANE]
        self._last_obs = obs   # update for next step's emergency brake check
        return obs, reward, terminated, truncated, info


# =============================================================================
# STAGE 2 WRAPPER — Longitudinal Control (Throttle/Brake + Energy)
# =============================================================================
class LongitudinalControlWrapper(gym.Wrapper):
    """
    Stage 2: Agent learns energy-efficient speed control.
    - Action space reduced to 1D: [throttle/brake]
    - Steering is provided by the frozen Stage 1 model
    - Reward focuses on energy EFFICIENCY (Wh/m), speed maintenance, and safety

    KEY FIX: Energy is now measured as efficiency (Wh per meter), NOT raw
    consumption (Wh/s). This prevents the agent from learning "go slow = win".
    """

    def __init__(self, env: gym.Env, lateral_model_path: str):
        super().__init__(env)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Load frozen lateral model
        print(f"[Stage 2] Loading lateral model from: {lateral_model_path}")
        self.lateral_model = SAC.load(lateral_model_path, device="cpu")
        self.lateral_model.policy.set_training_mode(False)

        self.prev_accel = 0.0
        self.last_obs = None
        self.W = cfg.REWARD_STAGE2

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_accel = 0.0
        self.last_obs = obs
        return obs, info

    def step(self, action):
        throttle = float(action[0])

        # Get steering from frozen Stage 1 model
        steer_action, _ = self.lateral_model.predict(self.last_obs, deterministic=True)
        steer = float(steer_action[0]) if isinstance(steer_action, np.ndarray) else float(steer_action)

        full_action = np.array([steer, throttle], dtype=np.float32)
        obs, base_reward, terminated, truncated, info = self.env.step(full_action)
        self.last_obs = obs

        # --- Stage 2 Reward ---
        # Start with terminal rewards (base env always returns 0.0)
        reward = _apply_terminal_reward(info)

        if not terminated or info.get("is_success", 0):
            # R_energy_efficiency: Wh per meter (NOT raw Wh/s)
            r_efficiency = compute_energy_efficiency(obs)

            # Speed rewards — the core fix for speed collapse
            r_speed_target, r_speed_maintain, r_too_slow = compute_speed_rewards(obs)

            # R_progress: Reward for forward movement. Scaled so 15m/s evaluates closer to 1.0
            r_progress = obs[IDX_VELOCITY] * (cfg.MAX_SPEED / 15.0)

            # R_safety_leader: Dynamics Penalty based on Time-To-Collision (TTC) instead of fixed meters
            # This fixes the bug where agent decelerated constantly to artificially raise safety distance
            leader_dist_m = obs[IDX_L_DIST] * cfg.MAX_DIST
            velocity_m = obs[IDX_VELOCITY] * cfg.MAX_SPEED
            safe_dist = velocity_m * 1.5 + 5.0  # 1.5s headway gap + 5m minimum gap
            r_leader = max(0.0, (safe_dist - leader_dist_m) / safe_dist) if leader_dist_m < safe_dist else 0.0

            # R_tls: Penalty for approaching red light at high speed
            tls_state = obs[IDX_TLS_STATE]
            tls_dist = obs[IDX_TLS_DIST]
            speed_ratio = obs[IDX_VELOCITY]  # already normalized
            r_tls = 0.0
            if tls_state < 0.5 and tls_dist < 0.3 and speed_ratio > 0.2:
                r_tls = speed_ratio  # faster approach to red = bigger penalty

            # R_smooth_accel: Penalty for jerky throttle
            accel_delta = abs(throttle - self.prev_accel)
            r_smooth = accel_delta

            reward += (
                self.W["energy_efficiency"] * r_efficiency +
                self.W["speed_target"]      * r_speed_target +
                self.W["speed_maintain"]    * r_speed_maintain +
                self.W["too_slow"]          * r_too_slow +
                self.W["progress"]          * r_progress +
                self.W["safety_leader"]     * r_leader +
                self.W["tls_violation"]     * r_tls +
                self.W["smooth_accel"]      * r_smooth
            )

        self.prev_accel = throttle
        return obs, reward, terminated, truncated, info


# =============================================================================
# ADAPTIVE REWARD SHAPING MODULE (ACARS — Idea 3)
# =============================================================================
class AdaptiveRewardShaping:
    """ACARS module for adaptive reward weight calculation based on performance."""
    def __init__(self, base_weights, targets, window=20):
        self.base = base_weights.copy()
        self.targets = targets
        self.performance_buffer = {k: deque(maxlen=window) for k in base_weights}
        
    def add_episode_stats(self, episode_stats):
        """Append average episode component rewards."""
        for k in self.base:
            if k in episode_stats:
                self.performance_buffer[k].append(episode_stats[k])
                
    def get_weights(self):
        """Compute adjusted weights based on performance gap."""
        adjusted = {}
        for key, base_w in self.base.items():
            if key in self.targets and len(self.performance_buffer[key]) > 0:
                current_avg = float(np.mean(self.performance_buffer[key]))
                target = self.targets[key]
                
                # Check mapping context for components.
                # Assuming positive base_w means "maximize" (reward),
                # negative base_w means "minimize" (penalty).
                if base_w > 0:  
                    # Gap > 0 if current < target (underperforming)
                    gap = (target - current_avg) / max(abs(target), 1e-6)
                else:           
                    # Gap > 0 if current > target (underperforming penalty)
                    gap = (current_avg - target) / max(abs(target), 1e-6)
                    
                # Sigmoid scaling: gap>0 -> scale>1 (amplify weight)
                scale = 1.0 + 0.5 * np.tanh(2.0 * gap)  # range [0.5, 1.5]
                adjusted[key] = base_w * scale
            else:
                adjusted[key] = base_w
        return adjusted


# =============================================================================
# STAGE 3 WRAPPER — Full Control (Policy Fusion Fine-Tuning)
# =============================================================================
class FullControlWrapper(gym.Wrapper):
    """
    Stage 3: Full autonomous control with combined global objective.
    - Action space is full 2D: [steering, throttle]
    - Model is initialized via Transfer Learning from Stage 1 or Stage 2
    - Reward combines ALL objectives with balanced weights
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Action space remains 2D from base env
        self.prev_steer = 0.0
        self.prev_accel = 0.0
        self.W = cfg.REWARD_STAGE3.copy()
        
        # ACARS setup
        self.use_acars = getattr(cfg, "USE_ACARS", False)
        if self.use_acars:
            self.acars = AdaptiveRewardShaping(cfg.REWARD_STAGE3, cfg.ACARS_TARGETS, cfg.ACARS_WINDOW)
            self.ep_comps = {k: 0.0 for k in self.W}
            self.ep_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_steer = 0.0
        self.prev_accel = 0.0
        if self.use_acars:
            self.ep_comps = {k: 0.0 for k in self.W}
            self.ep_steps = 0
            # attach weights to info so callback can log them at step 0 if needed
            info["acars_weights"] = self.W.copy()
        return obs, info

    def step(self, action):
        steer = float(action[0])
        throttle = float(action[1])

        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # --- Stage 3 Reward (Unified — Full Objective) ---
        # Start with terminal rewards (base env always returns 0.0)
        reward = _apply_terminal_reward(info)

        if not terminated or info.get("is_success", 0):
            # Speed rewards — use ALL 3 values (r_too_slow critical to prevent speed collapse)
            r_speed_target, _, r_too_slow = compute_speed_rewards(obs)

            # Progress (distance diff matching sumo_env calculate_reward)
            dist = self.env.unwrapped._get_dist_to_destination()
            if not hasattr(self, "prev_dist"): self.prev_dist = dist
            r_progress = float(np.clip(self.prev_dist - dist, -1.0, 1.0))
            self.prev_dist = dist

            # Safety — leader (TTC-based: 1.5s headway + 5m gap)
            leader_dist_m = obs[IDX_L_DIST] * cfg.MAX_DIST
            velocity_m = obs[IDX_VELOCITY] * cfg.MAX_SPEED
            safe_dist = velocity_m * 1.5 + 5.0
            r_leader = max(0.0, (safe_dist - leader_dist_m) / safe_dist) if leader_dist_m < safe_dist else 0.0

            # Smooth accel — penalty for jerky throttle
            r_smooth_accel = abs(throttle - self.prev_accel)

            # Energy efficiency (Wh/m)
            r_efficiency = compute_energy_efficiency(obs)

            # TLS violation: match sumo_env exactly (red light crossing)
            r_tls = 0.0
            unwrapped = self.env.unwrapped
            if getattr(unwrapped, "_prev_tls_was_red", False):
                veh_data = getattr(unwrapped, "veh_data", None)
                if veh_data and veh_data.get("road_id", "").startswith(":"):
                    r_tls = 1.0  # Mapped to W_RED_LIGHT

            # Time keeping (survival penalty)
            r_time = 1.0

            # ACARS metrics gathering
            raw_comps = {
                "speed_target": r_speed_target,
                "too_slow": r_too_slow,
                "progress": r_progress,
                "energy_efficiency": r_efficiency,
                "smooth_accel": r_smooth_accel,
                "safety_leader": r_leader,
                "tls_violation": r_tls,
                "time": r_time
            }
            
            if self.use_acars:
                self.ep_steps += 1
                for k, v in raw_comps.items():
                    if k in self.ep_comps:
                        self.ep_comps[k] += v

            reward += sum(self.W.get(k, 0.0) * v for k, v in raw_comps.items())

        # Update ACARS at end of episode
        if (terminated or truncated) and self.use_acars:
            avg_comps = {k: v / max(1, self.ep_steps) for k, v in self.ep_comps.items()}
            self.acars.add_episode_stats(avg_comps)
            self.W = self.acars.get_weights()
            info["acars_weights"] = self.W.copy()

        self.prev_steer = steer
        self.prev_accel = throttle
        return obs, reward, terminated, truncated, info
