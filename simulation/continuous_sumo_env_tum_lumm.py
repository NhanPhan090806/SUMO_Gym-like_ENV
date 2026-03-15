# Continuous action space SUMO gym-like environment
# Vehicle model: Vinfast VF9 (HEV/EV)
# Goal: reach destination with minimum steps, minimum CO2/energy, and safe driving
#Bản này là bản cập nhật sửa để nâng cấp lên

import os
import sys
import traci
import time
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --- SUMO PATH SETUP ---
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")


# =============================================================================
# CONSTANTS  (single source of truth — never hardcode these elsewhere)
# =============================================================================

MAX_SPEED  = 55.6   # m/s  (~200 km/h)
MAX_ACCEL  = 4.15   # m/s²
MAX_DECEL  = 6.0    # m/s²
MAX_ELEC   = 120.0  # Wh/s
MAX_SLOPE  = 20.0   # degrees
MAX_DIST   = 100.0  # metres  (radar / neighbour horizon)
TARGET_DIST = 35.0  # metres  (desired following distance)
MAX_LAT_OFFSET = 1.6  # metres  (half standard lane width)


# =============================================================================
# ENVIRONMENT
# =============================================================================

class SumoEnv(gym.Env):
    """
    Single-agent continuous SUMO environment.

    Action space  : Box(2,)  →  [steering ∈ [-1,1],  throttle ∈ [-1,1]]
    Obs space     : Box(24,) →  fully normalised to [-1, 1] / [0, 1]
    """

    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        render: bool = False,
        map_config=None,
        VTYPE_ID: str = "custom_passenger_car",
        TRAFFIC_SCALE: float = 5.0,
        test_mode: bool = False,
        test_route: str = "TestMap/test_route.rou.xml",
        imperfection: float = 0.5,
        impatience: float = 0.5,
        delay: int = 0,
    ) -> None:
        super().__init__()

        if map_config is None:
            map_config = ["maps/TestMap/osm.sumocfg"]

        # --- identifiers ---
        self.VEH_ID       = "my_ego_car"
        self.VTYPE_ID     = VTYPE_ID
        self.TRAFFIC_SCALE = TRAFFIC_SCALE
        self.render_mode  = render
        self.test_mode    = test_mode
        self.test_route   = test_route
        self.delay        = delay
        self.imperfection = imperfection
        self.impatience   = impatience

        # expose physics constants as instance attrs (used in reward / obs)
        self.MAX_SPEED  = MAX_SPEED
        self.MAX_ACCEL  = MAX_ACCEL
        self.MAX_DECEL  = MAX_DECEL
        self.MAX_ELEC   = MAX_ELEC
        self.MAX_SLOPE  = MAX_SLOPE
        self.MAX_DIST   = MAX_DIST
        self.TARGET_DIST = TARGET_DIST

        # episode limits
        self.MAX_EPISODE_STEPS = 1000
        self.step_count = 0

        # maps
        self.maps = [map_config] if isinstance(map_config, str) else map_config

        # --- spaces ---
        # Action: [steering, throttle]  both in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation (24 values, all normalised):
        #  [0]  speed            (norm by MAX_SPEED)
        #  [1]  acceleration     (norm by MAX_ACCEL/MAX_DECEL)
        #  [2]  elec consumption (norm by MAX_ELEC)
        #  [3]  norm lane index  [0, 1]
        #  [4]  road slope       (norm by MAX_SLOPE)
        #  [5]  lateral offset   (norm by MAX_LAT_OFFSET)
        #  [6]  heading error    (placeholder, always 0 — see comments)
        #  [7]  vehicle heading  (norm: /360 → [0,1])
        #  [8-15] surroundings  (4 × [dist, rel_speed])
        #  [16] leader dist      [0,1]
        #  [17] leader rel speed
        #  [18] speed limit      (norm by MAX_SPEED)
        #  [19] can_go_left      {0,1}
        #  [20] can_go_right     {0,1}
        #  [21] TLS dist         [0,1]
        #  [22] TLS state        {0=red, 1=green}
        #  [23] dist to turn     [0,1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )

        # state carry-overs
        self.last_known_dist: float = 0.0
        self._success: bool = False
        self.stuck_time: int = 0
        self._prev_action: np.ndarray = np.zeros(2, dtype=np.float32)

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _veh_exists(self) -> bool:
        try:
            return self.VEH_ID in traci.vehicle.getIDList()
        except Exception:
            return False

    # -------------------------------------------------------------------------
    def _get_surroundings(self) -> list:
        """
        Returns 8 values: [LF_dist, LF_rel_speed, LB_dist, LB_rel_speed,
                            RF_dist, RF_rel_speed, RB_dist, RB_rel_speed]
        All normalised to [0,1] / [-1,1].
        Default = far (1.0) and same speed (0.0).

        BUG FIX: back-vehicle distances were previously NOT abs()-ed, so
        min(negative_dist, MAX_DIST) always returned the negative distance,
        producing nonsensical negative normalised values.
        """
        data = {
            "LF": [1.0, 0.0], "LB": [1.0, 0.0],
            "RF": [1.0, 0.0], "RB": [1.0, 0.0],
        }

        my_speed = traci.vehicle.getSpeed(self.VEH_ID)

        def _process(neighbours, front_key, back_key):
            best_front = float("inf")
            best_back  = float("inf")
            for n_id, raw_dist in neighbours:
                abs_dist = abs(raw_dist)
                n_speed  = traci.vehicle.getSpeed(n_id)
                if raw_dist > 0:                          # vehicle is ahead
                    if abs_dist < best_front:
                        best_front = abs_dist
                        data[front_key] = [
                            min(abs_dist, MAX_DIST) / MAX_DIST,
                            (my_speed - n_speed) / MAX_SPEED,
                        ]
                else:                                     # vehicle is behind
                    if abs_dist < best_back:
                        best_back = abs_dist
                        data[back_key] = [
                            min(abs_dist, MAX_DIST) / MAX_DIST,   # FIX: was using raw (negative) dist
                            (my_speed - n_speed) / MAX_SPEED,
                        ]

        _process(traci.vehicle.getNeighbors(self.VEH_ID, 2), "LF", "LB")  # left  (0b10)
        _process(traci.vehicle.getNeighbors(self.VEH_ID, 1), "RF", "RB")  # right (0b01)

        return data["LF"] + data["LB"] + data["RF"] + data["RB"]

    # -------------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """
        Build and return the 24-element normalised observation vector.

        BUG FIX: velocity and acceleration were previously returned RAW
        (un-normalised) while every other feature was normalised, creating a
        severe scale mismatch that destabilises neural-net training.
        """
        if not self._veh_exists():
            return np.zeros(24, dtype=np.float32)

        # ── EGO PHYSICS (8) ──────────────────────────────────────────────────
        velocity     = traci.vehicle.getSpeed(self.VEH_ID) / MAX_SPEED           # FIX: now normalised
        raw_accel    = traci.vehicle.getAcceleration(self.VEH_ID)
        # Normalise acceleration: positive → [0,1], negative → [-1,0]
        if raw_accel >= 0:
            acceleration = raw_accel / MAX_ACCEL                                  # FIX: now normalised
        else:
            acceleration = raw_accel / MAX_DECEL                                  # FIX: now normalised
        acceleration = float(np.clip(acceleration, -1.0, 1.0))

        try:
            lane_idx    = traci.vehicle.getLaneIndex(self.VEH_ID)
            road_id     = traci.vehicle.getRoadID(self.VEH_ID)
            total_lanes = traci.edge.getLaneNumber(road_id)
            norm_lane   = lane_idx / max(1, total_lanes - 1)

            slope       = traci.vehicle.getSlope(self.VEH_ID) / MAX_SLOPE

            # Lateral offset: normalise to [-1, 1] relative to lane half-width
            lat_offset  = traci.vehicle.getLateralLanePosition(self.VEH_ID) / MAX_LAT_OFFSET  # FIX: now normalised
            lat_offset  = float(np.clip(lat_offset, -1.0, 1.0))

            # Heading error: SUMO doesn't expose independent yaw vs lane-angle,
            # so this stays 0 to avoid feeding a degenerate / noisy signal.
            heading_error = 0.0

            # Vehicle heading (proxy for "where are my wheels pointing")
            # getAngle returns [0, 360); normalise to [0, 1]
            heading     = traci.vehicle.getAngle(self.VEH_ID) / 360.0

        except Exception:
            norm_lane = lane_idx = 0.0
            slope = lat_offset = heading_error = heading = 0.0

        elec = float(np.clip(
            traci.vehicle.getElectricityConsumption(self.VEH_ID) / MAX_ELEC,
            0.0, 1.0
        ))

        ego_features = [velocity, acceleration, elec, norm_lane,
                        slope, lat_offset, heading_error, heading]

        # ── SURROUNDINGS (8) ─────────────────────────────────────────────────
        surroundings = self._get_surroundings()

        # ── LEADER (2) ───────────────────────────────────────────────────────
        try:
            leader = traci.vehicle.getLeader(self.VEH_ID, dist=MAX_DIST)
            if leader:
                l_dist      = leader[1] / MAX_DIST
                l_rel_speed = (traci.vehicle.getSpeed(self.VEH_ID) -
                               traci.vehicle.getSpeed(leader[0])) / MAX_SPEED
            else:
                l_dist, l_rel_speed = 1.0, 0.0
        except Exception:
            l_dist, l_rel_speed = 1.0, 0.0

        # ── INFRASTRUCTURE (6) ───────────────────────────────────────────────
        try:
            lane_id = traci.vehicle.getLaneID(self.VEH_ID)

            if lane_id == "":
                speed_limit = 1.0
                can_left = can_right = 0.0
                tls_dist = tls_state = turn_dist = 1.0
            else:
                speed_limit = traci.lane.getMaxSpeed(lane_id) / MAX_SPEED
                road_id2    = traci.vehicle.getRoadID(self.VEH_ID)
                num_lanes   = traci.edge.getLaneNumber(road_id2)
                can_left    = 1.0 if lane_idx < (num_lanes - 1) else 0.0
                can_right   = 1.0 if lane_idx > 0               else 0.0

                tls_data = traci.vehicle.getNextTLS(self.VEH_ID)
                if tls_data:
                    tls_dist  = tls_data[0][2] / MAX_DIST
                    tls_state = 1.0 if tls_data[0][3].lower() == "g" else 0.0
                else:
                    tls_dist, tls_state = 1.0, 1.0

                lane_len  = traci.lane.getLength(lane_id)
                lane_pos  = traci.vehicle.getLanePosition(self.VEH_ID)
                turn_dist = min(lane_len - lane_pos, MAX_DIST) / MAX_DIST

        except Exception:
            speed_limit = 1.0
            can_left = can_right = 0.0
            tls_dist = tls_state = turn_dist = 1.0

        infra = [speed_limit, can_left, can_right, tls_dist, tls_state, turn_dist]

        # ── COMBINE ──────────────────────────────────────────────────────────
        obs = np.array(
            ego_features + surroundings + [l_dist, l_rel_speed] + infra,
            dtype=np.float32,
        )
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    # -------------------------------------------------------------------------
    def _get_dist_to_destination(self) -> float:
        """Estimate remaining route distance in metres."""
        try:
            current_edge = traci.vehicle.getRoadID(self.VEH_ID)
            if current_edge.startswith(":"):
                return self.last_known_dist or 450.0

            if hasattr(self, "current_route_edges") and current_edge in self.current_route_edges:
                indices = [i for i, e in enumerate(self.current_route_edges) if e == current_edge]
                idx     = indices[-1]
                dist    = sum(
                    traci.lane.getLength(f"{e}_0")
                    for e in self.current_route_edges[idx:]
                )
                dist -= traci.vehicle.getLanePosition(self.VEH_ID)
                self.last_known_dist = dist
                return dist

        except Exception:
            pass
        return 450.0

    # -------------------------------------------------------------------------
    def _calculate_reward(self, action: np.ndarray) -> tuple[float, float, float]:
        """
        Compute step reward and return (reward, wiggle_penalty, safety_penalty)
        so that the caller can log them without re-computing.

        BUG FIX: previously, both _calculate_reward AND step() maintained
        self.prev_action independently.  _calculate_reward updated it first,
        so step()'s wiggle stat was always 0.  Now prev_action is owned
        exclusively by step(); this function receives it as an argument.

        Returns
        -------
        total_reward   : float
        wiggle_penalty : float  [0, 1]  — for info logging
        safety_penalty : float  [0, 1]  — for info logging
        """
        W_SPEED   =  1.0
        W_TIME    = -0.1
        W_ENERGY  = -0.4
        W_COMFORT = -0.3
        W_SAFETY  = -1.0

        TIME_HEADWAY = 2.0   # seconds
        MIN_GAP      = 5.0   # metres

        cur_speed = traci.vehicle.getSpeed(self.VEH_ID)

        # 1. Speed reward
        speed_reward = float(np.clip(cur_speed / MAX_SPEED, 0.0, 1.0))

        # 2. Energy penalty
        elec = traci.vehicle.getElectricityConsumption(self.VEH_ID)
        energy_penalty = float(np.clip(elec / MAX_ELEC, 0.0, 1.0))

        # 3. Comfort / wiggle penalty  (uses _prev_action set by step())
        delta          = np.abs(action - self._prev_action)
        wiggle_penalty = float((delta[0] * 0.7) + (delta[1] * 0.3))

        # 4. Safety penalty (dynamic time-headway model)
        required_safe_dist = cur_speed * TIME_HEADWAY + MIN_GAP
        safety_penalty     = 0.0
        try:
            leader = traci.vehicle.getLeader(self.VEH_ID, MAX_DIST)
            if leader is not None:
                gap = leader[1]
                if gap < required_safe_dist:
                    safety_penalty = float(np.clip(1.0 - gap / required_safe_dist, 0.0, 1.0))
        except Exception:
            pass

        # 5. Sanitise
        speed_reward   = np.nan_to_num(speed_reward)
        energy_penalty = np.nan_to_num(energy_penalty)
        wiggle_penalty = np.nan_to_num(wiggle_penalty)
        safety_penalty = np.nan_to_num(safety_penalty)

        total = (
            speed_reward   * W_SPEED  +
            W_TIME                    +
            energy_penalty * W_ENERGY +
            wiggle_penalty * W_COMFORT +
            safety_penalty * W_SAFETY
        )

        return float(total), wiggle_penalty, safety_penalty

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # 1. Close any running simulation
        try:
            traci.close()
        except Exception:
            pass
        time.sleep(0.5)

        # 2. Pick map / route
        if self.test_mode:
            active_map = self.maps[0]
            route_arg  = ["-a", self.test_route]
        else:
            active_map = random.choice(self.maps)
            route_arg  = []

        # 3. Reset episode state
        self.step_count    = 0
        self.stuck_time    = 0
        self._success      = False
        self._prev_action  = np.zeros(2, dtype=np.float32)
        self.last_known_dist = 0.0

        # 4. Start SUMO
        sumo_bin = "sumo-gui" if self.render_mode else "sumo"
        sumo_cmd = [
            sumo_bin, "-c", active_map,
            *route_arg,
            "--start", "--quit-on-end",
            "--device.emissions.probability", "1.0",
            "--scale",              str(self.TRAFFIC_SCALE),
            "--delay",              str(self.delay),
            "--no-step-log",        "true",
            "--time-to-teleport",   "-1",
            "--collision.action",   "remove",
            "--collision.check-junctions", "true",
            "--no-warnings",        "true",
        ]
        traci.start(sumo_cmd)

        # 5. Warm-up (let background traffic settle)
        for _ in range(random.randint(300, 500)): traci.simulationStep()

        # 6. Configure vehicle type
        try:
            existing_types = traci.vehicletype.getIDList()
            src = "DEFAULT_VEHTYPE" if "DEFAULT_VEHTYPE" in existing_types else existing_types[0]
            traci.vehicletype.copy(src, self.VTYPE_ID)
            traci.vehicletype.setVehicleClass(self.VTYPE_ID, "passenger")
            traci.vehicletype.setColor(self.VTYPE_ID, (0, 255, 0))
            traci.vehicletype.setParameter(self.VTYPE_ID, "mass",   "2911")
            traci.vehicletype.setLength(self.VTYPE_ID, "5.1181")
            traci.vehicletype.setEmissionClass(self.VTYPE_ID, "MMPEVEM")
            traci.vehicletype.setParameter(self.VTYPE_ID, "has.battery.device",         "true")
            traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.capacity",    "123000.00")
            traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.chargeLevel", "123000.00")
            # Apply imperfection / impatience to all background traffic types
            for vt in existing_types:
                traci.vehicletype.setParameter(vt, "sigma",      str(self.imperfection))
                traci.vehicletype.setParameter(vt, "impatience", str(self.impatience))
        except Exception as e:
            print(f"[SumoEnv] vType setup error: {e}")

        # 7. Spawn ego vehicle on fixed route
        spawned = False
        if not self.test_mode:
            try:
                # NOTE: Update this edge list to match your .net.xml file exactly.
                fixed_edges = [f"E{i}" for i in range(1, 14)]
                route_id    = "fixed_training_route"
                traci.route.add(route_id, fixed_edges)
                self.current_route_edges = fixed_edges

                traci.vehicle.add(
                    self.VEH_ID, route_id, departPos="free", typeID=self.VTYPE_ID
                )
                for _ in range(50):
                    traci.simulationStep()
                    if self._veh_exists():
                        spawned = True
                        break

                if spawned:
                    traci.vehicle.setSpeedMode(self.VEH_ID, 0)
                    traci.vehicle.setLaneChangeMode(self.VEH_ID, 0)
                    total_len = sum(
                        traci.lane.getLength(f"{e}_0")
                        for e in fixed_edges
                        if f"{e}_0" in traci.lane.getIDList()
                    )
                    self.last_known_dist = total_len
                else:
                    print("[SumoEnv] Could not spawn vehicle — retrying reset.")
                    try:
                        traci.vehicle.remove(self.VEH_ID)
                    except Exception:
                        pass
                    return self.reset(seed=seed, options=options)

            except Exception as e:
                print(f"[SumoEnv] Route setup error: {e}")
                traci.close()
                raise

        # 8. GUI tracking
        if self.render_mode and spawned and self._veh_exists():
            try:
                traci.gui.trackVehicle("View #0", self.VEH_ID)
                traci.gui.setZoom("View #0", 2000)
            except Exception:
                pass

        return self._get_obs(), {}

    # =========================================================================
    # STEP
    # =========================================================================

    def step(self, action: np.ndarray):
        self.step_count += 1

        # ── Apply action ──────────────────────────────────────────────────────
        steer_cmd = float(action[0])
        accel_cmd = float(action[1])

        if not self._veh_exists():
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0, True, False,
                self._build_info(0.0, 0.0, 0.0, 0.0, 0, "collision"),
            )

        desired_accel = accel_cmd * (MAX_ACCEL if accel_cmd >= 0 else MAX_DECEL)
        delta_time    = traci.simulation.getDeltaT()
        traci.vehicle.setAcceleration(self.VEH_ID, desired_accel, delta_time)

        # Lane change on threshold
        LC_THRESHOLD = 0.3
        if abs(steer_cmd) > LC_THRESHOLD:
            target_offset = 1 if steer_cmd > 0 else -1
            try:
                cur_lane  = traci.vehicle.getLaneIndex(self.VEH_ID)
                edge_id   = traci.vehicle.getRoadID(self.VEH_ID)
                num_lanes = traci.edge.getLaneNumber(edge_id)
                new_lane  = cur_lane + target_offset
                if 0 <= new_lane < num_lanes:
                    traci.vehicle.changeLane(self.VEH_ID, new_lane, 2.0)
            except Exception:
                pass

        # ── Simulation tick ───────────────────────────────────────────────────
        traci.simulationStep()

        reward      = 0.0
        terminated  = False
        truncated   = False
        sum_speed   = 0.0
        accum_energy = 0.0
        wiggle_stat  = 0.0
        safety_stat  = 0.0
        reason       = "running"

        # ── Collision / teleport detection ────────────────────────────────────
        if not self._veh_exists():
            terminated = True
            was_teleported = self.VEH_ID in traci.simulation.getStartingTeleportIDList()
            reward = -50.0 if was_teleported else -150.0
            reason = "off_road" if was_teleported else "collision"
            obs    = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, reward, terminated, truncated, \
                   self._build_info(0.0, 0.0, 0.0, 0.0, 0, reason)

        # Check explicit collision list (car may still be in scene one tick after)
        col_ids = set()
        for c in traci.simulation.getCollisions():
            col_ids.add(c.collider)
            col_ids.add(c.victim)
        if self.VEH_ID in col_ids:
            terminated    = True
            self._success = False
            reward        = -200.0
            reason        = "collision"
            obs           = self._get_obs()
            return obs, reward, terminated, truncated, \
                   self._build_info(0.0, 0.0, 0.0, 0.0, 0, reason)

        # ── Vehicle is alive — collect data ───────────────────────────────────
        ego_speed = traci.vehicle.getSpeed(self.VEH_ID)
        sum_speed = ego_speed

        e_cons = traci.vehicle.getElectricityConsumption(self.VEH_ID)
        if e_cons is None or np.isnan(e_cons):
            e_cons = 0.0
        accum_energy = e_cons

        # ── Step reward (FIX: prev_action managed here, not inside reward fn) ─
        step_r, wiggle_stat, safety_stat = self._calculate_reward(action)
        self._prev_action = action.copy()   # update AFTER reward uses it
        reward += step_r

        # ── Stuck detection ───────────────────────────────────────────────────
        # FIX: previously stuck_time was incremented both inside the sim loop
        # and again after the loop, double-counting on every stuck step.
        # Now it is only updated once, here.
        leader_info = traci.vehicle.getLeader(self.VEH_ID, 7.0)
        next_tls    = traci.vehicle.getNextTLS(self.VEH_ID)
        is_leader_stopped = (
            leader_info is not None and
            traci.vehicle.getSpeed(leader_info[0]) < 0.1
        )
        is_red_light = (
            bool(next_tls) and
            next_tls[0][2] < 10.0 and
            next_tls[0][3].lower() in ("r", "u", "y")
        )

        if ego_speed < 0.1:
            if is_leader_stopped or is_red_light:
                reward      -= 0.01        # stopped for a valid reason
            else:
                reward      -= 1.0         # unjustified stop
                self.stuck_time += 1
        else:
            self.stuck_time = 0

        if reward < -0.5:
            reward -= 0.2  # small extra nudge when already penalised

        # ── Goal check ────────────────────────────────────────────────────────
        if not self._success:
            try:
                cur_edge = traci.vehicle.getRoadID(self.VEH_ID)
                if (
                    hasattr(self, "current_route_edges") and
                    self.current_route_edges and
                    cur_edge == self.current_route_edges[-1] and
                    self._get_dist_to_destination() < 5.0
                ):
                    self._success = True
            except Exception:
                pass

        # ── Terminal conditions ───────────────────────────────────────────────
        if self._success:
            terminated = True
            reward    += 200.0
            reason     = "goal"

        elif self.stuck_time > 50 and not terminated:
            terminated = True
            reward    -= 120.0
            reason     = "stuck"

        elif self.step_count >= self.MAX_EPISODE_STEPS:
            truncated  = True
            reason     = "timeout"

        info_success = 1 if self._success else 0

        obs = self._get_obs()
        return obs, reward, terminated, truncated, \
               self._build_info(sum_speed, accum_energy, wiggle_stat,
                                safety_stat, info_success, reason)

    # =========================================================================
    # INFO HELPER
    # =========================================================================

    def _build_info(
        self,
        real_speed: float,
        real_energy: float,
        wiggle: float,
        safety: float,
        is_success: int,
        reason: str,
    ) -> dict:
        """
        Build the info dict with consistent keys.

        KEY FIX: the original code used "success_reason" here but MetricsWrapper
        in ppo.py reads info.get("reason", ...).  The key is now "reason"
        everywhere.
        """
        return {
            "real_speed":  float(real_speed),
            "real_energy": float(real_energy),
            "wiggle":      float(wiggle),
            "safety":      float(safety),
            "is_success":  int(is_success),
            "reason":      reason,          # FIX: was "success_reason" — mismatched key
        }

    # =========================================================================
    # CLOSE
    # =========================================================================

    def close(self):
        try:
            traci.close()
        except Exception:
            pass


# =============================================================================
# QUICK SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    env = SumoEnv(
        map_config="TestMap/osm.sumocfg",
        render=True,
        test_mode=False,
        test_route="TestMap/test_route.rou.xml",
        delay=100,
    )

    obs, info = env.reset()
    print(f"Observation shape : {obs.shape}")
    print(f"Obs min/max       : {obs.min():.3f} / {obs.max():.3f}")

    total_reward = 0.0
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"Step {env.step_count:>3d} | act={action} | "
            f"rew={reward:+.2f} | spd={obs[0]:.2f} | "
            f"energy={obs[2]:.2f} | reason={info['reason']}"
        )
        if terminated or truncated:
            print(f"Episode done — reason: {info['reason']}  total_reward={total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0

    env.close()
    print("Smoke test complete.")