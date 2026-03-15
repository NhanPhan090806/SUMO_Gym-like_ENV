# env_random.py (Optimized Version)
import os
import sys
import math
import traci
import time
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

class SumoEnv(gym.Env):
    def __init__(self, render: bool = True, map_config = ["maps/TestMap/osm.sumocfg"],
              VTYPE_ID = "custom_passenger_car", TRAFFIC_SCALE = 0.5,
              test_mode: bool = False, test_route = "TestMap/test_route.rou.xml",
              imperfection = 0.5, impatience = 0.5, delay = 0) -> None:
        super().__init__()

        self.VEH_ID = "my_ego_car"
        self.VTYPE_ID = VTYPE_ID
        self.TRAFFIC_SCALE = TRAFFIC_SCALE
        self.render_mode: bool = render
        self.test_mode = test_mode
        self.test_route = test_route
        self.delay = delay
        self.step_count = 0
        self.MAX_EPISODE_STEPS = 1001

        # --- CONSTANTS ---
        self.MAX_SPEED = 55.6
        self.MAX_ACCEL = 4.15
        self.MAX_DECEL = 6.0
        self.MAX_ELEC = 120
        self.MAX_SLOPE = 20
        self.MAX_DIST = 100
        self.TARGET_SPEED_RATIO = 0.9   # bám 90% tốc độ giới hạn
        self.MIN_DESIRED_SPEED  = 3.0   # m/s — dưới mức này bị phạt too_slow

        self.maps = [map_config] if isinstance(map_config, str) else map_config
        self.imperfection = imperfection
        self.impatience = impatience

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # 21 chiều: 6 ego + 2 leader + 3 infra + 8 surroundings + 2 lane availability
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)

        self.last_known_dist = 0.0

        # --- CACHE STORAGE ---
        self.veh_data = {}
        # OPT 1: Cache turn_info — vẫn cần cho step() để kích hoạt sumo_rescue_active
        self._turn_info_cache = (0.0, 1.0, 0.0)
        # OPT 2: Cache độ dài các edge trong route (static, không đổi trong episode)
        self._route_edge_lengths: dict = {}

    def _veh_exists(self):
        try:
            return self.VEH_ID in traci.vehicle.getIDList()
        except Exception:
            return False

    def _update_cache(self):
        """Gọi API Traci 1 lần và lưu tất cả thông tin cần thiết vào biến self.veh_data"""
        if not self._veh_exists():
            self.veh_data = None
            return
        try:
            self.veh_data = {
                "speed":      traci.vehicle.getSpeed(self.VEH_ID),
                "accel":      traci.vehicle.getAcceleration(self.VEH_ID),
                "elec":       traci.vehicle.getElectricityConsumption(self.VEH_ID),
                "lane_idx":   traci.vehicle.getLaneIndex(self.VEH_ID),
                "lane_id":    traci.vehicle.getLaneID(self.VEH_ID),
                "road_id":    traci.vehicle.getRoadID(self.VEH_ID),
                "slope":      traci.vehicle.getSlope(self.VEH_ID),
                "lat_offset": traci.vehicle.getLateralLanePosition(self.VEH_ID),
                "lane_pos":   traci.vehicle.getLanePosition(self.VEH_ID),
                "leader":     traci.vehicle.getLeader(self.VEH_ID, dist=self.MAX_DIST),
                "tls":        traci.vehicle.getNextTLS(self.VEH_ID),
                "can_left":   1.0 if traci.vehicle.couldChangeLane(self.VEH_ID, 1) else 0.0,
                "can_right":  1.0 if traci.vehicle.couldChangeLane(self.VEH_ID, -1) else 0.0,
            }
            # OPT 3: Cập nhật turn_info cho step() (sumo_rescue_active), không dùng trong obs nữa
            self._turn_info_cache = self._get_next_turn_info()
        except Exception:
            self.veh_data = None

    def _get_passenger_edges(self) -> list:
        all_edges = traci.edge.getIDList()
        valid_edges = []
        for edge_id in all_edges:
            if edge_id.startswith(":") or edge_id.startswith("!"):
                continue
            lane_id = f"{edge_id}_0"
            try:
                allowed = traci.lane.getAllowed(lane_id)
                if not allowed or "passenger" in allowed:
                    valid_edges.append(edge_id)
            except:
                continue
        return valid_edges

    # ------------------------------------------------------------------ #
    #  ROUTE-AWARE LANE GUIDANCE — chỉ dùng nội bộ cho sumo_rescue_active
    # ------------------------------------------------------------------ #
    def _get_turn_direction_numeric(self, from_edge: str, to_edge: str) -> float:
        try:
            def edge_heading(edge_id):
                shape = traci.edge.getShape(edge_id)
                if len(shape) < 2: return None
                x1, y1 = shape[0]; x2, y2 = shape[-1]
                return math.degrees(math.atan2(y2 - y1, x2 - x1))

            a1 = edge_heading(from_edge)
            a2 = edge_heading(to_edge)
            if a1 is None or a2 is None: return 0.0
            diff = (a2 - a1 + 360) % 360
            if diff > 180: diff -= 360
            return float(np.clip(-diff / 180.0, -1.0, 1.0))
        except Exception:
            return 0.0

    def _get_next_turn_info(self) -> tuple:
        """
        Trả về (turn_dir, turn_dist_norm, lane_offset) — chỉ dùng trong step()
        để tính sumo_rescue_active (turn_dist_n <= 0.3), không đưa vào obs nữa.
        """
        default = (0.0, 1.0, 0.0)
        if not self.veh_data: return default
        try:
            current_edge = self.veh_data["road_id"]
            if current_edge.startswith(":"): return default
            if not hasattr(self, "current_route_edges") or not self.current_route_edges: return default
            if current_edge not in self.current_route_edges: return default

            indices  = [i for i, x in enumerate(self.current_route_edges) if x == current_edge]
            curr_idx = indices[-1]
            if curr_idx >= len(self.current_route_edges) - 1: return default

            next_edge = self.current_route_edges[curr_idx + 1]
            turn_dir  = self._get_turn_direction_numeric(current_edge, next_edge)

            lane_id  = self.veh_data["lane_id"]
            if not lane_id: return (turn_dir, 1.0, 0.0)

            # OPT 4: Dùng cache độ dài làn thay vì gọi API mỗi bước
            if lane_id not in self._route_edge_lengths:
                self._route_edge_lengths[lane_id] = traci.lane.getLength(lane_id)
            lane_len = self._route_edge_lengths[lane_id]

            dist_left = max(0.0, lane_len - self.veh_data["lane_pos"])
            turn_dist_norm = float(np.clip(dist_left / max(lane_len, 1.0), 0.0, 1.0))

            num_lanes    = traci.edge.getLaneNumber(current_edge)
            correct_lane = None
            for li in range(num_lanes):
                try:
                    for link in traci.lane.getLinks(f"{current_edge}_{li}"):
                        if traci.lane.getEdgeID(link[0]) == next_edge:
                            correct_lane = li
                            break
                except Exception:
                    continue
                if correct_lane is not None: break

            if correct_lane is None: return (turn_dir, turn_dist_norm, 0.0)

            raw_offset  = correct_lane - self.veh_data["lane_idx"]
            lane_offset = float(np.clip(raw_offset / max(1, num_lanes - 1), -1.0, 1.0))
            return (turn_dir, turn_dist_norm, lane_offset)
        except Exception:
            return default

    def _get_surroundings(self):
        # OPT 5: 1 xe gần nhất mỗi hướng trái/phải (trước + sau) = 8 chiều
        my_speed = self.veh_data["speed"] if self.veh_data else 0.0
        result = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # LF, LB, RF, RB (dist, relspeed)

        def process_side(neighbors, f_idx, b_idx):
            closest_f = float("inf")
            closest_b = float("inf")
            for n_id, dist in neighbors:
                try:
                    n_speed = traci.vehicle.getSpeed(n_id)
                except:
                    continue
                if dist > 0:
                    if dist < closest_f:
                        closest_f = dist
                        result[f_idx]     = min(dist, self.MAX_DIST) / self.MAX_DIST
                        result[f_idx + 1] = (my_speed - n_speed) / self.MAX_SPEED
                else:
                    adist = abs(dist)
                    if adist < closest_b:
                        closest_b = adist
                        result[b_idx]     = min(adist, self.MAX_DIST) / self.MAX_DIST
                        result[b_idx + 1] = (my_speed - n_speed) / self.MAX_SPEED

        try:
            process_side(traci.vehicle.getNeighbors(self.VEH_ID, 2), 0, 2)  # trái
            process_side(traci.vehicle.getNeighbors(self.VEH_ID, 1), 4, 6)  # phải
        except:
            pass

        return result

    def _get_obs(self):
        if self.veh_data is None:
            return np.zeros(21, dtype=np.float32)

        d = self.veh_data

        try:
            # 1. EGO PHYSICS (6 chiều)
            velocity     = np.clip(d["speed"] / self.MAX_SPEED, 0.0, 2.0)
            acceleration = np.clip(d["accel"] / self.MAX_ACCEL, -1.0, 1.0)
            elec         = np.clip(d["elec"]  / self.MAX_ELEC,  0.0, 5.0)

            lane_idx    = d["lane_idx"]
            road_id     = d["road_id"]
            total_lanes = traci.edge.getLaneNumber(road_id)
            norm_lane   = lane_idx / max(1, total_lanes - 1)

            slope      = np.clip(d["slope"]      / self.MAX_SLOPE, -1.0, 1.0)
            lat_offset = np.clip(d["lat_offset"], -10.0, 10.0)

            # 2. SURROUNDINGS (8 chiều)
            surroundings = self._get_surroundings()

            # 3. LEADER (2 chiều)
            leader = d["leader"]
            if leader:
                l_dist      = min(leader[1], self.MAX_DIST) / self.MAX_DIST
                # OPT 6: Gọi getSpeed cho leader vẫn cần thiết (không có trong cache chung)
                l_rel_speed = (d["speed"] - traci.vehicle.getSpeed(leader[0])) / self.MAX_SPEED
            else:
                l_dist, l_rel_speed = 1.0, 0.0

            # 4. INFRASTRUCTURE (3 chiều)
            lane_id     = d["lane_id"]
            speed_limit = traci.lane.getMaxSpeed(lane_id) / self.MAX_SPEED if lane_id else 1.0

            tls_data = d["tls"]
            if tls_data:
                tls_dist  = min(tls_data[0][2], self.MAX_DIST) / self.MAX_DIST
                tls_state = 1.0 if tls_data[0][3].lower() == 'g' else 0.0
            else:
                tls_dist, tls_state = 1.0, 1.0

            # 5. LANE AVAILABILITY (2 chiều)
            can_left  = d.get("can_left", 0.0)
            can_right = d.get("can_right", 0.0)

            # turn_dir / turn_dist_n / lane_offset đã bỏ khỏi obs
            # SUMO mode 514 đảm nhận việc đổi làn; _turn_info_cache vẫn dùng trong step()

            obs_list = [
                # EGO PHYSICS  [0..5]
                velocity, acceleration, elec, norm_lane, slope, lat_offset,
                # LEADER       [6..7]
                l_dist, l_rel_speed,
                # INFRA        [8..10]
                speed_limit, tls_dist, tls_state,
                # LANE AVAIL   [11..12]
                can_left, can_right,
                # SURROUNDINGS [13..20]
            ] + surroundings   # 8 chiều

            obs = np.array(obs_list, dtype=np.float64)
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            obs = np.clip(obs, -5.0, 5.0)
            return obs.astype(np.float32)

        except Exception:
            return np.zeros(21, dtype=np.float32)

    def _get_dist_to_destination(self):
        try:
            if not self.veh_data: return 1100.0
            current_edge = self.veh_data["road_id"]

            if current_edge.startswith(":"):
                return self.last_known_dist if self.last_known_dist else 1100.0

            if hasattr(self, "current_route_edges") and current_edge in self.current_route_edges:
                indices = [i for i, x in enumerate(self.current_route_edges) if x == current_edge]
                idx = indices[-1]
                remaining_edges = self.current_route_edges[idx:]

                # OPT 7: Dùng cache độ dài edge thay vì gọi API traci mỗi step
                dist = 0.0
                for e in remaining_edges:
                    key = f"{e}_0"
                    if key not in self._route_edge_lengths:
                        self._route_edge_lengths[key] = traci.lane.getLength(key)
                    dist += self._route_edge_lengths[key]

                dist -= self.veh_data["lane_pos"]
                self.last_known_dist = dist
                return dist
            return 1100.0
        except:
            return 1100.0

    def _calculate_reward(self, action):
        if not self.veh_data: return 0.0
        d = self.veh_data

        # ------------------------------------------------------------------ #
        #  TRỌNG SỐ                                                           #
        # ------------------------------------------------------------------ #
        W_SPEED_TARGET  =  1.0   # bell-curve bám tốc độ giới hạn
        W_TOO_SLOW      = -1.5   # phạt chạy dưới MIN_DESIRED_SPEED
        W_PROGRESS      =  0.85  # tiến về đích
        W_ENERGY        = -0.15  # hiệu suất năng lượng (Wh/m, không phải Wh/s)
        W_COMFORT       = -0.05  # jerk ga/phanh
        W_SAFETY        = -1.0   # TTC-based: 1.5s headway + 5m gap

        W_RED_LIGHT     = -50.0  # vượt đèn đỏ/vàng (phạt cứng 1 lần)
        W_TIME          = -0.2   # penalty sống sót từng bước

        cur_speed = d["speed"]
        lane_id   = d["lane_id"]

        # ------------------------------------------------------------------ #
        #  1. SPEED — bell-curve theo tốc độ giới hạn (từ wrappers.py)       #
        #  Thay thế linear cũ: không còn phạt cứng khi < 3 m/s               #
        # ------------------------------------------------------------------ #
        try:
            speed_limit_ms = traci.lane.getMaxSpeed(lane_id) if lane_id else self.MAX_SPEED
        except:
            speed_limit_ms = self.MAX_SPEED

        target_speed = max(speed_limit_ms * self.TARGET_SPEED_RATIO, self.MIN_DESIRED_SPEED)
        speed_error  = (cur_speed - target_speed) / max(target_speed, 1.0)
        r_speed_target = float(np.exp(-3.0 * speed_error ** 2))  # [0, 1], đỉnh tại target

        # Too-slow penalty: tăng tuyến tính từ 0 (tại MIN) đến 2.0 (khi dừng hẳn)
        if cur_speed < self.MIN_DESIRED_SPEED:
            r_too_slow = 2.0 * (1.0 - cur_speed / self.MIN_DESIRED_SPEED)
        else:
            r_too_slow = 0.0

        # ------------------------------------------------------------------ #
        #  2. PROGRESS — tiến về đích                                        #
        # ------------------------------------------------------------------ #
        dist = self._get_dist_to_destination()
        if not hasattr(self, "prev_dist"): self.prev_dist = dist
        progress_reward = float(np.clip(self.prev_dist - dist, -1.0, 1.0))
        self.prev_dist = dist

        # ------------------------------------------------------------------ #
        #  3. ENERGY EFFICIENCY — Wh/m thay vì Wh/s (từ wrappers.py)        #
        #  Ngăn speed-collapse: xe chậm tốn Wh/m cao hơn xe nhanh            #
        # ------------------------------------------------------------------ #
        elec = abs(d["elec"])
        if cur_speed > 0.5:
            wh_per_meter   = elec / cur_speed
            energy_penalty = float(np.clip(wh_per_meter / 0.5, 0.0, 1.0))
        else:
            energy_penalty = 1.0  # dừng hẳn → hiệu suất tệ nhất

        # ------------------------------------------------------------------ #
        #  4. COMFORT — jerk ga/phanh                                        #
        # ------------------------------------------------------------------ #
        if not hasattr(self, "prev_action"): self.prev_action = action
        accel_jerk = float(np.abs(action[1] - self.prev_action[1]))
        self.prev_action = action

        # ------------------------------------------------------------------ #
        #  5. SAFETY — TTC-based (từ wrappers.py)                            #
        #  safe_dist = 1.5s headway + 5m gap                                 #
        #  Thay thế exp-penalty cũ dùng khoảng cách cố định                  #
        # ------------------------------------------------------------------ #
        leader = d["leader"]
        safe_dist = cur_speed * 1.5 + 5.0
        if leader is not None:
            leader_dist_m = leader[1]
            if leader_dist_m < safe_dist:
                safety_penalty = float(np.clip(
                    (safe_dist - leader_dist_m) / safe_dist, 0.0, 1.0
                ))
            else:
                safety_penalty = 0.0
        else:
            safety_penalty = 0.0

        # ------------------------------------------------------------------ #
        #  6. RED LIGHT — phạt 1 lần khi xe vượt qua vạch dừng đèn đỏ/vàng #
        # ------------------------------------------------------------------ #
        road_id = d["road_id"]
        red_light_penalty = 0.0
        if road_id.startswith(":") and getattr(self, "_prev_tls_was_red", False):
            red_light_penalty = 10.0   # W_RED_LIGHT × 1.0 = -50.0 / lần vượt

        tls_data = d["tls"]
        if tls_data:
            tls_dist_raw  = tls_data[0][2]
            tls_state_str = tls_data[0][3].lower()
            self._prev_tls_was_red = (
                ('r' in tls_state_str or 'y' in tls_state_str)
                and tls_dist_raw < 3.0
            )
        else:
            self._prev_tls_was_red = False

        # ------------------------------------------------------------------ #
        #  TỔNG HỢP                                                          #
        # ------------------------------------------------------------------ #
        r_speed_target    = np.nan_to_num(r_speed_target)
        r_too_slow        = np.nan_to_num(r_too_slow)
        progress_reward   = np.nan_to_num(progress_reward)
        energy_penalty    = np.nan_to_num(energy_penalty)
        accel_jerk        = np.nan_to_num(accel_jerk)
        safety_penalty    = np.nan_to_num(safety_penalty)
        red_light_penalty = np.nan_to_num(red_light_penalty)

        return (r_speed_target    * W_SPEED_TARGET)  + \
               (r_too_slow        * W_TOO_SLOW)      + \
               (progress_reward   * W_PROGRESS)      + \
               (energy_penalty    * W_ENERGY)        + \
               (accel_jerk        * W_COMFORT)       + \
               (safety_penalty    * W_SAFETY)        + \
               (red_light_penalty * W_RED_LIGHT)     + \
               W_TIME

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        try:
            traci.close()
        except Exception:
            pass
        time.sleep(0.5)

        if self.test_mode:
            active_map = self.maps[0]
            route_arg  = ["-a", self.test_route]
        else:
            active_map = random.choice(self.maps)
            route_arg  = []

        self.step_count = 0
        self.veh_data   = None
        # OPT 8: Xoá cache độ dài edge khi reset episode (map mới có thể khác)
        self._route_edge_lengths = {}
        self._turn_info_cache    = (0.0, 1.0, 0.0)
        self._prev_tls_was_red   = False

        SumoBinary = "sumo-gui" if self.render_mode else "sumo"
        SumoCMD = [SumoBinary, "-c", active_map] + route_arg + \
                ["--start", "--quit-on-end",
                "--device.emissions.probability", "1.0",
                "--scale", str(self.TRAFFIC_SCALE),
                "--delay", str(self.delay),
                "--no-step-log", "true",
                "--time-to-teleport", "-1",
                "--collision.action", "remove",
                "--collision.check-junctions", "true",
                "--no-warnings", "true"]

        traci.start(SumoCMD)
        self._success = False

        if hasattr(self, "prev_action"): del self.prev_action
        if hasattr(self, "prev_dist"):   del self.prev_dist

        # Warmup
        for _ in range(random.randint(300, 600)): traci.simulationStep()

        try:
            existing_types = traci.vehicletype.getIDList()
            source_type    = "DEFAULT_VEHTYPE"
            if source_type not in existing_types and len(existing_types) > 0:
                source_type = existing_types[0]

            traci.vehicletype.copy(source_type, self.VTYPE_ID)
            traci.vehicletype.setVehicleClass(self.VTYPE_ID, "passenger")
            traci.vehicletype.setColor(self.VTYPE_ID, (0, 255, 0))
            traci.vehicletype.setParameter(self.VTYPE_ID, "mass",   "2911")
            traci.vehicletype.setLength(self.VTYPE_ID, "5.1181")
            traci.vehicletype.setEmissionClass(self.VTYPE_ID, "MMPEVEM")

            traci.vehicletype.setParameter(self.VTYPE_ID, "has.battery.device",                    "true")
            traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.capacity",               "123000.00")
            traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.chargeLevel",            "123000.00")
            traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.rechargeEfficiency",     "0.8")
            traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.maxRegenerationAcceleration", "2.0")

            for v_type in existing_types:
                traci.vehicletype.setParameter(v_type, "sigma",      str(self.imperfection))
                traci.vehicletype.setParameter(v_type, "impatience", str(self.impatience))
        except Exception:
            pass

        spawned = False
        if self.test_mode:
            while self.VEH_ID not in traci.vehicle.getIDList():
                traci.simulationStep()
            traci.vehicle.setType(self.VEH_ID, self.VTYPE_ID)
            traci.vehicle.setSpeedMode(self.VEH_ID, 0)
            traci.vehicle.setLaneChangeMode(self.VEH_ID, 0)
            for _ in range(50):
                traci.simulationStep()
                if self.VEH_ID in traci.vehicle.getIDList():
                    spawned = True
                    break
        else:
            self.drivable_edges = self._get_passenger_edges()

            for attempt in range(20):
                if not self.drivable_edges: break
                start_edge   = random.choice(self.drivable_edges)
                route_edges  = [start_edge]
                visited_edges = {start_edge}   # OPT 9: set thay list → O(1) lookup
                current_len  = 0.0
                try:
                    current_len += traci.lane.getLength(f"{start_edge}_0")
                except:
                    continue

                curr_edge_id = start_edge
                dead_end     = False
                while current_len < 1100.0:
                    try:
                        num_lanes = traci.edge.getLaneNumber(curr_edge_id)
                    except:
                        dead_end = True
                        break

                    # OPT 10: Dùng set cho valid_next_edges để tránh duplicate check O(n)
                    valid_next_set = set()
                    for lane_idx in range(num_lanes):
                        try:
                            links = traci.lane.getLinks(f"{curr_edge_id}_{lane_idx}")
                        except:
                            continue
                        for link in links:
                            next_lane_id = link[0]
                            try:
                                next_edge_id = traci.lane.getEdgeID(next_lane_id)
                            except:
                                continue
                            if (not next_edge_id.startswith(":")
                                    and next_edge_id in self.drivable_edges
                                    and next_edge_id not in visited_edges):
                                valid_next_set.add(next_edge_id)

                    if not valid_next_set:
                        dead_end = True
                        break

                    next_edge = random.choice(list(valid_next_set))
                    route_edges.append(next_edge)
                    visited_edges.add(next_edge)
                    try: current_len += traci.lane.getLength(f"{next_edge}_0")
                    except: pass
                    curr_edge_id = next_edge

                if not dead_end and current_len >= 1100.0:
                    try:
                        route_id = f"route_{random.randint(0, 999999)}"
                        traci.route.add(route_id, route_edges)
                        self.current_route_edges = route_edges
                        traci.vehicle.add(self.VEH_ID, route_id, departPos="free", typeID=self.VTYPE_ID)
                        for _ in range(50):
                            traci.simulationStep()
                            if self.VEH_ID in traci.vehicle.getIDList():
                                spawned = True
                                break
                        if spawned:
                            traci.vehicle.setSpeedMode(self.VEH_ID, 0)
                            traci.vehicle.setLaneChangeMode(self.VEH_ID, 0)
                            self.last_known_dist = current_len
                            self.prev_dist       = current_len
                            break
                        else:
                            try: traci.vehicle.remove(self.VEH_ID)
                            except: pass
                    except Exception:
                        try: traci.vehicle.remove(self.VEH_ID)
                        except: pass
                        continue

            if not spawned:
                return self.reset(seed=seed, options=options)

        if self.render_mode and spawned:
            if self.VEH_ID in traci.vehicle.getIDList():
                traci.gui.trackVehicle("View #0", self.VEH_ID)
                traci.gui.setZoom("View #0", 1001)

        self.stuck_time = 0
        self._update_cache()

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.step_count += 1
        steer_cmd = action[0]
        accel_cmd = action[1]

        desired_accel = accel_cmd * self.MAX_ACCEL if accel_cmd >= 0 else accel_cmd * self.MAX_DECEL

        SIM_STEPS = 1

        if not self._veh_exists():
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, \
                {"real_speed": 0, "reason": "already_dead", "is_success": 0}

        # --- Chuyển đổi LaneChangeMode theo bối cảnh khoảng cách rẽ ---
        _, turn_dist_n, _ = self._turn_info_cache
        sumo_rescue_active = turn_dist_n <= 0.3
        if sumo_rescue_active:
            traci.vehicle.setLaneChangeMode(self.VEH_ID, 514)
        else:
            traci.vehicle.setLaneChangeMode(self.VEH_ID, 0)

        # Áp dụng action
        traci.vehicle.setAcceleration(self.VEH_ID, desired_accel, duration=0.5)
        # Khi SUMO đang cứu xe (mode 514), không phát lệnh changeLane từ agent
        if not sumo_rescue_active:
            LC_THRESHOLD     = 0.15
            current_lane_idx = traci.vehicle.getLaneIndex(self.VEH_ID)
            if steer_cmd < -LC_THRESHOLD:
                traci.vehicle.changeLane(self.VEH_ID, max(0, current_lane_idx - 1), 0.5)
            elif steer_cmd > LC_THRESHOLD:
                try:
                    edge_id   = traci.vehicle.getRoadID(self.VEH_ID)
                    num_lanes = traci.edge.getLaneNumber(edge_id)
                    traci.vehicle.changeLane(self.VEH_ID, min(num_lanes - 1, current_lane_idx + 1), 0.5)
                except: pass

        reward             = 0.0
        terminated         = False
        truncated          = False
        accumulated_energy = 0.0
        sum_speed          = 0.0
        valid_steps        = 0
        termination_reason = "running"

        for _ in range(SIM_STEPS):
            traci.simulationStep()
            self._update_cache()

            if self.veh_data is None:
                terminated = True
                teleport_list = traci.simulation.getStartingTeleportIDList()
                if self.VEH_ID in teleport_list:
                    reward -= 50.0
                    termination_reason = "teleport"
                else:
                    reward -= 200.0
                    termination_reason = "collision"
                break

            ego_speed = self.veh_data["speed"]
            leader    = self.veh_data["leader"]

            # Context Aware Stuck Detection
            is_red_light = False
            tls_data = self.veh_data["tls"]
            if tls_data and tls_data[0][2] < 20.0:
                state = tls_data[0][3].lower()
                if 'r' in state or 'y' in state: is_red_light = True

            # OPT 11: Dùng speed từ cache
            is_leader_stopped = (leader is not None and traci.vehicle.getSpeed(leader[0]) < 0.5)

            if ego_speed < 0.5 and not (is_red_light or is_leader_stopped):
                reward -= 0.5
                self.stuck_time += 1
            else:
                self.stuck_time = 0

            sum_speed   += ego_speed
            valid_steps += 1
            e = self.veh_data["elec"]
            accumulated_energy += e if not np.isnan(e) else 0.0

            reward += self._calculate_reward(action) / SIM_STEPS

            if self._success_check():
                terminated = True
                reward += 200.0
                self._success = True
                termination_reason = "goal"
                break

        if not terminated:
            if self.stuck_time > 100:
                terminated = True
                reward -= 400.0
                termination_reason = "stuck_too_long"
            elif self.step_count >= self.MAX_EPISODE_STEPS:
                truncated = True
                termination_reason = "timeout"
                _, _, final_offset = self._turn_info_cache
                if abs(final_offset) > 0.01:
                    reward -= 50.0

        # OPT 12: Chỉ build route_info khi cần
        route_info = ""
        if (terminated or truncated) and termination_reason in ("stuck_too_long", "timeout", "teleport"):
            if hasattr(self, "current_route_edges"):
                route_info = " -> ".join(self.current_route_edges)

        obs           = self._get_obs()
        avg_real_speed = sum_speed / max(1, valid_steps)

        safety_val = 0.0
        if self.veh_data and self.veh_data["leader"]:
            dist = self.veh_data["leader"][1]
            if dist < 20: safety_val = 1.0 - (dist / 20.0)

        info = {
            "real_speed":  avg_real_speed,
            "real_energy": accumulated_energy,
            "safety":      safety_val,
            "step_reward": reward,
            "is_success":  1 if self._success else 0,
            "reason":      termination_reason,
            "route":       route_info,
        }

        return obs, reward, terminated, truncated, info

    def _success_check(self):
        if not self.veh_data: return False
        try:
            current_edge = self.veh_data["road_id"]
            if current_edge.startswith(":"): return False

            if hasattr(self, "current_route_edges") and self.current_route_edges:
                if current_edge == self.current_route_edges[-1]:
                    lane_len = traci.lane.getLength(self.veh_data["lane_id"])
                    pos      = self.veh_data["lane_pos"]
                    if pos > (lane_len - 20.0):
                        return True
        except: pass
        return False

    def close(self):
        try:
            traci.close()
        except:
            pass

if __name__ == "__main__":
    env = SumoEnv(map_config="TestMap/osm.sumocfg", render=True, test_mode=False, test_route="TestMap/test_route.rou.xml", delay=100)
    obs, info = env.reset()
    print(f"Init Obs Shape: {obs.shape}")
    total_reward = 0
    print("Starting Loop...")
    for i in range(50):
        action = env.action_space.sample()
        action[1] = np.random.uniform(-1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {env.step_count} | Action: {action} | Reward: {reward:.2f} | Speed: {obs[0]:.2f} | Energy: {obs[2]:.2f}")
        if terminated or truncated:
            print("Episode Finished!")
            obs, info = env.reset()
    env.close()
    print("Test Complete.")