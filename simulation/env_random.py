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
        self.TARGET_DIST = 35.0

        self.maps = [map_config] if isinstance(map_config, str) else map_config
        self.imperfection = imperfection
        self.impatience = impatience

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)

        self.last_known_dist = 0.0

        # --- CACHE STORAGE ---
        self.veh_data = {}
        # OPT 1: Cache turn_info để tránh gọi nhiều lần/step
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
            }
            # OPT 3: Cập nhật turn_info ngay trong cache để dùng chung toàn bộ step
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
    #  ROUTE-AWARE LANE GUIDANCE                                          #
    # ------------------------------------------------------------------ #
    def _get_turn_direction_numeric(self, from_edge: str, to_edge: str) -> float:
        """
        Tính hướng rẽ khi chuyển từ from_edge → to_edge.
        Trả về [-1, +1]:  -1 = rẽ trái/U-turn  |  0 = thẳng  |  +1 = rẽ phải
        """
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
        Nhìn trước trong current_route_edges để tìm khúc rẽ tiếp theo.
        Trả về (turn_dir, turn_dist_norm, lane_offset):
          turn_dir      : hướng rẽ [-1=trái … +1=phải, 0=thẳng]
          turn_dist_norm: tỉ lệ còn lại trên edge hiện tại [1=mới vào/xa rẽ, 0=sắp rẽ]
          lane_offset   : số làn cần dịch [-1=trái … +1=phải], 0 = đang đúng làn rồi
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

            # Tìm làn nào trên current_edge kết nối sang next_edge
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
        # Theo dõi 2 xe gần nhất phía trước + 2 xe phía sau mỗi bên (trái/phải)
        # Tổng: 8 xe × 2 giá trị (dist, relspeed) = 16 chiều
        # Layout: [LF1, LF1_v, LF2, LF2_v, LB1, LB1_v, LB2, LB2_v,
        #          RF1, RF1_v, RF2, RF2_v, RB1, RB1_v, RB2, RB2_v]
        result = [1.0, 0.0] * 8  # 8 slots mặc định (xa / tốc độ bằng 0)
        my_speed = self.veh_data["speed"] if self.veh_data else 0.0

        def process_side(neighbors, base_f, base_b):
            """Điền 2 xe gần nhất phía trước (base_f) và 2 xe phía sau (base_b)."""
            fronts, backs = [], []
            for n_id, dist in neighbors:
                try:
                    n_speed = traci.vehicle.getSpeed(n_id)
                except:
                    continue
                if dist > 0:
                    fronts.append((dist, n_speed))
                else:
                    backs.append((abs(dist), n_speed))
            fronts.sort(key=lambda x: x[0])
            backs.sort(key=lambda x: x[0])
            for slot, (d, spd) in enumerate(fronts[:2]):
                idx = base_f + slot * 2
                result[idx]     = min(d, self.MAX_DIST) / self.MAX_DIST
                result[idx + 1] = (my_speed - spd) / self.MAX_SPEED
            for slot, (d, spd) in enumerate(backs[:2]):
                idx = base_b + slot * 2
                result[idx]     = min(d, self.MAX_DIST) / self.MAX_DIST
                result[idx + 1] = (my_speed - spd) / self.MAX_SPEED

        try:
            # Layout 16 chiều:
            # [0..3]  = LF1,LF1v,LF2,LF2v  |  [4..7]  = LB1,LB1v,LB2,LB2v
            # [8..11] = RF1,RF1v,RF2,RF2v   |  [12..15]= RB1,RB1v,RB2,RB2v
            process_side(traci.vehicle.getNeighbors(self.VEH_ID, 2), 0, 4)   # trái:  F@0, B@4
            process_side(traci.vehicle.getNeighbors(self.VEH_ID, 1), 8, 12)  # phải:  F@8, B@12
        except:
            pass

        return result

    def _get_obs(self):
        if self.veh_data is None:
            return np.zeros(30, dtype=np.float32)

        d = self.veh_data

        try:
            # 1. EGO PHYSICS
            velocity     = np.clip(d["speed"] / self.MAX_SPEED, 0.0, 2.0)
            acceleration = np.clip(d["accel"] / self.MAX_ACCEL, -1.0, 1.0)
            elec         = np.clip(d["elec"]  / self.MAX_ELEC,  0.0, 5.0)

            lane_idx    = d["lane_idx"]
            road_id     = d["road_id"]
            total_lanes = traci.edge.getLaneNumber(road_id)
            norm_lane   = lane_idx / max(1, total_lanes - 1)

            slope      = np.clip(d["slope"]      / self.MAX_SLOPE, -1.0, 1.0)
            lat_offset = np.clip(d["lat_offset"], -10.0, 10.0)

            # 2. SURROUNDINGS
            surroundings = self._get_surroundings()

            # 3. LEADER
            leader = d["leader"]
            if leader:
                l_dist      = min(leader[1], self.MAX_DIST) / self.MAX_DIST
                # OPT 6: Gọi getSpeed cho leader vẫn cần thiết (không có trong cache chung)
                l_rel_speed = (d["speed"] - traci.vehicle.getSpeed(leader[0])) / self.MAX_SPEED
            else:
                l_dist, l_rel_speed = 1.0, 0.0

            # 4. INFRASTRUCTURE
            lane_id     = d["lane_id"]
            speed_limit = traci.lane.getMaxSpeed(lane_id) / self.MAX_SPEED if lane_id else 1.0

            tls_data = d["tls"]
            if tls_data:
                tls_dist  = min(tls_data[0][2], self.MAX_DIST) / self.MAX_DIST
                tls_state = 1.0 if tls_data[0][3].lower() == 'g' else 0.0
            else:
                tls_dist, tls_state = 1.0, 1.0

            # 5. ROUTE-AWARE LANE GUIDANCE — dùng cache, không gọi lại hàm
            turn_dir, turn_dist_n, lane_offset = self._turn_info_cache

            obs_list = [
                velocity, acceleration, elec, norm_lane, slope, lat_offset,
                l_dist, l_rel_speed,
                speed_limit, turn_dir, turn_dist_n, tls_dist, tls_state, lane_offset
            ] + surroundings

            obs = np.array(obs_list, dtype=np.float64)
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            obs = np.clip(obs, -5.0, 5.0)
            return obs.astype(np.float32)

        except Exception:
            return np.zeros(30, dtype=np.float32)

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

        W_SPEED    =  1.2
        W_PROGRESS =  0.8
        W_ENERGY   = -0.10
        W_COMFORT  = -0.05   # chỉ jerk ga/phanh — KHÔNG phạt tay lái để xe dám chuyển làn
        W_SAFETY   = -0.8
        W_TIME     = -0.2
        # W_LANE / W_LANE_OK đã bỏ — SUMO mode 514 đảm nhiệm việc đưa xe về đúng làn

        # --- Progress ---
        dist = self._get_dist_to_destination()
        if not hasattr(self, "prev_dist"): self.prev_dist = dist
        progress_reward = np.clip(self.prev_dist - dist, -1.0, 1.0)
        self.prev_dist = dist

        # --- Speed ---
        cur_speed    = d["speed"]
        speed_reward = cur_speed / self.MAX_SPEED
        if cur_speed < 3.0:
            speed_reward -= 0.8

        # --- Energy ---
        energy_penalty = np.clip(d["elec"] / self.MAX_ELEC, 0.0, 1.0)

        # --- Comfort: CHỈ jerk của ga/phanh (action[1]), KHÔNG tính tay lái (action[0]) ---
        if not hasattr(self, "prev_action"): self.prev_action = action
        accel_jerk = float(np.abs(action[1] - self.prev_action[1]))
        self.prev_action = action

        # --- Safety ---
        safety_penalty = 0.0
        leader = d["leader"]
        if cur_speed <= 10.0:   target_dist = 15.0
        elif cur_speed <= 20.0: target_dist = 30.0
        else:                   target_dist = 50.0
        if leader is not None and leader[1] < target_dist:
            safety_penalty = float(np.exp(-(leader[1] / target_dist)))

        # --- Tổng hợp ---
        speed_reward    = np.nan_to_num(speed_reward)
        progress_reward = np.nan_to_num(progress_reward)
        energy_penalty  = np.nan_to_num(energy_penalty)
        accel_jerk      = np.nan_to_num(accel_jerk)
        safety_penalty  = np.nan_to_num(safety_penalty)

        return (speed_reward    * W_SPEED)    + \
               (progress_reward * W_PROGRESS) + \
               (accel_jerk      * W_COMFORT)  + \
               (safety_penalty  * W_SAFETY)   + \
               (energy_penalty  * W_ENERGY)   + \
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
        # Dùng giá trị từ cache (đã tính ở cuối bước trước) để tránh gọi hàm thêm
        _, turn_dist_n, lane_offset = self._turn_info_cache
        # Nếu đang ở 30% cuối của đường VÀ đang sai làn → Bật tự động chuyển làn để cứu (514)
        sumo_rescue_active = turn_dist_n <= 0.2
        if sumo_rescue_active:
            traci.vehicle.setLaneChangeMode(self.VEH_ID, 514)
        else:
            traci.vehicle.setLaneChangeMode(self.VEH_ID, 0)

        # Áp dụng action
        traci.vehicle.setAcceleration(self.VEH_ID, desired_accel, duration=0.5)
        # Khi SUMO đang cứu xe (mode 514), không phát lệnh changeLane từ agent
        # để tránh xung đột hai bên → gây dao động qua lại giữa các làn
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

                # Trong hàm step()
        for _ in range(SIM_STEPS):
            traci.simulationStep()
            
            # KIỂM TRA VA CHẠM TRƯỚC KHI CẬP NHẬT CACHE
            colliding_ids = traci.simulation.getCollidingVehiclesIDList()
            if self.VEH_ID in colliding_ids:
                terminated = True
                reward = -200.0  # Phạt nặng vì đâm xe
                termination_reason = "collision"
                self.veh_data = None # Đánh dấu xe đã mất
                break # Thoát vòng lặp ngay lập tức

            self._update_cache() # Cập nhật thông tin nếu xe còn sống
            
            # Kiểm tra các lý do khác (Về đích, Teleport...)
            if self.veh_data is None:
                if self._success_check():
                    reward = 200.0
                    termination_reason = "goal"
                else:
                    termination_reason = "removed_unknown"
                terminated = True
                break

            ego_speed = self.veh_data["speed"]
            leader    = self.veh_data["leader"]

            # Context Aware Stuck Detection
            is_red_light = False
            tls_data = self.veh_data["tls"]
            if tls_data and tls_data[0][2] < 20.0:
                state = tls_data[0][3].lower()
                if 'r' in state or 'y' in state: is_red_light = True

            # OPT 11: Dùng speed từ cache (leader speed cần gọi API vì không có trong cache chính)
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

        if not hasattr(self, "prev_action"): self.prev_action = action
        wiggle_stat  = float(np.abs(action[1] - self.prev_action[1]))
        self.prev_action = action

        if not terminated:
            if self.stuck_time > 100:
                terminated = True
                reward -= 400.0
                termination_reason = "stuck_too_long"
            elif self.step_count >= self.MAX_EPISODE_STEPS:
                truncated = True
                termination_reason = "timeout"
                # Phạt thêm nếu timeout mà vẫn đang sai làn
                _, _, final_offset = self._turn_info_cache
                if abs(final_offset) > 0.01:
                    reward -= 50.0

        # OPT 12: Chỉ build route_info khi cần (khi episode kết thúc có lỗi)
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
            "wiggle":      wiggle_stat,
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