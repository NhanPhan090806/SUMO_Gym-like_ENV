#Đây là bản hiện tại đang chạy tốt
# Continuous action space SUMO gym-like env
import os
import sys
import traci
import time
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# All vehicle data are based on Vinfast VF9 model
# (https://static-cms-prod.vinfastauto.us/cms-vinfast-us/Specs/VF-9-Spec.pdf)

# GOAL: TO GET THE EGO CAR TO GO ON A RANDOM ROUT WHICH THE LEAST STEP POSSIBLE AND LEAST CO2 EMISSION AND FUEL CONSUMPTION 
# Discrete action space (for ease of implementation =))).), each action lasts for ~10 simulationstep


# --- BOILERPLATE SETUP ---
# check if SUMO is set up appropriately
if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("Please declare environment variable 'SUMO_HOME'")



class SumoEnv(gym.Env):
	def __init__(self, render: bool = True, map_config = ["maps/TestMap/osm.sumocfg"], 
			  VTYPE_ID = "custom_passenger_car", TRAFFIC_SCALE = 5.0,
			  test_mode: bool = False, test_route = "TestMap/test_route.rou.xml",
			  imperfection = 0.5, impatience = 0.5, delay = 0) -> None:
		super().__init__()


		self.VEH_ID = "my_ego_car" # if this is changed, change the .rou.xml too
		self.VTYPE_ID = VTYPE_ID
		self.TRAFFIC_SCALE = TRAFFIC_SCALE
		self.render_mode: bool = render
		self.test_mode = test_mode
		self.test_route = test_route
		self.delay = delay
		self.step_count = 0 # Initialize counter
		self.MAX_EPISODE_STEPS = 1000 # Force stop after 2000 steps (approx 30 mins sim time)
		# Get the absolute path to the 'simulation' folder
		curr_dir = os.path.dirname(os.path.abspath(__file__))
		# Go up one level to 'Sumo' folder
		root_dir = os.path.dirname(curr_dir)
		
		# --- CENTRALIZED PHYSICS CONSTANTS ---
		self.MAX_SPEED = 55.6 # m/s
		self.MAX_ACCEL = 4.15 # m/s^2
		self.MAX_DECEL = 6.0  # m/s^2
		self.MAX_ELEC = 120   # Wh/s
		self.MAX_SLOPE = 20   # degrees
		self.MAX_DIST = 100 # meters
		self.TARGET_DIST = 35.0 # meters, target for leader distant


		# Handle string input and fix path
		self.maps = [map_config] if isinstance(map_config, str) else map_config
		self.imperfection = imperfection
		self.impatience = impatience
		# 2 continuous action box: [steering, throttle]
		self.action_space = spaces.Box(
			low=-1.0,
			high=1.0,
			shape=(2,),
			dtype=np.float32
		)

		# for 1-agent system
		# Ego Physics (4old + 4new): [Speed, Acceleration, Normalized_Lane_Index, 
		# Fuel/Electricity Consumption (both in one as this case is using HEV)]
		# Added: (just for the AI to learn "intent", not real physics, as SUMO handles these very tidy, 
		# so tidy that you don't need to explicitly define the exact value)
			# Lateral Lane Position: Distance from the center of the lane (in meters). -1.5 means left of center, 1.5 means right.
			# Heading Error: Difference between Car Angle and Lane Angle (in degrees or radians).
			# Current Steering Angle: The current angle of the front wheels. The agent needs to know "where are my wheels pointing right now?" to avoid oscillating.
			# Road Slope: The incline of the road

		# Front Radar (2): [Leader_Distance, Leader_Rel_Speed]
		# Side Awareness (8):
			# Front-Left: [Dist, Rel_Speed]
			# Back-Left: [Dist, Rel_Speed]
			# Front-Right: [Dist, Rel_Speed]
			# Back-Right: [Dist, Rel_Speed]
		# Infrastructure (6): [Speed_Limit, Can_Go_Left(0/1), Can_Go_Right(0/1), 
		# TLS_Dist, TLS_State(Red/Green), Dist_To_Turn]
		self.observation_space = spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(22,),
			dtype=np.float32
			)
		
		self.last_known_dist = 0.0
	
	def _veh_exists(self):
		try:
			return self.VEH_ID in traci.vehicle.getIDList()
		except Exception:
			return False
		
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
		random.shuffle(valid_edges)
		return valid_edges
	
	def _get_surroundings(self):
		# Constants for normalization

		# Default values: [Dist=1.0 (Far), RelSpeed=0.0 (same speed)]
		# Order: [Left-front, Left-back, Right-front, Right-back]
		data = {
			"L_F_Dist": 1.0, "L_F_RelSpeed": 0.0,
			"L_B_Dist": 1.0, "L_B_RelSpeed": 0.0,
			"R_F_Dist": 1.0, "R_F_RelSpeed": 0.0,
			"R_B_Dist": 1.0, "R_B_RelSpeed": 0.0,
		}
		
		my_speed = traci.vehicle.getSpeed(self.VEH_ID)

		# check left lane
		# The '2' is a binary bitset (0b010) meaning "Left Only"
		# for traci.vehicl.getNeighbors: if the return distance is > 0, the car is at front (R_F, L_F)
		# else, the car is at back (R_B, L_B)
		left_cars = traci.vehicle.getNeighbors(self.VEH_ID, 2)

		closest_front = float("inf")
		closest_back = float("inf")

		for n_id, dist in left_cars:
			if dist > 0: # dist > 0 => check front
				if dist < closest_front:
					closest_front = dist
					n_speed = traci.vehicle.getSpeed(n_id)
					data["L_F_Dist"] = min(dist, self.MAX_DIST) / self.MAX_DIST # normalization and double check
					data["L_F_RelSpeed"] = (my_speed - n_speed) / self.MAX_SPEED # type: ignore
			else: # Back (dist <= 0)
				if abs(dist) < closest_back:
					closest_back = abs(dist)
					n_speed = traci.vehicle.getSpeed(n_id)
					data["L_B_Dist"] = min(dist, self.MAX_DIST) / self.MAX_DIST # normalization and double check
					data["L_B_RelSpeed"] = (my_speed - n_speed) / self.MAX_SPEED # type: ignore

		
		# check right lane
		# The '1' is a binary bitset (0b01) meaning "Right Only"
		right_cars = traci.vehicle.getNeighbors(self.VEH_ID, 1)

		closest_front = float("inf")
		closest_back = float("inf")

		for n_id, dist in right_cars:
			if dist > 0: # front
				if dist < closest_front:
					closest_front = dist
					n_speed = traci.vehicle.getSpeed(n_id)
					data["R_F_Dist"] = min(dist, self.MAX_DIST) / self.MAX_DIST
					data["R_F_RelSpeed"] = (my_speed - n_speed) / self.MAX_SPEED # type: ignore
			else: # back
				if abs(dist) < closest_back:
					closest_back = abs(dist)
					n_speed = traci.vehicle.getSpeed(n_id)
					data["R_B_Dist"] = min(dist, self.MAX_DIST) / self.MAX_DIST
					data["R_B_RelSpeed"] = (my_speed - n_speed) / self.MAX_SPEED # type: ignore
		
		output = [stats for key, stats in data.items()]
		return output
	


	def _get_obs(self):
		if not self._veh_exists():
			return np.zeros(22, dtype=np.float32)

		try:
			# 1. EGO PHYSICS (Tối ưu còn 6 biến thay vì 8)
			velocity = traci.vehicle.getSpeed(self.VEH_ID) / self.MAX_SPEED
			acceleration = traci.vehicle.getAcceleration(self.VEH_ID) / self.MAX_ACCEL
			elec = traci.vehicle.getElectricityConsumption(self.VEH_ID) / self.MAX_ELEC
			
			lane_idx = traci.vehicle.getLaneIndex(self.VEH_ID)
			road_id = traci.vehicle.getRoadID(self.VEH_ID)
			total_lanes = traci.edge.getLaneNumber(road_id)
			norm_lane = lane_idx / max(1, total_lanes - 1)

			slope = traci.vehicle.getSlope(self.VEH_ID) / self.MAX_SLOPE
			lat_offset = traci.vehicle.getLateralLanePosition(self.VEH_ID)
			
			# Đã loại bỏ heading_error và steer_angle

			# 2. SURROUNDINGS (8 biến - giữ nguyên)
			surroundings = self._get_surroundings()

			# 3. LEADER (2 biến - giữ nguyên)
			leader = traci.vehicle.getLeader(self.VEH_ID, dist=self.MAX_DIST)
			l_dist, l_rel_speed = (leader[1]/self.MAX_DIST, (velocity - traci.vehicle.getSpeed(leader[0]))/self.MAX_SPEED) if leader else (1.0, 0.0)

			# 4. INFRASTRUCTURE (6 biến - giữ nguyên)
			lane_id = traci.vehicle.getLaneID(self.VEH_ID)
			speed_limit = traci.lane.getMaxSpeed(lane_id) / self.MAX_SPEED if lane_id else 1.0
			can_left = 1.0 if lane_idx < (total_lanes - 1) else 0.0
			can_right = 1.0 if lane_idx > 0 else 0.0
			
			tls_data = traci.vehicle.getNextTLS(self.VEH_ID)
			tls_dist, tls_state = (tls_data[0][2]/self.MAX_DIST, (1.0 if tls_data[0][3].lower() == 'g' else 0.0)) if tls_data else (1.0, 1.0)
			
			turn_dist = min(traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(self.VEH_ID), self.MAX_DIST) / self.MAX_DIST if lane_id else 1.0

			# Kết hợp thành vector 22 chiều
			obs = np.array([
				velocity, acceleration, elec, norm_lane, slope, lat_offset, # Ego (6)
				l_dist, l_rel_speed,                                        # Leader (2)
				speed_limit, can_left, can_right, tls_dist, tls_state, turn_dist # Infra (6)
			] + surroundings, dtype=np.float32)                             # Surround (8)

			return np.nan_to_num(obs, nan=0.0)

		except Exception:
			return np.zeros(22, dtype=np.float32)
	
	def _get_dist_to_destination(self):
		try:
			current_edge = traci.vehicle.getRoadID(self.VEH_ID)
			
			# Xử lý Internal Edge (giống như bạn đã làm)
			if current_edge.startswith(":"):
				return self.last_known_dist if self.last_known_dist else 450.0

			if hasattr(self, "current_route_edges") and current_edge in self.current_route_edges:
				# Tìm tất cả các vị trí của cạnh này trong route (để xử lý vòng lặp)
				indices = [i for i, x in enumerate(self.current_route_edges) if x == current_edge]
				
				# Chọn index gần nhất với tiến độ hiện tại (hoặc đơn giản là dùng index cuối cùng tìm thấy)
				# Ở đây ta dùng index cuối cùng để tránh việc bị reset về đầu route
				idx = indices[-1] 
				
				remaining_edges = self.current_route_edges[idx:]
				dist = 0.0
				for e in remaining_edges:
					dist += traci.lane.getLength(f"{e}_0")

				dist -= traci.vehicle.getLanePosition(self.VEH_ID)
				self.last_known_dist = dist
				return dist
			
			return 450.0 # Nếu lạc đường, coi như còn rất xa đích
		except:
			return 450.0


	# reward function
	# -----------------------------------------------------------
	# PHẦN CHỈNH SỬA: REWARD FUNCTION CÂN BẰNG LẠI
	# -----------------------------------------------------------
	def _calculate_reward(self, action):
		# --- CẤU HÌNH TRỌNG SỐ MỚI ---
		# Tăng trọng số tốc độ và tiến độ để khuyến khích di chuyển
		W_SPEED = 1.0      # Tăng từ 0.5 lên 1.5
		W_PROGRESS = 0.4   # Tăng từ 0.5 lên 1.0
		
		# Giảm nhẹ các hình phạt để Agent "dám" thử nghiệm
		W_ENERGY = -0.05   # Giảm từ -0.6 xuống -0.05 (Ban đầu đừng quan tâm tiết kiệm điện)
		W_COMFORT = -0.05  # Giảm từ -0.6 xuống -0.05 (Để Agent thoải mái đánh lái)
		W_SAFETY = -0.8    # Giữ nguyên phạt an toàn
		
		# QUAN TRỌNG: Phạt tồn tại (Time penalty)
		# Ép agent phải hoàn thành nhanh, đứng yên là chết dần
		W_TIME = -0.1      

		# 1. Tính toán Progress (Tiến độ về đích)
		dist = self._get_dist_to_destination()
		if not hasattr(self, "prev_dist"):
			self.prev_dist = dist
		
		# Thưởng khi khoảng cách tới đích giảm đi
		progress_raw = self.prev_dist - dist
		progress_reward = np.clip(progress_raw, -1.0, 1.0) # Clip để tránh bug dịch chuyển tức thời
		self.prev_dist = dist

		# 2. Tính toán Speed (Vận tốc)
		cur_speed = traci.vehicle.getSpeed(self.VEH_ID)
		# Thưởng trực tiếp theo % tốc độ tối đa (Linear)
		speed_reward = cur_speed / self.MAX_SPEED
		
		# Phạt nặng nếu đi quá chậm (dưới 1m/s ~ 3.6km/h) mà không có lý do
		if cur_speed < 1.0:
			speed_reward -= 0.5 # Phạt thêm vào speed reward

		# 3. Tính toán Energy (Năng lượng)
		elec = traci.vehicle.getElectricityConsumption(self.VEH_ID)
		energy_penalty = elec / self.MAX_ELEC 
		energy_penalty = np.clip(energy_penalty, 0.0, 1.0)

		# 4. Tính toán Comfort (Độ êm ái)
		if not hasattr(self, "prev_action"):
			self.prev_action = action
		action_delta = np.abs(action - self.prev_action)
		wiggle_penalty = np.mean(action_delta)
		self.prev_action = action

		# 5. Tính toán Safety (An toàn với xe trước)
		safety_penalty = 0.0
		leader = traci.vehicle.getLeader(self.VEH_ID, self.MAX_DIST)
		
		# Logic Dynamic Target Distance
		if cur_speed <= 10.0: target_dist = 15.0
		elif cur_speed <= 20.0: target_dist = 30.0
		else: target_dist = 50.0

		if leader is not None:
			leader_dist = leader[1]
			if leader_dist < target_dist:
				# Phạt mũ (Exponential) để agent sợ khi quá gần
				safety_penalty = np.exp(-(leader_dist / target_dist)) 
			else:
				safety_penalty = 0.0

		# Clean NaN values
		speed_reward = np.nan_to_num(speed_reward)
		progress_reward = np.nan_to_num(progress_reward)
		energy_penalty = np.nan_to_num(energy_penalty)
		wiggle_penalty = np.nan_to_num(wiggle_penalty)
		safety_penalty = np.nan_to_num(safety_penalty)

		# TỔNG HỢP REWARD
		reward = (speed_reward * W_SPEED) + \
				 (progress_reward * W_PROGRESS) + \
				 (wiggle_penalty * W_COMFORT) + \
				 (safety_penalty * W_SAFETY) + \
				 (energy_penalty * W_ENERGY) + \
				 W_TIME

		return reward

	def reset(self, seed=None, options=None):
		super().reset(seed=seed, options=options)
		
		# 1. Close previous simulation safely
		try:
			traci.close()
		except Exception:
			pass
		time.sleep(1.0) 
		
		# 2. Configuration
		if self.test_mode:
			active_map = self.maps[0]
			route_arg = ["-a", self.test_route]
		else:
			active_map = random.choice(self.maps)
			route_arg = []
			
		self.step_count = 0
		
		SumoBinary: str = "sumo-gui" if self.render_mode else "sumo"
		SumoCMD: list[str] = [SumoBinary, "-c", active_map] + route_arg + \
				["--start", "--quit-on-end",
				"--device.emissions.probability", "1.0",
				"--scale", str(self.TRAFFIC_SCALE),
				"--delay", str(self.delay),
				"--no-step-log", "true",
				"--time-to-teleport", "-1",
				"--collision.action", "remove",
				"--collision.check-junctions", "true",
				"--no-warnings", "true"]
		
		# 3. Start SUMO
		traci.start(SumoCMD)
		
		self._success = False

		# Clear history
		if hasattr(self, "prev_action"): del self.prev_action
		if hasattr(self, "prev_dist"): del self.prev_dist
		
		# Warmup
		for _ in range(random.randint(300, 600)): traci.simulationStep()

		# 4. Setup Vehicle Type
		try:
			existing_types = traci.vehicletype.getIDList()
			source_type = "DEFAULT_VEHTYPE"
			if source_type not in existing_types and len(existing_types) > 0:
				source_type = existing_types[0]

			traci.vehicletype.copy(source_type, self.VTYPE_ID)
			traci.vehicletype.setVehicleClass(self.VTYPE_ID, "passenger")
			traci.vehicletype.setColor(self.VTYPE_ID, (0, 255, 0)) 
			traci.vehicletype.setParameter(self.VTYPE_ID, "mass", "2911")
			traci.vehicletype.setLength(self.VTYPE_ID, "5.1181")
			traci.vehicletype.setEmissionClass(self.VTYPE_ID, "MMPEVEM")
			
			# EV Parameters
			traci.vehicletype.setParameter(self.VTYPE_ID, "has.battery.device", "true")
			traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.capacity", "123000.00")
			traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.chargeLevel", "123000.00")
			# Hiệu suất thu hồi năng lượng khi phanh (0.0 - 1.0)
			traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.rechargeEfficiency", "0.8")
			# Lực phanh tối đa có thể tái sinh
			traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.maxRegenerationAcceleration", "2.0")
			
			# Imperfection
			for v_type in existing_types:
				traci.vehicletype.setParameter(v_type, "sigma", str(self.imperfection))
				traci.vehicletype.setParameter(v_type, "impatience", str(self.impatience))

		except Exception as e:
			print(f"Error defining vType: {e}")

		# 5. FIXED ROUTE GENERATION (E1 -> E25)
		spawned = False
		
		if self.test_mode:
			# ... (Keep existing test logic if needed) ...
			pass
		else:
			# --- CREATE FIXED ROUTE ---
			try:
				# Generate list ["E1", "E2", ..., "E25"]
				# Ensure these IDs match exactly with your .net.xml file
				fixed_edges = [f"E{i}" for i in range(1, 14)]
				
				route_id = "fixed_training_route"
				
				# Add route to SUMO
				traci.route.add(route_id, fixed_edges)
				self.current_route_edges = fixed_edges
				
				# Add Vehicle
				traci.vehicle.add(self.VEH_ID, route_id, departPos="free", typeID=self.VTYPE_ID)
				
				# Wait for insertion
				for _ in range(50):
					traci.simulationStep()
					if self.VEH_ID in traci.vehicle.getIDList():
						spawned = True
						break
				
				if spawned:
					# Configure
					traci.vehicle.setSpeedMode(self.VEH_ID, 0)
					traci.vehicle.setLaneChangeMode(self.VEH_ID, 0)
					
					# Calculate total distance for Reward Normalization
					total_len = 0.0
					for e in fixed_edges:
						try:
							total_len += traci.lane.getLength(f"{e}_0")
						except:
							pass
					
					self.last_known_dist = total_len
					self.prev_dist = total_len
				else:
					print("Error: Could not spawn vehicle on fixed route E1->E25. Check if E1 is blocked.")
					# Clean up pending vehicle to prevent errors in next episode
					try: traci.vehicle.remove(self.VEH_ID) 
					except: pass
					return self.reset(seed=seed, options=options)

			except Exception as e:
				print(f"Fixed Route Setup Error: {e}")
				# If E1...E25 don't exist or aren't connected, this will crash
				traci.close()
				raise e

		# 6. Finalize
		if self.render_mode and spawned:
			if self.VEH_ID in traci.vehicle.getIDList():
				traci.gui.trackVehicle("View #0", self.VEH_ID)
				traci.gui.setZoom("View #0", 2000)

		self.stuck_time = 0
		obs = self._get_obs()
		info = {}
		return obs, info

	
	# -----------------------------------------------------------
	# HÀM STEP ĐÃ TỐI ƯU HÓA (Dựa trên Code 2 + Logic Code 1)
	# -----------------------------------------------------------
	def step(self, action):
		self.step_count += 1

		# 1. Physics Setup & Action Mapping
		steer_cmd = action[0]
		accel_cmd = action[1]
		
		# Mapping gia tốc
		if accel_cmd >= 0:
			desired_accel = accel_cmd * self.MAX_ACCEL
		else:
			desired_accel = accel_cmd * self.MAX_DECEL 

		SIM_STEPS = 1 
		
		# Safety check đầu vào
		if not self._veh_exists():
			return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, \
				{"real_speed": 0, "reason": "already_dead", "is_success": 0}

		# Thực thi lệnh điều khiển (Action Application)
		# Áp dụng gia tốc
		traci.vehicle.setAcceleration(self.VEH_ID, desired_accel, duration=0.5)
		
		# Áp dụng lái (Lane Change) - Logic gọn của Code 1
		LC_THRESHOLD = 0.3
		current_lane_idx = traci.vehicle.getLaneIndex(self.VEH_ID)
		if steer_cmd < -LC_THRESHOLD: 
			# Rẽ phải (giảm index) - Cần check max(0, ...)
			traci.vehicle.changeLane(self.VEH_ID, max(0, current_lane_idx - 1), 1.0)
		elif steer_cmd > LC_THRESHOLD: 
			# Rẽ trái (tăng index) - Cần check min(num_lanes, ...)
			# Lấy số lane của cạnh hiện tại để tránh lỗi out of bounds
			try:
				edge_id = traci.vehicle.getRoadID(self.VEH_ID)
				num_lanes = traci.edge.getLaneNumber(edge_id)
				traci.vehicle.changeLane(self.VEH_ID, min(num_lanes - 1, current_lane_idx + 1), 1.0)
			except: pass

		# 2. Simulation Loop
		reward = 0.0
		terminated = False
		truncated = False
		
		accumulated_energy = 0.0 
		sum_speed = 0.0
		valid_steps = 0 
		
		# Biến lưu lý do kết thúc
		termination_reason = "running"

		for _ in range(SIM_STEPS):
			traci.simulationStep()

			# --- CRITICAL: CHECK EXISTENCE IMMEDIATELY ---
			if self.VEH_ID not in traci.vehicle.getIDList():
				terminated = True
				# Phân loại lý do chết: Teleport hay Collision
				teleport_list = traci.simulation.getStartingTeleportIDList()
				if self.VEH_ID in teleport_list:
					reward -= 50.0 
					termination_reason = "teleport" # Lỗi map hoặc vi phạm lane
				else:
					reward -= 200.0 
					termination_reason = "collision" # Đâm xe khác
				break 
			
			# --- CONTEXT AWARE STUCK DETECTION (Logic từ Code 1) ---
			ego_speed = traci.vehicle.getSpeed(self.VEH_ID)
			leader = traci.vehicle.getLeader(self.VEH_ID, 10.0)
			
			# Check Đèn Đỏ
			next_tls = traci.vehicle.getNextTLS(self.VEH_ID)
			is_red_light = False
			if next_tls and next_tls[0][2] < 20.0: # Cách đèn < 20m
				state = next_tls[0][3].lower()
				if 'r' in state or 'y' in state: is_red_light = True
			
			# Check Xe Trước Dừng
			is_leader_stopped = (leader is not None and traci.vehicle.getSpeed(leader[0]) < 0.5)

			# Logic: Nếu dừng mà KHÔNG PHẢI do đèn đỏ hay xe trước -> Phạt
			if ego_speed < 0.5 and not (is_red_light or is_leader_stopped):
				reward -= 0.5 # Phạt dặm thêm mỗi step đứng yên vô lý
				self.stuck_time += 1
			else:
				self.stuck_time = 0 # Reset nếu di chuyển hoặc dừng hợp lý

			# --- DATA COLLECTION ---
			sum_speed += ego_speed
			valid_steps += 1
			try:
				e = traci.vehicle.getElectricityConsumption(self.VEH_ID)
				accumulated_energy += e if not np.isnan(e) else 0.0
			except: pass

			# --- REWARD CALCULATION ---
			reward += self._calculate_reward(action) / SIM_STEPS

			# --- SUCCESS CHECK ---
			if self._success_check():
				terminated = True
				reward += 200.0
				self._success = True
				termination_reason = "goal"
				break
		
		# 3. Post-Processing & Info
		
		# Tính toán Wiggle cho Info
		if not hasattr(self, "prev_action"): self.prev_action = action
		action_delta = np.abs(action - self.prev_action)
		wiggle_stat = np.mean(action_delta)
		self.prev_action = action

		# Xử lý Timeout và Stuck quá lâu
		if not terminated:
			if self.stuck_time > 100: # Kẹt quá 100 bước (khoảng 10s-20s)
				terminated = True
				reward -= 400.0
				termination_reason = "stuck_too_long"
			elif self.step_count >= self.MAX_EPISODE_STEPS:
				truncated = True
				termination_reason = "timeout"

		obs = self._get_obs()
		avg_real_speed = sum_speed / max(1, valid_steps)

		# Tính Safety Penalty cho Info (để log lại xem Agent lái ẩu thế nào)
		safety_val = 0.0
		if self._veh_exists():
			leader = traci.vehicle.getLeader(self.VEH_ID, self.MAX_DIST)
			if leader:
				dist = leader[1]
				# Dùng logic đơn giản cho info logging
				if dist < 20: safety_val = 1.0 - (dist/20.0) 

		info = {
			"real_speed": avg_real_speed,
			"real_energy": accumulated_energy, 
			"wiggle": wiggle_stat,
			"safety": safety_val, 
			"step_reward": reward,
			"is_success": 1 if self._success else 0,
			"reason": termination_reason 
		}

		return obs, reward, terminated, truncated, info

	# Helper function (Bạn cần paste cái này vào class SumoEnv nếu chưa có)
	def _success_check(self):
		if not self._veh_exists(): return False
		try:
			current_edge = traci.vehicle.getRoadID(self.VEH_ID)
			# Xử lý internal edge (dấu :) để không bị lỗi
			if current_edge.startswith(":"): return False 

			if hasattr(self, "current_route_edges") and self.current_route_edges:
				if current_edge == self.current_route_edges[-1]:
					# Nếu đã ở cạnh cuối cùng, kiểm tra vị trí xem đã gần hết đường chưa
					lane_len = traci.lane.getLength(traci.vehicle.getLaneID(self.VEH_ID))
					pos = traci.vehicle.getLanePosition(self.VEH_ID)
					if pos > (lane_len - 20.0): # Cách điểm cuối 20m
						return True
		except: pass
		return False


	def close(self):
		try:
			traci.close()
		except:
			pass



if __name__ == "__main__":
	# 1. Init
	# To Train (Random):
	env = SumoEnv(map_config="TestMap/osm.sumocfg", render=True, test_mode=False, test_route="TestMap/test_route.rou.xml", delay=100)
	
	# To Test (Fixed XML):
	# env = SumoEnv(map_config="TestMap/osm.sumocfg", render=True, test_mode=True, test_xml="my_route.rou.xml")
	
	# 2. Reset
	obs, info = env.reset()
	print(f"Init Obs Shape: {obs.shape}")
	
	done = False
	total_reward = 0
	
	# 3. Loop (Test for 50 steps)
	print("Starting Loop...")
	for i in range(50):
		# Random action
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
