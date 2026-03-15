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
			shape=(24,),
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
		# safety check: if vehicle is dead -> return zeros
		if self.VEH_ID not in traci.vehicle.getIDList():
			return np.zeros(24, dtype=np.float32)
		
		

		# EGO PHYSICS (4) --------
		velocity = traci.vehicle.getSpeed(self.VEH_ID) # *
		acceleration = traci.vehicle.getAcceleration(self.VEH_ID) # *
		# becareful here, getAcceleration returns the current accelerate (dynamic)
		# getAccel returns the max acceleration of a vehicle type (static)

		try:
			# lane index
			lane_idx = traci.vehicle.getLaneIndex(self.VEH_ID)
			road_id = traci.vehicle.getRoadID(self.VEH_ID)
			total_lanes = traci.edge.getLaneNumber(road_id)
			norm_lane = lane_idx / max(1, total_lanes - 1) # type: ignore # *

			slope = traci.vehicle.getSlope(self.VEH_ID) / self.MAX_SLOPE # type: ignore # *
			lat_offset = traci.vehicle.getLateralLanePosition(self.VEH_ID) # *

			# car_angle = traci.vehicle.getAngle(self.VEH_ID)
			# edge_angle = traci.vehicle.getAngle(self.VEH_ID) # getting exact lane angle is kinda complex in traci
			# heading_error = ((car_angle - edge_angle) % 360) / 360.0  # type: ignore # *

			# Heading error is conceptually important, but in SUMO lane-change control
			# vehicle yaw is not independently controllable, so this signal is degenerate.
			heading_error = 0.0

			steer_angle = traci.vehicle.getAngle(self.VEH_ID) / 360 # type: ignore # *

		except: 
			norm_lane = 0.0
			slope = 0

		# HEV energy: Just take whatever SUMO gives us. 
		# Fuel: Max ~12000/s. Elec: Max ~120 Wh/s		
		# If you want the agent to be smart enough to know "I am currently charging," 
		# you can split the Energy input into two fields. Fuel Burn (0.0 to 1.0), Battery Flow (-1.0 to 1.0) (pos.: draining, neg.: charging)
		elec = traci.vehicle.getElectricityConsumption(self.VEH_ID) / self.MAX_ELEC # type: ignore
	


		# LEADER (2) --------
		try:
			leader = traci.vehicle.getLeader(self.VEH_ID, dist=self.MAX_DIST) # output = leader_id, leader_dist
			if leader:
				l_dist = leader[1] / self.MAX_DIST # *
				l_speed = traci.vehicle.getSpeed(leader[0])
				l_rel_speed = (traci.vehicle.getSpeed(self.VEH_ID) - l_speed) / self.MAX_SPEED  # type: ignore # *
			else:
				l_dist = 1.0 # far away
				l_rel_speed = 0.0
		except:
			l_dist, l_rel_speed = 0.0, 0.0

		# surroundings (8)
		surroundings = self._get_surroundings()

		# infrastructure (6)
		try:
			# the Lane ID (id) is a unique string identifier for a specific lane (e.g., "edge1_0"), 
			# while the Lane Index (index) is its numerical position (starting from 0 for the rightmost lane) within an edge
			lane_id = traci.vehicle.getLaneID(self.VEH_ID)

			if lane_id == "":
				speed_limit = 1.0
				can_left, can_right = 0.0, 0.0
				tls_dist, tls_state, turn_dist = 1.0, 1.0, 1.0
			else:
				# Normal logic
				speed_limit = traci.lane.getMaxSpeed(laneID=lane_id) / self.MAX_SPEED
			
			# lane feasibility
			road_id = traci.vehicle.getRoadID(self.VEH_ID)
			num_lanes = traci.edge.getLaneNumber(road_id)

			can_left = 1.0 if lane_idx < (num_lanes - 1) else 0.0 # type: ignore # *
			can_right = 1.0 if lane_idx > 0 else 0.0 # type: ignore # *

			# traffic light state (tls)
			tls_data = traci.vehicle.getNextTLS(self.VEH_ID)
			if tls_data:
				# tls_data[0] is (tlsID, tlsIndex, dist, state)
				tls_dist = tls_data[0][2] / self.MAX_DIST # *
				
				# Simple logic: 'g'/'G' is Green (1.0), else Red (0.0)
				tls_state = 1.0 if tls_data[0][3].lower() == "g" else 0.0 # *
			else:
				tls_dist = 1.0
				tls_state = 1.0 # assume green

			# distance to turn
			# distance to end of the lane as a proxy
			lane_len = traci.lane.getLength(lane_id)
			lane_pos = traci.vehicle.getLanePosition(self.VEH_ID)
			turn_dist = min(lane_len - lane_pos, self.MAX_DIST) / self.MAX_DIST # *
		
		except:
			speed_limit, can_left, can_right, tls_dist, tls_state, turn_dist = 1.0, 0.0, 0.0, 1.0, 1.0, 1.0

		# combination
		obs = [velocity, acceleration, elec, norm_lane, slope, lat_offset, heading_error, steer_angle] + \
		surroundings + [l_dist, l_rel_speed] + \
		[speed_limit, can_left, can_right, tls_dist, tls_state, turn_dist] 

		obs = np.array(obs, dtype=np.float32)
		obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

		return obs
	
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
	def _calculate_reward(self, action):
		# --- CONFIGURATION ---
		# Weights (Adjusted for balance)
		W_SPEED = 1.0       # Primary goal: Move
		W_TIME = -0.1       # Existence penalty: Force completion
		W_ENERGY = -0.4     # Secondary: Efficiency
		W_COMFORT = -0.3    # Tertiary: Smoothness
		W_SAFETY = -1.0     # Critical: Don't crash
		
		# Physics Constants
		TIME_HEADWAY = 2.0  # Safe seconds behind leader
		MIN_GAP = 5.0       # Minimum meters gap even when stopped

		# 1. SPEED & PROGRESS REWARD
		# We use simple speed ratio. Progress is implicitly handled by speed.
		# If speed is high, we are making progress.
		cur_speed = traci.vehicle.getSpeed(self.VEH_ID)
		speed_reward = np.clip(cur_speed / self.MAX_SPEED, 0.0, 1.0)

		# 2. ENERGY PENALTY
		# Normalize: 1.0 = Max consumption
		elec = traci.vehicle.getElectricityConsumption(self.VEH_ID)
		# Using a soft clip because instant consumption can spike
		energy_penalty = np.clip(elec / self.MAX_ELEC, 0.0, 1.0)

		# 3. COMFORT (WIGGLE) PENALTY
		if not hasattr(self, "prev_action"):
			self.prev_action = action
		
		# Calculate absolute change in steering and throttle
		# We penalize steering wiggle more than throttle wiggle
		delta = np.abs(action - self.prev_action)
		wiggle_penalty = (delta[0] * 0.7) + (delta[1] * 0.3) 
		self.prev_action = action

		# 4. SAFETY PENALTY (Dynamic Time Headway)
		# 2-second rule: Safe Distance = Speed * 2s + Minimum Gap
		# This scales linearly with speed, removing the "Jump" bug.
		required_safe_dist = (cur_speed * TIME_HEADWAY) + MIN_GAP
		
		leader = traci.vehicle.getLeader(self.VEH_ID, self.MAX_DIST)
		safety_penalty = 0.0
		
		if leader is not None:
			leader_dist = leader[1]
			if leader_dist < required_safe_dist:
				# Exponential penalty as we get closer
				# If dist = safe_dist, penalty = 0
				# If dist = 0, penalty = 1
				safety_penalty = 1.0 - (leader_dist / required_safe_dist)
				safety_penalty = np.clip(safety_penalty, 0.0, 1.0)

		# 5. SANITIZATION
		speed_reward = np.nan_to_num(speed_reward)
		energy_penalty = np.nan_to_num(energy_penalty)
		wiggle_penalty = np.nan_to_num(wiggle_penalty)
		safety_penalty = np.nan_to_num(safety_penalty)

		# 6. TOTAL CALCULATION
		# Note: Added W_TIME (constant negative) to prevent the car from just parking to save energy.
		reward = (
			(speed_reward * W_SPEED) + 
			(W_TIME) + 
			(energy_penalty * W_ENERGY) + 
			(wiggle_penalty * W_COMFORT) + 
			(safety_penalty * W_SAFETY)
		)

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
		for _ in range(300): traci.simulationStep()

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

	
	def step(self, action):
		
		self.step_count += 1

		# 1. Physics Setup
		steer_cmd = action[0]
		accel_cmd = action[1]
		
		if accel_cmd >= 0:
			desired_accel = accel_cmd * self.MAX_ACCEL
		else:
			desired_accel = accel_cmd * self.MAX_DECEL

		SIM_STEPS = 1
		delta_time = SIM_STEPS * traci.simulation.getDeltaT()
		
		if not self._veh_exists():
			info = {
			"real_speed": 0.0,
			"real_energy": 0.0, 
			"wiggle": 0.0,
			"safety": 0.0, 
			"step_reward": 0.0,
			"is_success": 0,
		}

			obs = np.zeros(self.observation_space.shape, dtype=np.float32)
			return obs, 0.0, True, False, info
		
		traci.vehicle.setAcceleration(self.VEH_ID, desired_accel, delta_time)
		# current_speed = traci.vehicle.getSpeed(self.VEH_ID)
		# target_speed = current_speed + (desired_accel * delta_time)
		# target_speed = max(0.0, min(target_speed, self.MAX_SPEED))
		
		# traci.vehicle.slowDown(vehID=self.VEH_ID, speed=target_speed, duration=delta_time)

		# Lateral control (Same as before)
		LC_THRESHOLD = 0.3
		target_lane_offset = 0 
		if steer_cmd < -LC_THRESHOLD: target_lane_offset = -1 
		elif steer_cmd > LC_THRESHOLD: target_lane_offset = 1 

		if target_lane_offset != 0:
			try:
				current_lane = traci.vehicle.getLaneIndex(self.VEH_ID)
				edge_id = traci.vehicle.getRoadID(self.VEH_ID)
				num_lanes = traci.edge.getLaneNumber(edge_id)
				desired_lane = current_lane + target_lane_offset 
				if 0 <= desired_lane < num_lanes: 
					traci.vehicle.changeLane(self.VEH_ID, desired_lane, 2.0)
			except:
				pass

		# 2. Simulation Loop
		reward = 0.0
		terminated = False
		truncated = False
		
		accumulated_energy = 0.0 
		sum_speed = 0.0
		valid_steps = 0 

		for _ in range(SIM_STEPS):
			traci.simulationStep()

			# --- FIX START: CHECK EXISTENCE FIRST ---
			# We must check if the car is alive BEFORE asking for its RoadID
			# Trong hàm step(), sau khi chạy simulationStep()
			if self.VEH_ID in traci.vehicle.getIDList():
				ego_speed = traci.vehicle.getSpeed(self.VEH_ID)
				
				# Check Stuck Type
				leader = traci.vehicle.getLeader(self.VEH_ID, 7.0) # Khoảng cách check kẹt là 7m
				is_leader_stopped = (leader is not None and traci.vehicle.getSpeed(leader[0]) < 0.1)
				
				# Kiểm tra đèn đỏ (Nếu có trong bản đồ)
				next_tls = traci.vehicle.getNextTLS(self.VEH_ID)
				is_red_light = (next_tls and next_tls[0][2] < 10.0 and next_tls[0][3].lower() in ['r', 'u', 'y'])

				if ego_speed < 0.1:
					if is_leader_stopped or is_red_light:
						# Dừng đúng quy định: Không phạt hoặc phạt rất nhẹ để tránh "lười"
						stuck_penalty = -0.01 
					else:
						# Tự nhiên dừng: Phạt nặng
						stuck_penalty = -1.0 
						self.stuck_time += 1
				else:
					stuck_penalty = 0.0
					self.stuck_time = 0

				reward += stuck_penalty
			if self.VEH_ID not in traci.vehicle.getIDList():
				terminated = True

				# Teleport detection
				teleported_ids = traci.simulation.getStartingTeleportIDList()
				was_teleported = self.VEH_ID in teleported_ids

				if was_teleported:
					reward -= 50
				else:
					reward -= 150 # Collision
				
				# Exit the loop immediately so we don't query a dead car
				break 
			# ----------------------------------------

			# --- NOW IT IS SAFE TO QUERY THE CAR ---
			# Goal Check
			try:
				current_edge = traci.vehicle.getRoadID(self.VEH_ID)
				dist_to_goal = self._get_dist_to_destination()
				
				if not self._success:
					# Check if we are on the last edge of the route
					# (Use try/except in case route is empty or index out of bounds)
					if hasattr(self, "current_route_edges") and self.current_route_edges:
						is_at_last_edge = (current_edge == self.current_route_edges[-1])
						if is_at_last_edge and dist_to_goal < 5.0: # Increased to 5m for easier detection
							self._success = True
			except Exception:
				# If something weird happens (like edge ID mismatch), ignore this step
				pass
			
			# Check Collisions (Explicit)
			collision_list = traci.simulation.getCollisions()
			col_ids = [c.collider for c in collision_list] + [c.victim for c in collision_list]
			if self.VEH_ID in col_ids:
				terminated = True
				self._success = False
				reward -= 200 
				break 
			
			# Data Collection
			try:
				v = traci.vehicle.getSpeed(self.VEH_ID)
				sum_speed += v
				valid_steps += 1
				
				e_cons = traci.vehicle.getElectricityConsumption(self.VEH_ID)
				if e_cons is None or np.isnan(e_cons): e_cons = 0.0
				accumulated_energy += e_cons
			except:
				pass
			
			# Step Reward
			reward += self._calculate_reward(action) / SIM_STEPS
			if (sum_speed/max(1, valid_steps)) < 1.0:
				reward -= 0.2

		# --- AFTER LOOP ---

		if self._success:
			terminated = True
			reward += 200
			info_success = 1

		if self.step_count >= self.MAX_EPISODE_STEPS:
			truncated = True
		
		obs = self._get_obs()
			
		# CALCULATE AVERAGE SPEED
		# If valid_steps is 0 (instant crash), avg_speed is 0.
		avg_real_speed = sum_speed / max(1, valid_steps)

		# Stuck Detection (Use Average Speed now)
		if avg_real_speed < 0.5:
			self.stuck_time += 1
		else:
			self.stuck_time = 0 
		
		if self._success:
			info_success = 1
		elif self.stuck_time > 50 and not terminated:
			terminated = True
			reward -= 120
			info_success = 0
		else:
			info_success = 0

		

		if not hasattr(self, "prev_action"):
			self.prev_action = action
		
		# calculate action change in steering/gas
		action_delta = np.abs(action - self.prev_action)
		wiggle_stat = np.mean(action_delta) # speed or lane changing count as wiggling
		self.prev_action = action

		# get leader
		if self._veh_exists():
			leader = traci.vehicle.getLeader(self.VEH_ID, self.MAX_DIST)
		else:
			leader = None

		if leader is not None:
			leader_dist = leader[1]

			# safety penalty can be used for calculating reward and report error in ego veh and leader
			if leader_dist < self.TARGET_DIST: 
				# strong penalty when too close
				safety_penalty = (1 - (leader_dist / self.TARGET_DIST))
				safety_penalty = np.clip(safety_penalty, 0.0, 1.0)
			else:
				# mild penalty when too far
				safety_penalty = 0.1 * (leader_dist - self.TARGET_DIST) / self.TARGET_DIST
				safety_penalty = np.clip(safety_penalty, 0.0, 1.0)
		else:
			safety_penalty = 0.0

		# ... inside step(), at the very end ...

		# Logic to determine why the episode ended
		reason = "running" # Default (should only appear if episode is NOT done)
		
		if self._success:
			reason = "goal"
		elif truncated:
			reason = "timeout"
		elif terminated:
			# If terminated but not a goal, it's a failure.
			# Check the reward to guess the type of failure.
			if reward < -120:
				reason = "collision" # Hard crash
			elif self.stuck_time > 50:
				reason = "stuck"
			else:
				reason = "off_road" # Teleport / Lane Violation (The -50 penalty case)
		
		info = {
			"real_speed": avg_real_speed,
			"real_energy": accumulated_energy, 
			"wiggle": wiggle_stat,
			"safety": safety_penalty, 
			"step_reward": reward,
			"is_success": info_success,
			"success_reason": reason # Use the new logic variable
		}

		return obs, reward, terminated, truncated, info


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
