import os
import sys
import traci
import time
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces


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
	def __init__(self, render: bool = True, map_config = "maps/TestMap/osm.sumocfg", 
			  VEH_ID = "my_ego_car", VTYPE_ID = "custom_passenger_car", TRAFFIC_SCALE = 5.0,
			  imperfection = 0.5, impatience = 0.5) -> None:
		super().__init__()


		self.VEH_ID = VEH_ID
		self.VTYPE_ID = VTYPE_ID
		self.TRAFFIC_SCALE = TRAFFIC_SCALE
		self.render_mode: bool = render
		# Get the absolute path to the 'simulation' folder
		curr_dir = os.path.dirname(os.path.abspath(__file__))
		# Go up one level to 'Sumo' folder
		root_dir = os.path.dirname(curr_dir)
		
		# Handle string input and fix path
		self.maps = [map_config] if isinstance(map_config, str) else map_config
		self.imperfection = imperfection
		self.impatience = impatience
		# 5 actions: Brake, Coast, Gas, Left, Right (For Left and Right, we just change the lane, no steering wheel)
		self.action_space = spaces.Discrete(5)

		# for 1-agent system
		# Ego Physics (4): [Speed, Acceleration, Normalized_Lane_Index, 
		# Fuel/Electricity Consumption (both in one as this case is using HEV)]
		# Front Radar (2): [Leader_Distance, Leader_Rel_Speed]
		# Side Awareness (8):
			# Front-Left: [Dist, Rel_Speed]
			# Back-Left: [Dist, Rel_Speed]
			# Front-Right: [Dist, Rel_Speed]
			# Back-Right: [Dist, Rel_Speed]
		# Infrastructure (6): [Speed_Limit, Can_Go_Left(0/1), Can_Go_Right(0/1), 
		# TLS_Dist, TLS_State(Red/Green), Dist_To_Turn]
		self.observation_space = spaces.Box(
			low=0.0,
			high=1.0,
			shape=(20,),
			dtype=np.float32
			)
		
		self.step_count = 0
	
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
		MAX_DIST = 100.0 # Max distance of the "sensor"
		MAX_SPEED = 30.0 # max relative speed

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
					data["L_F_Dist"] = min(dist, MAX_DIST) / MAX_DIST # normalization and double check
					data["L_F_RelSpeed"] = (my_speed - n_speed) / MAX_SPEED # type: ignore
			else: # Back (dist <= 0)
				if abs(dist) < closest_back:
					closest_back = abs(dist)
					n_speed = traci.vehicle.getSpeed(n_id)
					data["L_B_Dist"] = min(dist, MAX_DIST) / MAX_DIST # normalization and double check
					data["L_B_RelSpeed"] = (my_speed - n_speed) / MAX_SPEED # type: ignore

		
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
					data["R_F_Dist"] = min(dist, MAX_DIST) / MAX_DIST
					data["R_F_RelSpeed"] = (my_speed - n_speed) / MAX_SPEED # type: ignore
			else: # back
				if abs(dist) < closest_back:
					closest_back = abs(dist)
					n_speed = traci.vehicle.getSpeed(n_id)
					data["R_B_Dist"] = min(dist, MAX_DIST) / MAX_DIST
					data["R_B_RelSpeed"] = (my_speed - n_speed) / MAX_SPEED # type: ignore
		
		output = [stats for key, stats in data.items()]
		return output
	


	def _get_obs(self):
		# safety check: if vehicle is dead -> return zeros
		if self.VEH_ID not in traci.vehicle.getIDList():
			return np.zeros(20, dtype=np.float32)
		
		# Constants
		MAX_SPEED = 30.0 # m/s
		MAX_ACCEL = 5.0 # m/(s^2)
		MAX_DIST = 100.0
		MAX_FUEL = 12000 # mg/s (due to test data)
		MAX_ELEC = 120 # Wh/s
		

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
		except:
			norm_lane = 0.0

		# HEV energy: Just take whatever SUMO gives us. 
		# Fuel: Max ~12000/s. Elec: Max ~120 Wh/s		
		# If you want the agent to be smart enough to know "I am currently charging," 
		# you can split the Energy input into two fields. Fuel Burn (0.0 to 1.0), Battery Flow (-1.0 to 1.0) (pos.: draining, neg.: charging)
		f = traci.vehicle.getFuelConsumption(self.VEH_ID) / MAX_FUEL # type: ignore
		e = traci.vehicle.getElectricityConsumption(self.VEH_ID) / MAX_ELEC # type: ignore
		energy = f + max(e, 0) # *


		# LEADER (2) --------
		try:
			leader = traci.vehicle.getLeader(self.VEH_ID, dist=MAX_DIST) # leader_id, leader_dist
			if leader:
				l_dist = leader[1] / MAX_DIST # *
				l_speed = traci.vehicle.getSpeed(leader[0])
				l_rel_speed = (traci.vehicle.getSpeed(self.VEH_ID) - l_speed) / MAX_SPEED  # type: ignore # *
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

			speed_limit = traci.lane.getMaxSpeed(laneID=lane_id) / MAX_SPEED # type: ignore # *

			# lane feasibility
			road_id = traci.vehicle.getRoadID(self.VEH_ID)
			num_lanes = traci.edge.getLaneNumber(road_id)

			can_left = 1.0 if lane_idx < (num_lanes - 1) else 0.0 # type: ignore # *
			can_right = 1.0 if lane_idx > 0 else 0.0 # type: ignore # *

			# traffic light state (tls)
			tls_data = traci.vehicle.getNextTLS(self.VEH_ID)
			if tls_data:
				# tls_data[0] is (tlsID, tlsIndex, dist, state)
				tls_dist = tls_data[0][2] / MAX_DIST # *
				
				# Simple logic: 'g'/'G' is Green (1.0), else Red (0.0)
				tls_state = 1.0 if tls_data[0][3].lower() == "g" else 0.0 # *
			else:
				tls_dist = 1.0
				tls_state = 1.0 # assume green

			# distance to turn
			# distance to end of the lane as a proxy
			lane_len = traci.lane.getLength(lane_id)
			lane_pos = traci.vehicle.getLanePosition(self.VEH_ID)
			turn_dist = min(lane_len - lane_pos, MAX_DIST) / MAX_DIST # *
		
		except:
			speed_limit, can_left, can_right, tls_dist, tls_state, turn_dist = 1.0, 0.0, 0.0, 1.0, 1.0, 1.0

		# combination
		obs = [velocity, acceleration, energy, norm_lane] + surroundings + [l_dist, l_rel_speed] + \
		[speed_limit, can_left, can_right, tls_dist, tls_state, turn_dist] 

		return np.array(obs, dtype=np.float32)
	

	# reward function
	def _calculate_rewards(self) -> float:
		# goal: fast, safety, energy efficiency

		# get data
		v = traci.vehicle.getSpeed(self.VEH_ID)
		f = traci.vehicle.getFuelConsumption(self.VEH_ID)
		e = traci.vehicle.getElectricityConsumption(self.VEH_ID)

		# Normalize (approx.)
		rew_speed = v / 30.0


		# energy cost
		# when it comes to cost, we treat 1 Wh/s ~ 300mg/s
		cost_fuel = f / 12000.0 # type: ignore
		cost_elec = max(0.0, e) / 120.0 # type: ignore
		energy_cost = -(cost_fuel + cost_elec)

		combined_reward: float = rew_speed + energy_cost


		return combined_reward



	def reset(self, seed=None, options=None):

		super().reset(seed=seed,options=options)
		
		# try to close the environment
		try:
			traci.close()
		except:
			pass

		active_map = random.choice(self.maps)

		# set up the launch
		SumoBinary: str = "sumo-gui" if self.render_mode else "sumo"
		SumoCMD: list[str] = [SumoBinary, "-c", active_map, "--start", "--quit-on-end",
				"--device.emissions.probability", "1.0",
				"--scale", str(self.TRAFFIC_SCALE),
				"--no-step-log", "true",
				"--no-warnings", "true"]
		

		# launch sumo
		traci.start(SumoCMD)


		# 1. SETUP VEHICLE TYPE
		try:
			existing_types = traci.vehicletype.getIDList()
			source_type = "DEFAULT_VEHTYPE"
			if source_type not in existing_types and len(existing_types) > 0:
				source_type = existing_types[0]

			# Copy base type
			traci.vehicletype.copy(source_type, self.VTYPE_ID)
			traci.vehicletype.setVehicleClass(self.VTYPE_ID, "passenger")
			traci.vehicletype.setColor(self.VTYPE_ID, (0, 255, 0)) 
			traci.vehicletype.setLength(self.VTYPE_ID, 5.0)
			traci.vehicletype.setEmissionClass(self.VTYPE_ID, "MMPEVEM")

			# --- HEV PARAMETERS START ---
			# Enable Emission (Fuel) and Battery (Elec) devices
			traci.vehicletype.setParameter(self.VTYPE_ID, "has.emissions.device", "true")
			traci.vehicletype.setParameter(self.VTYPE_ID, "has.battery.device", "true")
			traci.vehicletype.setParameter(self.VTYPE_ID, "has.elecHybrid.device", "true")
			
			# Set Battery Capacity (e.g., 13600 Wh buffer)
			# According to Toyota 2026 Prius Plug-in Hybrid 
			# (https://www.toyota.ca/en/vehicles/prius-plug-in-hybrid/models-specifications/#:~:text=64-,Battery%20Capacity,-13.6)
			traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.capacity", "13600.00")
			
			# Set Switching Threshold (e.g., Engine turns on if speed > 10m/s)
			# This ensures low-speed crawling uses electricity
			traci.vehicletype.setParameter(self.VTYPE_ID, "device.elecHybrid.minBatteryCharge", "0.10")
			# --- HEV PARAMETERS END ---

			# Apply Imperfection to Background Traffic
			for v_type in existing_types:
				traci.vehicletype.setImperfection(v_type, self.imperfection)
				traci.vehicletype.setImpatience(v_type, self.impatience)

		except Exception as e:
			print(f"Error defining vType: {e}")

		# we load a random map every episode so we load drivable roads every episodes
		self.drivable_edges = self._get_passenger_edges()


		# spawn the agent
		spawned = False
		ego_veh_tracked = False # Tracking vehicle: on?
		while not spawned: # try to spawn the agent until success
			try:
				# advance 1 step to make sure the env is ready
				traci.simulationStep()

				# pick a random route
				edge_start = random.choice(self.drivable_edges)
				edge_end = random.choice(self.drivable_edges)

				# successfully pick a route that starting edge is different from ending edge
				if edge_start != edge_end:
					# ask SUMO for a rout
					route = traci.simulation.findRoute(edge_start, edge_end, self.VTYPE_ID)

					# ensure that the route is long enough
					if route.edges and len(route.edges) > 5: # type: ignore
						route_id = f"route_{random.randint(0, 1000000)}"
						# Define route ID
						traci.route.add(route_id, route.edges) # type: ignore

						# add the car
						traci.vehicle.add(self.VEH_ID, route_id, departPos="0", typeID=self.VTYPE_ID)

						# disable safety guards and make the world imperfect
						traci.vehicle.setSpeedMode(self.VEH_ID, 0)
						traci.vehicle.setLaneChangeMode(self.VEH_ID, 0)

						# move simulation until we can spawn our ego car
						traci.simulationStep()
						if self.VEH_ID in traci.vehicle.getIDList():
							print(f"Vehicle {self.VEH_ID} has successfully entered the road network!")
							if not ego_veh_tracked and self.render_mode:
								ego_veh_tracked = True
								traci.gui.trackVehicle("View #0", self.VEH_ID)
								traci.gui.setZoom("View #0", 600)
							spawned = True
			except:
				continue # retry if sumo raises error

		obs = self._get_obs()
		info = {}
		return obs, info		

	
	def step(self, action):

		# - apply action -

		# get current status
		current_speed = traci.vehicle.getSpeed(self.VEH_ID)
		lane_change = 0
		target_speed = current_speed

		# Constants for action logic
		ACCEL_STEP = 1.0 # how much speed increase per action (m/s^2)
		BRAKE_STEP = 1.0 # how much speed decrease per action (m/s^2)
		MAX_SPEED = 30.0 # hard cap

		# actions: 
		# Brake, Coast, Gas, Left, Right (For Left and Right, we just change the lane, no steering wheel)
		if action == 0: # Brake
			target_speed = max(0.0, current_speed - BRAKE_STEP) # type: ignore
		elif action == 1: # Coast
			target_speed = current_speed 
		elif action == 2: # Gas
			target_speed = min(MAX_SPEED, current_speed + ACCEL_STEP) # type: ignore
		# lane change: as the lane index starting from 0 (on the right most) to the leftmost
		# => lane decrease => go right, else go left
		elif action == 3: # left
			lane_change = 1
		elif action == 4: # Right
			lane_change = -1
		
		# Send Speed
		traci.vehicle.setSpeed(vehID=self.VEH_ID, speed=target_speed)

		# If lane change
		if lane_change != 0:
			current_lane = traci.vehicle.getLaneIndex(self.VEH_ID)
			edge_id = traci.vehicle.getRoadID(self.VEH_ID)
			num_lanes = traci.edge.getLaneNumber(edgeID=edge_id)

			target_lane = current_lane + lane_change # type: ignore
			# Only pass the command if the lane change is possible
			if 0 <= target_lane and target_lane < num_lanes: # target_lane belongs to [0, num_lanes]
				# Duration makes it smooth, not instantaneous teleportation
				traci.vehicle.changeLane(self.VEH_ID, target_lane, 1.5)

		# Run simulation
		reward = 0.0
		terminated = False  # natural game-over/done
		truncated = False # step count exceeds limit

		# for every 10 steps, we make action once (already above)
		SIM_STEPS = 10
		for _ in range(SIM_STEPS):
			traci.simulationStep()

			# check crash
			# "collisions" includes hitting other cars or walls
			if self.VEH_ID in traci.simulation.getCollisions():
				terminated = True
				reward -= 1000.0 # big penalty
				break

			# check if vehicle finished/disappeared (Arrived or Teleported due to error)
			if self.VEH_ID not in traci.vehicle.getIDList():
				terminated = True
				# if didn't crash, assume finished
				reward += 500.0
				break
			
			# calculate step reward
			reward += self._calculate_rewards()

		# update states
		obs = self._get_obs()
		info = {}

		return obs, reward, terminated, truncated, info


	def close(self):
		try:
			traci.close()
		except:
			pass



if __name__ == "__main__":
	# 1. Init
	env = SumoEnv(map_config="TestMap/osm.sumocfg", render=True) # Set render=True to watch
	
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
		
		obs, reward, terminated, truncated, info = env.step(action)
		total_reward += reward
		
		print(f"Step {i} | Action: {action} | Reward: {reward:.2f} | Speed: {obs[0]:.2f} | Energy: {obs[2]:.2f}")
		
		if terminated or truncated:
			print("Episode Finished!")
			obs, info = env.reset()
			
	env.close()
	print("Test Complete.")