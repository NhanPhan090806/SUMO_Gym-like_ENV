import os
import sys
import traci

# --- BOILERPLATE SETUP ---
if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("Please declare environment variable 'SUMO_HOME'")

sumoBinary = "sumo-gui" 
# Enable emissions for everyone
sumoCmd = [sumoBinary, "-c", "osm.sumocfg", "--start", "--device.emissions.probability", "1.0"]

# --- VARIABLES ---
VEH_ID = "my_ego_car"
tracked_vehicle_active = False

print("Starting SUMO...")
traci.start(sumoCmd)

ran_route = []
step = 0
while step < 3600 or True:
	try:
		traci.simulationStep()

		# ==========================================================
		# 1. ROBUST SPAWN LOGIC (Main Loop Level)
		# ==========================================================
		if step == 10:
			print(f"Attempting to spawn {VEH_ID}...")
			
			route_list = traci.route.getIDList()
			spawned_successfully = False

			# Loop through ALL routes until we find one that accepts a car
			for route_id in route_list:
				if route_id in ran_route:
					continue
				try:
					# Try to add the vehicle to this specific route
					traci.vehicle.add(VEH_ID, route_id, typeID="DEFAULT_VEHTYPE")
					
					# --- IF WE GET HERE, IT WORKED! ---
					spawned_successfully = True
					tracked_vehicle_active = True
					
					# A. Make it Green and Big
					traci.vehicle.setColor(VEH_ID, (0, 255, 0))
					
					# B. Lock Camera ("View #0" is standard, but we catch errors just in case)
					try:
						traci.gui.trackVehicle("View #0", VEH_ID)
						traci.gui.setZoom("View #0", 1000) # Zoom level 1000 is usually good
					except traci.TraCIException:
						print("Could not control GUI (maybe running in non-GUI mode?)")
					
					# C. FREEZE THE CAR (Speed = 0)
					traci.vehicle.setSpeed(VEH_ID, 0)
					
					print(f"SUCCESS! {VEH_ID} spawned on '{route_id}'. Camera Locked. Vehicle Frozen.")
					ran_route.append(route_id)
					break # Stop looking for routes
				except traci.TraCIException:
					# This route was invalid (e.g. bike path), try the next one
					continue
			
			if not spawned_successfully:
				print("CRITICAL ERROR: Could not find ANY valid route for a passenger car.")

		# ==========================================================
		# 2. UNFREEZE LOGIC
		# ==========================================================
		# At step 100 (roughly 10 seconds later), let the car drive
		if step == 100 and tracked_vehicle_active:
			print(f"Releasing {VEH_ID} to drive normally...")
			traci.vehicle.setSpeed(VEH_ID, -1) # -1 returns control to SUMO driver

		# ==========================================================
		# 3. TRACKING & STATS
		# ==========================================================
		# Check if car is currently in the simulation
		current_vehicles = traci.vehicle.getIDList()
		
		if VEH_ID in current_vehicles:
			tracked_vehicle_active = True
			
			# Calculate stats
			co2 = traci.vehicle.getCO2Emission(VEH_ID)
			fuel = traci.vehicle.getFuelConsumption(VEH_ID)
			speed = traci.vehicle.getSpeed(VEH_ID)

			if step % 10 == 0:
				status = "FROZEN" if speed == 0 else "DRIVING"
				print(f"Step {step} [{status}] | Speed: {speed:.2f} m/s | CO2: {co2:.2f} | Fuel: {fuel:.2f}")

		else:
			# If we were tracking it, but it's gone now
			if tracked_vehicle_active:
				print(f"Vehicle {VEH_ID} has left the simulation.")
				#traci.close()
				#traci.start(sumoCmd)
				tracked_vehicle_active = False
				
				# Check for crash
				collisions = traci.simulation.getCollisions()
				for c in collisions:
					if c.collider == VEH_ID or c.victim == VEH_ID:
						print(f"!!! ALERT: {VEH_ID} CRASHED, RESTART THE SIMULATION !!!")
						traci.close()
						traci.start(sumoCmd)

		step += 1


	except KeyboardInterrupt:
		traci.close
		break

traci.close()