import os
import sys
import time
import traci

# 1. SETUP: Check for SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# 2. CONFIGURATION: Point to your specific config file
# Based on your screenshot, the config file is 'osm.sumocfg'
sumoBinary = "sumo-gui" # Use "sumo" for command line only, "sumo-gui" to watch it
sumoCmd = [sumoBinary, "-c", "osm.sumocfg", "--start", "--scale", "5"]

# 3. START SUMO
print("Starting SUMO...")
traci.start(sumoCmd)

# Variables to track our "controlled" car
target_vehicle_id = "my_ego_vehicle"
step = 0

while step < 3600: # Run for 3600 simulation steps
    
    # 4. INJECT/CONTROL: Simulation Logic happens BEFORE the step
    
    # Example: Check if any vehicles are running, and pick one to control
    # (Since we don't know your Edge IDs, we will hijack the first car we see)
    active_vehicles = traci.vehicle.getIDList()
    
    if len(active_vehicles) > 0:
        # If we haven't picked a car yet, pick the first one
        if target_vehicle_id not in active_vehicles:
            target_vehicle_id = active_vehicles[0]
            # Change its color to RED so you can spot it
            traci.vehicle.setColor(target_vehicle_id, (255, 0, 0)) 
            print(f"Now controlling vehicle: {target_vehicle_id}")

        # --- CONTROL THE SPECIFIC CAR ---
        # Force the car to slow down to 5 m/s
        traci.vehicle.setSpeedMode(target_vehicle_id, 0) # Disable auto-braking (dangerous!)
        traci.vehicle.setSpeed(target_vehicle_id, 5.0)
        
        # Example: Try to change lane to the left (1) or right (-1)
        traci.vehicle.changeLaneRelative(target_vehicle_id, 1, duration=5)

    # --- ADJUST ENVIRONMENT ---
    # Example: Manipulate traffic lights
    # tls_list = traci.trafficlight.getIDList()
    # if tls_list:
    #    traci.trafficlight.setPhase(tls_list[0], 0) # Force a specific phase

    # 5. ADVANCE SIMULATION
    if step < 20:
        traci.simulationStep()
    else:
        traci.simulationStep()
        time.sleep(10)
        
    step += 1

# 6. CLEANUP
traci.close()
print("Simulation finished.")