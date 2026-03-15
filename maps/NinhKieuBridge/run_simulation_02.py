import os
import sys
import traci
import time
import random

# --- BOILERPLATE SETUP ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

sumoBinary = "sumo-gui"  # open the gui SUMO, without -gui, there will be no gui

# --- CONFIGURATION ---
VEH_ID = "my_ego_car"
VTYPE_ID = "custom_passenger_car" 
TRAFFIC_SCALE = 5.0  # <--- NEW: Set this to 2.0 or 3.0 to increase traffic

sumoCmd = [
    sumoBinary, 
    "-c", "osm.sumocfg", 
    "--start", 
    "--device.emissions.probability", "1.0", 
    "--quit-on-end",
    "--delay", "100",
    "--scale", str(TRAFFIC_SCALE) # <--- APPLIES THE TRAFFIC SCALE
]

def get_passenger_edges():
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

print(f"Starting Simulation Script with Traffic Scale: {TRAFFIC_SCALE}...")

try:
    while True:
        print("\n--- INITIALIZING NEW EPISODE ---")
        traci.start(sumoCmd)

        # Get all vehicle types currently loaded in the simulation
        v_types = traci.vehicletype.getIDList()
        sigma = 0.5
        for v_type in v_types:
            # Set sigma (0.0 = perfect robot, 0.5 = normal, 1.0 = highly imperfect/erratic)
            traci.vehicletype.setImperfection(v_type, sigma) 
            
            # Optional: Make them impatient (push their way through)
            traci.vehicletype.setImpatience(v_type, 0.5) 

        print(f"Traffic behavior updated: Sigma set to {sigma}")
        
        # 1. SETUP VEHICLE TYPE
        try:
            existing_types = traci.vehicletype.getIDList()
            source_type = "DEFAULT_VEHTYPE"
            # check if the default source type is legit, 
            # else take the very first element of existed type 
            if source_type not in existing_types and len(existing_types) > 0:
                source_type = existing_types[0]

            traci.vehicletype.copy(source_type, VTYPE_ID)
            traci.vehicletype.setVehicleClass(VTYPE_ID, "passenger")
            traci.vehicletype.setColor(VTYPE_ID, (0, 255, 0)) # R, G, B
            traci.vehicletype.setLength(VTYPE_ID, 5.0)
            print(f"Created vType '{VTYPE_ID}'")
        except Exception as e:
            print(f"Error defining vType: {e}")

        drivable_edges = get_passenger_edges()
        print(f"Map scanned. Found {len(drivable_edges)} roads for cars.")

        step = 0

        # command to spawn vehicle used? 
        # (please note that spawning command used 
        # does not mean vehicle spawned successfully.
        # If there is a traffic jam at the spawn point, SUMO puts your car in a "waiting queue.")
        vehicle_spawned_command = False

        # tracking vehicle mode: on?, this similarly means that the vehicle spawned successfully?
        # because if the vehicle is on the road, you can track it, unless, there's nothing to track
        tracked_vehicle_active = False


        # even when the vehicle is spawned successfully, 
        # we freeze it for a little bit of time for the computer to successfully set up the environment
        vehicle_unfreezed = False

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            # =========================
            # 1. SPAWN COMMAND (Step 10)
            # =========================
            if step == 5:
                print(f"Finding route for {VEH_ID}...")
                random.shuffle(drivable_edges)
                
                spawn_found = False
                for _ in range(50): 
                    start = random.choice(drivable_edges)
                    end = random.choice(drivable_edges)
                    if start == end: continue
                    
                    try:
                        route = traci.simulation.findRoute(start, end, vType=VTYPE_ID)
                        if route.edges and len(route.edges) > 5:
                            rt_id = f"route_{step}"
                            traci.route.add(rt_id, route.edges)
                            traci.vehicle.add(VEH_ID, rt_id, typeID=VTYPE_ID, departPos="0")
                            traci.vehicle.setSpeed(VEH_ID, 0)
                            
                            print(f"Command sent: Spawn on {start}.")
                            vehicle_spawned_command = True
                            spawn_found = True
                            break
                    except:
                        continue
                
                if not spawn_found:
                    print("No valid route found. Restarting.")
                    break 

            # =========================
            # 2. WAIT FOR INSERTION
            # =========================
            if vehicle_spawned_command and not tracked_vehicle_active:
                current_vehs = traci.vehicle.getIDList()
                if VEH_ID in current_vehs:
                    print(f"Vehicle {VEH_ID} has successfully entered the road network!")
                    tracked_vehicle_active = True
                    traci.gui.trackVehicle("View #0", VEH_ID)
                    traci.gui.setZoom("View #0", 600)
                else:
                    if step > 200: # Wait longer if traffic is heavy (scaled up)
                        print("Timeout: Vehicle could not enter (too much traffic?). Restarting.")
                        break

            # =========================
            # 3. UNFREEZE
            # =========================
            # this is for letting your env variable to be set up, change this due to your computer's specs

            if tracked_vehicle_active and step == 69:
                # 0 = Disable all checks (Gap checks, brake checks, etc.)
                # 31 = Default (All checks on)
                traci.vehicle.setSpeedMode(VEH_ID, 0)
                # Optional: Disable lane change safety checks if you control steering too
                traci.vehicle.setLaneChangeMode(VEH_ID, 0)
                vehicle_unfreezed = True
                print(">>> GO! Vehicle released to drive. <<<")
                # the second parameter of traci.vehicle.setSpeed is the speed (m/s) of the car if no command is passed
                # and if it is -1, the SUMO default driver will take control
                traci.vehicle.setSpeed(VEH_ID, 20)

            # =========================
            # 4. MONITOR & DATA
            # =========================
            if tracked_vehicle_active:
                
                # A. CHECK EXISTENCE FIRST (Fixes the crash)
                if VEH_ID not in traci.vehicle.getIDList():
                    print("Vehicle finished route (arrived).")
                    time.sleep(2)
                    break 

                # B. CHECK CRASH
                collisions = traci.simulation.getCollisions()
                for c in collisions:
                    if c.collider == VEH_ID or c.victim == VEH_ID:
                        print(f"!!! CRASH DETECTED on {traci.vehicle.getRoadID(VEH_ID)} !!!")
                        time.sleep(3)
                        # We break here, which leads to outer loop restart
                        break 
                else: 
                    # This 'else' belongs to the 'for' loop. 
                    # It runs only if NO break occurred (no crash).
                    
                    # C. PRINT STATS (Safe now because we know vehicle exists)
                    if step % 10 == 0 and vehicle_unfreezed:
                        try:
                            co2 = traci.vehicle.getCO2Emission(VEH_ID)
                            fuel = traci.vehicle.getFuelConsumption(VEH_ID)
                            speed = traci.vehicle.getSpeed(VEH_ID)
                            status = "FROZEN" if speed == 0 else "DRIVING"
                            print(f"Step {step} [{status}] | Speed: {speed:.2f} m/s | CO2: {co2:.2f} | Fuel: {fuel:.2f}")
                        except traci.TraCIException:
                            # In case vehicle disappears mid-millisecond (rare)
                            pass

            step += 1
        
        try:
            traci.close()
        except:
            pass
        time.sleep(1)

except KeyboardInterrupt:
    try:
        traci.close()
    except:
        pass