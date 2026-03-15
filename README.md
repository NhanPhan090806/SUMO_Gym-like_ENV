***

# Autonomous HEV Driving with Deep Reinforcement Learning

This project implements a custom Gymnasium environment for the SUMO (Simulation of Urban MObility) traffic simulator. It trains an autonomous agent to control a Hybrid Electric Vehicle (HEV) with the goals of maximizing speed, minimizing energy consumption (fuel + electricity), and ensuring safety.

The agent is trained using the Proximal Policy Optimization (PPO) algorithm provided by the Stable Baselines 3 library.

## Project Features

*   **Custom Gymnasium Environment:** A fully compliant Gym environment (`SumoEnv`) that handles TraCI communication.
*   **HEV Simulation:** Implements a rule-based Energy Management System (EMS) to calculate Fuel vs. Battery usage based on physics thresholds (mimicking a Toyota Prius logic).
*   **360-Degree Awareness:** The agent receives normalized sensor data regarding surrounding vehicles (Front, Back, Left, Right).
*   **Discrete Action Space:** Simplified control scheme handling longitudinal (gas/brake) and lateral (lane change) movements.
*   **Robust Training Pipeline:** Includes callbacks for model checkpointing, best-model saving, and detailed logging.
*   **Random Map Training:** Handle map input as a list or a string to osm.sumocfg file(s) to load random or one dedicated map

## Prerequisites

To run this project, you must have the following software installed:

1.  **Python 3.8+**
2.  **SUMO Simulator:** You must have SUMO installed and the `SUMO_HOME` environment variable set.
    *   Ubuntu/Debian: `sudo apt-get install sumo sumo-tools sumo-doc`
    *   Windows: Download the installer from the Eclipse SUMO website.

## Installation

1.  Clone this repository.
2.  Install the required Python packages:

```bash
pip install gymnasium numpy stable-baselines3 shimmy traci
```

## Project Structure

```text
Sumo_RL/
├── maps/                   # Contains SUMO network and configuration files
│   └── TestMap/
│       └── osm.sumocfg
├── models/                 # Stores trained agents
│   ├── checkpoints/        # Periodic backups during training
│   └── highest_reward/     # The best performing model found so far
├── reports/                # Tensorboard logs
├── simulation/             # Custom Environment Package
│   ├── __init__.py
│   └── sumo_env.py         # Main Environment Class (Gym Interface)
├── train_RL.py             # Script to train the PPO agent
├── test_RL.py              # Script to visualize/test trained models
└── README.md
```

## Technical Details

### Observation Space
The agent receives a normalized vector of size **20** (`Box(20,)`).

| Index | Feature | Description |
| :--- | :--- | :--- |
| 0 | Speed | Ego vehicle speed (Normalized) |
| 1 | Acceleration | Ego vehicle acceleration |
| 2 | Energy Rate | Combined Fuel + Electricity cost |
| 3 | Lane Index | Normalized position of current lane |
| 4 | L_Front_Dist | Distance to nearest Left-Front vehicle |
| 5 | L_Front_RelSpd | Relative speed of Left-Front vehicle |
| 6 | L_Back_Dist | Distance to nearest Left-Back vehicle |
| 7 | L_Back_RelSpd | Relative speed of Left-Back vehicle |
| 8 | R_Front_Dist | Distance to nearest Right-Front vehicle |
| 9 | R_Front_RelSpd | Relative speed of Right-Front vehicle |
| 10 | R_Back_Dist | Distance to nearest Right-Back vehicle |
| 11 | R_Back_RelSpd | Relative speed of Right-Back vehicle |
| 12 | Leader_Dist | Distance to vehicle directly ahead |
| 13 | Leader_RelSpd | Relative speed of leader |
| 14 | Speed Limit | Max speed of current road |
| 15 | Can Go Left | 1.0 if left lane exists, else 0.0 |
| 16 | Can Go Right | 1.0 if right lane exists, else 0.0 |
| 17 | TLS Distance | Distance to next Traffic Light |
| 18 | TLS State | 1.0 if Green, 0.0 if Red/Yellow |
| 19 | Turn Distance | Distance to end of current edge/junction |

### Action Space
The agent operates in a discrete action space (`Discrete(5)`).

| Action Index | Command | Description |
| :--- | :--- | :--- |
| 0 | Brake | Decreases target speed (Regenerative braking allowed) |
| 1 | Coast | Maintains current target speed |
| 2 | Accelerate | Increases target speed (Consumes Energy) |
| 3 | Change Left | Initiates a lane change to the left |
| 4 | Change Right | Initiates a lane change to the right |

*Note: Each action is repeated for 10 simulation steps (~1.0 second) to ensure smooth control.*

### Reward Function
The reward is calculated at every simulation step:

`Reward = Normalized_Speed - Energy_Cost + Bonuses/Penalties`

*   **Speed:** Positive reward for moving closer to the speed limit.
*   **Energy:** Negative reward proportional to Fuel consumption (mg/s) and Electricity usage (Wh/s).
*   **Safety:** Large penalty (-1000) for collisions.
*   **Completion:** Bonus (+500) for finishing the route successfully.

## Usage

### Training
To train the agent from scratch (without GUI to maximize speed):

```bash
python train_RL.py
```

*   Models are saved automatically to `models/checkpoints` on Keyboard Interrupt (Ctrl+C).
*   The best performing model is saved to `models/highest_reward`.
*   Logs are output to the console every 10 steps.

### Testing / Visualization
To watch the trained agent drive in the SUMO GUI:

```bash
python test_RL.py
```

1.  Run the script.
2.  The script will list all available models in the `models/` directory.
3.  Enter the number corresponding to the model you wish to load.
4.  The SUMO GUI will open, and the agent will control the green vehicle.

## Acknowledgments
*   **SUMO Team:** For the traffic simulation engine.
*   **Stable Baselines 3:** For the PPO implementation.
*   **Gymnasium:** For the environment standard.