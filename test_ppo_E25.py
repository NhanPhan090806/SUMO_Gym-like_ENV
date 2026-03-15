"""
test_ppo_map1_fixed_route.py — Evaluate a trained PPO model on a FIXED route
                                E1 → E25 on maps/map1/run.sumocfg.

Usage:
    python test_ppo_map1_fixed_route.py --model models/tianshou_ppo/best_policy_XXXXXXXX.pth
    python test_ppo_map1_fixed_route.py --model models/tianshou_ppo/best_policy_XXXXXXXX.pth --no-gui
    python test_ppo_map1_fixed_route.py --model models/tianshou_ppo/best_policy_XXXXXXXX.pth --episodes 50 --delay 50

Results are saved to: reports/test/ppo/<model_name>/map1_fixed_route_<timestamp>.csv
"""

import os
import sys
import csv
import time
import math
import random
import argparse
import torch
import traci
import numpy as np
from datetime import datetime

# ── Tianshou ──────────────────────────────────────────────────────────────────
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.policy import PPOPolicy

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please set the 'SUMO_HOME' environment variable.")


# ═══════════════════════════════════════════════════════════════════════════════
# FIXED CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MAP_CONFIG   = "maps/map1/run.sumocfg"
FIXED_EDGES  = [f"E{i}" for i in range(1, 26)]   # E1 … E25
ROUTE_ID     = "fixed_test_route"
VEH_ID       = "my_ego_car"
VTYPE_ID     = "custom_passenger_car"

# Must mirror your training network exactly
HIDDEN_SIZES  = [256, 256]
OBS_SHAPE     = (30,)
ACT_SHAPE     = (2,)

LR            = 3e-4
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
MAX_GRAD_NORM = 0.3
VF_COEF       = 0.25
ENT_COEF      = 0.05

MAX_SPEED     = 55.6
MAX_ACCEL     = 4.15
MAX_DECEL     = 6.0
MAX_ELEC      = 120.0
MAX_SLOPE     = 20.0
MAX_DIST      = 100.0
MAX_EPISODE_STEPS = 1000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_HEADER = [
    "episode", "steps", "ep_reward",
    "avg_speed", "total_energy",
    "avg_wiggle", "avg_safety",
    "success", "reason"
]


# ═══════════════════════════════════════════════════════════════════════════════
# POLICY
# ═══════════════════════════════════════════════════════════════════════════════

def build_policy():
    import gymnasium as gym
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=ACT_SHAPE, dtype=np.float32)
    net    = Net(OBS_SHAPE, hidden_sizes=HIDDEN_SIZES, device=DEVICE)
    actor  = ActorProb(net, ACT_SHAPE, device=DEVICE, unbounded=True).to(DEVICE)
    critic = Critic(net, device=DEVICE).to(DEVICE)
    optim  = torch.optim.Adam(
        set(list(actor.parameters()) + list(critic.parameters())), lr=LR
    )
    policy = PPOPolicy(
        actor, critic, optim,
        dist_fn=torch.distributions.Normal,
        action_space=action_space,
        discount_factor=GAMMA,
        gae_lambda=GAE_LAMBDA,
        max_grad_norm=MAX_GRAD_NORM,
        vf_coef=VF_COEF,
        ent_coef=ENT_COEF,
        action_scaling=True,
        action_bound_method="clip",
    )
    return policy


def load_policy(model_path: str) -> PPOPolicy:
    policy = build_policy()
    policy.load_state_dict(torch.load(model_path, map_location=DEVICE))
    policy.eval()
    print(f"Loaded weights: {model_path}")
    return policy


def select_action(policy: PPOPolicy, obs: np.ndarray) -> np.ndarray:
    """Deterministic action — use the Gaussian mean (no sampling noise)."""
    with torch.no_grad():
        obs_t  = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, _ = policy.actor(obs_t)   # logits = (mu, sigma)
        mu = logits[0]
        action = mu.squeeze(0).cpu().numpy()
    return np.clip(action, -1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMO HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def start_sumo(render: bool, delay: int):
    try:
        traci.close()
    except Exception:
        pass
    time.sleep(0.3)

    binary  = "sumo-gui" if render else "sumo"
    cmd = [
        binary, "-c", MAP_CONFIG,
        "--start", "--quit-on-end",
        "--device.emissions.probability", "1.0",
        "--delay",            str(delay),
        "--no-step-log",      "true",
        "--time-to-teleport", "-1",
        "--collision.action", "remove",
        "--collision.check-junctions", "true",
        "--no-warnings",      "true",
    ]
    traci.start(cmd)


def setup_vehicle_type():
    """Copy DEFAULT_VEHTYPE → VTYPE_ID with EV parameters."""
    try:
        existing = traci.vehicletype.getIDList()
        src = "DEFAULT_VEHTYPE" if "DEFAULT_VEHTYPE" in existing else existing[0]
        traci.vehicletype.copy(src, VTYPE_ID)
        traci.vehicletype.setVehicleClass(VTYPE_ID, "passenger")
        traci.vehicletype.setColor(VTYPE_ID, (0, 255, 0))
        traci.vehicletype.setParameter(VTYPE_ID, "mass",   "2911")
        traci.vehicletype.setLength(VTYPE_ID, "5.1181")
        traci.vehicletype.setEmissionClass(VTYPE_ID, "MMPEVEM")
        traci.vehicletype.setParameter(VTYPE_ID, "has.battery.device",                "true")
        traci.vehicletype.setParameter(VTYPE_ID, "device.battery.capacity",           "123000.00")
        traci.vehicletype.setParameter(VTYPE_ID, "device.battery.chargeLevel",        "123000.00")
        traci.vehicletype.setParameter(VTYPE_ID, "device.battery.rechargeEfficiency", "0.8")
        traci.vehicletype.setParameter(VTYPE_ID, "device.battery.maxRegenerationAcceleration", "2.0")
    except Exception as e:
        print(f"[WARN] vehicle type setup: {e}")


def spawn_vehicle() -> tuple[bool, float]:
    """
    Add the fixed route E1→E25 and spawn the ego vehicle.
    Returns (spawned: bool, total_route_len: float).
    """
    try:
        traci.route.add(ROUTE_ID, FIXED_EDGES)
    except Exception:
        pass  # Route may already exist from a previous episode reset

    total_len = 0.0
    for e in FIXED_EDGES:
        try:
            total_len += traci.lane.getLength(f"{e}_0")
        except Exception:
            pass

    traci.vehicle.add(VEH_ID, ROUTE_ID, departPos="free", typeID=VTYPE_ID)

    for _ in range(50):
        traci.simulationStep()
        if VEH_ID in traci.vehicle.getIDList():
            traci.vehicle.setSpeedMode(VEH_ID, 0)
            traci.vehicle.setLaneChangeMode(VEH_ID, 0)
            return True, total_len

    try:
        traci.vehicle.remove(VEH_ID)
    except Exception:
        pass
    return False, total_len


def reset_episode(render: bool, delay: int) -> tuple[np.ndarray, float]:
    """
    (Re)start SUMO, warm up traffic, spawn ego vehicle.
    Returns (initial_obs, total_route_len).
    """
    start_sumo(render, delay)

    # Warm-up: let background traffic populate
    warmup = random.randint(100, 200)
    for _ in range(warmup):
        traci.simulationStep()

    setup_vehicle_type()

    spawned, total_len = spawn_vehicle()
    if not spawned:
        print("[WARN] Spawn failed, retrying episode …")
        traci.close()
        return reset_episode(render, delay)

    if render and VEH_ID in traci.vehicle.getIDList():
        traci.gui.trackVehicle("View #0", VEH_ID)
        traci.gui.setZoom("View #0", 2000)

    obs = get_obs(total_len)
    return obs, total_len


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVATION  (mirrors env_random._get_obs / _update_cache)
# ═══════════════════════════════════════════════════════════════════════════════

def get_veh_data() -> dict | None:
    if VEH_ID not in traci.vehicle.getIDList():
        return None
    try:
        return {
            "speed":      traci.vehicle.getSpeed(VEH_ID),
            "accel":      traci.vehicle.getAcceleration(VEH_ID),
            "elec":       traci.vehicle.getElectricityConsumption(VEH_ID),
            "lane_idx":   traci.vehicle.getLaneIndex(VEH_ID),
            "lane_id":    traci.vehicle.getLaneID(VEH_ID),
            "road_id":    traci.vehicle.getRoadID(VEH_ID),
            "slope":      traci.vehicle.getSlope(VEH_ID),
            "lat_offset": traci.vehicle.getLateralLanePosition(VEH_ID),
            "lane_pos":   traci.vehicle.getLanePosition(VEH_ID),
            "leader":     traci.vehicle.getLeader(VEH_ID, dist=MAX_DIST),
            "tls":        traci.vehicle.getNextTLS(VEH_ID),
        }
    except Exception:
        return None


def get_surroundings(my_speed: float) -> list:
    """
    Mirrors env_random._get_surroundings():
    8 xe × 2 giá trị (dist, rel_speed) = 16 chiều
    Layout: [LF1,LF1v, LF2,LF2v, LB1,LB1v, LB2,LB2v,
             RF1,RF1v, RF2,RF2v, RB1,RB1v, RB2,RB2v]
    """
    result = [1.0, 0.0] * 8  # 8 slots mặc định

    def process_side(neighbors, base_f, base_b):
        fronts, backs = [], []
        for n_id, dist in neighbors:
            try:
                n_speed = traci.vehicle.getSpeed(n_id)
            except Exception:
                continue
            if dist > 0:
                fronts.append((dist, n_speed))
            else:
                backs.append((abs(dist), n_speed))
        fronts.sort(key=lambda x: x[0])
        backs.sort(key=lambda x: x[0])
        for slot, (d, spd) in enumerate(fronts[:2]):
            idx = base_f + slot * 2
            result[idx]     = min(d, MAX_DIST) / MAX_DIST
            result[idx + 1] = (my_speed - spd) / MAX_SPEED
        for slot, (d, spd) in enumerate(backs[:2]):
            idx = base_b + slot * 2
            result[idx]     = min(d, MAX_DIST) / MAX_DIST
            result[idx + 1] = (my_speed - spd) / MAX_SPEED

    try:
        # Layout 16 chiều:
        # [0..3]  = LF1,LF1v,LF2,LF2v  |  [4..7]  = LB1,LB1v,LB2,LB2v
        # [8..11] = RF1,RF1v,RF2,RF2v   |  [12..15]= RB1,RB1v,RB2,RB2v
        process_side(traci.vehicle.getNeighbors(VEH_ID, 2), 0, 4)   # trái
        process_side(traci.vehicle.getNeighbors(VEH_ID, 1), 8, 12)  # phải
    except Exception:
        pass
    return result


def _edge_heading(edge_id: str) -> float | None:
    """Tính góc hướng của một edge (degrees), dùng cho get_turn_info."""
    try:
        shape = traci.edge.getShape(edge_id)
        if len(shape) < 2:
            return None
        x1, y1 = shape[0]
        x2, y2 = shape[-1]
        return math.degrees(math.atan2(y2 - y1, x2 - x1))
    except Exception:
        return None


def get_turn_info(d: dict) -> tuple:
    """
    Tính (turn_dir, turn_dist_n, lane_offset) cho route cố định FIXED_EDGES.
    Mirrors env_random._get_next_turn_info().
    - turn_dir      : [-1=trái … +1=phải, 0=thẳng]
    - turn_dist_n   : tỉ lệ còn lại trên edge hiện tại  [1=xa, 0=sắp rẽ]
    - lane_offset   : số làn cần dịch  [-1=trái … +1=phải, 0=đúng rồi]
    """
    default = (0.0, 1.0, 0.0)
    try:
        current_edge = d["road_id"]
        if current_edge.startswith(":"):
            return default
        if current_edge not in FIXED_EDGES:
            return default

        indices  = [i for i, x in enumerate(FIXED_EDGES) if x == current_edge]
        curr_idx = indices[-1]
        if curr_idx >= len(FIXED_EDGES) - 1:
            return default

        next_edge = FIXED_EDGES[curr_idx + 1]

        # Hướng rẽ
        a1 = _edge_heading(current_edge)
        a2 = _edge_heading(next_edge)
        if a1 is None or a2 is None:
            turn_dir = 0.0
        else:
            diff = (a2 - a1 + 360) % 360
            if diff > 180:
                diff -= 360
            turn_dir = float(np.clip(-diff / 180.0, -1.0, 1.0))

        # Khoảng cách còn lại trên edge hiện tại
        lane_id = d["lane_id"]
        if not lane_id:
            return (turn_dir, 1.0, 0.0)
        try:
            lane_len = traci.lane.getLength(lane_id)
        except Exception:
            return (turn_dir, 1.0, 0.0)
        dist_left     = max(0.0, lane_len - d["lane_pos"])
        turn_dist_n   = float(np.clip(dist_left / max(lane_len, 1.0), 0.0, 1.0))

        # Làn nào trên current_edge kết nối sang next_edge
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
            if correct_lane is not None:
                break

        if correct_lane is None:
            return (turn_dir, turn_dist_n, 0.0)

        raw_offset  = correct_lane - d["lane_idx"]
        lane_offset = float(np.clip(raw_offset / max(1, num_lanes - 1), -1.0, 1.0))
        return (turn_dir, turn_dist_n, lane_offset)
    except Exception:
        return default


def get_obs(route_len: float) -> np.ndarray:
    d = get_veh_data()
    if d is None:
        return np.zeros(30, dtype=np.float32)
    try:
        velocity     = np.clip(d["speed"]  / MAX_SPEED,  0.0,  2.0)
        acceleration = np.clip(d["accel"]  / MAX_ACCEL, -1.0,  1.0)
        elec         = np.clip(d["elec"]   / MAX_ELEC,   0.0,  5.0)

        lane_idx     = d["lane_idx"]
        road_id      = d["road_id"]
        total_lanes  = traci.edge.getLaneNumber(road_id)
        norm_lane    = lane_idx / max(1, total_lanes - 1)

        slope        = np.clip(d["slope"] / MAX_SLOPE, -1.0, 1.0)
        lat_offset   = np.clip(d["lat_offset"], -10.0, 10.0)

        surroundings = get_surroundings(d["speed"])

        leader = d["leader"]
        if leader:
            l_dist      = min(leader[1], MAX_DIST) / MAX_DIST
            l_rel_speed = (d["speed"] - traci.vehicle.getSpeed(leader[0])) / MAX_SPEED
        else:
            l_dist, l_rel_speed = 1.0, 0.0

        lane_id     = d["lane_id"]
        speed_limit = traci.lane.getMaxSpeed(lane_id) / MAX_SPEED if lane_id else 1.0

        tls_data = d["tls"]
        if tls_data:
            tls_dist  = min(tls_data[0][2], MAX_DIST) / MAX_DIST
            tls_state = 1.0 if tls_data[0][3].lower() == "g" else 0.0
        else:
            tls_dist, tls_state = 1.0, 1.0

        # Route-aware lane guidance — dùng get_turn_info() thực
        turn_dir, turn_dist_n, lane_offset = get_turn_info(d)

        obs_list = [
            velocity, acceleration, elec, norm_lane, slope, lat_offset,
            l_dist, l_rel_speed,
            speed_limit, turn_dir, turn_dist_n, tls_dist, tls_state, lane_offset,
        ] + surroundings

        obs = np.array(obs_list, dtype=np.float64)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = np.clip(obs, -5.0, 5.0)
        return obs.astype(np.float32)
    except Exception:
        return np.zeros(30, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def success_check(d: dict | None) -> bool:
    if not d:
        return False
    try:
        current_edge = d["road_id"]
        if current_edge.startswith(":"):
            return False
        if current_edge == FIXED_EDGES[-1]:
            lane_len = traci.lane.getLength(d["lane_id"])
            if d["lane_pos"] > (lane_len - 20.0):
                return True
    except Exception:
        pass
    return False


def step_env(
    action: np.ndarray,
    step_count: int,
    prev_action: np.ndarray,
    route_len: float,
    stuck_time: int,
) -> tuple:
    """
    Apply action, advance simulation, compute reward/done/info.
    Returns (obs, reward, terminated, truncated, info, new_stuck_time, new_prev_action).
    """
    steer_cmd, accel_cmd = action[0], action[1]
    desired_accel = accel_cmd * MAX_ACCEL if accel_cmd >= 0 else accel_cmd * MAX_DECEL

    if VEH_ID not in traci.vehicle.getIDList():
        obs = np.zeros(30, dtype=np.float32)
        return obs, 0.0, True, False, {"real_speed": 0.0, "reason": "already_dead", "is_success": 0}, stuck_time, action

    # ── SUMO rescue: bật mode 514 khi gần cuối edge VÀ đang sai làn ─────────
    # Mirrors env_random.step() sumo_rescue_active logic
    _d_pre = get_veh_data()
    if _d_pre is not None:
        _, _turn_dist_n, _lane_offset = get_turn_info(_d_pre)
        sumo_rescue_active = _turn_dist_n <= 0.3
    else:
        sumo_rescue_active = False

    if sumo_rescue_active:
        traci.vehicle.setLaneChangeMode(VEH_ID, 514)
    else:
        traci.vehicle.setLaneChangeMode(VEH_ID, 0)

    # Apply acceleration
    traci.vehicle.setAcceleration(VEH_ID, desired_accel, duration=0.5)

    # Apply lane change — chỉ khi SUMO không đang cứu xe (tránh xung đột)
    if not sumo_rescue_active:
        LC_THRESHOLD = 0.3
        current_lane = traci.vehicle.getLaneIndex(VEH_ID)
        if steer_cmd < -LC_THRESHOLD:
            traci.vehicle.changeLane(VEH_ID, max(0, current_lane - 1), 1.0)
        elif steer_cmd > LC_THRESHOLD:
            try:
                n_lanes = traci.edge.getLaneNumber(traci.vehicle.getRoadID(VEH_ID))
                traci.vehicle.changeLane(VEH_ID, min(n_lanes - 1, current_lane + 1), 1.0)
            except Exception:
                pass

    traci.simulationStep()
    d = get_veh_data()

    reward            = 0.0
    terminated        = False
    truncated         = False
    termination_reason = "running"
    accumulated_energy = 0.0
    sum_speed          = 0.0

    if d is None:
        terminated = True
        teleport_list = traci.simulation.getStartingTeleportIDList()
        if VEH_ID in teleport_list:
            reward -= 50.0
            termination_reason = "teleport"
        else:
            reward -= 200.0
            termination_reason = "collision"
    else:
        ego_speed = d["speed"]
        sum_speed = ego_speed

        # Stuck detection (skip if red light or leader stopped)
        is_red = False
        tls_data = d["tls"]
        if tls_data and tls_data[0][2] < 20.0:
            state = tls_data[0][3].lower()
            if "r" in state or "y" in state:
                is_red = True

        leader = d["leader"]
        leader_stopped = leader is not None and traci.vehicle.getSpeed(leader[0]) < 0.5

        if ego_speed < 0.5 and not (is_red or leader_stopped):
            reward    -= 0.5
            stuck_time += 1
        else:
            stuck_time = 0

        e = d["elec"]
        accumulated_energy = e if not np.isnan(e) else 0.0

        # ── Reward (mirrors _calculate_reward) ──────────────────────────────
        W_SPEED, W_PROGRESS, W_ENERGY = 1.2, 0.8, -0.05
        W_COMFORT, W_SAFETY, W_TIME   = -0.05, -0.8, -0.2

        current_edge = d["road_id"]
        dist = 0.0
        if not current_edge.startswith(":") and current_edge in FIXED_EDGES:
            indices = [i for i, x in enumerate(FIXED_EDGES) if x == current_edge]
            idx = indices[-1]
            for e_id in FIXED_EDGES[idx:]:
                try:
                    dist += traci.lane.getLength(f"{e_id}_0")
                except Exception:
                    pass
            dist -= d["lane_pos"]
        else:
            dist = route_len   # fallback

        progress_reward = np.clip(route_len - dist, -1.0, 1.0)
        route_len       = dist   # update for next step

        speed_reward = ego_speed / MAX_SPEED
        if ego_speed < 3.0:
            speed_reward -= 0.8

        energy_penalty = np.clip(accumulated_energy / MAX_ELEC, 0.0, 1.0)

        action_delta   = np.abs(action - prev_action)
        wiggle_penalty = float(np.mean(action_delta))

        safety_penalty = 0.0
        if leader is not None:
            l_dist = leader[1]
            target = 15.0 if ego_speed <= 10.0 else (30.0 if ego_speed <= 20.0 else 50.0)
            if l_dist < target:
                safety_penalty = np.exp(-(l_dist / target))

        for val in (speed_reward, progress_reward, energy_penalty, wiggle_penalty, safety_penalty):
            val = np.nan_to_num(val)

        reward += (
            speed_reward    * W_SPEED +
            progress_reward * W_PROGRESS +
            wiggle_penalty  * W_COMFORT +
            safety_penalty  * W_SAFETY +
            energy_penalty  * W_ENERGY +
            W_TIME
        )

        if success_check(d):
            terminated = True
            reward    += 200.0
            termination_reason = "goal"

    if not terminated:
        if stuck_time > 100:
            terminated = True
            reward    -= 400.0
            termination_reason = "stuck_too_long"
        elif step_count >= MAX_EPISODE_STEPS:
            truncated = True
            termination_reason = "timeout"

    # Safety stat
    safety_val = 0.0
    if d and d.get("leader"):
        dist_to_leader = d["leader"][1]
        if dist_to_leader < 20:
            safety_val = 1.0 - (dist_to_leader / 20.0)

    # Wiggle stat
    action_delta  = np.abs(action - prev_action)
    wiggle_stat   = float(np.mean(action_delta))
    avg_real_speed = sum_speed   # single-step here; accumulated by caller

    info = {
        "real_speed":  avg_real_speed,
        "real_energy": accumulated_energy,
        "wiggle":      wiggle_stat,
        "safety":      safety_val,
        "is_success":  1 if termination_reason == "goal" else 0,
        "reason":      termination_reason,
    }

    obs = get_obs(route_len)
    return obs, reward, terminated, truncated, info, stuck_time, action


# ═══════════════════════════════════════════════════════════════════════════════
# CSV
# ═══════════════════════════════════════════════════════════════════════════════

def init_csv(filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)
    print(f"Results → {filepath}")


def append_csv(filepath: str, row: list):
    with open(filepath, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_test(args):
    model_path = args.model
    n_episodes = args.episodes
    render     = not args.no_gui
    delay      = args.delay

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    report_dir = os.path.join("reports", "test", "ppo", model_name)
    timestamp  = datetime.now().strftime("%d%m%Y_%H%M%S")
    csv_path   = os.path.join(report_dir, f"map1_fixed_route_{timestamp}.csv")

    init_csv(csv_path)
    policy = load_policy(model_path)

    all_rewards   = []
    all_speeds    = []
    all_energies  = []
    success_count = 0

    print(f"\n{'='*62}")
    print(f"  Model    : {model_name}")
    print(f"  Map      : {MAP_CONFIG}")
    print(f"  Route    : E1 → E25  ({len(FIXED_EDGES)} edges)")
    print(f"  Episodes : {n_episodes}")
    print(f"  GUI      : {'ON' if render else 'OFF'}")
    print(f"  Device   : {DEVICE}")
    print(f"{'='*62}\n")

    for ep in range(1, n_episodes + 1):
        obs, route_len = reset_episode(render, delay)

        ep_reward     = 0.0
        ep_energy     = 0.0
        ep_speed_sum  = 0.0
        ep_wiggle_sum = 0.0
        ep_safety_sum = 0.0
        ep_steps      = 0
        stuck_time    = 0
        prev_action   = np.zeros(2, dtype=np.float32)
        done          = False

        while not done:
            action = select_action(policy, obs)
            obs, reward, terminated, truncated, info, stuck_time, prev_action = step_env(
                action, ep_steps + 1, prev_action, route_len, stuck_time
            )
            ep_reward     += reward
            ep_energy     += info["real_energy"]
            ep_speed_sum  += info["real_speed"]
            ep_wiggle_sum += info["wiggle"]
            ep_safety_sum += info["safety"]
            ep_steps      += 1
            done = terminated or truncated

        try:
            traci.close()
        except Exception:
            pass

        avg_speed  = ep_speed_sum  / max(1, ep_steps)
        avg_wiggle = ep_wiggle_sum / max(1, ep_steps)
        avg_safety = ep_safety_sum / max(1, ep_steps)
        success    = info["is_success"]
        reason     = info["reason"]

        if success:
            success_count += 1

        all_rewards.append(ep_reward)
        all_speeds.append(avg_speed)
        all_energies.append(ep_energy)

        append_csv(csv_path, [
            ep, ep_steps,
            f"{ep_reward:.2f}",
            f"{avg_speed:.2f}",
            f"{ep_energy:.2f}",
            f"{avg_wiggle:.4f}",
            f"{avg_safety:.4f}",
            success,
            reason,
        ])

        icon = "✓" if success else "✗"
        print(
            f"[{ep:>3}/{n_episodes}] {icon}  "
            f"steps={ep_steps:>4}  reward={ep_reward:>8.2f}  "
            f"speed={avg_speed:>5.2f} m/s  "
            f"energy={ep_energy:>8.2f}  reason={reason}"
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  SUMMARY — map1 fixed route E1→E25  ({n_episodes} episodes)")
    print(f"{'='*62}")
    print(f"  Success rate : {success_count}/{n_episodes}  ({100*success_count/n_episodes:.1f} %)")
    print(f"  Avg reward   : {np.mean(all_rewards):.2f}  ± {np.std(all_rewards):.2f}")
    print(f"  Avg speed    : {np.mean(all_speeds):.2f} m/s")
    print(f"  Avg energy   : {np.mean(all_energies):.2f}")
    print(f"  Results CSV  : {csv_path}")
    print(f"{'='*62}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a trained PPO policy on the fixed route E1→E25 in maps/map1."
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the .pth policy file  (e.g. models/tianshou_ppo/best_policy_....pth)"
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of test episodes (default: 50)"
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Disable SUMO GUI (headless mode)"
    )
    parser.add_argument(
        "--delay", type=int, default=0,
        help="SUMO GUI animation delay in ms — use e.g. 50 to slow down (default: 0)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_test(parse_args())