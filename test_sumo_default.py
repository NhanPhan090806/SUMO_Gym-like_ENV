import os
import sys
import time
import numpy as np
import traci
from datetime import datetime
import csv

# Thêm đường dẫn nếu cần
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation.env_random_copy import SumoEnv

def run_sumo_default_test(episodes=20, render=True, delay=20):
    # Khởi tạo Env
    env = SumoEnv(
        render=render,
        map_config=["maps/map_grid_tuned_no_car/run.sumocfg"], # Dùng map bạn đang training
        test_mode=False, # Để nó tạo route ngẫu nhiên
        delay=delay
    )

    results = []
    print(f"\n{'='*60}")
    print(f"BẮT ĐẦU KIỂM TRA ROUTE VỚI SUMO DEFAULT CONTROL")
    print(f"{'='*60}\n")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        
        # --- KÍCH HOẠT CHẾ ĐỘ TỰ LÁI MẶC ĐỊNH CỦA SUMO ---
        # SpeedMode 31: Bật tất cả các tính năng an toàn, tuân thủ luật, phanh khi cần.
        # LaneChangeMode 1621: Chuyển làn chiến lược (để theo route), hợp tác và an toàn.
        if env._veh_exists():
            traci.vehicle.setSpeedMode(env.VEH_ID, 31)
            traci.vehicle.setLaneChangeMode(env.VEH_ID, 512) # 512 = strategic lane changing only (the vehicle will not change lanes if it is not on its route)

        done = False
        ep_steps = 0
        ep_reward = 0
        
        while not done:
            # Thay vì gọi env.step(action), chúng ta gọi simulationStep trực tiếp
            # để tránh việc env.step áp dụng setAcceleration thủ công.
            traci.simulationStep()
            ep_steps += 1
            
            # Cập nhật cache để các hàm check của env hoạt động đúng
            env._update_cache()
            
            # Kiểm tra xem xe còn tồn tại không
            if not env._veh_exists():
                done = True
                # Lấy lý do từ logic của env
                # Lưu ý: Vì không gọi env.step, ta phải tự mô phỏng lại logic kiểm tra của nó
                if env._success_check():
                    reason = "goal"
                    success = 1
                else:
                    teleport_list = traci.simulation.getStartingTeleportIDList()
                    reason = "teleport/collision" if env.VEH_ID in teleport_list else "removed"
                    success = 0
            elif ep_steps >= env.MAX_EPISODE_STEPS:
                done = True
                reason = "timeout"
                success = 0
            elif env.stuck_time > 150: # Nếu SUMO mặc định cũng đứng yên > 150 bước
                done = True
                reason = "stuck_default_logic"
                success = 0
            
            # Giả lập stuck detection như trong env
            if env.veh_data and env.veh_data["speed"] < 0.1:
                env.stuck_time += 1
            else:
                env.stuck_time = 0

            if done:
                break

        status_icon = "✓" if reason == "goal" else "✗"
        print(f"[{ep:>3}/{episodes}] {status_icon} Route Length: {env.last_known_dist:.1f}m | Steps: {ep_steps:>4} | Reason: {reason}")
        
        results.append({
            "episode": ep,
            "success": 1 if reason == "goal" else 0,
            "steps": ep_steps,
            "reason": reason,
            "dist": env.last_known_dist
        })

    env.close()

    # --- TỔNG KẾT ---
    successes = sum(r['success'] for r in results)
    print(f"\n{'='*60}")
    print(f"TỔNG KẾT:")
    print(f"Tỷ lệ Route khả thi: {successes}/{episodes} ({100*successes/episodes:.1f}%)")
    
    stuck_routes = [r for r in results if r['reason'] in ["stuck_default_logic", "timeout"]]
    if stuck_routes:
        print(f"Cảnh báo: Có {len(stuck_routes)} route khiến ngay cả SUMO mặc định cũng bị kẹt.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_sumo_default_test(episodes=10, render=True, delay=20)