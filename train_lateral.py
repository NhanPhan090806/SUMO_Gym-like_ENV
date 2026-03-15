# train_lateral.py
# Stage 1: Lateral Control — Lane Keeping & Lane Changing
# Algorithm: SAC (Soft Actor-Critic)
# The agent only controls steering; throttle is fixed.

import os
import sys
import random
from datetime import datetime

import torch
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

import config as cfg
from simulation.sumo_env import SumoEnv
from wrappers import LateralControlWrapper
from callbacks import CurriculumLogCallback


def main():
    print("=" * 60)
    print("  STAGE 1: LATERAL CONTROL (Lane Keeping / Changing)")
    print("  Algorithm: SAC | Timesteps:", f"{cfg.STAGE1_TIMESTEPS:,}")
    print("=" * 60)

    # --- Reproducibility ---
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    # --- Environment ---
    train_env = LateralControlWrapper(
        SumoEnv(render=False, traffic_scale=cfg.TRAFFIC_SCALE)
    )
    eval_env = LateralControlWrapper(
        SumoEnv(render=False, traffic_scale=cfg.TRAFFIC_SCALE)
    )

    # --- Callbacks ---
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(cfg.STAGE1_MODEL_DIR, "best/"),
        log_path=cfg.STAGE1_LOG_DIR,
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=1_000,
        save_path=os.path.join(cfg.STAGE1_MODEL_DIR, "checkpoints/"),
        name_prefix="sac_lateral",
    )

    log_cb = CurriculumLogCallback(
        log_dir=cfg.STAGE1_LOG_DIR,
        stage_name="STAGE1_LATERAL",
        target_timesteps=cfg.STAGE1_TIMESTEPS,
        verbose=1,
    )

    all_callbacks = CallbackList([eval_cb, ckpt_cb, log_cb])

    # --- Model ---
    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=0,
        device="auto",
        **cfg.SAC_PARAMS,
    )

    # --- Train ---
    final_path = os.path.join(cfg.STAGE1_MODEL_DIR, "sac_lateral_final")

    try:
        model.learn(
            total_timesteps=cfg.STAGE1_TIMESTEPS,
            callback=all_callbacks,
        )
        model.save(final_path)
        print(f"\n[STAGE 1] Training complete! Model saved: {final_path}")

    except KeyboardInterrupt:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        interrupted_path = os.path.join(cfg.STAGE1_MODEL_DIR, f"sac_lateral_interrupted_{ts}")
        model.save(interrupted_path)
        print(f"\n[STAGE 1] Interrupted! Saved: {interrupted_path}")

    except Exception as e:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        crash_path = os.path.join(cfg.STAGE1_MODEL_DIR, f"sac_lateral_CRASH_{ts}")
        model.save(crash_path)
        print(f"\n[STAGE 1] CRASH! Emergency save: {crash_path}")
        raise e

    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
