# train_longitudinal.py
# Stage 2: Longitudinal Control — Energy Efficiency & Safety
# Algorithm: SAC (Soft Actor-Critic)
# The agent controls throttle/brake; steering is provided by frozen Stage 1 model.

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
from wrappers import LongitudinalControlWrapper
from callbacks import CurriculumLogCallback


def find_stage1_model() -> str:
    """Locate the best available Stage 1 model."""
    candidates = [
        os.path.join(cfg.STAGE1_MODEL_DIR, "sac_lateral_final.zip"),
        os.path.join(cfg.STAGE1_MODEL_DIR, "best", "best_model.zip"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    # Search for interrupted/crash backups
    if os.path.exists(cfg.STAGE1_MODEL_DIR):
        zips = [
            os.path.join(cfg.STAGE1_MODEL_DIR, f)
            for f in os.listdir(cfg.STAGE1_MODEL_DIR) if f.endswith(".zip")
        ]
        if zips:
            return max(zips, key=os.path.getmtime)

    raise FileNotFoundError(
        f"No Stage 1 model found in {cfg.STAGE1_MODEL_DIR}. "
        "Please run train_lateral.py first."
    )


def main():
    print("=" * 60)
    print("  STAGE 2: LONGITUDINAL CONTROL (Energy & Safety)")
    print("  Algorithm: SAC | Timesteps:", f"{cfg.STAGE2_TIMESTEPS:,}")
    print("=" * 60)

    # --- Reproducibility ---
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    # --- Find Stage 1 model ---
    lateral_model_path = find_stage1_model()
    print(f"  Using lateral model: {lateral_model_path}")

    # --- Environment ---
    train_env = LongitudinalControlWrapper(
        SumoEnv(render=False, traffic_scale=cfg.TRAFFIC_SCALE),
        lateral_model_path=lateral_model_path,
    )
    eval_env = LongitudinalControlWrapper(
        SumoEnv(render=False, traffic_scale=cfg.TRAFFIC_SCALE),
        lateral_model_path=lateral_model_path,
    )

    # --- Callbacks ---
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(cfg.STAGE2_MODEL_DIR, "best/"),
        log_path=cfg.STAGE2_LOG_DIR,
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=os.path.join(cfg.STAGE2_MODEL_DIR, "checkpoints/"),
        name_prefix="sac_longitudinal",
    )

    log_cb = CurriculumLogCallback(
        log_dir=cfg.STAGE2_LOG_DIR,
        stage_name="STAGE2_LONGITUDINAL",
        target_timesteps=cfg.STAGE2_TIMESTEPS,
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
    final_path = os.path.join(cfg.STAGE2_MODEL_DIR, "sac_longitudinal_final")

    try:
        model.learn(
            total_timesteps=cfg.STAGE2_TIMESTEPS,
            callback=all_callbacks,
        )
        model.save(final_path)
        print(f"\n[STAGE 2] Training complete! Model saved: {final_path}")

    except KeyboardInterrupt:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(cfg.STAGE2_MODEL_DIR, f"sac_longitudinal_interrupted_{ts}")
        model.save(path)
        print(f"\n[STAGE 2] Interrupted! Saved: {path}")

    except Exception as e:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(cfg.STAGE2_MODEL_DIR, f"sac_longitudinal_CRASH_{ts}")
        model.save(path)
        print(f"\n[STAGE 2] CRASH! Emergency save: {path}")
        raise e

    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
