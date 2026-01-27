# train_cql_v2.py
import os
import numpy as np
import pandas as pd
from pathlib import Path

import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.replay_buffers import ReplayBuffer
from d3rlpy.callbacks import PeriodicEval
from d3rlpy.metrics import EnvironmentEvaluator
from d3rlpy.preprocessing import StandardObservationScaler, MinMaxActionScaler

# ====== 请替换为你自己的环境类 ======
from your_env_module import SimpleTradingEnv  # ← 替换为你的 env 所在模块名


def load_validation_env(csv_path: str, initial_cash: float = 100_000):
    """从 OHLCV CSV 创建验证环境"""
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return SimpleTradingEnv(
        df=df,
        initial_cash=initial_cash,
        # commission=0.001,      # 与训练一致
        # min_rebalance_gap=1,
        # max_position=1.0,
    )


def main():
    # ==============================
    # 配置路径
    # ==============================
    DATASET_PATH = "trading_dataset_v3.h5"      # 训练数据 .h5
    VALIDATION_CSV = "sz.000513_val.csv" # 验证集 OHLCV CSV
    MODEL_SAVE_DIR = "models"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # ==============================
    # 1. 加载训练 dataset (.h5)
    # ==============================
    print("🔄 Loading training dataset from .h5...")
    # buffer = ReplayBuffer(capacity=10_000_000)
    # dataset = MDPDataset.load(DATASET_PATH, buffer=buffer)
    train_dataset = d3rlpy.dataset.ReplayBuffer.load("./dataset/sz.000513/train_dataset.h5", d3rlpy.dataset.InfiniteBuffer()) 
    print(f"✅ Loaded {len(dataset)} transitions ({dataset.n_episodes} episodes)")

    # ==============================
    # 2. 创建验证环境 (from CSV)
    # ==============================
    print("🧪 Creating validation environment from CSV...")
    validation_env = load_validation_env(VALIDATION_CSV)
    evaluator = EnvironmentEvaluator(env=validation_env, n_trials=1)

    # ==============================
    # 3. 配置 CQL 算法
    # ==============================
    cql = d3rlpy.algos.CQL(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        conservative_weight=2.0,
        alpha_learning_rate=0.0,
        alpha_threshold=10.0,
        initial_alpha=0.2,
        batch_size=256,
        observation_scaler=StandardObservationScaler(),
        action_scaler=MinMaxActionScaler(minimum=0.0, maximum=1.0),
        # use_gpu=False,  # 若有 GPU 可设为 True
    )

    # ==============================
    # 4. 设置最佳模型保存逻辑
    # ==============================
    best_score = -np.inf
    best_epoch = 0

    def save_best_model(algo, dataset, epoch, step, is_last):
        nonlocal best_score, best_epoch
        eval_result = evaluator(algo)
        total_return = eval_result["return"]  # final_value / initial_cash - 1

        print(f"[Epoch {epoch:02d}] Validation Return: {total_return:.4f}")

        if total_return > best_score:
            best_score = total_return
            best_epoch = epoch
            model_path = os.path.join(MODEL_SAVE_DIR, "cql_best_val_return.d3")
            algo.save(model_path)
            print(f"🎉 New best model (epoch={epoch}) saved to: {model_path}")

    callbacks = [
        PeriodicEval(
            evaluator=evaluator,
            interval=1,
            call_func=save_best_model,
        )
    ]

    # ==============================
    # 5. 开始训练（30 epochs）
    # ==============================
    n_epochs = 30
    n_steps_per_epoch = len(dataset)
    total_steps = n_epochs * n_steps_per_epoch

    print(f"🚀 Starting CQL training for {n_epochs} epochs ({total_steps:,} steps)...")
    cql.fit(
        dataset,
        n_steps=total_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        callbacks=callbacks,
        show_progress=True,
    )

    print(f"\n✅ Training finished.")
    print(f"🏆 Best validation return: {best_score:.4f} (epoch {best_epoch})")
    print(f"💾 Final model saved at: models/cql_best_val_return.d3")


if __name__ == "__main__":
    main()