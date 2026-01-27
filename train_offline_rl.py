import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import d3rlpy
from d3rlpy.dataset import MDPDataset
# from d3rlpy.metrics import evaluate_on_environment
from sklearn.model_selection import train_test_split
from d3rlpy.metrics import TDErrorEvaluator, AverageValueEstimationEvaluator, EnvironmentEvaluator
# 替换为你自己的模块
from simpletradingenv import SimpleTradingEnv

from concat_npz import load_all_data_for_stock

# ==============================
# 1. 加载所有 .npz 文件 → MDPDataset
# ==============================
def load_episodes_from_dir(dir_path: str, max_episodes: int = None):
    """从目录加载所有 .npz，返回 observations, actions, rewards, terminals"""
    obs_list, act_list, rew_list, term_list = [], [], [], []
    
    npz_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".npz")])
    if max_episodes:
        npz_files = npz_files[:max_episodes]
    
    for file in tqdm(npz_files, desc=f"Loading {os.path.basename(dir_path)}"):
        data = np.load(os.path.join(dir_path, file))
        obs_list.append(data["observations"])
        act_list.append(data["actions"])
        rew_list.append(data["rewards"])
        term_list.append(data["terminals"])
    
    # 拼接
    observations = np.concatenate(obs_list, axis=0).astype(np.float32)
    actions = np.concatenate(act_list, axis=0).astype(np.float32)
    rewards = np.concatenate(rew_list, axis=0).astype(np.float32)
    terminals = np.concatenate(term_list, axis=0).astype(bool)
    
    return observations, actions, rewards, terminals


def build_dataset(train_dir: str, val_dir: str):
    """构建训练集 + 验证集 MDPDataset"""
    print("🔍 Loading training episodes...")
    obs_train, act_train, rew_train, term_train = load_episodes_from_dir(train_dir)
    
    print("🔍 Loading validation episode...")
    obs_val, act_val, rew_val, term_val = load_episodes_from_dir(val_dir)
    
    # 创建 MDPDataset（d3rlpy 要求 action 是 2D: [N, action_dim]）
    dataset_train = MDPDataset(
        observations=obs_train,
        actions=act_train,
        rewards=rew_train,
        terminals=term_train,
        discrete_action=False  # 连续动作（仓位比例）
    )
    
    dataset_val = MDPDataset(
        observations=obs_val,
        actions=act_val,
        rewards=rew_val,
        terminals=term_val,
        discrete_action=False
    )
    
    print(f"✅ Train dataset: {len(dataset_train)} transitions")
    print(f"✅ Val dataset: {len(dataset_val)} transitions")
    
    return dataset_train, dataset_val


# ==============================
# 2. 自定义评估函数（计算金融指标）
# ==============================
def financial_evaluator(env, policy, episode_len: int = 250):
    """
    在给定 env 上运行策略，返回金融指标
    """
    obs = env.reset()
    total_values = []
    actions = []
    
    for _ in range(episode_len):
        action = policy(obs.reshape(1, -1))[0]  # d3rlpy policy 输出 [1,1]
        actions.append(action.item())
        obs, reward, done, info = env.step(action)
        total_values.append(info["total_value"])
        if done:
            break
    
    total_values = np.array(total_values)
    returns = np.diff(total_values) / total_values[:-1]
    returns = np.nan_to_num(returns)
    
    if len(returns) == 0:
        return {"sharpe": 0, "annual_return": 0, "max_drawdown": 0}
    
    # 年化收益（假设 250 交易日）
    annual_return = np.mean(returns) * 250
    # 夏普比率（无风险利率=0）
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(250)
    # 最大回撤
    cummax = np.maximum.accumulate(total_values)
    drawdown = (cummax - total_values) / cummax
    max_drawdown = np.max(drawdown)
    
    return {
        "sharpe": float(sharpe),
        "annual_return": float(annual_return),
        "max_drawdown": float(max_drawdown),
        "final_value": float(total_values[-1]),
        "avg_position": float(np.mean(actions))
    }


# ==============================
# 3. 主训练流程
# ==============================
def main():
    # 路径配置
    # DATA_ROOT = "datasets/policy_zoo"
    # TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    # VAL_DIR = os.path.join(DATA_ROOT, "val")
    TRAIN_DIR = "datasets/sz.000513"
    VAL_DIR = "datasets/sz.000513_val"
    MODEL_SAVE_DIR = "models/offline_cql"
    
    # 加载数据
    # dataset_train, dataset_val = build_dataset(TRAIN_DIR, VAL_DIR)
    
    dataset_train = load_all_data_for_stock("datasets/sz.000513")
    dataset_val = load_all_data_for_stock("datasets/sz.000513_val")
    
    # 划分训练/内部验证（可选）
    train_episodes, test_episodes = train_test_split(
        dataset_train.episodes, test_size=0.1, random_state=42
    )
    # internal_train = MDPDataset.from_episodes(train_episodes)
    # internal_test = MDPDataset.from_episodes(test_episodes)
    
    # 初始化算法（CQL 适合保守型金融任务）
    td3 = d3rlpy.algos.TD3Config(
        batch_size=256,
        # learning_rate=3e-4,
        # alpha_learning_rate=1e-4,
        # use_gpu=False  # 改为 True 如果有 GPU
        # conservative_weight=10.0  # 控制保守程度
    ).create(device="cpu")  # 或 "cuda"
    
    # 设置验证器（每 1000 步评估一次）
    val_df = pd.read_csv("sz.000513_val.csv")
    val_env = SimpleTradingEnv(
        val_df,
        # price_series=val_df['close'],
        # np.load(os.path.join(VAL_DIR, "ep_0000.npz"))["observations"][:, -1],  # 假设最后1维是price
        initial_cash=100_000,
        commission_buy=0.0003,
        commission_sell=0.0013,
        rebalance_band=0.05,
        enable_tplus1=False,
        window_size=10
    )
    
    def validation_scorer(algo, step):
        metrics = financial_evaluator(val_env, algo.predict, episode_len=len(val_env.price_series))
        print(f"\n[Step {step}] Val Metrics: {metrics}")
        return metrics["sharpe"]  # 以夏普为优化目标
    
    # 训练
    print("🚀 Starting offline RL training with CQL...")
    td3.fit(
        dataset_train,     #internal_train,
        # eval_episodes=internal_test,
        # evaluators=dataset_val,
        n_steps=50000,
        n_steps_per_epoch=1000,
        #scorers={
        #    "td_error": d3rlpy.metrics.td_error_scorer,
        #    "value_scale": d3rlpy.metrics.average_value_estimation_scorer,
        #    "validation_sharpe": validation_scorer
        #},
        show_progress=True,
        experiment_name="cql_trading",
        # logdir="d3rlpy_logs",
    )
    
    # 保存模型
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    td3.save_model(os.path.join(MODEL_SAVE_DIR, "final_td3.d3"))
    print(f"✅ Model saved to {MODEL_SAVE_DIR}")
    
    # 最终评估（在完整 2024 年）
    print("\n🎯 Final evaluation on full 2024 validation episode...")
    final_metrics = financial_evaluator(val_env, td3.predict, episode_len=len(val_env.price_series))
    print("Final Validation Metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # 保存结果
    with open(os.path.join(MODEL_SAVE_DIR, "validation_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)


if __name__ == "__main__":
    main()