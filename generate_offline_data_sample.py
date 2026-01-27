# 假设 prices 是某只 A 股的复权收盘价（numpy array）

from simpletradingenv import SimpleTradingEnv
import pandas as pd
import numpy as np

prices = pd.read_csv("sz.000513_d_origin.csv", index_col = "date")
prices = prices.replace(0,np.nan).dropna()

env = SimpleTradingEnv(
    price_series=prices,
    initial_cash=100_000,
    commission_buy=0.0003,
    commission_sell=0.0013,
    rebalance_band=0.05,
    take_profit_pct=0.2,
    stop_loss_pct=0.15,
    enable_tplus1=True
)

obs = env.reset(start_idx=50)  # 从第50天开始
observations, actions, rewards, terminals = [], [], [], []

for t in range(200):  # 最多跑200步
    action = np.random.uniform(0, 1)  # 行为策略（例如随机）
    next_obs, reward, done, info = env.step(action)
    
    observations.append(obs)
    actions.append(action)
    rewards.append(reward)
    terminals.append(done)
    
    if done:
        break
    obs = next_obs

# 保存为 .npz
np.savez(
    "random_strategy_episode.npz",
    observations=np.array(observations, dtype=np.float32),
    actions=np.array(actions, dtype=np.float32).reshape(-1, 1),
    rewards=np.array(rewards, dtype=np.float32),
    terminals=np.array(terminals, dtype=bool)
)