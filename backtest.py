# backtest.py
import numpy as np
import pandas as pd
# from offline_rl_stock_pipeline import StockTradingEnv  # 复用你的环境
from simpletradingenv import SimpleTradingEnv
from d3rlpy.algos import CQL, TD3,TD3Config
import d3rlpy
from d3rlpy.dataset import MDPDataset
# from offline_rl_stock_pipeline_a_share import StockTradingEnv
import gymnasium as gym
from gymnasium import spaces
import torch

import os

def backtest_model(model_path, df_test, initial_balance=1e6):
    """
    在测试集上回测策略
    :param model_path: 如 "cql_stock.d3"
    :param df_test: 测试用 OHLCV DataFrame（必须与训练数据时间不重叠！）
    :return: 回测结果字典
    """
    # 加载训练好的策略
    td3config = TD3Config(actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        # alpha_learning_rate=3e-4,
        batch_size=128,
        # n_epochs=50,           # 总训练轮数
        # use_gpu=False,         # 若有 GPU 可设为 True
        # scaler="standard",     # 自动标准化观测值
        # action_scaler="min_max"  # 标准化动作（对连续控制有益）
        action_scaler=None        # 标准化动作（对连续控制有益）)
    )
    
    model= TD3(td3config,device='cpu',enable_ddp=False)

    # 创建测试环境（注意：fee/slippage 应与训练一致）
    env = SimpleTradingEnv(
        df=df_test,
        # window_size=10,
        # fee_rate=0.001,
        # base_slippage=0.0005,
        # initial_balance=initial_balance
    )


    # model.load_model(model_path)
    # model.load_learnable(model_path)

    model = d3rlpy.load_learnable(model_path,device='cpu')
    model.build_with_env(env)

    obs  = env.reset()
    done = False
    net_worths = [initial_balance]
    positions = []
    actions = []

    while not done:
        # 使用训练好的策略预测动作
        action = model.predict(obs[None, :])[0]  # 注意：predict 接收 batch
        # action = model.predict(obs)  # 注意：predict 接收 batch
        actions.append(action)
        
        obs, reward, done, _, info = env.step(action)
        net_worths.append(info['net_worth'])
        # positions.append(info['position_shares'])
        positions.append(info['position_ratio'])

    # 计算指标
    returns = np.diff(net_worths) / net_worths[:-1]
    total_return = (net_worths[-1] - initial_balance) / initial_balance
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 else 0
    max_drawdown = np.max(np.maximum.accumulate(net_worths) - net_worths) / np.maximum.accumulate(net_worths)[-1]

    results = {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'final_net_worth': net_worths[-1],
        'total_fees': env.total_fees,
        'net_worth_series': net_worths,
        'positions': positions,
        'actions': actions
    }

    print("📈 回测结果:")
    print(f"  总收益率: {total_return:.2%}")
    print(f"  年化夏普率: {sharpe:.2f}")
    print(f"  最大回撤: {max_drawdown:.2%}")
    print(f"  最终净值: ${net_worths[-1]:,.2f}")

    return results

# 示例使用
if __name__ == "__main__":
    # 🔸 加载测试数据（必须是训练/验证之外的时间段！）
    df_test = pd.read_csv("sz.000513_test.csv")  # e.g., 最近6个月

    assert os.path.exists("./models/td3_best_val_return.d3"), "model not exists"
    results = backtest_model("./models/td3_best_val_return.d3", df_test)

    # 可选：绘图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(results['net_worth_series'], label='Net Worth')
    plt.title('Backtest Performance')
    plt.xlabel('Step')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig("backtest_result.png")
    # plt.show()