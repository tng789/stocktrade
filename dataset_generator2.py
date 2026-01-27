import numpy as np
import pandas as pd
from typing import List, Tuple, Callable
from d3rlpy.dataset import MDPDataset
import os

from tqdm import tqdm
import argparse
from datetime import datetime
import time
import tomllib
from pathlib import Path

from data_preparation import get_ready
from gymnasium import spaces

# 假设你已有这些模块（请根据实际路径调整）
from policies import (
    strong_trend_policy,
    rsi_extreme_policy,
    golden_cross_policy,
    random_noisy_policy,
    bollinger_policy,
    # add_technical_indicators
)

initial_cash = 100_000

def detect_bull_markets(
    df: pd.DataFrame,
    min_return: float = 0.10,      # 未来 window 天涨幅阈值（如 20 天涨 10%）
    window: int = 20,              # 涨幅观察窗口
    min_length: int = 30,          # 牛市最短持续天数
    max_drawdown: float = 0.15,    # 牛市中允许的最大回撤（如 15%）
    ) -> List[Tuple[int, int]]:
    """
    检测历史数据中的牛市区间。
    
    返回: List[(start_index, end_index), ...]
    - 所有索引基于 df 的行号（0-based）
    - 区间 [start, end] 闭区间，包含端点
    
    改进点：
    - 加入最大回撤约束，防止将长期震荡误判为单一牛市
    - 确保每个牛市有明确起点和终点
    """
    prices = df['close'].values
    n = len(prices)
    if n < window + min_length:
        return []
    
    bull_windows = []
    i = 0
    
    while i <= n - window - 1:
        # Step 1: 寻找潜在牛市起点（未来 window 天涨幅 ≥ min_return）
        future_return = (prices[i + window] / prices[i]) - 1
        if future_return >= min_return:
            start = i
            end = i + window
            peak_price = prices[end]  # 当前高点
            
            # Step 2: 向后延伸，但监控从高点的最大回撤
            while end < n - 1:
                end += 1
                current_price = prices[end]
                
                # 更新高点
                if current_price > peak_price:
                    peak_price = current_price
                
                # 计算当前回撤
                drawdown = (peak_price - current_price) / peak_price
                
                # 若回撤超过阈值，终止当前牛市
                if drawdown > max_drawdown:
                    end -= 1  # 回退到最后一个有效位置
                    break
            
            # Step 3: 验证牛市有效性
            length = end - start + 1
            total_return = (prices[end] / prices[start]) - 1
            
            if length >= min_length and total_return >= min_return * 0.5:
                bull_windows.append((start, end))
                i = end + 1  # 跳过已覆盖区域，避免重叠
            else:
                i += 1
        else:
            i += 1
    
    return bull_windows

def is_in_bull_window(t: int, bull_windows: List[Tuple[int, int]]) -> bool:
    """判断时间点 t 是否在任一牛市区间内"""
    for start, end in bull_windows:
        if start <= t <= end:
            return True
    return False

def get_obs(df:pd.DataFrame, cash:float, t:int, window_size:int = 60):
    """返回当前窗口的状态数据, 其中 t 在当前窗口的位置, 从window_size-1 开始"""
    
    # window size 60, [0:60]，实际是0-59， 今天作为窗口的最后一天
    data_in_window = df.iloc[t - window_size + 1 : t+1 ]

    price = data_in_window.iloc[-1]['CLOSE']
    trend_score = data_in_window.iloc[-1]['trend']

    data_in_np = techs.to_numpy()
    # 拉平
    flattened =  data_in_np.flatten()
    total_value = cash + position * price    # 总价值，现金+position*昨天的价格

    account_info = np.array([
        cash / initial_cash,
        position * price / initial_cash,
        trend_score
    ], dtype=np.float64)

    state = np.concatenate([flattened, account_info]).astype(np.float64)  
    return state

def generate_trading_dataset(
    df: pd.DataFrame,
    df_normed:pd.DataFrame,
    train_start: str = "1999-01-01",
    train_end: str = "2023-12-31",
    window_size: int = 60,
    episode_length: int = 250,
    rebalance_band: float = 0.05,
    hindsight_ratio: float = 0.15,      # 15% 的 episode 使用 hindsight 满仓（仅牛市）
    trend_ratio: float = 0.25,          # 25% 使用 strong_trend
    random_ratio: float = 0.40,         # 40% 随机/噪声
    other_ratio: float = 0.20,          # 20% 其他策略（RSI/MA等）
    slide_step: int = 1,                # 滑动步长（1天）
    trend_threshold = 0.45
) -> MDPDataset:
    
    # os.makedirs(output_dir, exist_ok=True)
    """
    生成高质量 offline RL dataset
    """
    # ===== 1. 时间切片 =====
    df = df.set_index('date')               # df是带有技术指标点的数据
    df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
    
    df['cash'] = 0.0
    df['position'] = 0
    df['action'] = 0.0

    df = df.sort_index()

    
    strategies = {
        "strong_trend": strong_trend_policy,
        "rsi":  rsi_extreme_policy,
        "ma":    golden_cross_policy,
        "random_noisy":    random_noisy_policy,
        "bollinger":  bollinger_policy,
        "hindsight": hindsight
        }
    
    total_days = df.shape[0] 
    if total_days < episode_length:
        raise ValueError("Training period too short!")
    
    # ===== 2. 检测牛市区间（用于 hindsight）=====
    bull_windows = detect_bull_markets(df, min_return=0.10, window=20, min_length=30)
    print(df.shape)
    # for p in bull_windows: print(p, p[1]-p[0])
    print(f"✅ Detected {len(bull_windows)} bull markets")
    
    # 与第一版相比，不再使用gym，而是自己模拟gym的算法，来计算obs, rewards, done等数据，加快速度

    # ===== 4. 存储 buffer =====
    observations, actions, rewards, terminals = [], [], [], []
    
    # ===== 5. 主循环：每天滑动生成 episode =====
    num_episodes = 0
    for start_idx in tqdm(range(0, total_days - episode_length - window_size + 1, slide_step), desc="Generating episodes"):
    # for start_idx in range(0, total_days - episode_length - window_size + 1, slide_step):
        # df_in_episode = df.iloc[start_idx:start_idx+episode_length]
        # === 策略选择 ===
        r = np.random.rand()
        if r < hindsight_ratio:
            base_policy = "hindsight"
        elif r < hindsight_ratio + trend_ratio:
            base_policy = "strong_trend"
        elif r < hindsight_ratio + trend_ratio + random_ratio:
            base_policy = "random_noisy"
        else:
            # 随机选一个震荡策略
            base_policy = np.random.choice(["rsi", "ma", "bollinger"])

        val_set_length = episode_length + window_size 
        # obs = env.reset(start_idx,data_length=val_set_length)        # 进入env的数据，60天窗口期，加上实际的250天数据，总共310天
        
        # === 生成一条 episode ===
        
        # for step in range(window_size, val_set_length):
        df_episode = df.iloc[start_idx:start_idx+episode_length]
        
        for step in range(episode_length):                                      # 总共250天数据，则循环250天， 中途不退出
            t = start_idx +  step + window_size  -1                             # t是当前数据在整个val集中的绝对位置
            assert t < df.shape[0]
            # print(f"episode_{start_idx:04d}\n {t=}")
            # 决定 action
            if base_policy == "hindsight":
                if is_in_bull_window(t, bull_windows):
                    action = 1.0  # 牛市满仓
                else:
                    # 不在牛市：降级为探索策略（避免无效满仓）
                    action = random_noisy_policy(df, t)
            else:
                if base_policy in strategies:
                    action = strategies[base_policy](df,t)
                else:
                    action = 0.5
            df_episode.iloc[step]['action'] = action 
            
            next_obs = get_obs(df_episode,100000,step,60)
            #  action生成完毕，执行
            # next_obs, reward, done, info = env.step(action)
            observations.append(next_obs)
            # assert not np.any(np.isnan(next_obs)), "Nan found"

            actions.append([float(action)])
            rewards.append(float(reward))

            if step >= episode_length - 1:
                terminals.append(bool(done))
            
        episode_name = f"episode_{start_idx:04d}"
        # print(episode_name, "done...")
        
        num_episodes += 1
        # if num_episodes % 1000 == 0:
        #    print(f"Generated {num_episodes} episodes...")
    
    assert terminals[-1],  "must be True"
    # ===== 6. 构建 dataset =====
    dataset = MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=bool),
        action_space=spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)
    )

    all_actions = np.concatenate([ep.actions for ep in dataset.episodes])
    
    # action_file = pd.DataFrame(all_actions)
    # action_file.to_csv("aaa.csv")

    print("动作/Weight Actions size:", all_actions.shape)
    print("动作/Weight Actions mean/std:", all_actions.mean(), all_actions.std())
    print(f" - Action > 0.7 ratio: {(all_actions > 0.7).mean():.2%}")
    print(f" - Action > 0.8 ratio: {(all_actions > 0.8).mean():.2%}")
    print(f" - Action > 0.9 ratio: {(all_actions > 0.9).mean():.2%}")
#    print(f"\n✅ Dataset generated!")
#    print(f" - Episodes: {len(dataset.episodes)=}")
#    print(f" - Transitions: {dataset.size()}")
#    print(f" - Action mean/std: {dataset.actions.mean():.3f} ± {dataset.actions.std():.3f}")
#    print(f" - Action > 0.8 ratio: {(dataset.actions > 0.8).mean():.2%}")
    
    return dataset

def make_val_df(df:pd.DataFrame, start_date, end_date, data_dir):
    code = df.iloc[0]['code']
    start_date_n = int(df[df['date'] >= start_date].index[0])
    end_date_n = int(df[df['date'] <= end_date].index[-1])

    val_data = df.iloc[start_date_n-59:end_date_n,:]
    val_data.to_csv(Path(data_dir)/code/f"{code}.val.csv",index=False) 

    total_days = val_data.shape[0]
    
    # for i in range(0,total_days-90, 15):
    i = 0
    while i <= total_days-90:    
        if i + 90 > total_days:
            df_period = val_data.iloc[i:]
        else:
            df_period = val_data.iloc[i:i+90]
        df_period.to_csv(Path(data_dir)/code/f"{code}.val.{i:02d}.csv",index=False)
        i += 15
    
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True, help="the datafile in csv")
    # parser.add_argument("--start_date", type=str, default="1999-01-01", help="the start date of train data")
    # parser.add_argument("--end_date", type=str, default="2023-12-31", help="the end date of train data")

    parser.add_argument("--dir", type=str, default="dataset", help="the directory of dataset stored")
    opt = parser.parse_args()
    return opt

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    
    opt = parse_opt()

    # 加载你的 OHLCV 数据（需有 'close', 'volume' 列，index 为 datetime）
    datafile = Path(opt.data)
    if not  datafile.exists():
        print(f"data file {opt.data} not found...")
        exit(0)
    
    # df = pd.read_csv(opt.data, index_col="date", parse_dates=True)
    df = pd.read_csv(opt.data, parse_dates=True)

    # 删去成交量为零的行
    df = df.replace(0,np.nan).dropna()
    df.reset_index(drop=True, inplace=True)

    # df = add_technical_indicators(df)  # 在原ohlcv的基础上添加技术指标
    # 取得股票代码 
    code = df.iloc[0]['code']
    
    with open(f"{code}.toml","rb") as f:
        cfg = tomllib.load(f)
     
    # print(cfg)
    home_dir = Path(".") / cfg['dataset_dir'] / code
    home_dir.mkdir(parents=True, exist_ok=True)
    # 划分测试集和验证集
    # df_test = df[df.index >=]
    # df_test.to_csv(f"{code}_test.csv")

    # df_val = df.loc[f"{cfg['dates']['val_start']}":f"{cfg['dates']['val_end']}"]

    # data_file_val = home_dir / f"{code}_val.csv"
    # df_val.to_csv(data_file_val)

    # 取得归一化的数据集（全部数据，含训练验证测试）
    df_ind, df_normed = get_ready(df)
    
    # 保存在csv
    df_ind.to_csv(home_dir/f"{code}_ind.csv")
    df_normed.to_csv(home_dir/f"{code}_normed.csv")
    # 为分阶段保存准备，暂时不用
    make_val_df(df_normed,"2024-01-01","2024-12-31",cfg['dataset_dir'])
    
    ch = input("......") 
    # 按照策略生成模拟交易数据集
    
#        ("strong_trend", strong_trend_policy),
#        ("random_behavior",behavior_policy)

        # 可添加更多策略...
    
    dataset = generate_trading_dataset(
        df_ind,
        df_normed,
        # f"./dataset/{code}/train",
        train_start =cfg['dates']['train_start'],
        train_end   =cfg['dates']['train_end'],
        hindsight_ratio=cfg['ratio']['hindsight'],
        trend_ratio=cfg['ratio']['trend'],
        random_ratio=cfg['ratio']['random'],
        other_ratio=cfg['ratio']['other'],
        slide_step=5,
        trend_threshold = cfg['trend_threshold']
    )
    
    # 保存
    h5_file = Path(".") / cfg['dataset_dir'] / code / f"{code}_train_dataset.h5"
    # os.makedirs(h5_dir, exist_ok=True)
    
    dataset.dump(h5_file)