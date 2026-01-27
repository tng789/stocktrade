import numpy as np
import pandas as pd
from typing import List, Tuple          #, Callable
from d3rlpy.dataset import MDPDataset

from tqdm import tqdm
import argparse
from datetime import datetime,timedelta
import tomli
import tomli_w
from pathlib import Path

from data_preparation import get_ready, fetch_stocks
from gymnasium import spaces

# 假设你已有这些模块（请根据实际路径调整）
from policies import (
    strong_trend_policy,
    rsi_extreme_policy,
    golden_cross_policy,
    random_noisy_policy,
    bollinger_policy
    # add_technical_indicators
)
from enhancedtradingenv import EnhancedTradingEnv  # 你的环境

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


def generate_trading_dataset(
    df: pd.DataFrame,
    df_normed:pd.DataFrame,
    train_start: str = "1999-01-01",
    train_end: str = "2023-12-31",
    window_size: int = 60,
    episode_length: int = 200,
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
    df = df.sort_index()

    
    # 技术指标计算，最初大约30天时间的数据不整齐，删去不用。
    # df = df.iloc[30:]

    # prices = df['close'].values
    # volumes = df['volume'].values if 'volume' in df.columns else np.ones_like(prices)
    total_days = df.shape[0] 
    if total_days < episode_length:
        raise ValueError("Training period too short!")
    
    # ===== 2. 检测牛市区间（用于 hindsight）=====
    bull_windows = detect_bull_markets(df, min_return=0.10, window=20, min_length=30)
    print(df.shape)
    # for p in bull_windows: print(p, p[1]-p[0])
    print(f"✅ Detected {len(bull_windows)} bull markets")
    
    # ===== 3. 初始化环境 =====
    env = EnhancedTradingEnv(
        df_normed,              #归一化数据进入gym环境
        mode="train",
        trend_threshold= trend_threshold,
        rebalance_band=rebalance_band,
    )
#    
    # ===== 4. 存储 buffer =====
    observations, actions, rewards, terminals = [], [], [], []
    daily_returns, trend_bonus=[],[]
     
    # ===== 5. 主循环：每天滑动生成 episode =====
    num_episodes = 0
    for start_idx in tqdm(range(0, total_days - episode_length - window_size + 1, slide_step), desc="Generating episodes"):
    # for start_idx in range(0, total_days - episode_length - window_size + 1, slide_step):
        # df_in_episode = df.iloc[start_idx:start_idx+episode_length]
        # === 策略选择 ===
        r = np.random.rand()
        if r < hindsight_ratio:                                 # ~ 0.15
            base_policy = "hindsight"
        elif r < hindsight_ratio + trend_ratio:                 # 0.15 ~ 0.4
            base_policy = "strong_trend"
        elif r < hindsight_ratio + trend_ratio + random_ratio:  #0.4 ~ 0.8
            base_policy = "random_noisy"                        #0.8 ~
        else:
            # 随机选一个震荡策略
            alt_policies = ["rsi", "ma", "bollinger"]
            base_policy = np.random.choice(alt_policies)

        val_set_length = episode_length + window_size - 1 
        env.reset(start_idx, data_length=val_set_length)        # 进入env的数据，60天窗口期，加上实际的250天数据，总共310天
        # episode_done = False
        
        # === 生成一条 episode ===
        
        # for step in range(window_size, val_set_length):
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
                if base_policy == "strong_trend":
                    action = strong_trend_policy(df, t)
                elif base_policy == "random_noisy":
                    action = random_noisy_policy(df, t)
                elif base_policy == "rsi":
                    action = rsi_extreme_policy(df, t)
                elif base_policy == "ma":
                    action = golden_cross_policy(df, t)
                elif base_policy == "bollinger":
                    action = bollinger_policy(df, t)
                else:
                    action = 0.5
            
            if action > 1 or action <0:
                print("must be in the range of [0,1]")
            #  action生成完毕，执行
            next_obs, reward, done, info = env.step(action)
            # 保存
            observations.append(next_obs)
            # assert not np.any(np.isnan(next_obs)), "Nan found"

            actions.append(float(action))
            rewards.append(float(reward))

            daily_returns.append(float(info["daily_return"]))
            trend_bonus.append(float(info["trend_bonus"]))

            if step > episode_length - 1:
                if not done:
                    print("should not reach here, something wrong...")
                    done = True
            terminals.append(bool(done))
            
            # if done:
            #    episode_done = True
            #    break
            # obs = next_obs
            
        # rewards_array  = np.array(rewards)
        # print(f"{np.mean(rewards_array)=} {np.var(rewards_array)=}  {np.std(rewards_array)=}")
        
        # episode_name = f"episode_{start_idx:04d}"
        # print(episode_name, "done...")
        
        num_episodes += 1
        # if num_episodes % 1000 == 0:
        #    print(f"Generated {num_episodes} episodes...")
    
    assert terminals[-1],  "the last one must be True, to mark the end of the episode"
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
    print(f" - Action < 0.3 ratio: {(all_actions < 0.3).mean():.4%}")
    print(f" - Action < 0.5 ratio: {(all_actions < 0.5).mean():.4%}")
    print(f" - Action > 0.7 ratio: {(all_actions > 0.7).mean():.4%}")
    print(f" - Action > 0.8 ratio: {(all_actions > 0.8).mean():.4%}")
    print(f" - Action > 0.9 ratio: {(all_actions > 0.9).mean():.4%}")

    return_array = np.array(daily_returns)
    bonus_array  = np.array(trend_bonus)
    print(f"{np.mean(return_array)=} {np.var(return_array)=}  {np.std(return_array)=}")
    print(f"{np.mean(bonus_array)=} {np.var(bonus_array)=}  {np.std(bonus_array)=}")
#    print(f"\n✅ Dataset generated!")
#    print(f" - Episodes: {len(dataset.episodes)=}")
#    print(f" - Transitions: {dataset.size()}")
#    print(f" - Action mean/std: {dataset.actions.mean():.3f} ± {dataset.actions.std():.3f}")
#    print(f" - Action > 0.8 ratio: {(dataset.actions > 0.8).mean():.2%}")
    
    return dataset

def make_val_df(df:pd.DataFrame, start_date, end_date, data_dir):
    code = df.iloc[0]['code']
    # start_timestamp = datetime.strptime(start_date, "%Y-%m-%d")
    # end_timestamp = datetime.strptime(end_date, "%Y-%m-%d")
    start_date_n = int(df[df['date'] >= start_date].index[0])
    end_date_n = int(df[df['date'] <= end_date].index[-1])

    val_data = df.iloc[start_date_n-59:end_date_n,:]
    val_data.to_csv(Path(data_dir)/code/f"{code}.val.csv",index=False) 

    #total_days = val_data.shape[0]
    
    # for i in range(0,total_days-90, 15):
    #i = 0
    #while i <= total_days-90:    
    #    if i + 90 > total_days:
    #        df_period = val_data.iloc[i:]
    #    else:
    #        df_period = val_data.iloc[i:i+90]
    #    df_period.to_csv(Path(data_dir)/code/f"{code}.val.{i:02d}.csv",index=False)
    #    i += 15
    
def merge(df_file:str, increment_df:pd.DataFrame)->pd.DataFrame:
    if Path(df_file).exists():
        df = pd.read_csv(df_file, parse_dates=True)
        df = pd.concat([df, increment_df], axis= 0).drop_duplicates()
        df.reset_index(drop=True, inplace=True)
    else:
        df = increment_df.copy()

    df.to_csv(df_file, index=False) 

    return df
    
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--code","-c", type=str, required=True, help="the datafile in csv")
    # parser.add_argument("--task","-t", type=str, default="",choices=['collect','train'], help="collect and normalize data, or generate dataset for training")
    parser.add_argument("--yes", "-y", action='store_true', help="cotniue to generate offline data")
    opt = parser.parse_args()
    return opt

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    
    opt = parse_opt()
    code = opt.code

    if Path(f"{code}.toml").exists():
        cfg_path = f"{code}.toml"
    else:
        cfg_path = "buffet.toml"

    with open(cfg_path,"rb") as f:
        cfg = tomli.load(f)

    home_dir = Path(".") / cfg['dataset_dir'] / code
    home_dir.mkdir(parents=True, exist_ok=True)

    datafile = home_dir/f"{code}.csv"
    indfile  = home_dir/f"{code}.ind.csv"
    normfile = home_dir/f"{code}.norm.csv"


    today = datetime.now().strftime("%Y-%m-%d")
    if not  datafile.exists():
        df = pd.DataFrame()
        start_date = "1999-01-01"
    else:
        df = pd.read_csv(datafile,parse_dates=True,skip_blank_lines=True)
        df = df.dropna(how='all')

        df['date'] = df['date'].str.replace("/","-")

        last_date = df.iloc[-1]['date']
        start_date = datetime.strptime(last_date,"%Y-%m-%d") + timedelta(days=1)
        start_date = start_date.strftime("%Y-%m-%d")

    end_date = today
    if start_date > end_date:
        new_transaction = pd.DataFrame()
    else:
        new_transaction = fetch_stocks(code, start_date, end_date)

    # 如果得到新数据的ohlcv数据，继续往前，计算技术指标，归一化，以及 生成离线数据
    days = new_transaction.shape[0]
    if days > 0 :                   # 有新数据
        # print(cfg)
        if cfg['code'] == "buffet":
            cfg['code'] = code
            df = new_transaction.copy()
            increment_ind, increment_normed = get_ready(df)
        else:
            df= pd.concat([df, new_transaction], axis=0).drop_duplicates()
            # 重置索引
            df.reset_index(drop=True, inplace=True)
            increment_ind, increment_normed = get_ready(df.iloc[-(days+60):])
     
        # 保存实际的ohlcv数据
        df.to_csv(datafile, index=False, date_format='%Y-%m-%d',encoding="gbk")
    
        # 划分测试集和验证集
        # df_test = df[df.index >=]
        # df_test.to_csv(f"{code}_test.csv")
    
        # df_val = df.loc[f"{cfg['dates']['val_start']}":f"{cfg['dates']['val_end']}"]
    
        # data_file_val = home_dir / f"{code}_val.csv"
        # df_val.to_csv(data_file_val)
    
        # 取得归一化的数据集（全部数据，含训练验证测试）
        # 窗口 至少60天
        
        # 保存到csv
        df_ind = merge(indfile, increment_ind)
        df_normed = merge(normfile,increment_normed)
    
        # 为分阶段保存准备，暂时不用
        make_val_df(df_normed,cfg['dates']['val_start'],cfg['dates']['val_end'],cfg['dataset_dir'])

    else:       #没有新数据
        print("no new ohlcv data fetched")

    go_ahead = opt.yes
    if not go_ahead:
        try:
            to_continue:str = input("continue to generate offline data? (yes)")
        except EOFError:
            to_continue = "N"
    
        if to_continue.lower().startswith("n"):
            print("ohlcv data downloaded and historical data updated.")
            go_ahead = False
        else:
            go_ahead = True

    if go_ahead:
        print("continue to generate mock transaction data")
        # 按照策略生成模拟交易数据集 保存
        if indfile.exists() and normfile.exists():
            df_ind = pd.read_csv(indfile, parse_dates=True)
            df_normed = pd.read_csv(normfile, parse_dates=True)

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
                slide_step=20,
                trend_threshold = cfg['trend_threshold']
            )
    
            # 保存
            code = opt.code
            h5_file = home_dir/f"{code}_train_dataset.h5"
            dataset.dump(h5_file)
        else:
            print("either tech indicators or normilized file exists")

    cfg['last_update'] = today
    with open(f"{code}.toml","wb") as f:
        tomli_w.dump(cfg, f)
     