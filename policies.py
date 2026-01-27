import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Callable
from tqdm import tqdm


# 假设你已定义 SimpleTradingEnv（可从上文复制）
# from simpletradingenv import SimpleTradingEnv  # 替换为你的实际模块名

import sys

max_lot = 1.0       # 满仓 1.0, 即，总资产中最多90%用于股票

def label_market_regime(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    简单市场状态标签（可扩展）
    - 1: 牛市（20日涨幅 > 10%）
    - -1: 熊市（20日跌幅 > 10%）
    - 0: 震荡
    """
    returns = df['close'].pct_change(window).fillna(0)
    regime = pd.Series(0, index=df.index)
    regime[returns > 0.10] = 1
    regime[returns < -0.10] = -1
    return regime

# ==============================
# 1. 技术指标预计算（无未来泄露）
# ==============================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close_price = df['close']
    
    # 移动平均线
    df['ma5'] = close_price.rolling(5, min_periods=1).mean()
    df['ma20'] = close_price.rolling(20, min_periods=1).mean()
    
    # 交叉信号（当前是否 ma5 > ma20）
    df['ma5_above_ma20'] = (df['ma5'] > df['ma20']).astype(int)
    
    # 布林带位置
    ma20 = close_price.rolling(20, min_periods=1).mean()
    std20 = close_price.rolling(20, min_periods=1).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    bb_pos = (close_price - bb_lower) / (bb_upper - bb_lower + 1e-8)  # ✅ 修复：close → close_price
    df['bb_position'] = np.clip(bb_pos, 0, 1)
    
    # RSI
    delta = close_price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 波动率（用于风控）
    log_ret = np.log(close_price / close_price.shift(1)).fillna(0)
    df['vol20'] = log_ret.rolling(20, min_periods=1).std() * np.sqrt(250)
    
    # ================
    # 新增：ADX 和 MACD
    # ================
    df = add_adx(df)
    df = add_macd(df)
    
    return df


# ==============================
# 辅助函数：计算 ADX（14日）
# ==============================
def add_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    high, low, close = df['high'], df['low'], df['close']
    
    # True Range
    tr0 = abs(high - low)
    tr1 = abs(high - close.shift(1))
    tr2 = abs(low - close.shift(1))
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=1).mean()
    
    # Directional Movement
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)
    
    plus_di = 100 * (plus_dm.rolling(window, min_periods=1).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window, min_periods=1).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    adx = dx.rolling(window, min_periods=1).mean()
    
    df['adx'] = adx.fillna(0)
    df['plus_di'] = plus_di.fillna(0)
    df['minus_di'] = minus_di.fillna(0)
    return df


# ==============================
# 辅助函数：计算 MACD（12,26,9）
# ==============================
def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    df = df.copy()
    close = df['close']
    ema_fast = close.ewm(span=fast, min_periods=1).mean()
    ema_slow = close.ewm(span=slow, min_periods=1).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=1).mean()
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = macd_line - signal_line
    return df


# ==============================
# 【新增】强趋势策略：ADX + MACD + Volume
# ==============================
def strong_trend_policy(df_ind, t: int) -> float:
    """
    强趋势策略：
    - ADX > 25：确认强趋势
    - MACD 金叉（hist 由负转正）：确认上涨动量
    - 成交量 > 20日均量 * 1.2：确认资金入场
    """
    if t < 50:  # 数据不足
        return 0.5
    
    adx = df_ind['adx'].iloc[t]
    macd_hist = df_ind['macd_hist'].iloc[t]
    macd_hist_prev = df_ind['macd_hist'].iloc[t-1]
    
    # 量能确认（需 volume 列存在）
    if 'volume' in df_ind.columns:
        vol_today = df_ind['volume'].iloc[t]
        vol_ma20 = df_ind['volume'].iloc[t-19:t+1].mean()
        volume_ok = vol_today > (vol_ma20 * 1.2)
    else:
        volume_ok = True  # 若无 volume，跳过
    
    # 趋势强度
    strong_trend = adx > 25
    
    # MACD 金叉（柱状图由负转正）
    macd_bullish = (macd_hist_prev <= 0) and (macd_hist > 0)
    
    if strong_trend and macd_bullish and volume_ok:
        return 1.0   # 满仓
    elif adx < 20:   # 震荡市
        return 0.2   # 低仓位
    else:
        # return 0.5   # 默认
        return 0.6   # 加0.1，期望增加模拟数据的持仓


# ==============================
# 其他策略（保持不变，仅修复 bollinger_policy）
# ==============================
def random_policy(obs, t: int):
    return np.random.uniform(0.0, max_lot)

def rsi_extreme_policy(df_ind, t: int) -> float:
    rsi = df_ind.iloc[t]['rsi']
    if rsi < 30:
        return max_lot
    elif rsi > 70:
        return 0.0
    else:
        # return 0.5
        return 0.65   # 加0.15，期望增加模拟数据的持仓

def rsi_low_vol_policy(df_ind, t: int) -> float:
    rsi = df_ind['rsi'].iloc[t]
    vol = df_ind['vol20'].iloc[t]
    if rsi < 30 and vol < 0.4:
        return max_lot
    elif rsi > 70 or vol > 0.6:
        return 0.0
    else:
        return 0.5

def smart_rsi_policy(df_ind, t: int) -> float:
    if t < 5:
        return 0.0
    rsi = df_ind['rsi'].iloc[t]
    vol_ratio = df_ind['volume'].iloc[t] / (df_ind['volume'].iloc[t-5:t].mean() + 1e-8)
    if rsi < 30 and vol_ratio > 1.2:
        return max_lot
    elif rsi > 70:
        return 0.0
    else:
        return 0.3

def bollinger_policy(df_ind, t: int) -> float:
    if 'bb_position' not in df_ind.columns:
        close_price = df_ind['close']
        ma20 = close_price.rolling(20, min_periods=1).mean()
        std20 = close_price.rolling(20, min_periods=1).std()
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        bb_pos = (close_price - bb_lower) / (bb_upper - bb_lower + 1e-8)  # ✅ 修复
        bb_pos = np.clip(bb_pos, 0, 1)
    else:
        bb_pos = df_ind['bb_position']
    pos = bb_pos.iloc[t]
    return float(np.clip(1 - pos, 0.0, max_lot))

def golden_cross_policy(df_ind, t: int) -> float:
    if t == 0:
        return 0.0
    prev_above = df_ind.iloc[t - 1]['ma5_above_ma20']
    curr_above = df_ind.iloc[t]['ma5_above_ma20']
    if curr_above == 1 and prev_above == 0:
        return max_lot
    elif curr_above == 0 and prev_above == 1:
        return 0.0
    else:
        return float(curr_above)

def volume_breakout_policy(df_ind, t: int) -> float:
    if t < 20:
        return 0.0
    close = df_ind['close'].iloc[t]
    high_20 = df_ind['high'].iloc[t-20:t].max()
    vol_today = df_ind['volume'].iloc[t]
    vol_ma5 = df_ind['volume'].iloc[t-5:t].mean()
    price_breakout = close > high_20
    volume_surge = vol_today > (vol_ma5 * 1.5)
    if price_breakout and volume_surge:
        return max_lot
    else:
        return 0.0

def volatility_adjusted_policy(df_ind, t: int) -> float:
    vol = df_ind['vol20'].iloc[t]
    if vol > 0.5:
        return 0.3
    elif vol < 0.2:
        return max_lot
    else:
        # return 0.6
        return 0.75                 # 原来 0.6，改0.75试试

def random_noisy_policy(df_ind, t: int) -> float:
    base = np.random.uniform(0.3, 1.0)
    noise = np.random.normal(0, 0.1)
    n = float(np.clip(base + noise, 0.0, max_lot))
    return n

def behavior_policy(df_ind, t:int):

    if np.random.rand() < 0.2:
        return max_lot
    else:
        # 简单 MA crossover
        prices = df_ind['close']
        if t >= 20:
            ma5 = np.mean(prices[t-4:t+1])
            ma20 = np.mean(prices[t-19:t+1])
            return 1.0 if ma5 > ma20 else 0.0
        else:
            return 0.5

# 这个方法用于保存成npz,现在用不上了。暂时保留着做参考。
#def generate_episodes(
#    df: pd.DataFrame,
#    behavior_policy: Callable[[np.ndarray], float],
#    output_dir: str,
#    window_size: int = 250,
#    slide_step: int = 1,
#    min_episode_length: int = 50,
#    env_kwargs: Dict = None,
#):
#    """
#    从单只股票日线数据生成多个 episodes
#    
#    Args:
#        df: 包含 'date', 'close' 的 DataFrame（按时间排序）
#        behavior_policy: 函数，输入 obs，输出 target_weight ∈ [0,1]
#        output_dir: 输出目录
#        window_size: 每个 episode 的最大长度（天）
#        slide_step: 滑动窗口步长
#        min_episode_length: 最短有效 episode 长度
#        env_kwargs: 传给 SimpleTradingEnv 的参数
#    """
#    os.makedirs(output_dir, exist_ok=True)
#    
#    # 确保日期列存在并排序
#    if 'date' not in df.columns:
#        raise ValueError("DataFrame must contain 'date' column")
#    df = df.sort_values('date').reset_index(drop=True)
#    
#    # 计算市场状态（用于元数据）
#    df['regime'] = label_market_regime(df)
#    
#    # 提取价格序列
#    prices = df['close'].values.astype(np.float64)
#    dates = df['date'].values
#    
#    total_records = df.shape[0]
#
#    meta_info = {}
#    episode_id = 0
#
#    # 滑动窗口划分
#    for start_idx in tqdm(range(0, total_records - window_size, slide_step), desc="Generating episodes"):
#        end_idx = start_idx + window_size
#        current_window_size = window_size               #当前的window_size，这是考虑最后的一个episode不一定是预设的200天
#
#        if end_idx > total_records: 
#            end_idx = total_records
#            current_window_size = end_idx - start_idx 
#
#        sub_prices = prices[start_idx:end_idx]
#        sub_dates = dates[start_idx:end_idx]
#        
#        # 创建环境
#        # env = SimpleTradingEnv(price_series=sub_prices, **env_kwargs)
#        working_df = df.iloc[start_idx:end_idx]
#        # env = SimpleTradingEnv(working_df,**env_kwargs)
#        env = SimpleTradingEnv(working_df)
#        obs = env.reset()
#        
#        # 存储轨迹
#        observations, actions, rewards, terminals = [], [], [], []
#        total_value_history = []
#        
#        for pace in range(current_window_size):
#            # action = behavior_policy(obs)
#            action = behavior_policy(working_df, pace)
#            assert action >= 0,  "action must be > 0"
#            next_obs, reward, done, info = env.step(action)
#            
#            observations.append(obs)
#            actions.append(action)
#            rewards.append(reward)
#            terminals.append(done)
#            total_value_history.append(info["total_value"])
#            
#            if done:
#                break
#            obs = next_obs
#        
#        # 过滤太短的 episode
#        if len(rewards) < min_episode_length:
#            continue
#
##        # assert len(rewards) == len(terminals)
##        assert len(observations) == len(actions) == len(rewards) == len(terminals) 
##        assert not np.any(np.isnan(rewards))
##        assert not np.any(np.isinf(rewards))
##
#        assert terminals[-1] , "last element of terminals must be True"
#
#        # actions_ = np.array(actions, dtype=np.float64)
#        # print(output_dir.split("/")[-1])
#        # print(actions_.mean(), actions_.std())
##        if actions_.ndim == 1:
##            actions_ = actions_.reshape(-1, 1)
#
#        # 保存 .npz
#        episode_name = f"episode_{episode_id:04d}"
#        np.savez_compressed(
#            os.path.join(output_dir, f"{episode_name}.npz"),
#            observations=np.array(observations, dtype=np.float64),
#            # actions=np.array(actions, dtype=np.float64).reshape(-1, 1),
#            actions=np.array(actions, dtype=np.float64),
#            rewards=np.array(rewards, dtype=np.float64),
#            terminals=np.array(terminals, dtype=bool)
#        )
#        
#        # 保存元数据
#        meta_info[episode_name] = {
#            "start_date": str(sub_dates[0]),
#            "end_date": str(sub_dates[len(rewards)-1]),  # 实际结束日
#            "planned_end_date": str(sub_dates[-1]),
#            "length": len(rewards),
#            "planned_length": window_size,
#            "terminated_early": terminals[-1] and len(rewards) < window_size,
#            "source_strategy": getattr(behavior_policy, '__name__', 'unknown'),
#            "start_regime": int(df.loc[start_idx, 'regime']),
#            "avg_regime": int(np.round(df.loc[start_idx:start_idx+len(rewards)-1, 'regime'].mean())),
#            "initial_price": float(sub_prices[0]),
#            "final_price": float(sub_prices[len(rewards)-1])
#        }
#        
#        episode_id += 1
#
#    # 保存元数据 JSON
#    with open(os.path.join(output_dir, "episodes_meta.json"), "w", encoding="utf-8") as f:
#        json.dump(meta_info, f, indent=2, ensure_ascii=False)
#
#    print(f"✅ 生成 {episode_id} 个 episodes，元数据已保存至 {output_dir}/episodes_meta.json")
#
#if __name__ == "__main__":
#    # 加载数据（示例）
#    # df = pd.read_csv("sz.000513.train.csv")  # 必须含 'date', 'close','volume'
#    df = pd.read_csv("sz.000513_d_origin.csv", index_col='date')  # 必须含 'date', 'close','volume'
#    df[df.index >= "2025-01-01"].to_csv("sz.000513_test.csv")       #测试用
#    
#    df = df.replace(0,np.nan).dropna()
#    
#    code = f"{df.iloc[0]['code']}"
#    
#    df = add_technical_indicators(df)
#    df = df.iloc[1:,:]
#    df[df.index < "2024-01-01"].to_csv("sz.000513_train.csv")           # 带技术指标的训练数据
#    
#    df_tmp = df[df.index > "2024-01-01"]
#    df_tmp[df_tmp.index < "2025-01-01"].to_csv("sz.000513_val.csv")
#    
#
#    df.to_csv("with_tech_indicators.csv")
#    # 生成多策略数据集
#    strategies = [
#        # ("random", random_policy),                    # 不用，可能导致频繁交易
#        # ("rsi_extreme", rsi_extreme_policy),
#        ("bollinger_band", bollinger_policy),
#        ("ma_crossover", golden_cross_policy),
#        ("volatility_weighted", volatility_adjusted_policy),
#        # ("volume_breakout", volume_breakout_policy),      # ← 新增量价策略
#        # ("random_walk", random_noisy_policy),
#        ("rsi_low_vol", rsi_low_vol_policy),             # ← 关键混合策略
#        # ("smart_rsi", smart_rsi_policy),
#        ("strong_trend", strong_trend_policy),
#        ("random_behavior",behavior_policy)
#
#        # 可添加更多策略...
#    ]
#    
#    # window_size = df.shape[0]
#
#    for stock in (["sz.000513_train"]):
#        purpose = stock.split("_")[-1]
#        for name, policy in strategies:
#            output_dir = f"datasets/{code}/{purpose}/{name}"
#            data = pd.read_csv(f"{stock}.csv")
#            generate_episodes(
#                df = data,
#                behavior_policy=policy,
#                output_dir=output_dir,
#                window_size=250,
#                # window_size=200,        #for validate data npz, 一年到不了250个交易日
#                slide_step=1,
#                min_episode_length=50
#            )