import numpy as np
import pandas as pd

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """计算 RSI（无未来函数）"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """计算平均真实波幅（ATR）"""
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr


def build_normalized_features(
    df: pd.DataFrame,
    include_price_window: bool = True,
    price_window_size: int = 10,
    clip_quantile: float = 0.99,  # 用于裁剪极端值
    ) -> np.ndarray:
    """
    构建归一化特征矩阵，适用于 RL 状态输入
    
    Args:
        df: OHLCV DataFrame，必须包含 ['open', 'high', 'low', 'close', 'volume']
        include_price_window: 是否包含原始价格窗口（用于 SimpleTradingEnv 兼容）
        price_window_size: 价格窗口长度（仅当 include_price_window=True 时使用）
        clip_quantile: 用于裁剪非比率型特征的分位数（如波动率）
    
    Returns:
        X: 归一化特征矩阵，shape [T, n_features]
    """
    df = df.copy()
    features = []
    names = []

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # ==============================
    # 1. 价格相对特征（天然归一化）
    # ==============================
    if include_price_window:
        # 保留原始价格窗口（用于 SimpleTradingEnv 的 _get_state）
        # 注意：实际使用时会在 env 内部做 price / current_price 归一化
        for i in range(price_window_size):
            offset = price_window_size - i
            if i == 0:
                feat = close
            else:
                feat = close.shift(offset)
            features.append(feat)
            names.append(f"price_t-{offset}")
    
    # MA 相对位置
    ma5 = close.rolling(5, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    features.append(ma5 / close)
    features.append(ma20 / close)
    names.extend(['ma5_ratio', 'ma20_ratio'])

    # EMA
    ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
    features.append(ema12 / close)
    features.append(ema26 / close)
    names.extend(['ema12_ratio', 'ema26_ratio'])

    # ==============================
    # 2. 动量类（压缩到 [-1, 1]）
    # ==============================
    roc5 = close.pct_change(5).fillna(0)
    roc20 = close.pct_change(20).fillna(0)
    # 使用 tanh 压缩，放大中小变动敏感度
    features.append(np.tanh(roc5 * 10))
    features.append(np.tanh(roc20 * 5))
    names.extend(['roc5_norm', 'roc20_norm'])

    # RSI（[0,100] → [0,1]）
    rsi = compute_rsi(close, 14)
    features.append(rsi / 100.0)
    names.append('rsi_norm')

    # MACD 信号（已中心化，用 tanh 压缩）
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False, min_periods=1).mean()
    macd_hist = (dif - dea) * 2
    features.append(np.tanh(macd_hist * 100))  # 缩放因子根据经验调整
    names.append('macd_hist_norm')

    # ==============================
    # 3. 波动率类（裁剪 + 压缩）
    # ==============================
    log_ret = np.log(close / close.shift(1)).fillna(0)
    vol20 = log_ret.rolling(20, min_periods=1).std() * np.sqrt(250)
    # 裁剪极端值（如 >99% 分位数）
    vol_clip = vol20.quantile(clip_quantile)
    vol20_clipped = np.clip(vol20, 0, vol_clip)
    # 归一化到 [0,1]
    vol20_norm = vol20_clipped / (vol_clip + 1e-8)
    features.append(vol20_norm)
    names.append('vol20_norm')

    # ATR 相对波动
    atr14 = compute_atr(high, low, close, 14)
    features.append(atr14 / close)
    names.append('atr_ratio')

    # ==============================
    # 4. 量价类
    # ==============================
    vol_ma5 = volume.rolling(5, min_periods=1).mean()
    vol_ratio = volume / (vol_ma5 + 1e-8)
    # 裁剪异常放量
    vol_ratio = np.clip(vol_ratio, 0, 5.0)  # 超过5倍均量视为极端
    vol_ratio_norm = vol_ratio / 5.0
    features.append(vol_ratio_norm)
    names.append('vol_ratio_norm')

    # 布林带位置（[0,1]）
    bb_ma20 = close.rolling(20, min_periods=1).mean()
    bb_std20 = close.rolling(20, min_periods=1).std()
    bb_upper = bb_ma20 + 2 * bb_std20
    bb_lower = bb_ma20 - 2 * bb_std20
    bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
    bb_position = np.clip(bb_position, 0, 1)
    features.append(bb_position)
    names.append('bb_position')

    # ==============================
    # 合并特征
    # ==============================
    X = pd.concat(features, axis=1)
    X.columns = names
    X = X.fillna(0).astype(np.float32)  # 前导 NaN 用 0 填充（合理假设）

    X.to_csv("sz.000513.features.csv")
    return X.values  # [T, n_features]


if __name__ == "__main__":
    # 加载数据
    df = pd.read_csv("sz.000513_d_origin.csv", parse_dates=['date'])
    df.replace(0, np.nan).dropna()

    df = df.sort_values('date').reset_index(drop=True)

    # 全局计算特征（2000–2025）
    feature_matrix = build_normalized_features(df, include_price_window=True, price_window_size=10)
   
    # 划分训练集（2000–2023）
    train_mask = df['date'] <= '2023-12-31'
    train_features = feature_matrix[train_mask]
    train_prices = df.loc[train_mask, 'close'].values
    
    # 划分测试集和验证集（2024–）
    temp_mask =  df['date'] >= '2024-01-01'
    df_temp = df[temp_mask]
    temp_features = feature_matrix[temp_mask]
    temp_prices = df.loc[temp_mask, 'close'].values
    
    val_mask = df_temp['date'] < "2025-01-01"
    val_features = temp_features[val_mask]
    val_values = df_temp.loc[val_mask,'close'].values

    # temp_mask = df['date'] >= '2024-01-01'
    # temp_features = feature_matrix[temp_mask]
    # temp_prices = df.loc[temp_mask, 'close'].values
    

    test_mask = df['date'] >= '2025-01-01'
    test_features = feature_matrix[test_mask]
    test_prices = df.loc[test_mask, 'close'].values