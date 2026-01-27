import numpy as np
import pandas as pd
class TrendStrengthDetector:
    """
    检测当前市场是否具备“可交易趋势”
    输出：0~1，值越大表示趋势越强、噪音越低
    """
    def __init__(self, lookback=60):
        self.lookback = lookback

    # def compute_trend_score(self, prices: np.ndarray) -> float:
    def compute_trend_score(self, prices: pd.Series) -> float:
        """
        计算趋势强度分数
        输入: 最近 N 日收盘价 (length >= lookback)
        输出: 0.0 ~ 1.0+
        """
        if len(prices) < self.lookback:
            return 0.0

        p = prices[-self.lookback:]

        # 1. 动量强度（标准化）
        momentum = (p.iloc[-1] / p.iloc[0]) - 1
        
        # 2. 波动率（对数收益率标准差）
        log_returns = np.diff(np.log(p))
        volatility = np.std(log_returns) * np.sqrt(250)             # 1年按照250个交易日计算
        
        # 3. 趋势效率比（Efficiency Ratio）
        # = |总价格变化| / 总路径长度 → 越接近1，趋势越干净
        price_change = abs(p.iloc[-1] - p.iloc[0])
        path_length = np.sum(np.abs(np.diff(p)))
        efficiency = price_change / (path_length + 1e-8)
        
        # 4. ADX 近似（简化版）
        up = np.maximum(np.diff(p), 0)
        down = np.maximum(-np.diff(p), 0)
        avg_up = np.mean(up[-14:])
        avg_down = np.mean(down[-14:])
        rs = avg_up / (avg_down + 1e-8)
        adx_approx = 100 * (rs / (1 + rs))  # 0~100

        # 综合打分（可调权重）
        score = (
            0.3 * np.clip(abs(momentum) * 10, 0, 1) +
            0.2 * np.clip(efficiency * 2, 0, 1) +
            0.3 * np.clip(adx_approx / 100, 0, 1) +
            0.2 * (1 - np.clip(volatility / 0.5, 0, 1))  # 适中波动最佳
        )
        
        return float(score)

    def should_trade(self, prices: np.ndarray, threshold: float = 0.45) -> bool:
        """
        是否启用 RL 策略？
        - 趋势强 → 启用 RL（精细择时）
        - 趋势弱 → 退守 B&H（避免震荡损耗）
        """
        score = self.compute_trend_score(prices)
        return score >= threshold
    


# -----------------------------------------------
# 使用方法 
# 在策略决策时
#trend_detector = TrendStrengthDetector(lookback=60)

# 获取最近价格（需从 env 或 df 中提取）
#recent_prices = df['close'].iloc[-70:].values  # 多取几天防边界

#if trend_detector.should_trade(recent_prices, threshold=0.45):
#    action = rl_model.predict(state)[0]
#    print(f"📈 趋势强 (score={trend_detector.compute_trend_score(recent_prices):.2f})，启用 RL 策略")
#else:
#    action = 1.0  # 退守满仓 B&H
#    print(f"📉 趋势弱，退守买入持有")