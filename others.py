
import numpy as np
import pandas as pd
from backtest_engine import Backtester  # 假设你有回测框架

from trend_detector import TrendStrengthDetector  # 假设你已有该模块
#“动态阈值”版本（根据波动率自适应）
class AdaptiveHybridPolicy(HybridPolicy):

    def __init__(self, rl_model, regime_detector, base_threshold=0.6, vol_window=20):
        super().__init__(rl_model, regime_detector, threshold=base_threshold)
        self.base_threshold = base_threshold
        self.vol_window = vol_window

    def _compute_dynamic_threshold(self, price_history: np.ndarray) -> float:
        """高波动 → 提高阈值（更严格）；低波动 → 降低阈值（更宽松）"""
        if len(price_history) < self.vol_window:
            return self.base_threshold
        
        log_returns = np.diff(np.log(price_history[-self.vol_window:]))
        volatility = np.std(log_returns) * np.sqrt(252)
        
        # 动态调整：vol ↑ → threshold ↑
        # 例如：vol=0.3（高）→ threshold=0.7；vol=0.1（低）→ threshold=0.5
        dynamic_th = self.base_threshold + 0.2 * (volatility - 0.2)
        return np.clip(dynamic_th, 0.4, 0.8)

    def predict(self, obs: np.ndarray, price_history: np.ndarray) -> float:
        dynamic_threshold = self._compute_dynamic_threshold(price_history)
        trend_score = self.regime_detector.compute_trend_score(price_history)
        
        if trend_score > dynamic_threshold:
            action = self.rl_model.predict(obs.reshape(1, -1))[0]
        else:
            action = self.fallback_action
        
        return float(np.clip(action, 0.0, 1.0))


# Regime 阈值自动优化脚本（基于历史数据）
# optimize_threshold.py

def evaluate_threshold(df, threshold, window=60):
    """评估单个阈值的表现"""
    detector = TrendStrengthDetector(lookback=window)
    actions = []
    for i in range(window, len(df)):
        prices = df['close'].iloc[:i+1].values
        score = detector.compute_trend_score(prices)
        action = 1.0 if score > threshold else 1.0  # 假设 fallback=B&H
        actions.append(action)
    
    # 简化：直接返回年化收益（实际应调用完整回测）
    returns = df['close'].pct_change().iloc[window:].values
    strategy_returns = returns * np.array(actions)
    annual_return = np.mean(strategy_returns) * 252
    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
    return annual_return, sharpe

def optimize_threshold(df, thresholds=np.linspace(0.3, 0.8, 11)):
    """在 2020–2024 数据上搜索最优阈值"""
    results = []
    for th in thresholds:
        ret, sr = evaluate_threshold(df[df.index >= '2020'], th)
        results.append((th, ret, sr))
        print(f"Threshold {th:.2f} → Annual Return: {ret:.2%}, Sharpe: {sr:.2f}")
    
    best = max(results, key=lambda x: x[2])  # 按 Sharpe 最优
    print(f"\n✅ 最优阈值: {best[0]:.2f} (Sharpe={best[2]:.2f})")
    return best[0]

# 使用示例
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# best_th = optimize_threshold(df)

#HybridPolicy 完整 Python 类（带注释）

class HybridPolicy:
    def __init__(
        self,
        rl_model,
        regime_detector: TrendStrengthDetector,
        threshold: float = 0.6,
        fallback_mode: str = "buy_and_hold",  # or "cash"
        asset_type: str = "equity"  # "equity", "crypto", "commodity"
    ):
        """
        混合策略：RL + Regime Filter
        
        Parameters:
        - rl_model: d3rlpy trained model (e.g., CQL)
        - regime_detector: 趋势强度检测器
        - threshold: 趋势强度阈值，高于此值启用 RL
        - fallback_mode: 'buy_and_hold' 或 'cash'
        - asset_type: 用于自动设置 fallback（可选）
        """
        self.rl_model = rl_model
        self.regime_detector = regime_detector
        self.threshold = threshold
        
        # 自动设置 fallback action based on asset type
        if fallback_mode == "auto":
            if asset_type in ["equity", "crypto"]:
                self.fallback_action = 1.0  # B&H
            else:
                self.fallback_action = 0.0  # 空仓
        elif fallback_mode == "buy_and_hold":
            self.fallback_action = 1.0
        else:  # "cash"
            self.fallback_action = 0.0

    def predict(self, obs: np.ndarray, price_history: np.ndarray) -> float:
        """
        决策函数
        
        Args:
        - obs: 当前 observation（必须包含技术指标，但不需价格）
        - price_history: 截至当前的所有收盘价，用于计算趋势强度
        
        Returns:
        - action: 仓位比例 [0.0, 1.0]
        """
        # 1. 计算当前趋势强度
        trend_score = self.regime_detector.compute_trend_score(price_history)
        
        # 2. 根据 regime 决策
        if trend_score > self.threshold:
            # 启用 RL 策略
            action = self.rl_model.predict(obs.reshape(1, -1))[0]
        else:
            # 退守安全策略
            action = self.fallback_action
        
        # 3. 安全裁剪
        return float(np.clip(action, 0.0, 1.0))
    

#推荐滑点模型（适用于股票/ETF）：用不上
def _apply_slippage(self, action, price, volume):
    """
    基于成交量的滑点模型
    - action: 目标仓位比例（0～1）
    - price: 当前价格
    - volume: 当日成交量
    """
    position_change = abs(action - self.current_position)
    if position_change < 0.01:  # 小于1%变动，忽略滑点
        return price
    
    # 滑点 ≈ (仓位变动比例) * (典型冲击成本)
    # 假设冲击成本 = 0.1% per 10% 仓位变动
    impact = 0.001 * (position_change / 0.1)
    slippage = price * impact
    
    # 买入时价格更高，卖出时更低
    if action > self.current_position:
        return price + slippage
    else:
        return price - slippage


