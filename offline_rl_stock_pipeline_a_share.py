# offline_rl_stock_pipeline_a_share.py
"""
A股专用 Offline RL 股票交易环境
- 100股整数倍交易
- 禁止做空（动作空间: [0.0, 1.0]）
- 真实订单执行模型
- 技术指标增强
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings("ignore")

class StockTradingEnv(gym.Env):
    def __init__(self, df, window_size=10, fee_rate=0.001, base_slippage=0.0005, initial_balance=1e6):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.fee_rate = fee_rate
        self.base_slippage = base_slippage
        self.initial_balance = float(initial_balance)

        # A股：仅支持多头（0~1），禁止做空
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # 状态维度：
        # - OHLCV × window_size
        # - 技术指标 (ma_ratio, rsi, vol_ratio)
        # - 账户状态 (position_ratio, balance_ratio)
        self.obs_dim = window_size * 5 + 3 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.reset()

    def _compute_indicators(self, idx):
        """计算技术指标"""
        start = max(0, idx - 30)
        window = self.df.iloc[start:idx+1]
        close = window['close'].values.astype(np.float64)
        volume = window['volume'].values.astype(np.float64)

        if len(close) < 20:
            return np.array([1.0, 50.0, 1.0], dtype=np.float32)                 #三个数字对应 ma_ratio, rsi, vol_ratio

        ma5 = np.mean(close[-5:])
        ma20 = np.mean(close[-20:])
        ma_ratio = ma5 / (ma20 + 1e-8)

        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain[:])
        avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss[:])
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        vol_today = volume[-1]
        vol_avg_10 = np.mean(volume[-10:])
        vol_ratio = vol_today / (vol_avg_10 + 1e-8)

        return np.array([ma_ratio, rsi, vol_ratio], dtype=np.float32)

    def _target_shares_from_ratio(self, target_ratio, price):
        """将目标仓位比例转换为最接近的100股整数倍（A股规则）"""
        if price <= 0 or target_ratio <= 0:
            return 0

        target_value = target_ratio * self.net_worth                    #target ratio是占净值的比例，
        target_shares_continuous = target_value / price                 #转换成股票的数量
        # 向下取整到100股倍数（保守执行）
        target_shares = int(target_shares_continuous // 100) * 100
        return max(0, target_shares)

    def _execute_order_by_shares(self, delta_shares, step_idx):
        """按股数执行订单，返回成交价和手续费"""
        if delta_shares == 0:
            return 0.0, 0.0

        row = self.df.iloc[step_idx]
        open_p, high_p, low_p, close_p = row[['open', 'high', 'low', 'close']].astype(float)
        volume = float(row['volume'])

        # 流动性因子
        order_value = abs(delta_shares) * close_p
        market_capacity = volume * close_p + 1e-8
        liquidity_factor = min(1.0, order_value / market_capacity)

        total_slippage = self.base_slippage * (1 + liquidity_factor)

        if delta_shares > 0:  # 买入
            exec_price = np.random.uniform(open_p, high_p)              #取开盘和最高之间的随机数值，买入
            exec_price *= (1 + total_slippage)
        else:  # 卖出
            exec_price = np.random.uniform(low_p, open_p)               #
            exec_price *= (1 - total_slippage)

        trade_value = abs(delta_shares) * exec_price
        fee = trade_value * self.fee_rate
        return exec_price, fee

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position_shares = 0  # 实际持股数（股）
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.total_fees = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        if self.current_step >= len(self.df):
            return np.zeros(self.obs_dim, dtype=np.float32)

        window = self.df.iloc[self.current_step - self.window_size : self.current_step]
        ohlcv_flat = window[['open', 'high', 'low', 'close', 'volume']].values.flatten().astype(np.float32)

        tech = self._compute_indicators(self.current_step - 1)

        current_close = self.df.iloc[self.current_step - 1]['close']
        position_value = self.position_shares * current_close
        position_ratio = position_value / self.net_worth if self.net_worth > 0 else 0.0
        balance_ratio = self.balance / self.initial_balance

        return np.concatenate([ohlcv_flat, tech, [position_ratio, balance_ratio]], dtype=np.float32)

    def step(self, action):
        target_ratio = np.clip(action[0], 0.0, 1.0)  # A股：0~1

        current_close = self.df.iloc[self.current_step]['close']
        target_shares = self._target_shares_from_ratio(target_ratio, current_close)

        delta_shares = target_shares - self.position_shares                             #delta可以是负数，表示减少，即卖出

        exec_price, fee = 0.0, 0.0
        if delta_shares != 0:
            exec_price, fee = self._execute_order_by_shares(delta_shares, self.current_step)
            self.total_fees += fee
            cash_flow = delta_shares * exec_price
            self.balance -= (cash_flow + fee)
            self.position_shares = target_shares

        self.net_worth = self.balance + self.position_shares * current_close
        reward = (self.net_worth - self.prev_net_worth) / self.initial_balance
        self.prev_net_worth = self.net_worth

        self.current_step += 1
        done = self.current_step >= len(self.df)

        obs = self._get_obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        info = {
            'net_worth': self.net_worth,
            'position_shares': self.position_shares,
            'position_ratio': (self.position_shares * current_close) / self.net_worth if self.net_worth > 0 else 0
        }
        return obs, reward, done, False, info


# ==============================
# 行为策略：均线交叉（A股版）
# ==============================
def ma_behavior_policy(obs, window_size=10, obs_per_step=5):
    closes = obs[3::obs_per_step]  # 提取所有 close
    if len(closes) < 20:
        return np.array([0.0], dtype=np.float32)
    # 5日均值
    ma5 = np.mean(closes[-5:])
    # 20日均值
    ma20 = np.mean(closes[-20:])
    if ma5 > ma20:
        return np.array([1.0], dtype=np.float32)  # 满仓
    else:
        return np.array([0.0], dtype=np.float32)  # 空仓


# ==============================
# 生成离线数据集（A股合规）
# ==============================
def generate_offline_dataset(df, output_path="stock_offline_data_a_share.npz"):
    env = StockTradingEnv(
        df=df,
        window_size=10,
        fee_rate=0.001,
        base_slippage=0.0005,
        initial_balance=1e6
    )

    obs_list, act_list, rew_list, done_list = [], [], [], []
    obs, _ = env.reset()
    done = False

    while not done:
        action = ma_behavior_policy(obs, window_size=10, obs_per_step=5)
        next_obs, reward, done, _, _ = env.step(action)

        obs_list.append(obs)
        act_list.append(action)
        rew_list.append(reward)
        done_list.append(done)

        obs = next_obs

    np.savez_compressed(
        output_path,
        observations=np.array(obs_list, dtype=np.float32),
        actions=np.array(act_list, dtype=np.float32),
        rewards=np.array(rew_list, dtype=np.float32),
        terminals=np.array(done_list, dtype=bool)
    )
    print(f"✅ A股合规离线数据集已保存至: {output_path}")
    print(f"   样本数: {len(obs_list)}")
    print(f"   动作范围: [0.0, 1.0]（禁止做空）")
    print(f"   交易单位: 100股整数倍")


# ==============================
# 示例使用
# ==============================
if __name__ == "__main__":
    # 替换为你的A股OHLCV CSV（列: open,high,low,close,volume）
    df = pd.read_csv("sz.000513_d_origin.csv",index_col='date')
    code = df.iloc[0]['code']

    df =df.replace(0,np.nan).dropna()

    df_train = df[df.index < "2024-01-01"]

    df_val = df[df.index >= "2024-01-01"]
    df_val = df_val[df_val.index < "2025-01-01"]
    df_val.to_csv(f"{code}_val.csv")
    
    df_test = df[df.index >= "2025-01-01"]
    df_test.to_csv(f"{code}_test.csv")

    print("📊 数据预览:")
    print(df.head())
    print(f"总K线数: {len(df)}")

    generate_offline_dataset(df_train, f"{code}_train.npz")
    generate_offline_dataset(df_val, f"{code}_val.npz")