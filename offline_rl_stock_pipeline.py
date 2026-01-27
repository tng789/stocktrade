# offline_rl_stock_pipeline.py
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 1. 增强版股票交易环境
# ==============================
class StockTradingEnv(gym.Env):
    def __init__(self, df, window_size=10, fee_rate=0.001, base_slippage=0.0005, initial_balance=1e6):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.fee_rate = fee_rate
        self.base_slippage = base_slippage
        self.initial_balance = float(initial_balance)

        # 动作：目标仓位比例 ∈ [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 状态维度：
        # - OHLCV × window_size
        # - 技术指标 (ma_ratio, rsi, vol_ratio)
        # - 账户状态 (position, balance_ratio)
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
            return np.array([1.0, 50.0, 1.0], dtype=np.float32)

        # MA5 / MA20
        ma5 = np.mean(close[-5:])
        ma20 = np.mean(close[-20:])
        ma_ratio = ma5 / (ma20 + 1e-8)

        # RSI(14)
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain[:])
        avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss[:])
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        # Volume Ratio
        vol_today = volume[-1]
        vol_avg_10 = np.mean(volume[-10:])
        vol_ratio = vol_today / (vol_avg_10 + 1e-8)

        return np.array([ma_ratio, rsi, vol_ratio], dtype=np.float32)

    def _execute_order(self, target_pos, current_pos, step_idx):
        """更真实的订单执行"""
        delta = target_pos - current_pos
        if abs(delta) < 1e-6:
            return 0.0, 0.0

        row = self.df.iloc[step_idx]
        open_p, high_p, low_p, close_p = row[['open', 'high', 'low', 'close']].astype(float)
        volume = float(row['volume'])

        # 流动性因子：订单规模 / 市场容量
        trade_value = abs(delta) * self.net_worth
        market_capacity = volume * close_p + 1e-8
        liquidity_factor = min(1.0, trade_value / market_capacity)

        # 滑点 = 基础滑点 + 流动性惩罚
        total_slippage = self.base_slippage * (1 + liquidity_factor)

        # 随机成交价（在OHLC范围内）
        if delta > 0:  # 买入
            exec_price = np.random.uniform(open_p, high_p)
            exec_price *= (1 + total_slippage)
        else:  # 卖出
            exec_price = np.random.uniform(low_p, open_p)
            exec_price *= (1 - total_slippage)

        fee = trade_value * self.fee_rate
        return exec_price, fee

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.total_fees = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        if self.current_step >= len(self.df):
            return np.zeros(self.obs_dim, dtype=np.float32)

        # 原始OHLCV窗口
        window = self.df.iloc[self.current_step - self.window_size : self.current_step]
        ohlcv_flat = window[['open', 'high', 'low', 'close', 'volume']].values.flatten().astype(np.float32)

        # 技术指标
        tech = self._compute_indicators(self.current_step - 1)

        # 账户状态
        balance_ratio = self.balance / self.initial_balance

        return np.concatenate([ohlcv_flat, tech, [self.position, balance_ratio]], dtype=np.float32)

    def step(self, action):
        target_pos = np.clip(action[0], -1.0, 1.0)

        # 执行订单
        exec_price, fee = self._execute_order(target_pos, self.position, self.current_step)
        self.total_fees += fee
        self.balance -= fee

        # 更新持仓（简化：按总资产比例）
        self.position = target_pos

        # 计算净值（使用当前收盘价）
        current_close = self.df.iloc[self.current_step]['close']
        self.net_worth = self.balance + self.position * self.initial_balance * (current_close / self.df.iloc[0]['close'])

        # 奖励：标准化的日收益
        reward = (self.net_worth - self.prev_net_worth) / self.initial_balance
        self.prev_net_worth = self.net_worth

        self.current_step += 1
        done = self.current_step >= len(self.df)

        obs = self._get_obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        info = {'net_worth': self.net_worth, 'position': self.position}

        return obs, reward, done, False, info


# ==============================
# 2. 行为策略：均线交叉
# ==============================
def ma_behavior_policy(obs, window_size=10, obs_per_step=5):
    n = window_size
    closes = obs[3::obs_per_step]  # 提取所有 close（每5个元素第4个）
    if len(closes) < 20:
        return np.array([0.0], dtype=np.float32)
    ma5 = np.mean(closes[-5:])
    ma20 = np.mean(closes[-20:])
    if ma5 > ma20:
        return np.array([1.0], dtype=np.float32)
    elif ma5 < ma20:
        return np.array([-1.0], dtype=np.float32)
    else:
        return np.array([0.0], dtype=np.float32)


# ==============================
# 3. 生成离线数据集
# ==============================
def generate_offline_dataset(df, output_path="stock_offline_data.npz"):
    env = StockTradingEnv(
        df=df,
        window_size=10,
        fee_rate=0.001,      # 0.1% 手续费
        base_slippage=0.0005 # 0.05% 基础滑点
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

    # 转换并保存
    np.savez_compressed(
        output_path,
        observations=np.array(obs_list, dtype=np.float32),
        actions=np.array(act_list, dtype=np.float32),
        rewards=np.array(rew_list, dtype=np.float32),
        terminals=np.array(done_list, dtype=bool)
    )
    print(f"✅ 离线数据集已保存至: {output_path}")
    print(f"   样本数: {len(obs_list)}")
    print(f"   状态维度: {obs_list[0].shape}")


# ==============================
# 4. 示例：从CSV加载并生成数据
# ==============================
if __name__ == "__main__":
    # 🔸 替换为你自己的 CSV 路径
    # CSV 必须包含列: open, high, low, close, volume
    df = pd.read_csv("sz.000513_d_origin.csv", index_col='date')

    df = df.replace(0,np.nan).dropna()
    # 可选：检查数据
    print("📊 数据预览:")
    print(df.head())
    print(f"总K线数: {len(df)}")

    # 生成离线数据集
    generate_offline_dataset(df, "stock_offline_data.npz")

    # 后续可用 d3rlpy 训练：
    #   from d3rlpy.dataset import MDPDataset
    #   dataset = MDPDataset.load("stock_offline_data.npz")
    #   cql = CQL(...); cql.fit(dataset)