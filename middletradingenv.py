import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from trendstrengthdetector import TrendStrengthDetector

class MiddleTradingEnv(gym.Env):
    def __init__(
        self,
        df,
        initial_cash=100000,      # 初始资金
        commission_buy=0.0003,       # 买入佣金（如万3）
        commission_sell=0.0013,      # 卖出佣金（万3 + 印花税0.1%）
        slippage_bp=2,               # 滑点（2 bp = 0.02%）
        max_weight=1.0,            # 最大仓位比例（1.0 = 100%）
        window_size=60,              # 状态中包含过去多少根K线
        rebalance_band=0.1,         # 再平衡带宽（如 ±5%）  加到10，减少交易,相比20%似乎效果更好
        take_profit_pct=1.0,        # 止盈线（25%）
        stop_loss_pct=1.0,          # 止损线（25%）
        enable_tplus1=False,          # 是否启用 T+1 限制
        lot_size=100                 # A股最小交易单位（1手=100股）
    ):
        # self.price_series = np.array(price_series, dtype=np.float64)
        super(MiddleTradingEnv, self).__init__()
        # price_series  = np.array(df['close'], dtype=np.float64)
        # volume_series = np.array(df['volume'], dtype=np.float64)
        # turn_series   = np.array(df['turn'], dtype=np.float64)
        self.df = df
        
        self.tech_columns = ['close','volume','turn', 'ma5', 'ma20',
                        'bb_position', 'rsi', 'vol20', 'adx', 'plus_di',
                        'minus_di', 'macd', 'macd_signal', 'macd_hist']

        # df_tech = add_technical_indicators(df)
        # self.df_tech = df_tech[self.tech_columns]
        self.price_series = np.array(df['close'], dtype=np.float64)

        self.T = len(self.price_series)
        if self.T <= window_size:
            raise ValueError("price_series too short for window_size")
        
        self.initial_cash = float(initial_cash)
        self.commission_buy = commission_buy
        self.commission_sell = commission_sell
        self.slippage_bp = slippage_bp
        self.max_weight = max_weight
        self.window_size = window_size
        self.rebalance_band = rebalance_band
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.enable_tplus1 = enable_tplus1
        self.lot_size = lot_size
        
        self._state_shape = None  # 新增：缓存状态形状
        self.trend_score = 0
        self.trend_detector = TrendStrengthDetector(lookback=window_size)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)

        # 状态维度：
        # - (close price, volume, turn) × window_size
        # - account info:
        #    self.cash / self.initial_cash,         #当前现金/原始资金
        #    (self.position * self.price_series[self.t]) / self.initial_cash,       #当前股票数量*股票价值 / 原始基金
        data_dim = len(self.tech_columns)*window_size
        account_info_dim = 2 
        trend_info_dim = 1
        obs_dim = data_dim + account_info_dim + trend_info_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64) 
        
        # self.feature_matrix = feature_matrix
        # self.T = len(feature_matrix)
        self.reset()

    def reset(self, start_idx=0, episode_length=250):               # episode length 250 用于模拟数据生成
        """重置环境，可指定起始时间"""
        self.t = start_idx + self.window_size - 1
        self.start_from = self.t                                    # 当前轮是从哪一步开始的

        self.cash = self.initial_cash
        self.position = 0.0                                         # 关键参数，即仓位,股票数量
        self.todays_buys = 0.0  # 今日买入股数（用于 T+1）
        self.done = False
        self.trend_score = 0 
        self.total_value = self.initial_cash

        self.episode_length = episode_length

        obs = self._get_state()
        if self._state_shape is None:
            self._state_shape = obs.shape

        return obs 

    def _env_market_feature_normalize(self,features:pd.DataFrame):
        data_in_window = features.copy()
        normalized = pd.DataFrame()

        for column in self.tech_columns:

            if column in (["adx", "plus_di", "minus_di", "turn", "rsi"]):             #本身百分比，还原到浮点数
                normalized[column] = data_in_window[column]/100 
            elif column in (['close','ma5','ma20']):                                  #价格系列，用z-score
                # normalized[column] = self._normalize(features[column])
                normalized[column] = features[column].copy()
            elif column in (["bb_position", "v20", "macd", "macd_signal", "macd_hist"]):  #已经归一化
                normalized[column] = data_in_window[column].copy()
            elif column in (["volume"]):                                              #成交量取对数再除以10，转换到1以下。
                normalized[column] = np.log10(data_in_window[column]+ 1e-8)/10
            else:
                pass
        
        #排个序，以保证顺序
        normalized = normalized.sort_index(axis=1)
        # rows_with_nan = normalized[normalized.isna().any(axis=1)]
        # print(f"{rows_with_nan=}")
        # 转换成numpy array
        normalized_np = normalized.to_numpy()
        # 拉平返回
        return normalized_np.flatten()

    
    def _normalize(self, prices:pd.Series, clip_std:int=3):
    
        # 只对价格序列采取z-score归一化计算
        data = np.array(prices)
        mean = np.nanmean(data)
        std = np.nanstd(data)
        
        if std == 0:                    # 标准差为零，则，返回一个全为零的array？怎么理解
            return np.zeros_like(data)
        
        # 先进行Z-Score标准化
        z_scores = (data - mean) / std
        
        # 裁剪极端值
        z_scores_clipped = np.clip(z_scores, -clip_std, clip_std)
        
        prices_normalized = np.tanh(z_scores_clipped / clip_std * 2)        #这一步貌似不需要啊
        # return prices_normalized
        return z_scores_clipped
    
    def _get_state(self):
        """返回归一化状态"""
        """正常状态获取（仅在 t < T 时调用）"""
        if self.t >= self.T:
            raise RuntimeError("_get_state() called at/after episode end")        
        
        # price_window = self.price_series[self.t - self.window_size : self.t + 1]            # 一进来就是10, [0:11]，实际是0-10， 今天编号10，前面0-9
        data_in_window = self.df.iloc[self.t - self.window_size + 1 : self.t+1]                           # window size 10, [0:10]，实际是0-9， 今天作为窗口的最后一天

        # tech_in_window = self.df_tech.iloc[self.t - self.window_size + 1 : self.t+1] 
        # norm_prices = price_window / price_window[-1]                                       # 总共11天,都除以今天的价格
        # prices_normalized = self._normalize(data_in_window['close'])
        # assert data_in_window.isna().any().any(), "data has nan"
        # for c in data_in_window.columns:
            # print(f"column {c} has Nan  {data_in_window[c].isnull().any()}")

        # market_features = np.stack([prices_normalized, volumes_normalized, turns_normalized], axis=1)  # shape: (10, 3)
        market_flat = self._env_market_feature_normalize(data_in_window)

        assert not np.any(np.isnan(market_flat)), "Nan found in market flat"

        self.trend_score = self.trend_detector.compute_trend_score(data_in_window['close']) 

        total_value = self.cash + self.position * self.price_series[self.t]                 # 总价值，现金+position*昨天的价格？
        account_info = np.array([
            self.cash / self.initial_cash,
            (self.position * self.price_series[self.t]) / self.initial_cash,
            self.trend_score
        ], dtype=np.float64)

        state = np.concatenate([market_flat, account_info]).astype(np.float64)  
        return state

    def _round_to_lot(self, shares):
        """按 A 股 100 股整数取整（向下取整）"""
        return (shares // self.lot_size) * self.lot_size

    def step(self, target_weight):
        """
        target_weight: 目标仓位比例 ∈ [0, 1]
        返回: next_state, reward, done, info
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")
        
        # 用current/old表示交易、调仓前的数据， target和next表示调仓后的数据。

        # ——————1 保存调仓前的状态 ——————
        current_cash = self.cash
        current_position = self.position

        current_price = self.price_series[self.t]                   #今天传入的数据的价格
        total_value_yesterday = self.total_value                      #前一轮的计算的结果，留给这一轮的总价/净值
        # self.cash + self.position * current_price
        current_weight = (self.position * current_price) / total_value_yesterday if total_value_yesterday > 0 else 0.0
        
#        # —————— 2. 检查止盈止损 ——————
#        # 当天的股价*股票数量 + 现金，与 原始总价相比 
#        total_value_if_hold = self.cash + current_position * current_price
#
#        # 计算当前总的盈利或亏损率
#        terminated_by_condition = False
#        profit_pct = (total_value_if_hold - self.initial_cash) / self.initial_cash
#        drawdown_pct = (self.initial_cash - total_value_if_hold) / self.initial_cash
#
#        if profit_pct >= self.take_profit_pct or drawdown_pct >= self.stop_loss_pct:
#            # 强制平仓, 卖光
#            trade_cost = self.position *(self.commission_sell-self.slippage_bp / 10000)
#            self.cash += self.position * current_price - trade_cost
#
#            self.position = 0.0
#            # self.todays_buys = 0.0
#            self.total_value = self.cash
#            terminated_by_condition = True
#
#            self.done = True
#            self.t += 1 
#            eps = 1e-8
#            # reward = np.log((self.total_value + eps) / (total_value_yesterday + eps))
#            reward = np.clip((self.total_value + eps) / (total_value_yesterday + eps)*100, -10.0, 10.0)         # 原来的太小， 不合适
#
#            # 不用关心是否到了episode末尾，直接赋0
#            next_state = np.zeros(self._state_shape, dtype=np.float64)
#
#            info = {
#                "total_value": self.total_value,
#                "position": self.position,
#                "trade_cost": trade_cost,
#                "terminated_by_condition": terminated_by_condition
#            }
#
#            return next_state, reward, self.done, info
        
        # ------------------------------------------------------------
        # 不再关心止盈止亏，正常操作
        # ------------------------------------------------------------
        # —————— 1. 裁剪目标仓位 ——————
        target_weight = np.clip(target_weight, 0.0, self.max_weight)

        # —————— 2. 判断是否需要调仓（带宽机制） ——————
        if abs(target_weight - current_weight) <= self.rebalance_band:
            # 不交易
            delta_position = 0.0
            trade_cost = 0.0
        else:
            if target_weight > current_weight:          #买入
                # 计算目标持仓市值
                target_value = target_weight * total_value_yesterday
                target_position_raw = target_value / (current_price*(1+self.commission_buy)*(1+self.slippage_bp/10000))                #

                # —————— 3. 资金约束：不能买超过现金 ——————
                #当前现金可以买多少, 含手续费和滑点
                max_affordable = self.cash / (current_price*(1+self.commission_buy)*(1+self.slippage_bp/10000))
                target_position_raw = min(target_position_raw, max_affordable)

                # —————— 5. A股整手限制 ——————
                target_position = self._round_to_lot(target_position_raw)
                delta_position = target_position - self.position

                # —————— 6. 计算交易成本 ——————
                # if delta_position > 0:  # 买入, 这里有bug，要保证self.cash必须大于0，待查！
                
                trade_value = delta_position * current_price
                commission = trade_value * self.commission_buy                  # 手续费
                slippage = trade_value * (self.slippage_bp / 10000)             # 滑点
                trade_cost = commission + slippage
                # self.cash -= trade_value + trade_cost
                cash = self.cash - trade_value - trade_cost
                if cash < 0:                    #如果现金不够则不买, 仓位不变       
                    target_position = self.position 
                else:
                    # 买入，现金减少，仓位变化
                    self.cash = cash                            # 剩余的现金
                    self.position = target_position             # 持仓仓位    
            # 卖出 
            elif target_weight < current_weight:  # 卖出
                # 计算目标持仓市值
                target_value = target_weight * total_value_yesterday
                target_position_raw = target_value / current_price                  #

                # —————— 5. A股整手限制 ——————
                target_position = self._round_to_lot(target_position_raw)
                delta_position =  self.position - target_position                   #要卖出的股票数量

                # —————— 6. 计算交易成本 ——————
                trade_value = delta_position * current_price
                commission = trade_value * self.commission_sell
                slippage = trade_value * (self.slippage_bp / 10000)
                trade_cost = commission + slippage

                self.cash += trade_value - trade_cost
                self.position = target_position
                
            else:
                trade_cost = 0.0

            # 计算总价值，交易完成后的总价值，留给下一轮用。
            # total_value_today = self.cash + self.position * current_price
            assert self.position >= 0, "Position cannot be less than 0" 
            
        # —————— 10. 计算奖励（对数收益） ——————
        total_value_today = self.cash + (self.position * current_price)                 # if not terminated_by_condition else 0)
        self.total_value = total_value_today

        scale = 10 
        weight_after_transaction = (current_price * self.position)/self.total_value
        # 防止除零
        eps = 1e-8
        # reward = np.log((total_value_today + eps) / (total_value_yesterday + eps))
        daily_return_raw = (self.total_value - total_value_yesterday) / (total_value_yesterday + eps)
        # reward = np.clip(daily_return_raw*100, -10.0, 10.0)         # 原来的太小， 不合适
        daily_return = np.clip(daily_return_raw*100, -10.0, 10.0)         

        # 鼓励在趋势中保持高仓位
        trend_score = self.trend_score         #self.trend_detector.compute_trend_score(self.price_series)
        trend_bonus = scale * 0.1 * weight_after_transaction * trend_score  # trend_score 来自 detector

        # 惩罚频繁切换（减少噪音交易）
        # switch_penalty = -0.01 * abs(action - last_action)
        # 计算交易后的weight
        switch_penalty = -0.02 * scale * abs(weight_after_transaction - current_weight)

        reward = daily_return + trend_bonus + switch_penalty

        # —————— 9. 检查是否到序列末尾 ——————
        self.t += 1
        # print(f"in Env\n{self.t=} {self.episode_length=} {self.window_size=}")
        if self.t - self.start_from > self.episode_length - self.window_size - 1:
            self.done = True
            next_state = np.zeros(self._state_shape, dtype=np.float64)
        else:
            self.done = False
            next_state = self._get_state()

        # —————— 11. 准备返回 ——————
        info = {
            "total_value": self.total_value,
            "position": self.position,
            "trade_cost": trade_cost,
            "terminated_by_condition": self.done,
            "next_step":self.t
        }

        return next_state, reward, self.done, info
    