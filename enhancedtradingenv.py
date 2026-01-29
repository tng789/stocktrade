import numpy as np
# import pandas as pd
# import gymnasium as gym
from gymnasium import spaces

from trendstrengthdetector import TrendStrengthDetector

# class EnhancedTradingEnv(gym.Env):
class EnhancedTradingEnv():
    def __init__(
        self,
        df,
        mode,                       # train for offline data generation, predict for validation and test...
        initial_cash=100000,        # 初始资金
        # position = 0,               # 初始的股票数量
        commission_buy=0.0003,      # 买入佣金（如万3）
        commission_sell=0.0013,     # 卖出佣金（万3 + 印花税0.1%）
        slippage_bp=2,              # 滑点（2 bp = 0.02%）
        max_weight=1.0,             # 最大仓位比例（1.0 = 100%）    #该参数废弃
        window_size=60,             # 状态中包含过去多少根K线
        rebalance_band=0.2,         # 再平衡带宽（如 ±5%）  加到20，减少交易,相比10%似乎效果更好
        # take_profit_pct=1.0,      # 止盈线（25%）
        # stop_loss_pct=1.0,        # 止损线（25%）
        # enable_tplus1=False,      # 是否启用 T+1 限制
        trend_threshold = 0.45, 
        lot_size=100                # A股最小交易单位（1手=100股）
    ):
        # self.price_series = np.array(price_series, dtype=np.float64)
        # super(EnhancedTradingEnv, self).__init__()
        # price_series  = np.array(df['close'], dtype=np.float64)
        # volume_series = np.array(df['volume'], dtype=np.float64)
        # turn_series   = np.array(df['turn'], dtype=np.float64)
        self.df = df
        self.mode = mode
        self.tech_columns = ['close','volume','turn', 'ma5', 'ma20',
                        'bb_position', 'rsi', 'vol20', 'adx', 'plus_di',
                        'minus_di', 'macd', 'macd_signal', 'macd_hist']

        # df_tech = add_technical_indicators(df)
        # self.df_tech = df_tech[self.tech_columns]
        self.price_series = np.array(df['CLOSE'], dtype=np.float64)

        self.T = len(self.price_series)
        if self.T < window_size:
            raise ValueError("price_series too short for window_size")
        
        self.initial_cash = float(initial_cash)
        self.commission_buy = commission_buy
        self.commission_sell = commission_sell
        self.slippage_bp = slippage_bp
        self.max_weight = max_weight
        self.window_size = window_size
        self.rebalance_band = rebalance_band
        # self.take_profit_pct = take_profit_pct
        # self.stop_loss_pct = stop_loss_pct
        # self.enable_tplus1 = enable_tplus1
        self.lot_size = lot_size
        
        self._state_shape = None  # 新增：缓存状态形状
        # self.position = position
        self.trend_threshold = trend_threshold
        self.trend_score = 0
        self.trend_detector = TrendStrengthDetector(lookback=window_size)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)

        returns = df['CLOSE'].pct_change().dropna() * initial_cash
        self.return_scale = np.std(returns) if len(returns) > 0 else initial_cash * 0.001
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
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float64) 
        
        # self.feature_matrix = feature_matrix
        # self.T = len(feature_matrix)
        # self.reset()

    def reset(self, start_idx=0, data_length=250):               # episode length 250 用于模拟数据生成
        """重置环境，可指定起始时间"""
        
        self.T = data_length
        
        self.t = start_idx + self.window_size - 1
        self.start_from = self.t                                    # 当前轮是从哪一步开始的

        self.cash = self.initial_cash
        self.position = 0.0                                         # 关键参数，即仓位,股票数量
        self.todays_buys = 0.0  # 今日买入股数（用于 T+1）
        self.done = False
        self.trend_score = self.df.iloc[self.t]['trend'] 
        self.total_value = self.initial_cash

        self.data_length = data_length
        # print(f"{self.data_length=}")
        
        self.operation = 0              # 0 for hold, 1 for buy, -1 for sell

        obs = self._get_state()
        if self._state_shape is None:
            self._state_shape = obs.shape

        return obs 

    def fforward(self, pace:int=1):
        self.t += pace
        if self.t > self.data_length:
            self.t = self.data_length -1

    def _get_state(self):
        """返回归一化状态"""
        """正常状态获取（仅在 t < T 时调用）"""
        # if self.t >= self.T:
            # raise RuntimeError("_get_state() called at/after episode end")        
        
        data_in_window = self.df.iloc[self.t - self.window_size + 1 : self.t+1]                           # window size 10, [0:10]，实际是0-9， 今天作为窗口的最后一天

        techs = data_in_window[self.tech_columns]
        techs = techs.sort_index(axis=1)        # 排个序，以防错位

        # self.trend_score = self.trend_detector.compute_trend_score(data_in_window['close']) 
        self.trend_score = self.df.iloc[self.t]['trend']

        data_in_np = techs.to_numpy()
        # 拉平
        flattened =  data_in_np.flatten()
        total_value = self.cash + self.position * self.price_series[self.t]                 # 总价值，现金+position*昨天的价格？
        account_info = np.array([
            self.cash / self.initial_cash,
            (self.position * self.price_series[self.t]) / self.initial_cash,
            self.trend_score
        ], dtype=np.float64)

        state = np.concatenate([flattened, account_info]).astype(np.float64)  
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
        
        # print(f"In Env, current step {self.t=}")
        # 用current/old表示交易、调仓前的数据， target和next表示调仓后的数据。

        obs = self._get_state()

        # ——————1 保存调仓前的状态 ——————
        current_cash = self.cash
        current_position = self.position

        current_price = self.price_series[self.t]                   #今天传入的数据的价格
        total_value_yesterday = self.total_value                      #前一轮的计算的结果，留给这一轮的总价/净值
        # self.cash + self.position * current_price
        current_weight = (self.position * current_price) / total_value_yesterday if total_value_yesterday > 0 else 0.0
        
        target_weight = np.clip(target_weight, 0.0, self.max_weight)
            
        delta = target_weight - current_weight              # 此delta是仓位的差，不是股票数量

        if self.df.iloc[self.t]['trend'] < self.trend_threshold:             # 趋势阈值暂定0.45
            #do not TRADE
            # delta = 0                         #用零表示不交易，训练时趋势不限制交易，而在推理时才限制。
            position_bonus = 0.1 * target_weight
        else:
            position_bonus = -0.2 * target_weight

        # —————— 2. 判断是否需要调仓（带宽机制） ——————
        # 小于带宽阈值，不交易。仓位大于0.8时，原阈值限制使得很难再提升仓位，故设定特殊条件，
        # 使得 仓位能达到0.9甚至以上 
        # 卖出时，如果目标为0，则delta不遵守rebalance规则
        special_condition1= (current_weight < target_weight and 
                                current_weight > 0.8 and 
                                delta > self.rebalance_band / 2)
        special_condition2 = (target_weight == 0)
        # 既不是上端也不是末端,遵守rebalance规则
        
        if self.mode == "predict":
            if not special_condition1 and not special_condition2:
                if abs(delta) < self.rebalance_band:
                    delta = 0
        else:
            pass


        if delta > 0 :          #买入
            # 计算目标持仓市值
            target_value = target_weight * total_value_yesterday
            target_position_raw = target_value / (current_price*(1+self.commission_buy)*(1+self.slippage_bp/10000))                #
    
            # —————— 3. 资金约束：不能买超过现金 ——————
            #当前现金可以买多少, 含手续费和滑点
            max_affordable = self.cash / (current_price*(1+self.commission_buy)*(1+self.slippage_bp/10000))
            # 不一定买得到 target_weight指定的数量，那就用手中cash买最大数量
            target_position_raw = min(target_position_raw, max_affordable + current_position)
    
            # —————— 5. A股整手限制 ——————
            # 在数据生成时，则不管整手限制
            if self.mode == "predict":
                target_position = self._round_to_lot(target_position_raw)
            else:
                target_position = target_position_raw

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
        elif delta < 0:  # 卖出
            # 计算目标持仓市值
            target_value = target_weight * total_value_yesterday
            target_position_raw = target_value / current_price                  #
    
            # —————— 5. A股整手限制 ——————
            # 预测时才使用整手限制，生成模拟数据时不用
            if self.mode == "predict":
                target_position = self._round_to_lot(target_position_raw)
            else:
                target_position = target_position_raw
                
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

        # scale = 100             #self.initial_cash * 0.001 
        weight_after_transaction = (current_price * self.position)/self.total_value
        # 防止除零
        eps = 1e-8
        # reward = np.log((total_value_today + eps) / (total_value_yesterday + eps))
        daily_return_raw = (self.total_value - total_value_yesterday) / (total_value_yesterday + eps)
        # reward = np.clip(daily_return_raw*100, -10.0, 10.0)         # 原来的太小， 不合适
        # daily_return = np.clip(daily_return_raw*100, -10.0, 10.0)     
        reward_scale = 0.0220                                   #阿里给的经验值,用于A股 
        daily_return = daily_return_raw / (reward_scale + 1e-8)  # scale to ～[-1, 1]

        # 鼓励在趋势中保持高仓位
        trend_score = self.trend_score         #self.trend_detector.compute_trend_score(self.price_series)
        # trend_bonus = weight_after_transaction * trend_score  # trend_score 来自 detector
        if daily_return_raw > 0:
            trend_bonus = 0.2 * target_weight * trend_score  # 鼓励顺势高仓
        else:
            trend_bonus = -0.1 * target_weight * trend_score  # 惩罚逆势高仓（力度可调）
        # trend_bonus = self.return_scale * 0.1 * position_bonus  # trend_score 来自 detector

        # 惩罚频繁切换（减少噪音交易）
        switch_penalty = -0.1 * abs(target_weight - current_weight)
        # 计算交易后的weight
        # switch_penalty = -0.02 * scale * abs(weight_after_transaction - current_weight)

        reward = daily_return + trend_bonus + switch_penalty
        
        # —————— 9. 检查是否到序列末尾 ——————
        # print(f"in Env\n{self.t=} {self.data_length=} {self.window_size=}")
        if self.t - self.start_from + 1  > self.data_length - self.window_size:
            self.done = True
            # next_state = np.zeros(self._state_shape, dtype=np.float64)
            next_state = obs
        else:
            self.done = False
            self.t += 1
            next_state = self._get_state()

        # —————— 11. 准备返回 ——————
        info = {
            "total_value": self.total_value,
            "position": self.position,
            "trade_cost": trade_cost,
            "terminated_by_condition": self.done,
            "next_step":self.t,
            "daily_return": daily_return,
            "trend_bonus":trend_bonus
        }

        return next_state, reward, self.done, info
    