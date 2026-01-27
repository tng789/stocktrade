import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SimpleTradingEnv(gym.Env):
    def __init__(
        self,
        df,
        # price_series,                # array of closing prices (T,)
        # feature_matrix,
        initial_cash=1_000_000,      # 初始资金
        commission_buy=0.0003,       # 买入佣金（如万3）
        commission_sell=0.0013,      # 卖出佣金（万3 + 印花税0.1%）
        slippage_bp=2,               # 滑点（2 bp = 0.02%）
        max_weight=1.0,            # 最大仓位比例（1.0 = 100%）
        window_size=10,              # 状态中包含过去多少根K线,  也就是说前面10天，今天是第11天
        rebalance_band=0.05,         # 再平衡带宽（如 ±5%）  加到20，减少交易
        take_profit_pct=1.0,        # 止盈线（25%）
        stop_loss_pct=1.0,          # 止损线（25%）
        enable_tplus1=False,          # 是否启用 T+1 限制
        lot_size=100                 # A股最小交易单位（1手=100股）
    ):
        # self.price_series = np.array(price_series, dtype=np.float64)
        super(SimpleTradingEnv, self).__init__()
        price_series = np.array(df['close'], dtype=np.float64)
        self.price_series = price_series

        self.T = len(price_series)
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

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)

        # 状态维度：
        # - OHLCV × window_size
        # - 技术指标 (ma_ratio, rsi, vol_ratio)
        # - 账户状态 (position_ratio, balance_ratio)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64) 
        
        # self.feature_matrix = feature_matrix
        # self.T = len(feature_matrix)
        self.reset()

    def reset(self, start_idx=None):
        """重置环境，可指定起始时间"""
        self.t = self.window_size if start_idx is None else start_idx
        self.cash = self.initial_cash
        self.position = 0.0                                         # 关键参数，即仓位,股票数量
        self.todays_buys = 0.0  # 今日买入股数（用于 T+1）
        self.done = False
        
        self.total_value = self.initial_cash

        obs = self._get_state()
        if self._state_shape is None:
            self._state_shape = obs.shape

        return obs 

#    def _get_state(self):
#        if self.t >= self.T:
#            raise RuntimeError("Out of bounds")
#        # 账户信息仍动态计算
#        total_value = self.cash + self.position * self.price_series[self.t]
#        account_info = np.array([
#            self.cash / self.initial_cash,
#            (self.position * self.price_series[self.t]) / self.initial_cash,
#        ], dtype=np.float64)
#        
#        # 拼接预计算特征 + 账户状态
#        tech_features = self.feature_matrix[self.t]
#        
#        return np.concatenate([tech_features, account_info]).astype(np.float64)

    def _get_state(self):
        """返回归一化状态"""
        """正常状态获取（仅在 t < T 时调用）"""
        if self.t >= self.T:
            raise RuntimeError("_get_state() called at/after episode end")        
        
        price_window = self.price_series[self.t - self.window_size : self.t + 1]            # 一进来就是10, [0:11]，实际是0-10， 今天编号10，前面0-9
        
        norm_prices = price_window / price_window[-1]                                       # 总共11天,都除以今天的价格

        total_value = self.cash + self.position * self.price_series[self.t]                 # 总价值，现金+position*昨天的价格？

        account_info = np.array([
            self.cash / self.initial_cash,
            (self.position * self.price_series[self.t]) / self.initial_cash,
        ], dtype=np.float64)

        return np.concatenate([norm_prices[:-1], account_info]).astype(np.float64)  

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
        
        # —————— 2. 检查止盈止损 ——————
        # 当天的股价*股票数量 + 现金，与 原始总价相比 
        total_value_if_hold = self.cash + current_position * current_price

        # 计算当前总的盈利或亏损率
        terminated_by_condition = False
        profit_pct = (total_value_if_hold - self.initial_cash) / self.initial_cash
        drawdown_pct = (self.initial_cash - total_value_if_hold) / self.initial_cash

        if profit_pct >= self.take_profit_pct or drawdown_pct >= self.stop_loss_pct:
            # 强制平仓, 卖光
            trade_cost = self.position *(self.commission_sell-self.slippage_bp / 10000)
            self.cash += self.position * current_price - trade_cost

            self.position = 0.0
            # self.todays_buys = 0.0
            self.total_value = self.cash
            terminated_by_condition = True

            self.done = True
            self.t += 1 
            eps = 1e-8
            # reward = np.log((self.total_value + eps) / (total_value_yesterday + eps))
            reward = np.clip((self.total_value + eps) / (total_value_yesterday + eps)*100, -10.0, 10.0)         # 原来的太小， 不合适

            # 不用关心是否到了episode末尾，直接赋0
            next_state = np.zeros(self._state_shape, dtype=np.float64)

            info = {
                "total_value": self.total_value,
                "position": self.position,
                "trade_cost": trade_cost,
                "terminated_by_condition": terminated_by_condition
            }

            return next_state, reward, self.done, info
        
        # ------------------------------------------------------------
        # 没有达到止盈止亏，正常操作
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

        # 防止除零
        eps = 1e-8
        # reward = np.log((total_value_today + eps) / (total_value_yesterday + eps))
        daily_return = (self.total_value - total_value_yesterday) / (total_value_yesterday + eps)
        reward = np.clip(daily_return*100, -10.0, 10.0)         # 原来的太小， 不合适

        # —————— 9. 检查是否到序列末尾 ——————
        self.t += 1
        if self.t >= self.T:
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
            "terminated_by_condition": terminated_by_condition
        }

        return next_state, reward, self.done, info
    
        # —————— 7. 推进时间 ——————
#     self.t += 1
#        if self.t >= self.T:
#            self.done = True
#            next_price = current_price   #或者用 last price
#        else:
#            next_price = self.price_series[self.t]
#
#        # next_price = self.price_series[self.t] if self.t < self.T else current_price
#        next_total_value = self.cash + self.position * next_price
#