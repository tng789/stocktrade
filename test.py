import numpy as np
import pandas as pd
import d3rlpy

from enhancedtradingenv import EnhancedTradingEnv  # 复用你的环境

from pathlib import Path
# from datetime import datetime
import argparse

window_size = 60

#class HybridPolicyWithHysteresis:
#    '''引入 hysteresis（迟滞/滞后）机制,类似恒温器：升温到 25°C 关空调，降温到 23°C 才开 → 避免频繁开关
#        参数建议
#        市场	high_th	low_th
#        A股/美股	0.65	0.45
#    '''
#    def __init__(self, rl_model, detector, high_th=0.65, low_th=0.45):
#        self.rl_model = rl_model
#        self.detector = detector
#        self.high_th = high_th   # 进入 RL 区域的阈值
#        self.low_th = low_th     # 退出 RL 区域的阈值
#        self.in_rl_mode = False  # 当前状态
#
#    def predict(self, obs, prices):
#        score = self.detector.compute_trend_score(prices)
#        
#        if self.in_rl_mode:
#            # 当前在 RL 模式：只有 score < low_th 才退出
#            if score < self.low_th:
#                self.in_rl_mode = False
#                return 1.0  # fallback to B&H
#            else:
#                return self.rl_model.predict(obs)[0]
#        else:
#            # 当前在 B&H 模式：只有 score > high_th 才进入 RL
#            if score > self.high_th:
#                self.in_rl_mode = True
#                return self.rl_model.predict(obs)[0]
#            else:
#                return 1.0
def calculate_trend():
    pass

def hysteresis(in_rl_mode, score, high_th=0.6, low_th=0.5):
    '''主要目标是屏蔽掉中间的振荡区域，减少操作。'''
    if in_rl_mode:
        # 当前在 RL 模式：只有 score < low_th 才退出
        if score < low_th:
            in_rl_mode = False
            # return 1.0  # fallback to B&H
    else:
        # 当前在 B&H 模式：只有 score > high_th 才进入 RL
        if score > high_th:
            in_rl_mode = True

    return in_rl_mode

def update(info:list, window:int, pace:int=1):
    result =[0]*(window-1)
    for i in range(len(info)-1):
        result.append(info[i])
        result = result + [0]*(pace-1)
    result.append(info[-1]) 
    return result

# ==============================
# 2. 自定义评估函数（计算金融指标 + Buy & Hold 基准）
# ==============================
def financial_evaluator(env, algo, in_batch = True, pace = 1):
    """
    在给定 env 上运行策略，返回金融指标
    """
    days_for_test = env.df.shape[0]
    obs = env.reset(start_idx=0, data_length=days_for_test)

    total_values = []
    actions = []
    raws = []
    position = []
    cash = []
    
    df = env.df.copy()
    # high_th=0.65
    # low_th=0.45
    in_rl_mode = True

    #走过前59天历史数据，确定趋势和策略模式
    for k in range(window_size-1):
        in_rl_mode = hysteresis(in_rl_mode,env.df.iloc[k]['trend'])

    # 真正开始做predict
    for j in range(window_size-1, days_for_test, pace):

        trend_score = env.df.iloc[env.t]['trend']
        in_rl_mode = hysteresis(in_rl_mode,trend_score)

        raw_action = algo.predict(obs[None,:])[0]  # 确定性策略，shape (1,)

        # 迟滞效应，避免策略的来回切换，明确在算法模式下，才使用算法
        if in_rl_mode:          
            action = raw_action
        else:                   # 否则，fallback to B&H
            action = [1.0]                  

#        # 如果趋势强，将仓位向 1.0 拉近
#        if trend_score > 0.5:
            # action = 0.7 * raw_action + 0.3 * 1.0  # 加权平均
#            action = raw_action
#        else:
#            action = raw_action
            # action = [1]
            # action = [env.position*env.df.iloc[env.t]['CLOSE']/env.total_value]
        
        actions.append(action[0])
        raws.append(raw_action[0])
        obs, reward, done, info = env.step(action[0])
        position.append(env.position)
        cash.append(env.cash)
        total_values.append(info["total_value"])
        if done:
            break
    
        env.fforward(pace-1)                #env.step已经走了一步

    if not in_batch :
#        df['total_value']  = [0] * (window_size-1) + total_values
#        df['actions']      = [0] * (window_size-1) + [x[0] for x in actions]
#        df['cash']         = [0] * (window_size-1) + cash
#        df['position']     = [0] * (window_size-1) + position
        df['total_value']  = update(total_values,window=60,pace=pace)
        df['actions']      = update(actions,window=60,pace=pace)
        df['raw_actions']      = update(raws,window=60,pace=pace)
        df['cash']         = update(cash,window=60,pace=pace)
        df['position']     = update(position,window=60,pace=pace)

    total_values = np.array(total_values)
    returns = np.diff(total_values) / total_values[:-1]
    returns = np.nan_to_num(returns)
    
    if len(returns) == 0:
        return {"sharpe": 0, "annual_return": 0, "max_drawdown": 0}
    
    # 年化收益（假设 250 交易日）
    annual_return = np.mean(returns) * 250
    # 夏普比率（无风险利率=0）
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(250)
    # 最大回撤
    cummax = np.maximum.accumulate(total_values)
    drawdown = (cummax - total_values) / cummax
    max_drawdown = np.max(drawdown)
    
    # --- Buy & Hold 基准 ---
    bh_initial_value = env.initial_cash
    # shares = bh_initial_value / prices[0]  # 全仓买入并持有
    shares = bh_initial_value / env.price_series[window_size-1]  # 全仓买入并持有
    bh_values = shares * env.price_series
    bh_returns = np.diff(bh_values) / bh_values[:-1]
    bh_returns = np.nan_to_num(bh_returns)

    bh_annual_return = np.mean(bh_returns) * 250
    bh_sharpe = np.mean(bh_returns) / (np.std(bh_returns) + 1e-8) * np.sqrt(250)
    bh_cummax = np.maximum.accumulate(bh_values)
    bh_drawdown = (bh_cummax - bh_values) / bh_cummax
    bh_max_drawdown = np.max(bh_drawdown)

    # --- 超额收益（Alpha）---
    alpha_over_bh = annual_return - bh_annual_return

    info =  {
        "sharpe": float(sharpe),
        "annual_return": float(annual_return),
        "max_drawdown": float(max_drawdown),
        "final_value": float(total_values[-1]),
        "avg_position": float(np.mean(actions)),
        # Buy & Hold 基准
        "bh_annual_return": float(bh_annual_return),
        "bh_sharpe": float(bh_sharpe),
        "bh_max_drawdown": float(bh_max_drawdown),
        # 超额表现
        "alpha_over_bh": float(alpha_over_bh)
    }

    return info if in_batch else (info, df)
    

def make_test_dataset(df:pd.DataFrame, start_date:str, end_date:str)->pd.DataFrame:
    
    raw_data = df[df['date']>=start_date]
    raw_data = raw_data[raw_data['date']<=end_date]
    
    first_date = raw_data.iloc[0]['date']

    num = raw_data[raw_data['date']== first_date].index[0]

    if num  < window_size:
        print("requires more data as window requires 60 days")
        return pd.DataFrame()
    
    start = num - window_size + 1
    end   = num + raw_data.shape[0]

    test_dataset = df.iloc[start:end]
    test_dataset.reset_index(drop=True, inplace=True)

    return test_dataset
    # test_dataset.to_csv(f"{code}.test.csv",index=False)



def parse_opt():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--data", type=str, required=True, help="the data file")
    parser.add_argument("--model", type=str, help="the model file")
    parser.add_argument("--code", type=str, required=True, help="the code of the share/stock")
    parser.add_argument("--pace", type=int, default=1, help="the step of days for predict action")
    parser.add_argument("--start_date", type=str, required=True, help="the start date of test data")
    parser.add_argument("--end_date", type=str, help="the end date of test data if not provided")

    # parser.add_argument("--dir", type=str, default="dataset", help="the directory of dataset stored")
    opt = parser.parse_args()
    return opt


if __name__ ==  "__main__":
    opt = parse_opt()

    home_dir = Path(".") /'dataset'/f"{opt.code}"
    print("🛠️  创建测试环境...")
    df_norm = pd.read_csv(home_dir / f"{opt.code}.norm.csv", parse_dates=True)
    
    end_date_dataset = df_norm.iloc[-1]['date']
    if opt.end_date is None:                    #命令行未提供的话，去库里最后一天
        end_date = end_date_dataset
    else:                                       # 否则，取 提供的和库中最后一天 两者中小的
        end_date = min(opt.end_date,end_date_dataset)

    df = make_test_dataset(df_norm,opt.start_date,end_date)

    print(f"从 {df.iloc[0]['date']} 到 {df.iloc[-1]['date']}，总共{df.shape[0]} 天")

    env_kwargs = {
        "initial_cash": 100000,
        "commission_buy": 0.0003,
        "commission_sell": 0.0013,
        "rebalance_band": 0.2,
        # "take_profit_pct": 1.0,
        # "stop_loss_pct": 1.0,
        "window_size": 60
    }   

    env =  EnhancedTradingEnv(df=df,mode="predict",**env_kwargs)

    if opt.model is None:       # 推理
        models = sorted(list(home_dir.glob("*.d3")))
        
        results = dict()
        for model in models: 
            model_name = str(model).split('/')[-1]
            print(f"🛠️  装入模型 {model_name}")
            cql = d3rlpy.load_learnable(model,device='cuda:0')

            info = financial_evaluator(env, cql, df.shape[0],pace = opt.pace)               #第三个参数待优化
            # for k,v in info.items():
                # print(k, v)

            results.update({model_name:info})
            print(f"🛠️ 模型 {model_name} 测试结束")
        result_df = pd.DataFrame.from_dict(results).T
        # print(result_df)
        result_path = home_dir/f"{opt.code}.results.csv"
        # result_df.to_csv(result_path, index=False)
        result_df.index.name = "model"
        result_df.to_csv(result_path)
    else:
        # 装入模型
        model = home_dir / opt.model
        cql = d3rlpy.load_learnable(model,device='cuda:0')
        # 推理
        info, result_df = financial_evaluator(env, cql, in_batch=False, pace=opt.pace)               
        result_path = home_dir/f"{opt.model}.result.csv"
        result_df.iloc[window_size-1:].to_csv(result_path, index=False)
        