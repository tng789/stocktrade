import pandas as pd
import numpy as np
from data_preparation import update_stock_data
from test import hysteresis
import d3rlpy

from enhancedtradingenv import EnhancedTradingEnv
from datetime import datetime
from pathlib import Path

import re
import tomli


window_size = 60
days_in_a_year = 250
# ==============================
# 2. 自定义评估函数（计算金融指标 + Buy & Hold 基准）
# 只做一天也就是当天的数据处理，计算年收益等指标在其它函数中完成
# ==============================
def evaluator(env, algo):
    """
    在给定 env 上运行策略，返回金融指标
    """
    days_for_test = env.df.shape[0]
    obs = env.reset(start_idx=0, data_length=days_for_test)

    df = env.df.copy()
    in_rl_mode = True

    #走过前59天历史数据，确定趋势和策略模式
    for k in range(window_size-1):
        in_rl_mode = hysteresis(in_rl_mode,df.iloc[k]['trend'])

    # 真正开始做predict

    trend_score = df.iloc[-1]['trend']
    in_rl_mode = hysteresis(in_rl_mode,trend_score)

    raw_action = algo.predict(obs[None,:])[0]  # 确定性策略，shape (1,)

    # 迟滞效应，避免策略的来回切换，明确在算法模式下，才使用算法
    if in_rl_mode:          
        action = raw_action
    else:                   # 否则，fallback to B&H
        action = [1.0]                  

    # 如果趋势强，将仓位向 1.0 拉近
    #if trend_score > 0.5:
    #   # action = 0.7 * raw_action + 0.3 * 1.0  # 加权平均
    #   # action = raw_action
    #else:
    #    action = raw_action
    #   # action = [1]
    #   # action = [env.position*env.df.iloc[env.t]['CLOSE']/env.total_value]
       
    #   actions.append(action[0])
    #   raws.append(raw_action[0])
    obs, reward, done, info = env.step(action[0])
    position = env.position
    cash = env.cash
    total_value = info["total_value"]

    info =  {
        "total_value": float(total_value),
        "action": float(action[0]),
        "position": int(position),
        "cash": float(cash),
    }

    return info
    
def finance_results(df:pd.DataFrame, initial_cash:int=100000):

    if df.shape[0] <= 1 :
        print("数据量太少，不足以计算年收益率等金融指标")
        return {
            "rl_annual_return":0.0, "rl_sharpe":0.0, "rl_max_drawdown":0.0,
            "bh_annual_return":0.0, "bh_sharpe":0.0, "bh_max_drawdown":0.0,
            "alpha_over_bh":0.0}

    total_values = np.array(df['total_value'])
    returns = np.diff(total_values) / total_values[:-1]
    returns = np.nan_to_num(returns)
    
    # if len(returns) == 0:
        # return {"sharpe": 0, "annual_return": 0, "max_drawdown": 0}
    
    # 年化收益（假设 250 交易日）
    rl_annual_return = np.mean(returns) * days_in_a_year
    # 夏普比率（无风险利率=0）
    rl_sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(days_in_a_year)
    # 最大回撤
    cummax = np.maximum.accumulate(total_values)
    drawdown = (cummax - total_values) / cummax
    rl_max_drawdown = np.max(drawdown)
    
    # --- Buy & Hold 基准 ---
    bh_initial_value = initial_cash
    # shares = bh_initial_value / prices[0]  # 全仓买入并持有
    shares = bh_initial_value / df.iloc[0]['close']  # 全仓买入并持有
    bh_values = shares * df['close'] 
    bh_returns = np.diff(bh_values) / bh_values[:-1]
    bh_returns = np.nan_to_num(bh_returns)

    bh_annual_return = np.mean(bh_returns) * days_in_a_year
    bh_sharpe = np.mean(bh_returns) / (np.std(bh_returns) + 1e-8) * np.sqrt(days_in_a_year)
    bh_cummax = np.maximum.accumulate(bh_values)
    bh_drawdown = (bh_cummax - bh_values) / bh_cummax
    bh_max_drawdown = np.max(bh_drawdown)

    # --- 超额收益（Alpha）---
    alpha_over_bh = rl_annual_return - bh_annual_return

    return {
        "rl_annual_return":rl_annual_return, "rl_sharpe":rl_sharpe, "rl_max_drawdown":rl_max_drawdown,
        "bh_annual_return":bh_annual_return, "bh_sharpe":bh_sharpe, "bh_max_drawdown":bh_max_drawdown,
        "alpha_over_bh":alpha_over_bh}

def update_results(df_result:pd.DataFrame, today:str, info:dict):

    info['date'] = today
    info_list = {k: [v] for k, v in info.items()}
    df = pd.DataFrame(info_list)

    df.loc[:,'date'] = pd.to_datetime(df['date'].copy())
    df_result.loc[:,'date'] = pd.to_datetime(df_result['date'].copy())
        
    df_result = pd.merge(df_result, df, on="date",how="inner" )

    return df_result

if __name__ == "__main__":
    today:str = datetime.now().strftime("%Y-%m-%d")

    # 读取已处理的股票列表，
    # 读取用户列表，文件在dataset目录下，文件名 users.txt，格式：
    # 名称 代码1 代码2 代码3 代码4
    
    dataset_home_dir = Path(".") /'dataset'

    users_file = Path(".") / "predictions" / "users.txt"
    models_file = dataset_home_dir / "models.csv"
    if not users_file.exists():         # or not models_file.exists():
        print("master list not existing, quited.")
        exit()
    
    #-------------------------------
    # 按照模型列表更新OHLCV数据
    #-------------------------------
    df_models = pd.read_csv(models_file)       # 模型列表,格式，code, model_name
    codes = df_models['code'].tolist()
    for code in codes:
        # 根据toml文件来生成数据， toml文件应该放在dataset下面，每个股票一个目录，toml在那里。要改动。后续是不是全进数据库，再议论
        cfg_path = Path(".") /f"{code}.toml"
        with open(cfg_path,"rb") as f:
            cfg = tomli.load(f)
        # 更新数据
        update_stock_data(code, cfg)


    #-------------------------------
    # 按照用户列表的模型进行预测
    #-------------------------------
    with open(users_file, 'rt', encoding='utf-8') as f:
        # 读取用户列表
        user_master_list = [line.rstrip('\n') for line in f if line.strip()]
    
    for user in user_master_list:
        name, *stocks = re.sub(r'\s+', ' ', user).split(" ")       
    
        user_home_dir   = Path(".") / 'predictions' / name
        if not user_home_dir.exists():
            user_home_dir.mkdir(parents=True)

        for code in stocks:
            trade_history = user_home_dir / f"{code}.csv"
            
            # 取历史交易记录，供env使用
            if not trade_history.exists():      # 没有交易记录,则取默认值，或者临时输入的值
                cash = 100000   
                position = 0 
                total_value = 100000
                rebalance_band = 0.2
            else:                               # 有交易记录,则取出上一次的数值
                df = pd.read_csv(trade_history)
                cash = df.iloc[-1]['cash']
                position = df.iloc[-1]['position']
                total_value = df.iloc[-1]['total_value']
                rebalance_band = df.iloc[-1]['rebalance_band']

            # 读取基础数据，不含技术指标，作为最后的展示给用户的数据
            bs_data = pd.read_csv(dataset_home_dir /code/f"{code}.csv",parse_dates=True)

            result_file = user_home_dir / f'{code}.transactions.csv'
            if not result_file.exists():    # 第一次运行，没有结果文件
                df_result = bs_data[bs_data['date'] ==today ]
            else:                           # 运行过，有结果文件
                df_result = pd.read_csv(result_file,parse_dates=True)
        
            # 准备推理所需数据
            df_norm = pd.read_csv(dataset_home_dir /code/f"{code}.norm.csv")
            dataset_size = df_norm.shape[0]

            # 数据集必须大于历史数据窗口，60天， 且，本脚本在每日收盘数据就绪后才能运行
            if dataset_size < window_size :         #or df_norm.iloc[-1]['date'] != today:
                continue

            df_test = df_norm.iloc[dataset_size-60:]            
            
            env_kwargs = {
                "initial_cash": cash,
                "rebalance_band": rebalance_band,
                "position": position,
                # "window_size": 60
            }   

            env =  EnhancedTradingEnv(df=df_test,mode="predict",**env_kwargs)
            # 加载模型
            model = df_models.loc[df_models['code']==code, 'model'].iloc[0]
            print(f"{code=}     {df_models=}")
            
            print(f"{type(model)=}")

            cql_file = dataset_home_dir / code / model 
            print(f"{cql_file=}")
            cql = d3rlpy.load_learnable(cql_file, device='cuda:0')

            # 做出预测，并按预测操作
            info = evaluator(env, cql)
            df_result = update_results(df_result, today, info)

            # 计算金融指标
            info = finance_results(df_result)
            df_result = update_results(df_result, today, info)

            # 保存结果
            df_result.to_csv(result_file, index=False)
            
            
            #get user preference
            # make environment
            # load model
            # predict
            # save result
            # calculate finance result
            # save result





    