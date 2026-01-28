import pandas as pd
from data_preparation import update_stock_data

import d3rlpy

from test import financial_evaluator
from enhancedtradingenv import EnhancedTradingEnv
from datetiem import datetime
from pathlib import Path

def update_predictions(df, home_dir):
    df_prediction = pd.read_csv(home_dir / f"{code}.prediction.csv")

    df = df[df['date', 'code', 'open','close', 'high','low','position','total_value','cash']]
    df_prediction = pd.concat([df_prediction, df]).drop_duplicates()
    df_prediction.to_csv(home_dir / f"{code}.prediction.csv", index= False)

if __name__ == "__main__":
    today:str = datetime.now().strftime("%Y-%m-%d")

    master_list_file = Path(".") / "dataset" / "master_list.txt"

    if not master_list_file.exists():
        print("stock master list not existing, quitted.")
        exit() 
    
    master_list = []
    with open(master_list_file, 'rt', encoding='utf-8') as f:
        # 去除每行末尾的换行符
        master_list =  [line.rstrip('\n') for line in f if line.strip()]

    # 股票代码列表，也是模型列表
    for line in master_list:
        code, model = line.split(" ")
        home_dir = Path(".") / "dataset" / f"{code}" 
        update_stock_data(code)
        
        df_norm = pd.read(Path(".")/"dataset"/f"{code}.norm.csv")
        if df_norm.shape[0] < 60 or df_norm.iloc[-1]['date'] != today:
            continue

        df_test = df_norm.iloc[-60:] 

        env_kwargs = {
            "initial_cash": 100000,
            "rebalance_band": 0.2,
            "window_size": 60
        }   

        env =  EnhancedTradingEnv(df=df_test,mode="predict",**env_kwargs)

        algo = home_dir / model 
        cql = d3rlpy.load_learnable(algo,device='cuda:0')

        info, result_df = financial_evaluator(env, cql, in_batch=False)

        update_predictions(result_df, code, home_dir)
    

    