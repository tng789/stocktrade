import pandas as pd
from data_preparation import update_stock_data

import d3rlpy

from test import financial_evaluator
from enhancedtradingenv import EnhancedTradingEnv
from datetime import datetime
from pathlib import Path

import re
import tomli
def update_predictions(df, home_dir):
    df_prediction = pd.read_csv(home_dir / f"{code}.prediction.csv")

    df = df[df['date', 'code', 'open','CLOSE', 'high','low','position','total_value','cash']]
    df_prediction = pd.concat([df_prediction, df]).drop_duplicates()
    df_prediction.to_csv(home_dir / f"{code}.prediction.csv", index= False)

if __name__ == "__main__":
    today:str = datetime.now().strftime("%Y-%m-%d")

    master_list_file = Path(".") / "dataset" / "master_list.txt"

    if not master_list_file.exists():
        print("stock master list not existing, quited.")
        exit() 
    
    master_list = []
    with open(master_list_file, 'rt', encoding='utf-8') as f:
        # 去除每行末尾的换行符
        master_list =  [line.rstrip('\n') for line in f if line.strip()]

    # 股票代码列表，也是模型列表
    for line in master_list:
        print(f"{line=}")
        code, model = re.sub(r'\s+', ' ', line).split(" ")
        print(f"{code=} {model=}")

        cfg_path = Path(".") /f"{code}.toml"
        with open(cfg_path,"rb") as f:
            cfg = tomli.load(f)

        home_dir = Path(".") / "dataset" / f"{code}" 
        update_stock_data(code, cfg)
        
        df_norm = pd.read_csv(home_dir /f"{code}.norm.csv")
        dataset_size = df_norm.shape[0]

        if dataset_size < 60 or df_norm.iloc[-1]['date'] != today:
            continue

        df_test = df_norm.iloc[dataset_size-60:] 
        print(df_test.shape)
        
        env_kwargs = {
            "initial_cash": 100000,
            "rebalance_band": 0.2,
            "window_size": 60
        }   

        env =  EnhancedTradingEnv(df=df_test,mode="predict",**env_kwargs)

        algo = home_dir / model 
        cql = d3rlpy.load_learnable(algo,device='cuda:0')

        info, result_df = financial_evaluator(env, cql, in_batch=False)
        print(result_df)

        update_predictions(result_df, code, home_dir)
    

    