import pandas as pd
import numpy as np

from policies import add_technical_indicators
from trendstrengthdetector import TrendStrengthDetector

from pathlib import Path
import sys

import baostock as bs

tech_columns = ['close','volume','turn', 'ma5', 'ma20',
                        'bb_position', 'rsi', 'vol20', 'adx', 'plus_di',
                        'minus_di', 'macd', 'macd_signal', 'macd_hist']

window_size = 60

def calc_trend_score(prices:pd.Series, window_size:int=60):
    
    scores = [np.nan]*(window_size-1)
    trend_calculator = TrendStrengthDetector()
     
    for k in range(window_size-1,prices.shape[0]):
        data = prices[k-window_size+1:k+1]
        trend_score = trend_calculator.compute_trend_score(data)
        scores.append(trend_score)
    return pd.Series(scores)
    
     
def zscore(prices:pd.Series, clip_std:int=3, window_size:int = 60):

    normed_price = prices.copy()
    normed_price.iloc[:window_size] = np.nan
    # pd.Series([np.nan] * n)

    # 只对价格序列采取z-score归一化计算
    for k in range(window_size-1,prices.shape[0]):
        data = np.array(prices[k-window_size+1:k+1])
        mean = np.nanmean(data)
        std = np.nanstd(data)
    
        if std == 0:                    # 标准差为零，则，返回一个全为零的array？怎么理解
            z_scores_clipped =  np.zeros_like(data)
            # z_scores_clipped = 0 
        else:
            # 先进行Z-Score标准化
            z_scores = (data - mean) / std
    
            # 裁剪极端值
            z_scores_clipped = np.clip(z_scores, -clip_std, clip_std)
        
        # tanh_data = np.tanh(z_scores_clipped / clip_std * 2)
        
        normed_price[k] = z_scores_clipped[-1]
        # normed_price[k] = tanh_data[-1]

    return normed_price

def normalize(df:pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    
    for column in tech_columns:

        if column in (["adx", "plus_di", "minus_di", "turn", "rsi"]):             #本身百分比，还原到浮点数
            normalized[column] = df[column]/100 
        elif column in (['close','ma5','ma20']):                                  #价格系列，用z-score
            normalized[column] = zscore(df[column])
        elif column in (["bb_position", "v20", "macd", "macd_signal", "macd_hist"]):  #已经归一化
            normalized[column] = df[column]
        elif column in (["volume"]):                                              #成交量取对数再除以10，转换到1以下。
            normalized[column] = np.log10(df[column]+ 1e-8)/10
        else:
            pass
    
    #排个序，以保证顺序
    normalized = normalized.sort_index(axis=1)
    return normalized
    # rows_with_nan = normalized[normalized.isna().any(axis=1)]
    # print(f"{rows_with_nan=}")
    # 转换成numpy array
    # normalized_np = normalized.to_numpy()
    # 拉平返回
    # return normalized_np.flatten()

def get_ready(df:pd.DataFrame)->pd.DataFrame:
    
    print("add tech indicators...")
    df_ind = add_technical_indicators(df)
    # df_ind.to_csv("tech.csv", index=False)
    # 归一化
    print("normalize prices and tech indicators...")
    df_normed = normalize(df_ind)
    # 添加趋势指标
    
    print("add trend score...")
    df_normed['trend'] = calc_trend_score(df['close'])
    df_normed['CLOSE'] = df['close']
    # df_normed.to_csv("normed.csv", index=False)
    
    # 保存
    # df_normed.iloc[59:].to_csv("abcd.csv", index=False)
    # print(df.shape)
    # trend = calc_trend_score(df['close'])
    # print(trend.head(100))
    start = window_size - 1
    return df_ind.iloc[start:], df_normed.iloc[start:]


def convert_to_float(df:pd.DataFrame)->pd.DataFrame:
    df = df.replace("", 0)
    for col in df.columns:
        if col not in ['date', 'code']:
            df[col] = df[col].astype(float)
    return df


def fetch_stocks(code:str, start_date:str, end_date:str, freq = 'd')->pd.DataFrame:

    cols = ",".join(['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'turn','peTTM','psTTM','pcfNcfTTM','pbMRQ'])
    
    empty_df = pd.DataFrame()
    
    entry = bs.login()
    if entry.error_code != '0':
        print(entry.error_msg)
        bs.logout()
        return empty_df
    
    rs = bs.query_history_k_data_plus(
            code,
            cols,
            start_date = start_date, 
            end_date   = end_date, 
            frequency  = freq,
            adjustflag= "2"      #复权类型，默认不复权：3；1：后复权；2：前复权。 固定不变。
    )

    if rs.error_code != '0':
        print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
        bs.logout()
        return empty_df

    data_list = []
    while rs.next():
        # 获取一条记录，将记录合并在一起
        bs_data = rs.get_row_data()
        data_list.append(bs_data)

    df = pd.DataFrame(data_list, columns=rs.fields)
    if df.shape[0] != 0:  
        # 删去成交量为零的行，重置索引
        df = convert_to_float(df)
        df = df.replace(0,np.nan).dropna()
        df.reset_index(drop=True, inplace=True)

        df.sort_values(by=['date'], ascending=True, inplace=True)

        print("the last date of ohlcv: ", df.iloc[-1]['date'])
    bs.logout()
    return df


def main():
    datafile = Path(sys.argv[1])
    if not  datafile.exists():
        print(f"data file {datafile} not found...")
        exit(0)
    
    df = pd.read_csv(datafile, parse_dates=True)

    # 删去成交量为零的行
    df = df.replace(0,np.nan).dropna()
    
    # 重置索引
    df.reset_index(drop=True, inplace=True)
    
    # 在原ohlcv的基础上添加技术指标

    df_normed = get_ready(df) 
    df_normed.to_csv("abcd.csv")


if __name__ == "__main__":
    main()

