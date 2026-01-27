import pandas as pd
import numpy as np

from pathlib import Path

from datetime import datetime
import argparse

window_size = 60
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True, help="the normilized data file in csv")
    parser.add_argument("--start_date", type=str, default="2025-01-01", help="the start date of train data")
    parser.add_argument("--end_date", type=str, default="2025-12-31", help="the end date of train data")

    # parser.add_argument("--dir", type=str, default="dataset", help="the directory of dataset stored")
    opt = parser.parse_args()
    return opt


def main()->None:
    opt = parse_opt()
    df_normed = pd.read_csv(opt.data, parse_dates=True)

    code = df_normed.iloc[0]['code']
    raw_data = df_normed[df_normed['date']>=opt.start_date]
    raw_data = raw_data[raw_data['date']<=opt.end_date]
    
    first_date = raw_data.iloc[0]['date']

    num = raw_data[raw_data['date']== first_date].index[0]

    assert num >=window_size-1, "requires more data as window requires 60 days"

    start = num - window_size + 1
    end   = num + raw_data.shape[0]

    test_dataset = df_normed.iloc[start:end]
    test_dataset.reset_index(drop=True, inplace=True)
    test_dataset.to_csv(f"{code}.test.csv",index=False)


if __name__ == "__main__":
    main()
