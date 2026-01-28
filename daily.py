from data_preparation import update_stock_data

from datetiem import datetime
from pathlib import Path

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
    for code in master_list:
        update_stock_data(code)
        result = predict(code)
        update_result(result)
    

    