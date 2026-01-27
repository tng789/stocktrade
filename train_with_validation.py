# train_with_validation.py
import numpy as np
import pandas as pd
import d3rlpy

from d3rlpy.algos import CQL, CQLConfig
from d3rlpy.logging import TensorboardAdapterFactory
from d3rlpy.preprocessing import StandardObservationScaler, MinMaxActionScaler  #, MultiplyRewardScaler

# from simpletradingenv import SimpleTradingEnv  # 复用你的环境
from enhancedtradingenv import EnhancedTradingEnv  # 复用你的环境
# from concat_npz import load_all_data_for_stock

import tomllib
from pathlib import Path
import sys
from datetime import datetime
import json

import argparse

# from d3rlpy.metrics import TDErrorEvaluator

def create_val_env_from_ohlcv(df_val, window_size=10, fee_rate=0.001, base_slippage=0.0002, initial_balance=1e6):
    
    """为验证创建 Gym 环境（用于 evaluate_on_environment）, 已废弃，保留只为kwargs用法"""
    
#    env_kwargs = {
#        "initial_cash": 100_000,
#        "commission_buy": 0.0003,
#        "commission_sell": 0.0013,
#        "rebalance_band": 0.05,
#        "take_profit_pct": 0.25,
#        "stop_loss_pct": 0.25,
#        "enable_tplus1": False,  # 日频无需 T+1
#        "window_size": 60
#    }   
#    
    # return MiddleTradingEnv( df=df_val, **env_kwargs)
    return EnhancedTradingEnv(df=df_val)

def buy_hold(prices:pd.Series,initial_cash:int=100000, window_size:int = 60)->dict:
    
    trading_days = 250

    if len(prices) < window_size:
        raise ValueError("price_series too short for window_size")

    # 最前面59天为历史数据，第60天是第一天正式数据，与训练保持同步
    # 以第一天的正式数据全仓买入并持有, 不计佣金和滑点
    shares = initial_cash / prices[window_size-1]  
    
    bh_values = shares * prices
    bh_returns = np.diff(bh_values) / bh_values[:-1]
    bh_returns = np.nan_to_num(bh_returns)

    bh_annual_return = np.mean(bh_returns) * trading_days            #按照1年250交易日计算

    bh_sharpe = np.mean(bh_returns) / (np.std(bh_returns) + 1e-8) * np.sqrt(trading_days)

    bh_cummax = np.maximum.accumulate(bh_values)
    bh_drawdown = (bh_cummax - bh_values) / bh_cummax
    bh_max_drawdown = np.max(bh_drawdown)
    
    metrics = {
        "annual_return": bh_annual_return,
        "sharpe": bh_sharpe,
        "max_drawdown": bh_max_drawdown
    }
    return metrics

# ==============================
# 2. 自定义评估函数（计算金融指标 + Buy & Hold 基准）
# ==============================
def financial_evaluator(env, algo, data_len: int = 90):  #episode_len是每个验证集的长度
    """
    在给定 env 上运行策略，返回金融指标
    """
    window_size = 60
    if data_len < window_size:
        raise RuntimeError('validation dataset length must be greater than 60. 60 for window size') 

    # env中的df是整个验证集，而且有技术指标 且是归一化之后的
    start = 0
    total_days = env.df.shape[0]

    obs = env.reset(start_idx=start, data_length=total_days)        #data_length含window_size和实际验证数据
    print(f"{start=} {env.data_length=} {env.t=}")

    total_values = []
    actions = []
    
    for kk in range(window_size,total_days):                        #实际验证数据的长度
        # action = policy(obs.reshape(1, -1))[0]  # d3rlpy policy 输出 [1,1]
        # action = algo.predict(obs)  # 确定性策略，shape (1,)
        
        # 策略：在市场趋势大于阈值，也就是向好的情况下，利用RL策略(目标是比买入持有要好些)来精确控制，在市场趋势疲软不景气时，有三种策略：
        # 1. 不交易，冻结交易
        # 2. 清仓，持币观望，待市场好起来再动手买入跟进
        # 3. 买入持有，长期向好的情况下，眼前不明朗，吃市场beta
        if env.trend_score > env.trend_threshold:
            action = algo.predict(obs[None,:])[0]  # 确定性策略，shape (1,)
        else:
            action = [1.0]                # 这个地方我还是疑问,长期向上资产 （如：沪深300、标普500、BTC）,置为1，不看好，置0
        
        actions.append(action)
        obs, reward, done, info = env.step(action[0])
        # print(f"{kk=} {action=} {reward=} {done=} {env.t=}") 
        total_values.append(info["total_value"])
        if done:
            break
    
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

    metrics = {
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_position": float(np.mean(actions)),
        "final_value":total_values[-1]
        
    } 
    
    return metrics

# --- Buy & Hold 基准 ---
#    bh_initial_value = env.initial_cash
#    # shares = bh_initial_value / prices[0]  # 全仓买入并持有
#    shares = bh_initial_value / env.price_series[env.window_size-1]  # 全仓买入并持有
#    bh_values = shares * env.price_series
#    bh_returns = np.diff(bh_values) / bh_values[:-1]
#    bh_returns = np.nan_to_num(bh_returns)
#
#    bh_annual_return = np.mean(bh_returns) * 250
#    bh_sharpe = np.mean(bh_returns) / (np.std(bh_returns) + 1e-8) * np.sqrt(250)
#    bh_cummax = np.maximum.accumulate(bh_values)
#    bh_drawdown = (bh_cummax - bh_values) / bh_cummax
#    bh_max_drawdown = np.max(bh_drawdown)
#
#    # --- 超额收益（Alpha）---
#    alpha_over_bh = annual_return - bh_annual_return
#
#    results.append( {
#        "sharpe": float(sharpe),
#        "annual_return": float(annual_return),
#        "max_drawdown": float(max_drawdown),
#        "final_value": float(total_values[-1]),
#        "avg_position": float(np.mean(actions)),
#        # Buy & Hold 基准
#        "bh_annual_return": float(bh_annual_return),
#        "bh_sharpe": float(bh_sharpe),
#        "bh_max_drawdown": float(bh_max_drawdown),
#        # 超额表现
#        "alpha_over_bh": float(alpha_over_bh),
#    })
#
#    start += 15    
#
#    return results

def financial_evaluator_(env, algo, data_len: int = 90):  #episode_len是每个验证集的长度
    """
    在给定 env 上运行策略，返回金融指标
    """
    window_size = 60
    if data_len < window_size:
        raise RuntimeError('validation dataset length must be greater than 60. 60 for window size') 

    # env中的df是整个验证集，而且有技术指标 且是归一化之后的
    start = 0
    total = env.df.shape[0]

    results = []

    while start < total-data_len + 1:    
        # length  = total-start if total-start < data_len else data_len
        # print(f"{length=}")
    
        obs = env.reset(start_idx=start, data_length=data_len)        #data_length含window_size和实际验证数据
        print(f"{start=} {env.data_length=} {env.t=}")

        total_values = []
        actions = []
        
        for kk in range(data_len-window_size):                        #实际验证数据的长度
            # action = policy(obs.reshape(1, -1))[0]  # d3rlpy policy 输出 [1,1]
            # action = algo.predict(obs)  # 确定性策略，shape (1,)
            if env.trend_score > env.trend_threshold:

                action = algo.predict(obs[None,:])[0]  # 确定性策略，shape (1,)
                # assert action >= 0, "action must be greater than 0"
                # assert action[0] >=  0,  "Action must be greater than zero"
            else:
                action = [1]
            
            # action[0]  = np.clip(action[0], 0.0, 1.0)
            # actions.append(action.item())
            actions.append(action)
            obs, reward, done, info = env.step(action[0])
            # print(f"{kk=} {action=} {reward=} {done=} {env.t=}") 
            total_values.append(info["total_value"])
            if done:
                break
        
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
        shares = bh_initial_value / env.price_series[env.window_size-1]  # 全仓买入并持有
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

        results.append( {
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
            "alpha_over_bh": float(alpha_over_bh),
        })

        start += 15    

    return results

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--code","-c", type=str, required=True, help="the datafile in csv")
    parser.add_argument("--epochs","-e", type=int, default=60, help="continue to train for more epochs by ignoring the number in toml")
    # parser.add_argument("--yes", "-y", action='store_true', help="cotniue to generate offline data")
    opt = parser.parse_args()
    return opt

def main():
    opt = parse_opt()
    share_code = opt.code
    # ==============================
    # 1. 配置路径（请根据实际情况修改）
    # ==============================
    with open(f"{share_code}.toml", "rb") as f:
        cfg = tomllib.load(f)
    
    home_dir = Path(".") / cfg['dataset_dir']/share_code 
    home_dir.mkdir(exist_ok=True)

    val_dataset = home_dir /f"{share_code}.val.csv"    # 验证期原始 OHLCV（用于构建完整 episode 回测）
    if not Path(val_dataset).exists():
        raise FileExistsError("validation dataset not exists")

#    # ==============================
#    # 2. 加载数据集
#    # ==============================
#    print("📂 加载训练集和验证集...")
#    train_dataset = load_all_data_for_stock(TRAIN_NPZ)
#    train_dataset.dump("trading_dataset_v2.h5")

    # train_dataset = MDPDataset.load("trading_dataset_v3.h5", buffer=buffer)
    # train_dataset = d3rlpy.dataset.ReplayBuffer.load("./dataset/sz.000513/train_dataset.h5", d3rlpy.dataset.InfiniteBuffer()) 
    train_dataset_file = home_dir / f"{share_code}_train_dataset.h5"
    train_dataset = d3rlpy.dataset.ReplayBuffer.load(train_dataset_file, d3rlpy.dataset.InfiniteBuffer()) 
#    print("训练集数据的动作空间: ", train_dataset.dataset_info.action_space)
#    
    # print("Dataset size:", train_dataset.size())
    print("Dataset episode size:", len(train_dataset.episodes))
#    # print(f" - Avg episode length: {train_dataset.size()/ len(train_dataset.episodes):.1f}")
#
    all_rewards = np.concatenate([ep.rewards for ep in train_dataset.episodes])
    all_actions = np.concatenate([ep.actions for ep in train_dataset.episodes])
    all_obs     = np.concatenate([ep.observations for ep in train_dataset.episodes])
#
#    # has_negatives = np.any(all_actions < 0)
#
    print("训练集 Dataset shape (rewards):", all_rewards.shape)
    print("奖励Reward mean/std:", all_rewards.mean(), all_rewards.std())
    print(all_rewards.min(), all_rewards.max())
    # print("Dataset shape (actions):", all_actions.shape)
    print("动作/Weight Actions mean/std:", all_actions.mean(), all_actions.std())
    print(f" - Action < 0.3 ratio: {(all_actions < 0.3).mean():.4%}")
    print(f" - Action < 0.5 ratio: {(all_actions < 0.5).mean():.4%}")
    print(f" - Action > 0.7 ratio: {(all_actions > 0.7).mean():.2%}")
    print(f" - Action > 0.8 ratio: {(all_actions > 0.8).mean():.2%}")
    print(f" - Action > 0.9 ratio: {(all_actions > 0.9).mean():.2%}")
    print("环境空间Observation mean/std:", all_obs.mean(), all_obs.std())
#    
    # ==============================
    # 3. 创建验证环境（用于完整 episode 评估）
    # ==============================

    # with h5py.File('trading_dataset_v3.h5', 'r') as f:
        # train_dataset = f['dataset']
#        observations = np.array(f['observations'])
#        actions = np.array(f['actions'])
#        rewards = np.array(f['rewards'])
#        terminals = np.array(f['terminals'])  # 或 'dones'，布尔类型
#
    # 2. 创建MDPDataset
#    dataset = MDPDataset(
#        observations=observations,
#        actions=actions,
#        rewards=rewards,
#        terminals=terminals,
#        # discrete_action=False  # 根据你的动作空间设置：False为连续，True为离散
#    )

    print("🛠️  创建验证环境...")
    df_val = pd.read_csv(val_dataset)
    metrics_bh = buy_hold(df_val['CLOSE'])

    print("🛠️ 验证集数据指标：")
    print(json.dumps(metrics_bh,indent=4))

    # val_env = create_val_env_from_ohlcv( df_val, window_size=60, fee_rate=0.001, base_slippage=0.0005)
    
    val_env = EnhancedTradingEnv(df=df_val, mode="predict")

    # val_env.reset(0, df_val.shape[0])

    # 1. 定义验证回调（每轮评估一次）
    # evaluator = EnvironmentEvaluator(val_env, n_trials=10)    ## 单次 episode 验证（因是 determinstic trading） 
    #evaluators = {
    #    "environment_reward": financial_evaluator,  # 这会计算平均 Reward
        # 你依然可以保留之前的离线指标作为参考
    #} 
        # env=val_env,   # 你的验证环境（2024年数据）
    
    d3rlpy.seed(53)
    # ==============================
    # 4. 初始化 CQL 模型
    # ==============================
    # scaler = StandardObservationScaler()
    #td3_config = TD3Config(
    #    actor_learning_rate=3e-4,
    #    critic_learning_rate=3e-4,
    #    batch_size=128,
    #    observation_scaler = StandardObservationScaler(),
    #    action_scaler=MinMaxActionScaler(minimum=0.0, maximum=1.0),
    #    reward_scaler=MultiplyRewardScaler(100)
    #)
    #.create(device='cpu',enable_ddp=False)
    cql_config = CQLConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        conservative_weight=1.0,        #越低越不保守，越能探索 
        batch_size=256,
        alpha_threshold = 10.0,
        initial_alpha = 0.2,
        observation_scaler=StandardObservationScaler(),
        # action_scaler=MinMaxActionScaler(minimum=0.0, maximum=1.0),
        action_scaler=MinMaxActionScaler(),
        # reward_scaler=MultiplyRewardScaler(100)
    )

    cql = CQL(cql_config,device='cuda:0',enable_ddp=False)
    # cql = CQL(cql_config,device='cpu',enable_ddp=False)
    
    # cql.build_with_dataset(train_dataset) 

    # ==============================
    # 5. 定义验证指标：在完整验证 episode 上的总回报
    # ==============================
#    def validation_return_scorer(algo, dataset):
#        """包装 evaluate_on_environment 以适配 scorer 接口"""
#        # eval_score = evaluate_qlearning_with_environment(val_env,n_trials=10)
#        # return eval_score(algo)
#        return  evaluate_qlearning_with_environment(algo, val_env,n_trials=10)
#
#    scorers = {
#        'validation_return': validation_return_scorer,
#        # 可选：添加其他指标
#        # 'td_error': td_error_scorer,
#    }
#    
    def validation_scorer(algo, val_env):

        metrics = financial_evaluator(val_env, algo, data_len=val_env.df.shape[0] )           #, episode_len=len(val_env.price_series))
        # print(f"\n[Step {step}] Val Metrics: {metrics}")
        print("Final Validation Metrics:")
        result_str = json.dumps(metrics, indent=4)
        print(result_str)
        
        # return metrics["final_value"]  # 以净值为优化目标
        return metrics

#    def log_loss(algo, epoch, total_step, *args, **kwargs):
#        if total_step % 500 == 0:
#            critic_loss = algo.learn_info.get("critic_loss", float("nan"))
#            actor_loss = algo.learn_info.get("actor_loss", float("nan"))
#            print(f"[Step {total_step:6d}] Critic: {critic_loss:.6f} | Actor: {actor_loss:.6f}") 
            
    # ==============================
    # 5. 训练回调：每轮验证 + 保存最佳
    # ==============================
    best_score = -np.inf
    # patience = 10
    latest_model_num =0
    patience_counter = 0
    best_epoch = 0
    # best_model_path = os.path.join(MODEL_SAVE_DIR, "cql_best_val_return.d3")

    def epoch_callback(algo, epoch, total_step):
        nonlocal best_score, patience_counter, metrics_bh, best_epoch, latest_model_num
        # current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        # model_path = home_dir/f"{share_code}.{current_time}.{epoch:03d}.d3"
        print(f"{latest_model_num=}")
        model_path = home_dir/f"{share_code}.{epoch+latest_model_num:03d}.d3"

        algo.save(model_path)
        print(f"🎉 新模型保存至: {model_path}")

        return 

#        print(f"\n[Epoch {epoch}] 开始验证...")
#        
#        val_score = validation_scorer(algo, val_env)
#        
#        score = val_score['annual_return'] - metrics_bh['annual_return']
#        print(f"score of the epoch: {score}")
#        if score > best_score:
#            best_score = score
#            val_score['epoch'] = epoch
#            best_epoch = epoch
#        
#            with open(home_dir/f"{current_time}.json","wt") as f:
#                json.dump(val_score,f)
#
#            patience_counter = 0
#        else:
#            patience_counter += 1
#            if patience_counter >= patience:
#                print("⏹️ 验证回报连续下降，建议早停（当前版本无法中断，继续训练...）")
#        print(f"[Epoch {epoch}]结束。 当前最佳验证发生在第 {best_epoch} 轮，得分 Alpha {best_score:.4f}\n")

    # ==============================
    # 6. 训练 + 验证 + 保存最佳模型
    # ==============================
    print("🚀 开始训练 cql...")

    # 4. 设置训练参数
    # ==============================

    # print(f"📊 batch_size={batch_size}, dataset_size={dataset_size}")
    # print(f"🔄 n_steps_per_epoch = {n_steps_per_epoch} (≈1 full pass per epoch)")        

    # factory = TensorboardAdapterFactory(root_dir="d3rlpy_logs", experiment_name= f"{share_code}.{datetime.now().strftime("%Y%m%d")}")
    
    n_steps_per_epoch = 5000     #train_dataset.size() 返回的是episode的数量，肯定不对，用transaction的数量也不好，仍建议固定一个数字，暂取10000

    if opt.epochs is None:                          # 无epochs，就是按照toml执行
        print("epochs from cmd line not provided")
        # for epoch in range(cql_config..n_epochs):
        n_epochs = cfg['train_epochs']                  # 轮数,最佳模型往往出现在前10~30轮。
        total_steps = n_epochs * n_steps_per_epoch

        cql = CQL(cql_config,device='cuda:0',enable_ddp=False)

    else:   # 有epochs则按照命令行参数来，指在原先模型上继续训练
        print(f"epochs from cmd line {opt.epochs}")
        n_epochs = opt.epochs                  # 轮数,最佳模型往往出现在前10~30轮。
        total_steps = n_epochs * n_steps_per_epoch

        # 找到最新的那个模型...
        models = sorted(list(home_dir.glob("*.d3")))
        if len(models) >= 1:
            latest_model = str(models[-1])
            latest_model_num    = int(latest_model.split(".")[-2])
            print(f"{latest_model=} {latest_model_num=}")
            # cql.load_model(latest_model)
            cql = d3rlpy.load_learnable(latest_model,device='cuda:0')
        else:
            # 没找到
            # latest_model_num = 0

            cql = CQL(cql_config,device='cuda:0',enable_ddp=False)
            # cql = CQL(cql_config,device='cpu',enable_ddp=False)
    
    cql.build_with_dataset(train_dataset) 
    # ==============================
    learn_info = cql.fit(                  
        train_dataset,
        n_steps= total_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        show_progress= True,
        logger_adapter=TensorboardAdapterFactory(root_dir=f"d3rlpy_logs/{share_code}"),
        # logger_adapter=factory,
        epoch_callback=epoch_callback
        # callback = log_loss
    )
        
    print(learn_info)

    # print(f"✅ 训练完成。最佳验证回报: {best_score:.4f}")
    print("✅ 训练完成 ")
    # print(f"📁 最佳模型: {best_model_path}")

if __name__ == "__main__":
    main()
    


# evaluators={"td_error": TDErrorEvaluator(train_dataset)}
#class LossLogger:
#    def __call__(self, algo, epoch, total_step):
#        print(f"Step {total_step}: critic_loss={algo.actor_loss}")
#        
#def load_dataset_from_npz(path):
#    """从 .npz 加载 MDPDataset"""
#    data = np.load(path)
#    return  MDPDataset(
#        observations=data['observations'],
#        actions=data['actions'],
#        rewards=data['rewards'],
#        terminals=data['terminals'],
#        action_space=spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
#    )
#