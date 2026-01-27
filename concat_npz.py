import numpy as np
from typing import List, Dict
from gymnasium.spaces import Box
import glob
import os
import sys

from d3rlpy.dataset import MDPDataset

# def load_dataset_from_multiple_npz(paths: List[str]) -> MDPDataset:
def load_all_data_for_stock(stock_dir: str):
    """
    加载多个 .npz 文件，合并为一个 MDPDataset
    paths: ['path1.npz', 'path2.npz', ...]
    """
    observations_list = []
    actions_list = []
    rewards_list = []
    terminals_list = []

    policy_dirs = os.listdir(stock_dir)                 #当前路径下 dataset/sz.000513
    for policy_dir in policy_dirs:
        policy_name = os.path.join(stock_dir,policy_dir)                 #各策略生成的路径
        if os.path.isfile(policy_name): continue
        for npz in os.listdir(policy_name):                                          #然后采用npz文件
            if not npz.endswith(".npz"): continue
    # for path in paths:
            path = os.path.join(policy_name,npz)                                     # 使用绝对路径
            # print(f"npz 文件名：{path}")
            # data = np.load(path, allow_pickle=True)
            data = np.load(path)
            obs = data['observations']
            act = data['actions']
            rew = data['rewards']
            term = data['terminals']
    
            obs  = np.nan_to_num(obs, nan=0)
            act  = np.nan_to_num(act, nan=0)

            # has_nan = np.any(np.isnan(obs))
            
            # 确保形状匹配（假设都是 (T, *)）
            if len(obs) != len(act) or len(rew) != len(term):
                raise ValueError(f"轨迹 {path} 数据长度不一致")
    
            observations_list.append(obs)
            actions_list.append(act)
            rewards_list.append(rew)
            terminals_list.append(term)
    # 拼接所有轨迹
        observations = np.concatenate(observations_list, axis=0)
        actions = np.concatenate(actions_list, axis=0)
        rewards = np.concatenate(rewards_list, axis=0)
        terminals = np.concatenate(terminals_list, axis=0)

    # 定义动作空间（需与训练时一致）
    action_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    if actions.ndim == 1:
        actions = actions.reshape(-1,1)

    print(f"Observations shape: {observations.shape}")  # 应为 (N, 14)
    print(f"Actions shape: {actions.shape}")            # 应为 (N, 1)
    print(f"Rewards shape: {rewards.shape}")            # 应为 (N,)
    print(f"Terminals shape: {terminals.shape}")        # 应为 (N,)

    return MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        action_space=action_space,
        action_size=1
        # action_type="continuous" 
    )
    

# def load_all_data_for_stock(stock_dir: str):
#    paths = glob.glob(os.path.join(stock_dir, "*.npz"))
#    return load_dataset_from_multiple_npz(paths)

# 使用



def inspect_dataset(dataset):
    print(f"Total steps: {dataset.size()}")
    print(f"action singature: {dataset.dataset_info.action_signature}")
    # print(f"{dataset.observations.shape[0]=}")  # observations 是 numpy array，shape[0] 为总步数    
    # print(dataset.rewards.shape)       # (N,)
    # print(dataset.terminals.shape)     # (N,)
#    print(f"Action range: [{dataset.dataset_info.action_size:.3f}]")
#    print(f"Reward mean: {dataset.dataset_info.reward_signature.shape:.3f})             #, std: {dataset.rewards.std():.3f}")
#    print(f"Terminal ratio: {dataset.dataset_info.terminals.sum() / dataset.size():.3f}")
#    
if __name__ == "__main__":
    # code  = sys.argv[1]
    code = "sz.000513" 
    stock_A_dataset = load_all_data_for_stock(f"datasets/{code}")
    inspect_dataset(stock_A_dataset)
    