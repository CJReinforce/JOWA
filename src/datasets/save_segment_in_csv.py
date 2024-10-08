import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from download import TRAIN_ENVS, capitalize_game_name

# episode index, start index, stop index (not included), environment
envs = [
    capitalize_game_name(env) for env in TRAIN_ENVS
]
save_name = 'train_envs_{}.csv'
dir_path = 'dataset_csv/'
K = 8
tau = 4

sample_from_start_dataset = []
sample_from_end_dataset = []
trajectories = 0

for env in tqdm(envs):
    df = pd.read_csv(os.path.join(dir_path, f'{env}.csv'))
    trajectories += len(df)

    for i in tqdm(range(len(df))):
        sfs_start_index = np.arange(df.loc[i, 'Start index'], df.loc[i, 'End index']+1, tau) - df.loc[i, 'Start index']
        sfs_end_index = sfs_start_index + K
        sfs_dataset_per_traj_per_env = {
            'Episode index': [i] * len(sfs_start_index),
            'Start index': sfs_start_index,
            'End index': sfs_end_index,
            'Environment': [env] * len(sfs_start_index)
        }
        sample_from_start_dataset.append(pd.DataFrame(sfs_dataset_per_traj_per_env))

        sfe_end_index = np.arange(df.loc[i, 'End index'], df.loc[i, 'Start index'], -tau) - df.loc[i, 'Start index']
        sfe_start_index = sfe_end_index - K
        sfe_dataset_per_traj_per_env = {
            'Episode index': [df.loc[i, 'Episode index in dataset']] * len(sfe_start_index),
            'Start index': sfe_start_index,
            'End index': sfe_end_index,
            'Environment': [env] * len(sfe_start_index)
        }
        sample_from_end_dataset.append(pd.DataFrame(sfe_dataset_per_traj_per_env))

sample_from_start_dataset = pd.concat(sample_from_start_dataset)
sample_from_end_dataset = pd.concat(sample_from_end_dataset)
sample_from_start_dataset.to_csv(os.path.join(dir_path, save_name.format('sample_from_start')), index=False)
sample_from_end_dataset.to_csv(os.path.join(dir_path, save_name.format('sample_from_end')), index=False)
# print(trajectories)