import gzip
import os
import random

import numpy as np
import pandas as pd
from download import ENVS, capitalize_game_name
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    
    
if __name__ == '__main__':
    path_format = 'original_dataset/{}/{}/replay_logs/$store$_{}_ckpt.{}.gz'

    # Get meta info of the whole dataset
    df = pd.DataFrame(columns=['Total episodes', 'Total steps', 'Steps per episode'], index=ENVS)

    for env_ in tqdm(ENVS):
        env = capitalize_game_name(env_)
        episodes, steps = 0, 0
        for index in range(1, 6):
            for epoch in range(0, 55):
                store = 'terminal'
                path = path_format.format(env, index, store, epoch)
                if os.path.exists(path):
                    with gzip.open(path, 'rb') as f:
                        done = np.load(f, allow_pickle=False)
                    if not np.all(np.logical_or(done == 0, done == 1)):
                        print(f"Warning: Env: {env}, Index: {index}, Epoch: {epoch}, Valid Nums: {np.logical_not(np.logical_or(done == 0, done == 1)).sum()}, Valid Done: {done[np.where(np.logical_not(np.logical_or(done == 0, done == 1)))[0]]}")
                    # Note that `done` in Google Atari is not always 0 or 1, sometimes happened at the end of the file!
                    done = np.clip(done, 0, 1)
                    episodes += done.sum()
                    steps += done.size
        df.loc[env_] = [episodes, steps, steps/episodes]

    # df.to_csv('Google_Atari_Dataset_Info.csv')


    # Sample 10e6 steps per env
    cumu_episodes = np.zeros((len(ENVS), 5, 54), dtype=int)
    sample_episodes_num = np.ceil(10e6 / df.loc[:, 'Steps per episode'].values).astype(int)
    sample_episodes = []
    set_seed(0)

    for index_env, env_ in enumerate(tqdm(ENVS)):
        sample_episodes_this_env = np.random.randint(0, df.loc[env_, 'Total episodes'], sample_episodes_num[index_env])  # mistake: randint -> choice, but faster
        sample_episodes.append(sample_episodes_this_env)
        
        env = capitalize_game_name(env_)
        for index_seed, index in enumerate(range(1, 6)):
            for index_epoch, epoch in enumerate(range(54)):
                store = 'terminal'
                path = path_format.format(env, index, store, epoch)
                # compute episodes in this epoch
                if os.path.exists(path):
                    with gzip.open(path, 'rb') as f:
                        done = np.load(f, allow_pickle=False)
                    done = np.clip(done, 0, 1)
                    episodes = done.sum()
                else:
                    episodes = 0
                
                # compute cumulative episodes
                if index_epoch == 0:
                    if index_seed == 0:
                        cumu_episodes[index_env, index_seed, index_epoch] = episodes
                    else:
                        cumu_episodes[index_env, index_seed, index_epoch] = cumu_episodes[index_env, index_seed-1, -1] + episodes
                else:
                    cumu_episodes[index_env, index_seed, index_epoch] = cumu_episodes[index_env, index_seed, index_epoch-1] + episodes

    max_length = max(len(arr) for arr in sample_episodes)
    sample_episodes_array = np.full((len(sample_episodes), max_length), -1)
    for i, arr in enumerate(sample_episodes):
        sample_episodes_array[i, :len(arr)] = arr
    sample_episodes_array = np.sort(sample_episodes_array, axis=1)

    # np.save('sample_episodes.npy', sample_episodes_array)  # (60, max(10e6 / steps per episode))
    # np.save('cumulative_episodes.npy', cumu_episodes)  # (60, 5, 54)

    # sample_episodes_array = np.load('sample_episodes.npy')
    # cumu_episodes = np.load('cumulative_episodes.npy')

    SAVE_DIR = './dataset_csv/'
    os.makedirs(SAVE_DIR, exist_ok=True)

    def main(i):
        episode_index_in_dataset = 0
        df = pd.DataFrame(columns=[
            'Episode index in dataset', 
            'Seed', 
            'Epoch', 
            'Start index', 
            'End index', 
            'Steps'
        ])
        
        for j in range(sample_episodes_array.shape[1]):
            if sample_episodes_array[i, j] == -1:
                continue
            else:
                find_flag = False
                for k in range(5):
                    for l in range(54):
                        if cumu_episodes[i, k, l] > sample_episodes_array[i, j]:
                            find_flag = True
                            break
                    if find_flag:
                        break
                
                if not find_flag:
                    print(f"ERROR!!! Env: {ENVS[i]}, Episode: {sample_episodes_array[i, j]}")
                    continue
                else:
                    # find indices in terminal
                    if l == 0:
                        if k == 0:
                            start_index = 0
                        else:
                            start_index = cumu_episodes[i, k-1, -1]
                    else:
                        start_index = cumu_episodes[i, k, l-1]
                    delta_index = sample_episodes_array[i, j] - start_index
                    path = path_format.format(capitalize_game_name(ENVS[i]), k+1, 'terminal', l)
                    with gzip.open(path, 'rb') as f:
                        done = np.load(f, allow_pickle=False)
                    done = np.clip(done, 0, 1)
                    where_terminated = np.where(done == 1)[0]
                    sample_start_index = where_terminated[delta_index-1] + 1 if delta_index != 0 else 0
                    sample_end_index = where_terminated[delta_index]
                    df.loc[episode_index_in_dataset] = [episode_index_in_dataset, k, l, sample_start_index, sample_end_index, sample_end_index - sample_start_index + 1]
                    episode_index_in_dataset += 1
        # save df
        df.to_csv(os.path.join(
                SAVE_DIR, 
                f'{capitalize_game_name(ENVS[i])}.csv'
            ), 
            index=False,
        )

    process_map(main, range(len(ENVS)), max_workers=45, chunksize=1)