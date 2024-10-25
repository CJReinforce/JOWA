import gzip
import os
import random
from functools import partial

import cv2
import numpy as np
import pandas as pd
from download import ENVS, TEST_ENVS, TRAIN_ENVS, capitalize_game_name
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    
def compute_cumulative_num_of_episodes(envs, num_agents):
    df = pd.DataFrame(columns=['Total episodes', 'Total steps', 'Steps per episode'], index=envs)
    env_choosed_indices = {}
    cumu_episodes = np.zeros((len(envs), num_agents, 50), dtype=int)

    for index_env, env_ in enumerate(tqdm(envs)):
        env = capitalize_game_name(env_) if env_[0].islower() else env_
        cumu_episodes_this_env, cumu_steps_this_env = 0, 0
        choosed_indices = sorted(np.random.choice(range(1, 6), num_agents, replace=False))  # use data from 2 agents
        env_choosed_indices[env] = choosed_indices
        
        for index_agent, index in enumerate(choosed_indices):
            for index_epoch, epoch in enumerate(range(1, 51)):
                path = path_format.format(env, index, 'terminal', epoch)
                # compute num of episodes in this epoch
                with gzip.open(path, 'rb') as f:
                    done = np.load(f, allow_pickle=False)
                done = np.clip(done, 0, 1)
                episodes = done.sum()
                
                cumu_episodes_this_env += episodes
                cumu_steps_this_env += done.size
                
                # compute cumulative index of episodes
                if index_epoch == 0:
                    if index_agent == 0:
                        cumu_episodes[index_env, index_agent, index_epoch] = episodes
                    else:
                        cumu_episodes[index_env, index_agent, index_epoch] = cumu_episodes[
                            index_env, index_agent-1, -1] + episodes
                else:
                    cumu_episodes[index_env, index_agent, index_epoch] = cumu_episodes[
                        index_env, index_agent, index_epoch-1] + episodes

        df.loc[env_] = [
            cumu_episodes_this_env, 
            cumu_steps_this_env, 
            cumu_steps_this_env/cumu_episodes_this_env
        ]
    
    return env_choosed_indices, cumu_episodes, df
        
def main(envs, env_choosed_indices, cumu_episodes, df, num_sample_episodes, env_index):
    set_seed(0)
    # index of downsampled trajectories
    sample_episodes_this_env = sorted(np.random.randint(
        0, df.loc[envs[env_index], 'Total episodes'], num_sample_episodes[env_index]))
    
    env = envs[env_index]
    env = capitalize_game_name(env) if env[0].islower() else env
    downsampled_steps = 0
    
    for j in tqdm(range(num_sample_episodes[env_index]), desc=env):
        # find the choosed episode in which agent & epoch
        find_flag = False
        for index_k, k in enumerate(env_choosed_indices[env]):
            for index_l, l in enumerate(range(1, 51)):
                if cumu_episodes[env_index, index_k, index_l] > sample_episodes_this_env[j]:
                    find_flag = True
                    break
            if find_flag:
                break
        
        if not find_flag:
            print(f"ERROR: Env: {env}, Episode: {sample_episodes_this_env[j]} not found!")
            continue
        else:
            # find the offset of choosed episode in the coresspoding agent & epoch
            if index_l == 0:
                if index_k == 0:
                    start_index = 0
                else:
                    start_index = cumu_episodes[env_index, index_k-1, -1]
            else:
                start_index = cumu_episodes[env_index, index_k, index_l-1]
            delta_index = sample_episodes_this_env[j] - start_index
            
            path = path_format.format(env, k, 'terminal', l)
            with gzip.open(path, 'rb') as f:
                done = np.load(f, allow_pickle=False)
            done = np.clip(done, 0, 1)
            
            # find the start and end index of the choosed episode in the coresspoding agent & epoch
            where_terminated = np.where(done == 1)[0]
            sample_start_index = where_terminated[delta_index-1] + 1 if delta_index != 0 else 0
            sample_end_index = where_terminated[delta_index] + 1
            downsampled_steps += sample_end_index - sample_start_index
            
            last_obs_path = None  # obs cache due to extensive time for reading obs gzip
            
            # save downsampled trajectory in structured dir
            for index, content in enumerate(['observation', 'action', 'reward', 'terminal']):
                # save_path: dataset/downsampled_dataset/trajectory/{env}/{index of trajectory in downsampled dataset}/{o,a,r,d}
                save_path = os.path.join(save_dir, f"{env}/{j}/{content}/")
                if os.path.exists(save_path) and \
                    ((index == 0 and len(os.listdir(save_path)) >= sample_end_index - sample_start_index) or \
                        (index != 0 and len(os.listdir(save_path)) >= 1)):
                    continue
                
                path = path_format.format(env, k, content, l)
                
                if index == 0 and path == last_obs_path:
                    array = all_obs_array[sample_start_index:sample_end_index]
                elif index == 0:
                    with gzip.open(path, 'rb') as f:
                        all_obs_array = np.load(f, allow_pickle=False)
                        array = all_obs_array[sample_start_index:sample_end_index]
                else:
                    with gzip.open(path, 'rb') as f:
                        all_array = np.load(f, allow_pickle=False)
                        array = all_array[sample_start_index:sample_end_index]
                
                os.makedirs(save_path, exist_ok=True)
                
                # {save_path..}/observation/{step in that trajectory}.png
                # {save_path..}/{a,r,d}/0.npy
                if index == 0:
                    last_obs_path = path
                    for i in range(len(array)):
                        cv2.imwrite(os.path.join(save_path, f"{i}.png"), array[i])
                else:
                    np.save(
                        os.path.join(save_path, "0.npy"), 
                        array if index != 3 else np.clip(array, 0, 1)
                    )
    
    return downsampled_steps


if __name__ == '__main__':
    path_format = 'dataset/original/{}/{}/replay_logs/$store$_{}_ckpt.{}.gz'
    save_dir = 'dataset/downsampled/trajectory'
    
    num_agents = 2  # use data from 2 agents
    num_steps_per_env = 10e6
    envs = TRAIN_ENVS

    set_seed(0)
    
    env_choosed_indices, cumu_episodes, df = compute_cumulative_num_of_episodes(envs, num_agents)
    
    num_sample_episodes = np.ceil(num_steps_per_env / df.loc[:, 'Steps per episode'].values).astype(int)
    df['Downsampled episodes'] = num_sample_episodes
    
    partial_main = partial(main, envs, env_choosed_indices, cumu_episodes, df, num_sample_episodes)
    results = process_map(partial_main, range(len(envs)), max_workers=64, chunksize=1)
    df['Downsampled steps'] = results
    print(df)