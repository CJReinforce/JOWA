import gzip
import os
import random
from functools import partial

import h5py
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
                if os.path.exists(path):
                    with gzip.open(path, 'rb') as f:
                        done = np.load(f, allow_pickle=False)
                    done = np.clip(done, 0, 1)
                    episodes = done.sum()

                    cumu_steps_this_env += done.size
                else:
                    episodes = 0
                
                cumu_episodes_this_env += episodes
                
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
        
def main(
    envs, 
    env_choosed_indices, 
    cumu_episodes, 
    df, 
    num_sample_episodes, 
    save_dir, 
    L, 
    tau, 
    traj_path,
    meta_path,
    seg_csv_path,
    env_index,
):
    set_seed(env_index)

    # index of downsampled trajectories
    sample_episodes_this_env = sorted(np.random.randint(
        0, df.loc[envs[env_index], 'Total episodes'], num_sample_episodes[env_index]))
    
    env = envs[env_index]
    env = capitalize_game_name(env) if env[0].islower() else env
    downsampled_steps = 0

    env_traj_data = {}
    segment_right_padding_dataset = []
    segment_left_padding_dataset = []
    last_obs_path = None  # obs cache due to extensive time for reading obs gzip
    all_obs_array = None
    all_array = None
    
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

            # save segment-level dataset in CSV format
            # right padding for pretraining
            segment_start_index_right_padding = np.arange(sample_start_index, sample_end_index, tau) - sample_start_index
            segment_end_index_right_padding = segment_start_index_right_padding + L
            segment_right_padding_dataset_per_traj = {
                'Episode index': [j] * len(segment_start_index_right_padding),
                'Start index': segment_start_index_right_padding,
                'End index': segment_end_index_right_padding,
                'Environment': [env] * len(segment_start_index_right_padding)
            }
            segment_right_padding_dataset.append(pd.DataFrame(segment_right_padding_dataset_per_traj))
            
            # left padding for training involving imagination
            segment_end_index_left_padding = np.arange(sample_end_index-1, sample_start_index, -tau) - sample_start_index
            segment_start_index_left_padding = segment_end_index_left_padding - L
            segment_left_padding_dataset_per_traj = {
                'Episode index': [j] * len(segment_start_index_left_padding),
                'Start index': segment_start_index_left_padding,
                'End index': segment_end_index_left_padding,
                'Environment': [env] * len(segment_start_index_left_padding)
            }
            segment_left_padding_dataset.append(pd.DataFrame(segment_left_padding_dataset_per_traj))

            # save downsampled trajectory in structured dir
            traj_data = {}
            
            for index, content in enumerate(['observation', 'action', 'reward', 'terminal']):
                path = path_format.format(env, k, content, l)
                
                if index == 0 and path == last_obs_path:
                    array = all_obs_array[sample_start_index:sample_end_index].copy()
                elif index == 0:
                    del all_obs_array

                    with gzip.open(path, 'rb') as f:
                        all_obs_array = np.load(f, allow_pickle=False)
                        array = all_obs_array[sample_start_index:sample_end_index].copy()
                    
                    last_obs_path = path
                else:
                    with gzip.open(path, 'rb') as f:
                        all_array = np.load(f, allow_pickle=False)
                        array = all_array[sample_start_index:sample_end_index].copy()
                
                traj_data[content] = array
            env_traj_data[j] = traj_data
            del all_array

    segment_right_padding_dataset = pd.concat(segment_right_padding_dataset)
    segment_left_padding_dataset = pd.concat(segment_left_padding_dataset)

    df.loc[env, 'Downsampled steps'] = downsampled_steps

    # save traj-level dataset in hdf5
    with h5py.File(os.path.join(traj_path, f"{env}.h5"), 'w') as f:
        for episode_idx, trajectories in env_traj_data.items():
            episode_group = f.create_group(str(episode_idx))

            observations = trajectories['observation']
            actions = trajectories['action']
            rewards = trajectories['reward']
            dones = trajectories['terminal']

            episode_group.create_dataset('observations', data=observations)
            episode_group.create_dataset('actions', data=actions)
            episode_group.create_dataset('rewards', data=rewards)
            episode_group.create_dataset('terminals', data=dones)
    print(f'Save {env} trajectory-level dataset in hdf5.')

    # save meta-info of trajectory-level dataset
    df.loc[[env], :].to_csv(os.path.join(meta_path, f"{env}.csv"), index=False)

    # save segment-level dataset in CSV format
    segment_right_padding_dataset.to_csv(f"{seg_csv_path}/{env}_right_padding.csv", index=False)
    segment_left_padding_dataset.to_csv(f"{seg_csv_path}/{env}_left_padding.csv", index=False)


if __name__ == '__main__':
    path_format = 'dataset/original/{}/{}/replay_logs/$store$_{}_ckpt.{}.gz'
    save_dir = 'dataset/downsampled/'
    
    num_agents = 2  # use data from 2 agents
    num_steps_per_env = 10e6  # num of transitions per env (10M)
    num_processes = 64
    num_splits = 1  # depend on RAM size
    total_envs = TRAIN_ENVS

    total_envs = list(
        map(
            lambda env: capitalize_game_name(env) if env[0].islower() else env, 
            total_envs
        )
    )
    
    # split trajectory into segments for training 
    L = 8  # segment length
    tau = 4  # split offset

    set_seed(0)

    meta_path = os.path.join(save_dir, "trajectory/meta")
    traj_path = os.path.join(save_dir, f"trajectory/data")
    seg_csv_path = os.path.join(save_dir, "segment/csv")
    os.makedirs(meta_path, exist_ok=True)
    os.makedirs(traj_path, exist_ok=True)
    os.makedirs(seg_csv_path, exist_ok=True)

    split_envs = np.array_split(total_envs, num_splits)
    split_envs = [list(split) for split in split_envs]

    for envs in split_envs:
        env_choosed_indices, cumu_episodes, df = compute_cumulative_num_of_episodes(
            envs, num_agents)
        
        num_sample_episodes = np.ceil(
            num_steps_per_env / df.loc[:, 'Steps per episode'].values).astype(int)
        df['Downsampled episodes'] = num_sample_episodes
        
        partial_main = partial(
            main, envs, env_choosed_indices, cumu_episodes, 
            df, num_sample_episodes, save_dir, L, tau,
            traj_path, meta_path, seg_csv_path
        )

        process_map(
            partial_main, range(len(envs)), 
            max_workers=num_processes, chunksize=1,
        )
    
    # check the num of transitions
    meta_files = []
    for file in os.listdir(meta_path):
        df = pd.read_csv(os.path.join(meta_path, file))
        meta_files.append(df)
    concat_meta_file = pd.concat(meta_files, ignore_index=True)
    print(concat_meta_file)