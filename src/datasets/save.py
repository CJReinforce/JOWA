import gzip
import os
from multiprocessing import Process

import cv2
import numpy as np
import pandas as pd
from download import ENVS, capitalize_game_name
from tqdm import tqdm

trajectory_dir = 'dataset_csv/'
path_format = 'original_dataset/{}/{}/replay_logs/$store$_{}_ckpt.{}.gz'
save_dir = 'offline_dataset/'

files = ['{}_episodes_dataset.csv'.format(capitalize_game_name(i)) for i in ENVS]
print(f"Processing {len(files)} files:", files)


def main(files):
    for file in tqdm(files):
        file_path = os.path.join(trajectory_dir, file)
        df = pd.read_csv(file_path)
        last_obs_path = None
        env = file.split('_')[0]
        
        # now_len = len(os.listdir(os.path.join(save_dir, env)))
                
        for i in tqdm(range(len(df)), desc=f"Env: {env}"):  # range(len(df)-1,-1,-1)  # tqdm(range(now_len-1, len(df))):
            start_idx = df.loc[i, 'Start index']
            end_idx = df.loc[i, 'End index']
            seed = df.loc[i, 'Seed'] + 1
            epoch = df.loc[i, 'Epoch']
            
            for index, content in enumerate(['observation', 'action', 'reward', 'terminal']):
                check_dir = os.path.join(save_dir, f"{env}/{i}/{content}/")
                if os.path.exists(check_dir) and ((index == 0 and len(os.listdir(check_dir)) >= df.loc[i, 'Steps']) or (index != 0 and len(os.listdir(check_dir)) >= 1)):
                    continue
                
                path = path_format.format(env, seed, content, epoch)
                
                if index == 0 and path == last_obs_path:
                    array = all_obs_array[start_idx:end_idx+1]
                elif index == 0:
                    with gzip.open(path, 'rb') as f:
                        all_obs_array = np.load(f, allow_pickle=False)
                        array = all_obs_array[start_idx:end_idx+1]
                else:
                    with gzip.open(path, 'rb') as f:
                        all_array = np.load(f, allow_pickle=False)
                        array = all_array[start_idx:end_idx+1]
                
                save_path = os.path.join(save_dir, f"{env}/{i}/{content}/")
                os.makedirs(save_path, exist_ok=True)
                
                if index == 0:
                    last_obs_path = path
                    for j in range(len(array)):
                        cv2.imwrite(os.path.join(save_path, f"{j}.png"), array[j])
                else:
                    np.save(os.path.join(save_path, "0.npy"), array)

            
if __name__ == "__main__":
    num_processes = len(files)
    
    processes = []
    for i in range(num_processes):
        p = Process(
            target=main, 
            args=(files[i * len(files) // num_processes:(i + 1) * len(files) // num_processes],)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # main(files)