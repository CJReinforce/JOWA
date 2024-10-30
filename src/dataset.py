import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from action_tokenizer import batch_tokenize_envs, tokenize_actions
from utils import capitalize_game_name


class AtariTrajectory(Dataset):
    def __init__(
        self, 
        data_dir, 
        csv_dir, 
        envs, 
        csv_suffix="_right_padding", 
        sequence_length=8
    ):
        envs = list(
            map(
                lambda env: capitalize_game_name(env), 
                envs
            )
        )

        self.traj_data = {
            env: h5py.File(os.path.join(data_dir, env + '.h5'), 'r') for env in envs
        }
        
        df = [pd.read_csv(os.path.join(csv_dir, f'{env}{csv_suffix}.csv')) for env in envs]
        self.segment_indices = pd.concat(df, ignore_index=True)
        
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.segment_indices)

    # get data from traj-level hdf5
    def __getitem__(self, idx):
        index_in_env, start, stop, env = self.segment_indices.iloc[idx]

        env = capitalize_game_name(env)
        trajectory = self.traj_data[env][str(index_in_env)]

        episode_terminal = trajectory['terminals'][:]
        episode_length = episode_terminal.shape[0]
        
        padding_length_right = max(0, stop - episode_length)
        padding_length_left = max(0, -start)
        
        def pad(x):
            pad_right = torch.nn.functional.pad(
                x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]
            ) if padding_length_right > 0 else x
            
            return torch.nn.functional.pad(
                pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]
            ) if padding_length_left > 0 else pad_right

        start = max(0, start)
        stop = min(episode_length, stop)
        
        observations = trajectory['observations'][start:stop]
        actions = trajectory['actions'][start:stop]
        rewards = trajectory['rewards'][start:stop]
        ends = np.clip(episode_terminal[start:stop], 0, 1)
        
        return {
            'observations': pad(
                torch.from_numpy(observations).to(torch.float32).unsqueeze(1)/255.0
            ),  # (L, 1, 84, 84), dtype: float32
            'actions': pad(
                torch.from_numpy(tokenize_actions(env, actions)).to(torch.long)
            ),  # (L,), dtype: long
            'rewards': pad(
                torch.from_numpy(rewards).to(torch.float32)
            ),  # (L,), dtype: float32
            'ends': pad(
                torch.from_numpy(ends).to(torch.long)
            ),  # (L,), dtype: long
            'mask_padding': torch.cat(
                (
                    torch.zeros(padding_length_left, dtype=torch.bool), 
                    torch.ones(ends.shape[0], dtype=torch.bool), 
                    torch.zeros(padding_length_right, dtype=torch.bool)
                ), 
                dim=0
            ),  # (L,), dtype: bool
            'envs': env,  # str --collate_fn--> (B,), dtype: long
        }
    
    def sample_batch(self, batch_num_samples):
        idx = np.random.randint(0, self.__len__(), size=(batch_num_samples,))
        return collate_fn([self.__getitem__(i) for i in idx])


def collate_fn(batch_samples):
    return {k: torch.stack([sample[k] for sample in batch_samples]) \
        if k != 'envs' else \
            torch.LongTensor(batch_tokenize_envs([sample[k] for sample in batch_samples])) \
                for k in batch_samples[0]}


if __name__ == "__main__":
    from time import time

    import h5py
    from tqdm import tqdm

    dataset = AtariTrajectory(
        'dataset/downsampled/trajectory/data', 
        'dataset/downsampled/segment/csv',
        envs=['Assault', 'BeamRider']
    )
    
    h5_time = []

    for idx in tqdm(range(len(dataset))):
        start_time = time()
        data = dataset.__getitem__(idx)
        end_time = time()
        h5_time.append(end_time - start_time)

        if idx == 0:
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(
                        f"Key: {key}, Value shape: {value.shape}, Value Dtype: {value.dtype}"
                    )
                else:
                    print(f"Key: {key}, Value Type: {type(value)}")
    
    print("hdf5 load time:", sum(h5_time))