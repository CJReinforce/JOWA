import os
import pickle

import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm

from action_tokenizer import batch_tokenize_envs, tokenize_actions
from utils import capitalize_game_name


class AtariTrajectory(Dataset):
    def __init__(
        self, 
        data_dir, 
        csv_dir, 
        envs, 
        csv_suffix="_right_padding", 
        sequence_length=8,
    ):
        envs = list(
            map(
                lambda env: capitalize_game_name(env), 
                envs
            )
        )

        self.data_dir = data_dir
        self.sequence_length = sequence_length
        
        df = [pd.read_csv(os.path.join(csv_dir, f'{env}{csv_suffix}.csv')) for env in envs]
        self.segment_indices = pd.concat(df, ignore_index=True)
    
    def __len__(self):
        return len(self.segment_indices)

    # get data from traj-level hdf5
    def __getitem__(self, idx):
        def pad(x):
            pad_right = torch.nn.functional.pad(
                x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]
            ) if padding_length_right > 0 else x
            
            return torch.nn.functional.pad(
                pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]
            ) if padding_length_left > 0 else pad_right

        index_in_env, start, stop, env = self.segment_indices.iloc[idx]
        env = capitalize_game_name(env)
        path = os.path.join(self.data_dir, env + '.h5')

        with h5py.File(path, 'r', swmr=True) as data:
            trajectory = data[str(index_in_env)]
            # trajectory = self.traj_data[env][str(index_in_env)]

            episode_terminal = trajectory['terminals'][:]
            episode_length = episode_terminal.shape[0]
            
            padding_length_right = max(0, stop - episode_length)
            padding_length_left = max(0, -start)
            
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


class AtariTrajWithObsToken(AtariTrajectory):
    def __getitem__(self, idx):
        def pad(x):
            pad_right = torch.nn.functional.pad(
                x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]
            ) if padding_length_right > 0 else x
            
            return torch.nn.functional.pad(
                pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]
            ) if padding_length_left > 0 else pad_right

        index_in_env, start, stop, env = self.segment_indices.iloc[idx]
        env = capitalize_game_name(env)
        path = os.path.join(self.data_dir, env + '.h5')

        with h5py.File(path, 'r', swmr=True) as data:
            trajectory = data[str(index_in_env)]
            # trajectory = self.traj_data[env][str(index_in_env)]

            episode_terminal = trajectory['terminals'][:]
            episode_length = episode_terminal.shape[0]
            
            padding_length_right = max(0, stop - episode_length)
            padding_length_left = max(0, -start)
            
            start = max(0, start)
            stop = min(episode_length, stop)
            
            observations = trajectory['observations'][start:stop]
            actions = trajectory['actions'][start:stop]
            rewards = trajectory['rewards'][start:stop]
        
        ends = np.clip(episode_terminal[start:stop], 0, 1)
        
        return {
            'observations': pad(
                torch.from_numpy(observations).to(torch.long)
            ),  # (L, 36), dtype: long
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


class AtariTrajWithObsTokenInMemory(AtariTrajectory):
    def __init__(
        self, 
        data_dir, 
        csv_dir, 
        envs, 
        csv_suffix="_right_padding", 
        sequence_length=8,
        show_pbar=False,
        local_rank=0,
    ):
        super().__init__(
            data_dir, 
            csv_dir, 
            envs, 
            csv_suffix, 
            sequence_length,
        )

        envs = list(
            map(
                lambda env: capitalize_game_name(env), 
                envs
            )
        )
        self.traj_data = {}
        self.load_data_into_memory(envs, show_pbar)
        
        # if not dist.is_initialized() or local_rank == 0:
        #     self.load_data_into_memory(envs, show_pbar)
        
        # if dist.is_initialized():
        #     dist.barrier()
        #     if self.traj_data == {}:
        #         self.load_data_into_memory(envs, False)
        #     dist.barrier()

    def load_data_into_memory(self, envs, show_pbar):
        for env in tqdm(
            envs, 
            disable=not show_pbar, 
            desc='Load obs token dataset into memory',
        ):
            self.traj_data[env] = {}

            with h5py.File(
                os.path.join(self.data_dir, env + '.h5'), 
                'r', swmr=True,
            ) as f:
                for idx in f.keys():
                    self.traj_data[env][idx] = {
                        'observations': torch.from_numpy(
                            f[idx]['observations'][()]
                        ),#.share_memory_(),
                        'rewards': torch.from_numpy(
                            f[idx]['rewards'][()]
                        ),#.share_memory_(),
                        'actions': torch.from_numpy(
                            f[idx]['actions'][()]
                        ),#.share_memory_(),
                        'terminals': torch.from_numpy(
                            f[idx]['terminals'][()]
                        ),#.share_memory_(),
                    }
    
    def __getitem__(self, idx):
        def pad(x):
            pad_right = torch.nn.functional.pad(
                x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]
            ) if padding_length_right > 0 else x
            
            return torch.nn.functional.pad(
                pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]
            ) if padding_length_left > 0 else pad_right

        index_in_env, start, stop, env = self.segment_indices.iloc[idx]
        env = capitalize_game_name(env)
        path = os.path.join(self.data_dir, env + '.h5')

        trajectory = self.traj_data[env][str(index_in_env)]

        episode_terminal = trajectory['terminals']
        episode_length = episode_terminal.shape[0]
        
        padding_length_right = max(0, stop - episode_length)
        padding_length_left = max(0, -start)
        
        start = max(0, start)
        stop = min(episode_length, stop)
        
        observations = trajectory['observations'][start:stop]
        actions = trajectory['actions'][start:stop]
        rewards = trajectory['rewards'][start:stop]
        
        ends = np.clip(episode_terminal[start:stop], 0, 1)
        
        return {
            'observations': pad(observations.to(torch.long)),  # (L, 36), dtype: long
            'actions': pad(
                torch.from_numpy(tokenize_actions(env, actions.numpy())).to(torch.long)
            ),  # (L,), dtype: long
            'rewards': pad(rewards.to(torch.float32)),  # (L,), dtype: float32
            'ends': pad(ends.to(torch.long)),  # (L,), dtype: long
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


def collate_fn(batch_samples):
    return {
        k: torch.stack([sample[k] for sample in batch_samples]) \
        if k != 'envs' else \
            torch.LongTensor(
                batch_tokenize_envs(
                    [sample[k] for sample in batch_samples]
                )
            ) for k in batch_samples[0]
    }


if __name__ == "__main__":
    from time import time

    dataset = AtariTrajectory(
        'dataset/downsampled/trajectory/data', 
        'dataset/downsampled/segment/csv',
        envs=['Assault']
    )
    
    dataset_token = AtariTrajWithObsTokenInMemory(
        'dataset/downsampled/trajectory/token_data', 
        'dataset/downsampled/segment/csv',
        envs=['Assault'],
        show_pbar=True,
    )

    h5_time = []
    h5_token_time = []

    for idx in tqdm(range(len(dataset))):
        start_time = time()
        data = {} # dataset.__getitem__(idx)
        end_time = time()
        h5_time.append(end_time - start_time)

        start_time = time()
        data_token = dataset_token.__getitem__(idx)
        end_time = time()
        h5_token_time.append(end_time - start_time)

        if idx == 0:
            print('Dataset with obs images.')
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(
                        f"Key: {key}, Value shape: {value.shape}, Value Dtype: {value.dtype}, Value Range: {value.min()} ~ {value.max()}"
                    )
                else:
                    print(f"Key: {key}, Value Type: {type(value)}")

            print('Dataset with obs tokens.')
            for key, value in data_token.items():
                if isinstance(value, torch.Tensor):
                    print(
                        f"Key: {key}, Value shape: {value.shape}, Value Dtype: {value.dtype}, Value Range: {value.min()} ~ {value.max()}"
                    )
                else:
                    print(f"Key: {key}, Value Type: {type(value)}")
        
        if idx >= 1e4:
            break
    
    print("hdf5 with obs images load time:", sum(h5_time))
    print("hdf5 with obs tokens load time:", sum(h5_token_time))