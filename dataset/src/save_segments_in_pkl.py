import pandas as pd
from torchvision.transforms import ToTensor
from download import capitalize_game_name
import os
import numpy as np
import torch
from PIL import Image
import pickle
from torch.utils.data import Dataset
from action_tokenizer import tokenize_actions
from tqdm import tqdm
from multiprocessing import Process
from functools import partial
from tqdm.contrib.concurrent import process_map


class AtariTrajectory(Dataset):
    def __init__(self, data_path, csv_path, sequence_length=8):
        self.data_path = data_path
        self.segment_indices = pd.read_csv(csv_path)
        self.sequence_length = sequence_length
        self.transform = ToTensor()
    
    def __len__(self):
        return len(self.segment_indices)

    def __getitem__(self, idx):
        index_in_env, start, stop, env = self.segment_indices.iloc[idx]

        env = capitalize_game_name(env) if env[0].islower() else env
        trajectory_path = os.path.join(self.data_path, env, str(index_in_env))

        episode_terminal = np.load(os.path.join(trajectory_path, 'terminal', '0.npy'))
        episode_length = episode_terminal.shape[0]
        
        padding_length_right = max(0, stop - episode_length)
        padding_length_left = max(0, -start)
        
        def pad(x):
            pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
            return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

        start = max(0, start)
        stop = min(episode_length, stop)
        
        def load_and_transform_image(i):
            obs = Image.open(os.path.join(trajectory_path, "observation", f"{i}.png")).convert('L')  # (84, 84)
            obs = self.transform(obs)
            return obs
        
        observations = [load_and_transform_image(i) for i in range(start, stop)]
            
        # load action, reward, terminal
        actions = np.load(os.path.join(trajectory_path, 'action', '0.npy'))[start:stop]
        rewards = np.load(os.path.join(trajectory_path, 'reward', '0.npy'))[start:stop]
        ends = np.clip(episode_terminal[start:stop], 0, 1)
        
        return {
            'observations': pad(torch.stack(observations).to(torch.float32)),  # (L, 1, 84, 84), dtype: float32
            'actions': pad(torch.from_numpy(tokenize_actions(env, actions)).to(torch.long)),  # (L,), dtype: long
            'rewards': pad(torch.from_numpy(rewards).to(torch.float32)),  # (L,), dtype: float32
            'ends': pad(torch.from_numpy(ends).to(torch.long)),  # (L,), dtype: long
            'mask_padding': torch.cat(
                (torch.zeros(padding_length_left, dtype=torch.bool), 
                torch.ones(ends.shape[0], dtype=torch.bool), 
                torch.zeros(padding_length_right, dtype=torch.bool)), 
                dim=0
            ),  # (L,), dtype: bool
            'envs': env,  # str
        }


def main(dataset, save_dir, i):
    file_path = os.path.join(save_dir, f'{i}.pkl')
    data = dataset.__getitem__(i)

    # pickle
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    num_processes = 64
    save_dir = 'dataset/downsampled/segment/data'
    trajectory_level_data_path = 'dataset/downsampled/trajectory/data'
    segment_level_csv_path = 'dataset/downsampled/segment/csv/15_training_games_segments_right_padding.csv'

    os.makedirs(save_dir, exist_ok=True)

    dataset = AtariTrajectory(
        data_path=trajectory_level_data_path, 
        csv_path=segment_level_csv_path, 
        sequence_length=8,
    )

    partial_main = partial(main, dataset, save_dir)
    process_map(
        partial_main, np.arange(len(dataset)), 
        max_workers=num_processes, chunksize=1,
    )