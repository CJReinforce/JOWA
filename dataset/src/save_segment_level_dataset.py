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


class AtariTrajectory(Dataset):
    def __init__(self, data_path, csv_path, sequence_length):
        self.data_path = data_path
        self.trajectories_ind = pd.read_csv(csv_path)
        self.sequence_length = sequence_length
        self.transform = ToTensor()
    
    def __len__(self):
        return len(self.trajectories_ind)

    def __getitem__(self, idx):
        index_in_env, start, stop, env = self.trajectories_ind.iloc[idx]
        env = capitalize_game_name(env) if env[0].islower() else env
        trajectory_path = os.path.join(self.data_path, env, str(index_in_env))
        episode_terminal = np.load(os.path.join(trajectory_path, 'terminal', '0.npy'))
        episode_length = episode_terminal.shape[0]
        
        # pad if episode_length < sequence_length
        padding_length_right = max(0, stop - episode_length)
        padding_length_left = max(0, -start)
        
        def pad(x):
            pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
            return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

        start = max(0, start)
        stop = min(episode_length, stop)
        
        # load observation with multi-threading
        def load_and_transform_image(i):
            obs = Image.open(os.path.join(trajectory_path, "observation", f"{i}.png")).convert('L')  # (84, 84)
            obs = self.transform(obs)
            return obs
        
        observations = [load_and_transform_image(i) for i in range(start, stop)]
            
        # load action, reward, terminal
        actions = np.load(os.path.join(trajectory_path, 'action', '0.npy'))[start:stop]
        rewards = np.load(os.path.join(trajectory_path, 'reward', '0.npy'))[start:stop]
        ends = np.clip(episode_terminal[start:stop], 0, 1)
        
        if ends.sum() <= 1:
            return {
                'observations': pad(torch.stack(observations).to(torch.float32)),  # (L, 1, 84, 84), dtype: float32
                'actions': pad(torch.from_numpy(tokenize_actions(env, actions)).to(torch.long)),  # (L,), dtype: long
                'rewards': pad(torch.from_numpy(rewards).to(torch.float32)),  # (L,), dtype: float32
                'ends': pad(torch.from_numpy(ends).to(torch.long)),  # (L,), dtype: uint8
                'mask_padding': torch.cat(
                    (torch.zeros(padding_length_left, dtype=torch.bool), 
                    torch.ones(ends.shape[0], dtype=torch.bool), 
                    torch.zeros(padding_length_right, dtype=torch.bool)), 
                    dim=0
                ),  # (L,), dtype: bool
                'envs': env,  # str --collate_fn--> (B,), dtype: long
            }
        else:
            print(f"Warning! episode {trajectory_path} contains more than one terminal.")
            idx = np.random.randint(0, len(self))
            return self.__getitem__(idx)


def main(dataset, idx_range):
    for i in tqdm(idx_range):
        save_dir = 'offline_segment_dataset/'
        _, _, _, env = dataset.trajectories_ind.iloc[i]
        env = capitalize_game_name(env) if env[0].islower() else env

        save_dir = os.path.join(save_dir, env)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'{i}.pkl')

        if os.path.exists(file_path):
            continue
        else:
            data = dataset.__getitem__(i)

            # pickle
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)


if __name__ == "__main__":
    train_dataset = AtariTrajectory(
        data_path='offline_dataset/', 
        csv_path='dataset_csv/train_envs_sample_from_end.csv', 
        sequence_length=8,
    )

    num_processes = 64
    idx_split = np.array_split(np.arange(len(train_dataset)), num_processes)
    # main(train_dataset, idx_split[0])
    processes = []

    for i in range(num_processes):
        p = Process(target=main, args=(train_dataset, idx_split[i]))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()