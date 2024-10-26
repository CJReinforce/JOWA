import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from action_tokenizer import batch_tokenize_envs


class AtariTrajectory(Dataset):
    def __init__(self, data_path, csv_path, sequence_length, envs=None):
        self.data_path = data_path
        
        df = pd.read_csv(csv_path)
        # envs: [Assault, Atlantis, ...]
        if envs is not None:
            df = df[df['Environment'].isin(envs)].reset_index(drop=True)
        self.trajectories_ind = df
        
        self.sequence_length = sequence_length
        self.transform = ToTensor()
    
    def __len__(self):
        return len(self.trajectories_ind)      

    # get data from structured dir
    def __getitem__(self, idx):
        index_in_env, start, stop, env = self.trajectories_ind.iloc[idx]
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
            
        actions = np.load(os.path.join(trajectory_path, 'action', '0.npy'))[start:stop]
        rewards = np.load(os.path.join(trajectory_path, 'reward', '0.npy'))[start:stop]
        ends = np.clip(episode_terminal[start:stop], 0, 1)
            
        if ends.sum() <= 1:
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
                'envs': env,  # str --collate_fn--> (B,), dtype: long
            }
        else:
            print(f"Warning! episode {trajectory_path} contains more than one terminal.")
            idx = np.random.randint(0, len(self))
            return self.__getitem__(idx)

    # get data from .pkl file
    def __getitem_backup__(self, idx):
        data_path = os.path.join(self.data_path, f"{idx}.pkl")
        
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            return data
        
        except:
            print('#' * 50)
            print(f'Warning: {data_path} is incompleted!')
            idx = np.random.randint(0, len(self))
            return self.__getitem__(idx)  
    
    def sample_batch(self, batch_num_samples):
        idx = np.random.randint(0, self.__len__(), size=(batch_num_samples,))
        return collate_fn([self.__getitem__(i) for i in idx])


def collate_fn(batch_samples):
    return {k: torch.stack([sample[k] for sample in batch_samples]) \
        if k != 'envs' else \
            torch.LongTensor(batch_tokenize_envs([sample[k] for sample in batch_samples])) \
                for k in batch_samples[0]}