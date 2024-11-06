import os
import warnings
from itertools import zip_longest

import h5py
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from tqdm import tqdm

from models.tokenizer import Tokenizer
from torch.cuda.amp import autocast
from utils import hydra_main

warnings.filterwarnings("ignore")

envs = [
    'Phoenix', 
    'Centipede', 
    'SpaceInvaders', 
    'Carnival', 
    'NameThisGame', 
    'Assault', 
]
tokenizer_ckpt_path = 'outputs/vqvae_and_world_model/2024-11-02/02-46-27/checkpoints/epoch_6_step_175806/tokenizer.pt'
num_gpus = 8


bs = 8192
traj_dir = 'dataset/downsampled/trajectory/data'
new_traj_dir = 'dataset/downsampled/trajectory/token_data'
os.makedirs(new_traj_dir, exist_ok=True)
meta_dir = 'dataset/downsampled/trajectory/meta'


def process_param_name(name: str) -> str:
    if name.startswith('module.'):
        name = name[7:]
    elif name.startswith('_orig_mod.module.'):
        name = name[17:]
    elif name.startswith('_orig_mod.'):
        name = name[10:]
    else:
        raise NotImplementedError
    return name


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


@hydra_main(config_path="../config/tokenizer", config_name="default.yaml")
def get_config(cfg):
    return cfg


def main(env, idx, cfg,):
    device = f'cuda:{idx}'
    print(f"Env: {env}; Device: {device}.")

    tokenizer = instantiate(cfg).eval()
    # load ckpt
    ckpt_token = torch.load(tokenizer_ckpt_path)
    token_dict = tokenizer.state_dict()
    for name, param in ckpt_token.items():
        token_dict[process_param_name(name)] = param
    tokenizer.load_state_dict(token_dict)
    tokenizer.to(device)

    # get the num of trajectories
    meta_path = os.path.join(meta_dir, env + '.csv')
    df = pd.read_csv(meta_path)
    episodes = df.iloc[0, -2]

    traj_path = os.path.join(traj_dir, env + '.h5')
    new_dataset = []

    for episode in tqdm(range(episodes), desc=env):
        # read trajectory data
        with h5py.File(traj_path, 'r') as dataset:
            traj = dataset[str(episode)]
            obs = traj['observations'][:]
            act = traj['actions'][:]
            rew = traj['rewards'][:]
            done = traj['terminals'][:]
        
        # encode obs into tokens with bs
        steps = len(obs)
        splits = np.array_split(np.arange(steps), int(np.ceil(steps/bs)))
        obs_tokens = []

        for split in splits:
            obs_split = torch.from_numpy(obs[split]).to(torch.float32).unsqueeze(1)/255.0
            obs_split = obs_split.to(device)
            
            with autocast(dtype=torch.bfloat16):
                with torch.no_grad():
                    obs_token = tokenizer.encode(
                        obs_split, should_preprocess=True,
                    ).tokens.cpu().numpy()
            
            obs_tokens.append(obs_token)
        
        # store new trajectory
        obs_tokens = np.concatenate(obs_tokens, axis=0)
        new_dataset.append((obs_tokens, act, rew, done))
    
    # save new dataset of this env
    with h5py.File(os.path.join(new_traj_dir, f"{env}.h5"), 'w') as f:
        for episode_idx, trajectories in enumerate(new_dataset):
            episode_group = f.create_group(str(episode_idx))

            observations = trajectories[0]
            actions = trajectories[1]
            rewards = trajectories[2]
            dones = trajectories[3]

            episode_group.create_dataset('observations', data=observations)
            episode_group.create_dataset('actions', data=actions)
            episode_group.create_dataset('rewards', data=rewards)
            episode_group.create_dataset('terminals', data=dones)
    print(f'Save {env} trajectory-level dataset in hdf5.')


if __name__ == "__main__":
    cfg = get_config()
    
    mp.set_start_method('spawn')

    for env_group in grouper(envs, num_gpus):
        processes = []
        for env in env_group:
            if env is not None:
                idx = envs.index(env) % num_gpus
                p = mp.Process(target=main, args=(env, idx, cfg))
                p.start()
                processes.append(p)
        
        for p in processes:
            p.join()