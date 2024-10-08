import functools
import os
import pickle
import random
import shutil
import sys
import time
import warnings
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from pprint import pprint
from typing import Dict

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm

from action_tokenizer import (
    ATARI_NUM_ACTIONS,
    GAME_NAMES,
    batch_tokenize_envs,
    tokenize_actions,
)
from agent_expand_kv_cache import Agent
from atari_env_wrapper import AtariEnvWrapper
from envs import SingleProcessEnv
from game import AgentEnv, Game
from make_reconstructions import make_reconstructions_of_trajectories
from models.world_model_all_in_one import WorldModel
from utils import capitalize_game_name, configure_optimizer, set_seed

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '32355'


class AtariTrajectory(Dataset):
    def __init__(self, data_path, csv_path, sequence_length, envs):
        self.data_path = data_path
        df = pd.read_csv(csv_path)
        self.trajectories_ind = df[df['Environment'].isin(envs)].reset_index(drop=True)
        self.sequence_length = sequence_length
        self.transform = ToTensor()
    
    def __len__(self):
        return len(self.trajectories_ind)

    def __getitem_backup__(self, idx):
        env = self.trajectories_ind.iloc[idx, -1]
        env = capitalize_game_name(env) if env[0].islower() else env
        data_path = os.path.join(self.data_path, env, f"{idx}.pkl")
        
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            return data
        
        except:
            print('#' * 50)
            print(f'Warning: {data_path} is incompleted!')
            idx = np.random.randint(0, len(self))
            return self.__getitem__(idx)        

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

    def sample_batch(self, batch_num_samples):
        idx = np.random.choice(self.__len__(), batch_num_samples, replace=False)
        return collate_fn([self.__getitem__(i) for i in idx])
    
    
class Trainer:
    def __init__(self, cfg: DictConfig, rank: int) -> None:
        dist.init_process_group(
            "nccl", 
            timeout=timedelta(seconds=7200000),  # avoid timeout when evaluating
            rank=rank, 
            world_size=cfg.training.world_size,
        )
        torch.cuda.set_device(rank)
        
        self.training_desc = 'world_model'
        if cfg.training.world_model.train_critic:
            self.training_desc += ' + critic'
        if cfg.training.tokenizer.should:
            self.training_desc = 'tokenizer + ' + self.training_desc
        
        self.is_main_process = rank == 0

        # Initialize wandb and saving dir
        if self.is_main_process:
            print(f"Train {self.training_desc} with {cfg.training.world_size} GPUs.")
            
            save_root_dir = Path(f'outputs_{cfg.common.group_name}_group/{self.training_desc.replace(" + ", "_plus_")}/{time.strftime("%Y-%m-%d/%H-%M-%S")}')
            save_root_dir.mkdir(exist_ok=False, parents=True)
            
            wandb.init(
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True,
                resume=True,
                dir=save_root_dir,
                **cfg.wandb
            )

            self.ckpt_dir = save_root_dir / 'checkpoints'
            self.media_dir = save_root_dir / 'media'
            self.reconstructions_dir = self.media_dir / 'reconstructions'

            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=False, parents=False)
            
        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
        self.start_epoch = 1
        self.device = torch.device(cfg.common.device)
        self.dtype = get_dtype(cfg.training.dtype)
        
        # tokenizer
        self.tokenizer = instantiate(cfg.tokenizer)
        self.tokenizer.to(self.device)
        self.tokenizer = torch.nn.parallel.DistributedDataParallel(self.tokenizer, device_ids=[rank])
        self.tokenizer = torch.compile(self.tokenizer, mode="max-autotune")
        
        # origin wm
        self.world_model_ori = WorldModel(
            obs_vocab_size=cfg.tokenizer.vocab_size,
            act_vocab_size=ATARI_NUM_ACTIONS,
            config_transformer=instantiate(cfg.world_model),
            config_critic=instantiate(cfg.actor_critic),
            device=self.device,
            name='origin wm',
        )
        self.world_model_ori.to(self.device)
        self.world_model_ori = torch.nn.parallel.DistributedDataParallel(self.world_model_ori, device_ids=[rank])
        self.world_model_ori = torch.compile(self.world_model_ori, mode="max-autotune")

        # 3_q_head wm
        cfg_clone = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_clone.actor_critic.num_q = 3
        cfg_clone.actor_critic.use_rem = True
        self.world_model_3_q_head = WorldModel(
            obs_vocab_size=cfg.tokenizer.vocab_size,
            act_vocab_size=ATARI_NUM_ACTIONS,
            config_transformer=instantiate(cfg_clone.world_model),
            config_critic=instantiate(cfg_clone.actor_critic),
            device=self.device,
            name='3_q_head wm',
        )
        self.world_model_3_q_head.to(self.device)
        self.world_model_3_q_head = torch.nn.parallel.DistributedDataParallel(self.world_model_3_q_head, device_ids=[rank])
        self.world_model_3_q_head = torch.compile(self.world_model_3_q_head, mode="max-autotune")

        # no_cql wm
        cfg_clone = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_clone.actor_critic.q_penalty = None
        self.world_model_no_cql = WorldModel(
            obs_vocab_size=cfg.tokenizer.vocab_size,
            act_vocab_size=ATARI_NUM_ACTIONS,
            config_transformer=instantiate(cfg_clone.world_model),
            config_critic=instantiate(cfg_clone.actor_critic),
            device=self.device,
            name='no_cql wm',
        )
        self.world_model_no_cql.to(self.device)
        self.world_model_no_cql = torch.nn.parallel.DistributedDataParallel(self.world_model_no_cql, device_ids=[rank])
        self.world_model_no_cql = torch.compile(self.world_model_no_cql, mode="max-autotune")

        # mse_loss wm
        cfg_clone = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_clone.actor_critic.td_loss = 'mse'
        self.world_model_mse = WorldModel(
            obs_vocab_size=cfg.tokenizer.vocab_size,
            act_vocab_size=ATARI_NUM_ACTIONS,
            config_transformer=instantiate(cfg_clone.world_model),
            config_critic=instantiate(cfg_clone.actor_critic),
            device=self.device,
            name='mse_loss wm',
        )
        self.world_model_mse.to(self.device)
        self.world_model_mse = torch.nn.parallel.DistributedDataParallel(self.world_model_mse, device_ids=[rank])
        self.world_model_mse = torch.compile(self.world_model_mse, mode="max-autotune")

        # # half_cql wm
        # cfg_clone = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        # cfg_clone.actor_critic.q_penalty = 'half_cql'
        # self.world_model_half_cql = WorldModel(
        #     obs_vocab_size=cfg.tokenizer.vocab_size,
        #     act_vocab_size=ATARI_NUM_ACTIONS,
        #     config_transformer=instantiate(cfg_clone.world_model),
        #     config_critic=instantiate(cfg_clone.actor_critic),
        #     device=self.device,
        #     name='half_cql wm',
        # )
        # self.world_model_half_cql.to(self.device)
        # self.world_model_half_cql = torch.nn.parallel.DistributedDataParallel(self.world_model_half_cql, device_ids=[rank])
        # self.world_model_half_cql = torch.compile(self.world_model_half_cql, mode="max-autotune")

        # no_rem wm
        cfg_clone = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_clone.actor_critic.num_q = 3
        cfg_clone.actor_critic.use_rem = False
        self.world_model_no_rem = WorldModel(
            obs_vocab_size=cfg.tokenizer.vocab_size,
            act_vocab_size=ATARI_NUM_ACTIONS,
            config_transformer=instantiate(cfg_clone.world_model),
            config_critic=instantiate(cfg_clone.actor_critic),
            device=self.device,
            name='3_q_head no_rem wm',
        )
        self.world_model_no_rem.to(self.device)
        self.world_model_no_rem = torch.nn.parallel.DistributedDataParallel(self.world_model_no_rem, device_ids=[rank])
        self.world_model_no_rem = torch.compile(self.world_model_no_rem, mode="max-autotune")

        # q_loss_not_backwards wm
        cfg_clone = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_clone.actor_critic.q_loss_backwards_wm = False
        self.world_model_not_backwards = WorldModel(
            obs_vocab_size=cfg.tokenizer.vocab_size,
            act_vocab_size=ATARI_NUM_ACTIONS,
            config_transformer=instantiate(cfg_clone.world_model),
            config_critic=instantiate(cfg_clone.actor_critic),
            device=self.device,
            name='q_loss_not_backwards wm',
        )
        self.world_model_not_backwards.to(self.device)
        self.world_model_not_backwards = torch.nn.parallel.DistributedDataParallel(self.world_model_not_backwards, device_ids=[rank])
        self.world_model_not_backwards = torch.compile(self.world_model_not_backwards, mode="max-autotune")

        # no_supervised_loss wm
        cfg_clone = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_clone.actor_critic.supervised_weight = 0.0
        self.world_model_no_supervise = WorldModel(
            obs_vocab_size=cfg.tokenizer.vocab_size,
            act_vocab_size=ATARI_NUM_ACTIONS,
            config_transformer=instantiate(cfg_clone.world_model),
            config_critic=instantiate(cfg_clone.actor_critic),
            device=self.device,
            name='no_supervised_loss wm',
        )
        self.world_model_no_supervise.to(self.device)
        self.world_model_no_supervise = torch.nn.parallel.DistributedDataParallel(self.world_model_no_supervise, device_ids=[rank])
        self.world_model_no_supervise = torch.compile(self.world_model_no_supervise, mode="max-autotune")

        # no_task_embed wm
        cfg_clone = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_clone.actor_critic.use_task_embed = False
        self.world_model_no_task_embed = WorldModel(
            obs_vocab_size=cfg.tokenizer.vocab_size,
            act_vocab_size=ATARI_NUM_ACTIONS,
            config_transformer=instantiate(cfg_clone.world_model),
            config_critic=instantiate(cfg_clone.actor_critic),
            device=self.device,
            name='no_task_embed wm',
        )
        self.world_model_no_task_embed.to(self.device)
        self.world_model_no_task_embed = torch.nn.parallel.DistributedDataParallel(self.world_model_no_task_embed, device_ids=[rank])
        self.world_model_no_task_embed = torch.compile(self.world_model_no_task_embed, mode="max-autotune")
        
        if self.is_main_process:
            print(f'Training dtype: {self.dtype}, seed: {cfg.common.seed}')
            tokenizer_params_num = sum(p.numel() for p in self.tokenizer.parameters()) / 10 ** 6
            print(f'{tokenizer_params_num:.2f}M parameters in tokenizer.')

            for world_model in self.all_wms:
                wm_params_num = sum(p.numel() for p in world_model.parameters()) / 10 ** 6
                print(f'{wm_params_num:.2f}M parameters in {str(world_model.module)}, {tokenizer_params_num+wm_params_num:.2f}M parameters in total.')
        
        group = cfg.common.group_name
        sample_from_start = not cfg.training.world_model.train_critic
        self.train_dataset = AtariTrajectory(
            data_path='offline_dataset/', 
            csv_path='dataset_csv/train_envs_sample_from_end.csv', 
            sequence_length=cfg.common.sequence_length,
            envs=cfg.common.envs,
        )
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, seed=cfg.common.seed)
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.training.world_model.batch_num_samples,
            collate_fn=collate_fn,
            num_workers=cfg.datasets.train.num_of_workers,
            sampler=self.train_sampler,
            pin_memory=True,
            prefetch_factor=2,
        )

        self.optimizer_tokenizer = torch.optim.Adam(self.tokenizer.parameters(), lr=cfg.training.tokenizer.learning_rate)
        # self.optimizer_alpha = torch.optim.Adam([self.world_model.module.log_alpha], lr=cfg.training.world_model.alpha_lr)
        
        self.optimizer_world_model_ori = configure_optimizer(
            self.world_model_ori, 
            cfg.training.world_model.learning_rate, 
            cfg.training.world_model.weight_decay, 
            cfg.training.world_model.critic_lr,
        )
        self.optimizer_world_model_3_q_head = configure_optimizer(
            self.world_model_3_q_head, 
            cfg.training.world_model.learning_rate, 
            cfg.training.world_model.weight_decay, 
            cfg.training.world_model.critic_lr,
        )
        self.optimizer_world_model_no_cql = configure_optimizer(
            self.world_model_no_cql, 
            cfg.training.world_model.learning_rate, 
            cfg.training.world_model.weight_decay, 
            cfg.training.world_model.critic_lr,
        )
        self.optimizer_world_model_mse = configure_optimizer(
            self.world_model_mse, 
            cfg.training.world_model.learning_rate, 
            cfg.training.world_model.weight_decay, 
            cfg.training.world_model.critic_lr,
        )
        # self.optimizer_world_model_half_cql = configure_optimizer(
        #     self.world_model_half_cql, 
        #     cfg.training.world_model.learning_rate, 
        #     cfg.training.world_model.weight_decay, 
        #     cfg.training.world_model.critic_lr,
        # )
        self.optimizer_world_model_no_rem = configure_optimizer(
            self.world_model_no_rem, 
            cfg.training.world_model.learning_rate, 
            cfg.training.world_model.weight_decay, 
            cfg.training.world_model.critic_lr,
        )
        self.optimizer_world_model_not_backwards = configure_optimizer(
            self.world_model_not_backwards, 
            cfg.training.world_model.learning_rate, 
            cfg.training.world_model.weight_decay, 
            cfg.training.world_model.critic_lr,
        )
        self.optimizer_world_model_no_supervise = configure_optimizer(
            self.world_model_no_supervise, 
            cfg.training.world_model.learning_rate, 
            cfg.training.world_model.weight_decay, 
            cfg.training.world_model.critic_lr,
        )
        self.optimizer_world_model_no_task_embed = configure_optimizer(
            self.world_model_no_task_embed, 
            cfg.training.world_model.learning_rate, 
            cfg.training.world_model.weight_decay, 
            cfg.training.world_model.critic_lr,
        )
        
        if cfg.initialization.load_tokenizer is not None:
            self.load_checkpoint(cfg.initialization.load_tokenizer, 'tokenizer')

        if cfg.initialization.load_world_model is not None:
            self.load_checkpoint(cfg.initialization.load_world_model, 'world_model')
        
        self.global_training_step = 0
        
        # parameters for evaluation
        h, w = 84, 84
        self.size = [h, 2*w] if self.cfg.evaluation.env.do_reconstruction else [h, w]
        self.env_token = torch.as_tensor(
            [GAME_NAMES.index(cfg.evaluation.env.env_name)], 
            dtype=torch.long, device=self.device
        )
        # self.best_return = -np.inf
        
        dist.barrier()
    
    # should modify: 
    # 1. the definitions of wm and optimizer in `__init__` function
    # 2. all_wms
    # 3. all_wm_optimizers
    # 4. the name of them in `save` function
    @property
    def all_wms(self):
        return self.world_model_ori, self.world_model_3_q_head, self.world_model_no_cql, self.world_model_mse, self.world_model_no_rem, self.world_model_not_backwards, self.world_model_no_supervise, self.world_model_no_task_embed

    @property
    def all_wm_optimizers(self):
        return self.optimizer_world_model_ori, self.optimizer_world_model_3_q_head, self.optimizer_world_model_no_cql, self.optimizer_world_model_mse, self.optimizer_world_model_no_rem, self.optimizer_world_model_not_backwards, self.optimizer_world_model_no_supervise, self.optimizer_world_model_no_task_embed

    def run(self) -> None:
        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):
            
            if self.is_main_process:
                print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
                start_time = time.time()

            if self.cfg.training.should:
                self.train_sampler.set_epoch(epoch)
                self.train_world_model(epoch)
                dist.barrier()

            eval_metrics = {}
            if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
                eval_metrics= self.eval_world_model(epoch)
                dist.barrier()
            
            if eval_metrics is None:
                eval_metrics = {}
            
            if self.is_main_process:
                eval_metrics.update({'epoch': epoch, 'duration': (time.time() - start_time) / 3600})
                wandb.log(eval_metrics)

        self.finish()

    def train_world_model(self, epoch: int) -> None:
        for wm in self.all_wms:
            wm.train()
        
        self.tokenizer.zero_grad()
        for wm in self.all_wms:
            wm.zero_grad()
        
        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_world_model = self.cfg.training.world_model

        for step, batch_ in enumerate(tqdm(
            self.train_dataloader, 
            disable=not self.is_main_process, 
            desc=f"Epoch {epoch}. Train {self.training_desc}".replace("world_model", "wm"),
            file=sys.stdout,
        )):
            
            intermediate_losses = defaultdict(float)
            batch = self._to_device(batch_)

            # train tokenizer
            if cfg_tokenizer.should:
                self.tokenizer.train()
                
                bs = batch['observations'].shape[0] * self.cfg.common.sequence_length
                if bs <= cfg_tokenizer.batch_num_samples:
                    random_idx = np.random.permutation(bs)
                else:
                    random_idx = np.random.choice(range(0, bs), size=(cfg_tokenizer.batch_num_samples,), replace=False)
                batch_tokenizer = {'observations': rearrange(batch['observations'], 'b t c h w -> (b t) 1 c h w')[random_idx]}

                with autocast(dtype=self.dtype):
                    losses = self.tokenizer.module.compute_loss(batch_tokenizer)
                    
                loss_total_step = losses.loss_total
                self.optimizer_tokenizer.zero_grad()
                loss_total_step.backward()

                # log losses
                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"tokenizer/train/{loss_name}"] += loss_value
                intermediate_losses[f"tokenizer/train/total_loss"] += loss_total_step.item()

                if cfg_tokenizer.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.tokenizer.parameters(), cfg_tokenizer.max_grad_norm)

                self.optimizer_tokenizer.step()

            self.tokenizer.eval()
            for wm in self.all_wms:
                wm.train()
                
            # train world_model
            all_intermediate_losses, all_wm_logs = {}, {}

            for optimizer_world_model, world_model in zip(self.all_wm_optimizers, self.all_wms):
                with autocast(dtype=self.dtype):
                    losses, wm_logs = world_model.module.compute_loss(
                        batch, 
                        self.tokenizer.module, 
                        train_critic=cfg_world_model.train_critic, 
                        imagine_horizon=cfg_world_model.imagine_horizon,
                        training=True,
                        # image_batch=self._to_device(self.train_dataset.sample_batch(batch_num_samples=cfg_world_model.batch_num_samples)),  # time consuming!!!
                    )
                
                loss_total_step = losses.loss_total
                optimizer_world_model.zero_grad()
                # self.optimizer_alpha.zero_grad()
                loss_total_step.backward()

                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"{str(world_model.module)}/train/{loss_name}"] += loss_value
                intermediate_losses[f"{str(world_model.module)}/train/total_loss"] += loss_total_step.item()
                
                all_intermediate_losses.update(intermediate_losses)
                all_wm_logs.update(wm_logs)

                # self.optimizer_alpha.step()  # no grad clipping for alpha

                if cfg_world_model.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(world_model.parameters(), cfg_world_model.max_grad_norm)

                optimizer_world_model.step()
                world_model.module.hard_update_target_Q()

            self.global_training_step += 1

            # all logs
            if self.global_training_step % self.cfg.training.log_interval == 0:
                all_intermediate_losses = self._gather_tensor(all_intermediate_losses)
                all_wm_logs = self._gather_tensor(all_wm_logs)

                if self.is_main_process:
                    logs = {
                        'training_step': self.global_training_step, 
                        **all_intermediate_losses, 
                        **all_wm_logs, 
                    }
                    wandb.log(logs)
        
        dist.barrier()
        self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)
        
    @torch.no_grad()
    def eval_world_model(self, train_epoch: int) -> None:
        self.tokenizer.eval()
        for wm in self.all_wms:
            wm.eval()
        
        cfg_eval_world_model = self.cfg.evaluation.world_model
        cfg_eval_env = self.cfg.evaluation.env
        eval_metric = {}

        if self.cfg.training.world_model.train_critic and (train_epoch % cfg_eval_world_model.critic_eval_epoch_frequency == 0):
            for world_model in self.all_wms:
                if str(world_model.module) in ['no_cql wm', 'mse_loss wm', 'q_loss_not_backwards wm']:
                    continue
                
                num_given_steps = cfg_eval_env.num_given_steps
                
                # create env
                env_fn = AtariEnvWrapper(cfg_eval_env.env_name).create_env
                test_env = SingleProcessEnv(env_fn)
                
                agent = Agent(
                    self.tokenizer.module, 
                    world_model.module, 
                    self.env_token, 
                    self.dtype, 
                    num_given_steps, 
                    self.device
                ).to(self.device)
                agent.eval()
                
                env = AgentEnv(
                    agent, 
                    test_env, 
                    keymap_name='atari', 
                    do_reconstruction=cfg_eval_env.do_reconstruction
                )
                
                game = Game(
                    env, 
                    keymap_name='empty', 
                    size=self.size, 
                    fps=int(cfg_eval_env.fps), 
                    verbose=bool(cfg_eval_env.header), 
                    record_mode=bool(cfg_eval_env.save_mode),
                    num_eval_episodes=int(cfg_eval_env.num_eval_episodes),
                )
                episode_info_collect = game.run(max_time=cfg_eval_env.max_time, num_given_steps=num_given_steps)
                episode_info_summary = {k: np.mean([i[k] for i in episode_info_collect]) for k, v in episode_info_collect[0].items() if isinstance(v, (int, float))}
                
                mean_return = torch.tensor([episode_info_summary['return']], dtype=torch.float32, device=self.device)
                ret_all = [torch.zeros_like(mean_return) for _ in range(self.cfg.training.world_size)]
                dist.all_gather(ret_all, mean_return)
                
                if self.is_main_process:
                    ret_all = torch.cat(ret_all, dim=0)
                    mean_ret_all = ret_all.mean().item()

                    # if mean_ret_all >= self.best_return:
                    #     self.best_return = mean_ret_all
                    #     self.save_checkpoint(train_epoch, save_agent_only=not self.cfg.common.do_checkpoint, best=True)

                    eval_metric.update({
                        "epoch": train_epoch, 
                        f"eval/{str(world_model.module)} return": mean_ret_all,
                        "eval/num_given_steps": num_given_steps,
                    })
        
        if cfg_eval_world_model.save_reconstructions and self.is_main_process:
            train_batch = self.train_dataset.sample_batch(batch_num_samples=30)
            batch = self._to_device(train_batch)

            for world_model in self.all_wms:
                with autocast(dtype=self.dtype):
                    make_reconstructions_of_trajectories(
                        batch, 
                        save_dir=self.reconstructions_dir, 
                        epoch=train_epoch, 
                        tokenizer=self.tokenizer.module, 
                        world_model=world_model.module,
                        enable_check=False,
                    )
                
        return eval_metric

    def save_checkpoint(self, epoch: int, save_agent_only: bool, best: bool = False) -> None:
        if self.is_main_process:
            step_ckpt_dir = self.ckpt_dir / f'{"best_ckpt_" if best else ""}epoch_{epoch}{"_step_" + str(self.global_training_step) if best else ""}'
            step_ckpt_dir.mkdir(exist_ok=False, parents=False)
            
            ckpts = [f for f in self.ckpt_dir.glob('epoch_*') if f.name.startswith('epoch_')]
            ckpts.sort(key=lambda p: p.stat().st_mtime)
            if len(ckpts) > self.cfg.training.max_ckpts:
                shutil.rmtree(ckpts[0])
            
            # save ckpt
            torch.save(self.tokenizer.state_dict(), step_ckpt_dir / f'tokenizer.pt')

            for wm in self.all_wms:
                torch.save(wm.state_dict(), step_ckpt_dir / f'{str(wm.module).replace(" ", "_")}.pt')
            
            # save optimizer
            if not save_agent_only:
                torch.save(self.optimizer_tokenizer.state_dict(), step_ckpt_dir / f'optimizer_tokenizer.pt')

                torch.save(self.optimizer_world_model_ori.state_dict(), step_ckpt_dir / f'optimizer_world_model_ori.pt')
                torch.save(self.optimizer_world_model_3_q_head.state_dict(), step_ckpt_dir / f'optimizer_world_model_3_q_head.pt')
                torch.save(self.optimizer_world_model_no_cql.state_dict(), step_ckpt_dir / f'optimizer_world_model_no_cql.pt')
                torch.save(self.optimizer_world_model_mse.state_dict(), step_ckpt_dir / f'optimizer_world_model_mse.pt')
                # torch.save(self.optimizer_world_model_half_cql.state_dict(), step_ckpt_dir / f'optimizer_world_model_half_cql.pt')
                torch.save(self.optimizer_world_model_no_rem.state_dict(), step_ckpt_dir / f'optimizer_world_model_no_rem.pt')
                torch.save(self.optimizer_world_model_not_backwards.state_dict(), step_ckpt_dir / f'optimizer_world_model_not_backwards.pt')
                torch.save(self.optimizer_world_model_no_supervise.state_dict(), step_ckpt_dir / f'optimizer_world_model_no_supervise.pt')
                torch.save(self.optimizer_world_model_no_task_embed.state_dict(), step_ckpt_dir / f'optimizer_world_model_no_task_embed.pt')

    def load_checkpoint(self, path, component) -> None:
        def process_param_name(name: str) -> str:
            if name.startswith('module.'):
                name = name[7:]
            elif name.startswith('_orig_mod.module.'):
                name = name[17:]
            elif name.startswith('_orig_mod.'):
                name = name[10:]
            else:
                pass
            return '_orig_mod.module.' + name
        
        if component == 'tokenizer':
            ckpt_token = torch.load(os.path.join(path, 'tokenizer.pt'), map_location=self.device)
            self.tokenizer.load_state_dict(ckpt_token)
            
            if self.cfg.initialization.load_optimizer_tokenizer:
                ckpt_opt_tokenizer = torch.load(os.path.join(path, 'optimizer_tokenizer.pt'), map_location=self.device)
                self.optimizer_tokenizer.load_state_dict(ckpt_opt_tokenizer)

        elif component == 'world_model':
            for wm in self.all_wms:
                ckpt_world = torch.load(os.path.join(path, f'{str(wm.module).replace(" ", "_")}.pt'), map_location=self.device)
                
                # world_model_dict = wm.state_dict()
                # for name, param in ckpt_world.items():
                #     if name.endswith('.indices'):
                #         continue
                #     elif name == "_orig_mod.module.pos_emb.weight":
                #         world_model_dict[name] = param[:world_model_dict[name].size()[0]]
                #     else:
                #         world_model_dict[name] = param
                
                wm.load_state_dict(ckpt_world)
            
            if self.cfg.initialization.load_optimizer_world_model:
                for optimizer, name in zip(self.all_wm_optimizers, ['optimizer_world_model_ori.pt', 'optimizer_world_model_3_q_head.pt', 'optimizer_world_model_no_cql.pt', 'optimizer_world_model_mse.pt', 'optimizer_world_model_no_rem.pt', 'optimizer_world_model_not_backwards.pt', 'optimizer_world_model_no_supervise.pt', 'optimizer_world_model_no_task_embed.pt']):
                    ckpt_opt_wm = torch.load(os.path.join(path, name), map_location=self.device)
                    optimizer.load_state_dict(ckpt_opt_wm)
                    
                    # find_pos_emb_flag = False
                    # for i in ckpt_opt_wm['state'].keys():
                    #     if ckpt_opt_wm['state'][i]['exp_avg'].shape == (740, 512):
                    #         find_pos_emb_flag = True
                    #         break
                    
                    # if find_pos_emb_flag:
                    #     ckpt_opt_wm['state'][i]['exp_avg'] = ckpt_opt_wm['state'][i]['exp_avg'][:296]
                    #     ckpt_opt_wm['state'][i]['exp_avg_sq'] = ckpt_opt_wm['state'][i]['exp_avg_sq'][:296]
                    #     optimizer.load_state_dict(ckpt_opt_wm)
                    # else:
                    #     raise Exception

            # self.world_model_ori.load_state_dict(torch.load("outputs_group/world_model_plus_critic/2024-08-19/21-20-53/checkpoints/epoch_7/world_model_ori.pt", map_location=self.device))
            # self.world_model_3_q_head.load_state_dict(torch.load("outputs_group/world_model_plus_critic/2024-08-17/09-37-34/checkpoints/epoch_24/world_model_3_q_heads.pt", map_location=self.device))
            # self.world_model_no_cql.load_state_dict(torch.load("outputs_group/world_model_plus_critic/2024-08-17/09-37-34/checkpoints/epoch_24/world_model_no_cql.pt", map_location=self.device))
            # self.world_model_mse.load_state_dict(torch.load("outputs_group/world_model_plus_critic/2024-08-17/09-37-34/checkpoints/epoch_24/world_model_mse.pt", map_location=self.device))
            # self.world_model_half_cql.load_state_dict(torch.load("outputs_group/world_model_plus_critic/2024-08-19/21-20-53/checkpoints/epoch_7/world_model_3_q_heads.pt", map_location=self.device))
            # self.world_model_no_rem.load_state_dict(torch.load("outputs_group/world_model_plus_critic/2024-08-19/21-20-53/checkpoints/epoch_7/world_model_no_cql.pt", map_location=self.device))
            # self.world_model_not_backwards.load_state_dict(torch.load("outputs_group/world_model_plus_critic/2024-08-19/21-20-53/checkpoints/epoch_7/world_model_mse.pt", map_location=self.device))
            # self.world_model_no_task_embed.load_state_dict(torch.load("outputs_group/tokenizer_plus_world_model/2024-08-15/13-12-45/checkpoints/epoch_8/world_model.pt", map_location=self.device))

            # if self.cfg.initialization.load_optimizer_world_model:
            #     ckpt_opt_wm = torch.load(os.path.join(path, 'optimizer_world_model_40M.pt'), map_location=self.device)
            #     self.optimizer_world_model_40M.load_state_dict(ckpt_opt_wm)

            #     ckpt_opt_wm = torch.load(os.path.join(path, 'optimizer_world_model_80M.pt'), map_location=self.device)
            #     self.optimizer_world_model_3_q_head.load_state_dict(ckpt_opt_wm)

            #     ckpt_opt_wm = torch.load(os.path.join(path, 'optimizer_world_model_200M.pt'), map_location=self.device)
            #     self.optimizer_world_model_no_cql.load_state_dict(ckpt_opt_wm)
            
            if self.cfg.initialization.load_start_epoch:
                self.start_epoch = int(path.split('_')[-1]) + 1
            
        else:
            raise NotImplementedError(f"component {component} is not implemented.")
        
        if self.is_main_process:
            print(f'Successfully loaded {component} from {path}.')

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}  # , non_blocking=True)

    def _gather_tensor(self, batch: Dict[str, float]) -> Dict[str, float]:
        res = defaultdict(float)
        for k, v in batch.items():
            tensor_v = torch.tensor(v, device=self.device)
            dist.all_reduce(tensor_v, op=dist.ReduceOp.SUM)
            mean_v = tensor_v / self.cfg.training.world_size
            res[k] = mean_v.item()
        return res
        
    def finish(self) -> None:
        wandb.finish()
        dist.destroy_process_group()


def collate_fn(batch_samples):
    return {k: torch.stack([sample[k] for sample in batch_samples]) \
        if k != 'envs' else \
            torch.LongTensor(batch_tokenize_envs([sample[k] for sample in batch_samples])) \
                for k in batch_samples[0]}

def get_dtype(dtype: str):
    return torch.float16 if dtype == 'float16' else torch.bfloat16 if dtype == 'bfloat16' else torch.float32

def hydra_main(*args, **kw):
    main = hydra.main(*args, **kw)
    def main_decorator(f):
        returned_values = []
        @functools.wraps(f)
        def f_wrapper(*args, **kw):
            ret = f(*args, **kw)
            returned_values.append(ret)
            return ret
        wrapped = main(f_wrapper)
        @functools.wraps(wrapped)
        def main_wrapper(*args, **kw):
            wrapped(*args, **kw)
            return returned_values[0] if len(returned_values) == 1 else returned_values
        return main_wrapper
    return main_decorator

@hydra_main(config_path="../config", config_name="trainer_world_model_plus_tokenizer_group")
def get_hydra_config(cfg: DictConfig) -> DictConfig:
    return cfg

def main(rank: int, cfg: DictConfig):
    trainer = Trainer(cfg, rank)
    trainer.run()


if __name__ == '__main__':
    config = get_hydra_config()
    print(f'{"Hydra Config":#^50}')
    pprint(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    print('#' * 50)

    world_size= config.training.world_size
    # ddp training
    mp.spawn(main, args=(config,), nprocs=world_size, join=True)
    
    # debugging
    # main(0, config)