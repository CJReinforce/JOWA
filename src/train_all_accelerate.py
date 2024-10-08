import functools
import os
import random
import shutil
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Any, Dict

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm

from action_tokenizer import ATARI_NUM_ACTIONS, batch_tokenize_envs, tokenize_actions
from make_reconstructions import make_reconstructions_of_trajectories
from models.world_model import WorldModel
from utils import capitalize_game_name, configure_optimizer, set_seed

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


class AtariTrajectory(Dataset):
    def __init__(self, data_path, csv_path, sequence_length):
        self.data_path = data_path
        self.trajectories_ind = pd.read_csv(csv_path, usecols=['Episode index in dataset', 'Environment'])
        self.sequence_length = sequence_length
        self.sample_from_start = False
        self.transform = ToTensor()
    
    def __len__(self):
        return len(self.trajectories_ind)

    def __getitem__(self, idx):
        index_in_env, env = self.trajectories_ind.iloc[idx]
        env = capitalize_game_name(env)
        trajectory_path = os.path.join(self.data_path, env, str(index_in_env))
        episode_terminal = np.load(os.path.join(trajectory_path, 'terminal', '0.npy'))
        episode_length = episode_terminal.shape[0]
        start, stop = self._sample_episodes_segments(episode_length, self.sequence_length, self.sample_from_start)
        
        # pad if episode_length < sequence_length
        padding_length_right = max(0, stop - episode_length)
        padding_length_left = max(0, -start)
        
        def pad(x):
            pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
            return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

        start = max(0, start)
        stop = min(episode_length, stop)
        
        # load observation
        def load_and_transform_image(i):
            obs = Image.open(os.path.join(trajectory_path, "observation", f"{i}.png")).convert('L')  # (84, 84)
            obs = self.transform(obs)
            return obs
        
        observations = [load_and_transform_image(i) for i in range(start, stop)]
            
        # load action, reward, terminal
        actions = np.load(os.path.join(trajectory_path, 'action', '0.npy'))[start:stop]
        rewards = np.load(os.path.join(trajectory_path, 'reward', '0.npy'))[start:stop]
        ends = episode_terminal[start:stop]
        
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
    
    # differ from original iris
    def _sample_episodes_segments(self, episode_length: int, sequence_length: int, sample_from_start: bool):
        if episode_length < sequence_length:
            if sample_from_start:
                start = 0
                stop = sequence_length
            else:
                start = episode_length - sequence_length
                stop = episode_length
        else:
            if sample_from_start:
                start = random.randint(0, episode_length - sequence_length)
                stop = start + sequence_length
            else:
                stop = random.randint(sequence_length, episode_length)
                start = stop - sequence_length
        return start, stop

    def sample_batch(self, batch_num_samples):
        idx = np.random.choice(self.__len__(), batch_num_samples, replace=False)
        return collate_fn([self.__getitem__(i) for i in idx])
    
    
class Trainer:
    def __init__(self, cfg: DictConfig, accelerator: Accelerator) -> None:
        self.accelerator = accelerator
        
        # Initialize wandb and saving dir
        if self.accelerator.is_local_main_process:
            save_root_dir = Path(f'outputs/critic/{time.strftime("%Y-%m-%d/%H-%M-%S")}')
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

        self.tokenizer = instantiate(cfg.tokenizer)
        self.tokenizer = torch.compile(self.tokenizer, mode="max-autotune")
        
        self.world_model = WorldModel(obs_vocab_size=cfg.tokenizer.vocab_size,
                                      act_vocab_size=ATARI_NUM_ACTIONS,
                                      config_transformer=instantiate(cfg.world_model),
                                      config_critic=instantiate(cfg.actor_critic),)
        self.world_model = torch.compile(self.world_model, mode="max-autotune")
        
        if self.accelerator.is_local_main_process:
            print(f'{sum(p.numel() for p in self.tokenizer.parameters())} parameters in tokenizer.')
            print(f'{sum(p.numel() for p in self.world_model.parameters())} parameters in world_model')

        self.train_dataset = AtariTrajectory(
            data_path='/data_ssd/filtered_google_atari_datasets/trajectories/train/', 
            csv_path='datasets/metainfo/train_env_cumulative_steps_after_filtering.csv', 
            sequence_length=cfg.common.sequence_length
        )
        self.valid_dataset = AtariTrajectory(
            data_path='/data_ssd/filtered_google_atari_datasets/trajectories/valid/', 
            csv_path='datasets/metainfo/train_env_cumulative_steps_for_valid_trajectories.csv',
            sequence_length=cfg.common.sequence_length
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.training.world_model.batch_num_samples,
            collate_fn=collate_fn,
            num_workers=cfg.datasets.train.num_of_workers,
            pin_memory=True
        )
        self.valid_dataloader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.cfg.evaluation.world_model.batch_num_samples,
            collate_fn=collate_fn,
            num_workers=cfg.datasets.test.num_of_workers,
            pin_memory=True
        )
        
        self.optimizer_tokenizer = torch.optim.Adam(self.tokenizer.parameters(), lr=cfg.training.tokenizer.learning_rate)
        self.optimizer_world_model = configure_optimizer(
            self.world_model, cfg.training.world_model.learning_rate, cfg.training.world_model.weight_decay, cfg.training.world_model.critic_lr)
        
        if cfg.initialization.load_tokenizer is not None:
            self.load_checkpoint(cfg.initialization.load_tokenizer, 'tokenizer')
        
        if cfg.initialization.load_world_model is not None:
            self.load_checkpoint(cfg.initialization.load_world_model, 'world_model')
            
        self.tokenizer, self.world_model, self.train_dataloader, self.valid_dataloader, \
            self.optimizer_tokenizer, self.optimizer_world_model = self.accelerator.prepare(
            self.tokenizer, self.world_model, self.train_dataloader, self.valid_dataloader, 
            self.optimizer_tokenizer, self.optimizer_world_model
        )
        
        self.global_training_step = 0
        self.accelerator.wait_for_everyone()

    def run(self) -> None:
        
        try:
            for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):

                if self.accelerator.is_local_main_process:
                    print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
                    start_time = time.time()

                if self.cfg.training.should:
                    self.train_agent(epoch)
                    self.accelerator.wait_for_everyone()

                if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
                    eval_metrics= self.eval_agent(epoch)
                    self.accelerator.wait_for_everyone()
                else:
                    eval_metrics = {}
                
                if self.accelerator.is_local_main_process:
                    eval_metrics.update({'epoch': epoch, 'duration': (time.time() - start_time) / 3600})
                    wandb.log(eval_metrics)
                    
        except KeyboardInterrupt:
            pass
        
        self.finish()

    def turn_train_mode(self, train=True) -> None:
        self.tokenizer.train(train)
        self.world_model.train(train)
        
    def zero_grad_optimizers(self) -> None:
        self.optimizer_tokenizer.zero_grad()
        self.optimizer_world_model.zero_grad()
        
    def zero_grad_models(self) -> None:
        self.tokenizer.zero_grad()
        self.world_model.zero_grad()
    
    def loss_backward_and_log(self, losses, intermediate_losses, name) -> None:
        loss_total_step = losses.loss_total
        self.accelerator.backward(loss_total_step)

        for loss_name, loss_value in losses.intermediate_losses.items():
            intermediate_losses[f"{name}/train/{loss_name}"] += loss_value
        intermediate_losses[f"{name}/train/total_loss"] += loss_total_step.item()
    
    def train_agent(self, epoch: int) -> None:
        self.turn_train_mode()
        self.zero_grad_models()

        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_world_model = self.cfg.training.world_model
                
        for batch in tqdm(self.train_dataloader, desc=f"Training all", file=sys.stdout, disable=not self.accelerator.is_local_main_process):
            intermediate_losses = defaultdict(float)
            self.zero_grad_optimizers()
                        
            # training of tokenizer
            # self.tokenizer.train()
            
            # with self.accelerator.autocast():
            #     tokenizer_losses = self.tokenizer.module.compute_loss(batch)
            
            # self.loss_backward_and_log(tokenizer_losses, intermediate_losses, 'tokenizer')
            
            # if cfg_tokenizer.max_grad_norm is not None:
            #     torch.nn.utils.clip_grad_norm_(self.tokenizer.parameters(), cfg_tokenizer.max_grad_norm)
            
            # self.optimizer_tokenizer.step()
            self.tokenizer.eval()

            # training of world model
            with self.accelerator.autocast():
                world_model_losses, world_model_logs = self.world_model.module.compute_loss(
                    batch, self.tokenizer.module, imagine_horizon=cfg_world_model.imagine_horizon)
            
            self.loss_backward_and_log(world_model_losses, intermediate_losses, 'world_model')
            
            if cfg_world_model.max_grad_norm is not None:
                self.accelerator.clip_grad_norm_(self.world_model.parameters(), cfg_world_model.max_grad_norm)
            
            self.optimizer_world_model.step()
            # self.optimizer_critic.step()
            
            # self.world_model.module.soft_update_target_Q()
            self.world_model.module.hard_update_target_Q()

            self.global_training_step += 1

            # log
            if self.global_training_step % self.cfg.training.log_interval == 0:
                intermediate_losses = self._gather_tensor(intermediate_losses)
                world_model_logs = self._gather_tensor(world_model_logs)
                logs = {'training_step': self.global_training_step, **intermediate_losses, **world_model_logs}
                if self.accelerator.is_local_main_process:
                    wandb.log(logs)
        
        self.accelerator.wait_for_everyone()            
        self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

    @torch.no_grad()
    def eval_agent(self, epoch: int) -> None:
        self.turn_train_mode(False)

        metrics_tokenizer, metrics_world_model = {}, {}

        cfg_tokenizer = self.cfg.evaluation.tokenizer
        cfg_world_model = self.cfg.evaluation.world_model
        cfg_actor_critic = self.cfg.evaluation.actor_critic

        if epoch > cfg_tokenizer.start_after_epochs:
            metrics_tokenizer = self.eval_component(self.agent.tokenizer, cfg_tokenizer.batch_num_samples, sequence_length=1)

        if epoch > cfg_world_model.start_after_epochs:
            metrics_world_model = self.eval_component(self.agent.world_model, cfg_world_model.batch_num_samples, sequence_length=self.cfg.common.sequence_length, tokenizer=self.agent.tokenizer)

        if epoch > cfg_actor_critic.start_after_epochs:
            self.inspect_imagination(epoch)

        if cfg_tokenizer.save_reconstructions:
            batch = self._to_device(self.test_dataset.sample_batch(batch_num_samples=3, sequence_length=self.cfg.common.sequence_length))
            make_reconstructions_of_trajectories(batch, save_dir=self.reconstructions_dir, epoch=epoch, tokenizer=self.agent.tokenizer)

        return [metrics_tokenizer, metrics_world_model]

    @torch.no_grad()
    def eval_component(self, component: nn.Module, batch_num_samples: int, sequence_length: int, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        steps = 0
        pbar = tqdm(desc=f"Evaluating {str(component)}", file=sys.stdout)
        for batch in self.test_dataset.traverse(batch_num_samples, sequence_length):
            batch = self._to_device(batch)

            losses = component.compute_loss(batch, **kwargs_loss)
            loss_total_epoch += losses.loss_total.item()

            for loss_name, loss_value in losses.intermediate_losses.items():
                intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value

            steps += 1
            pbar.update(1)

        intermediate_losses = {k: v / steps for k, v in intermediate_losses.items()}
        metrics = {f'{str(component)}/eval/total_loss': loss_total_epoch / steps, **intermediate_losses}
        return metrics

    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        if self.accelerator.is_local_main_process:
            step_ckpt_dir = self.ckpt_dir / f'epoch_{epoch}_step_{self.global_training_step}'
            step_ckpt_dir.mkdir(exist_ok=False, parents=False)
            
            # remove old checkpoints
            ckpts = list(self.ckpt_dir.glob('epoch_*_step_*'))
            ckpts.sort(key=lambda p: p.stat().st_mtime)
            if len(ckpts) > self.cfg.training.max_ckpts:
                shutil.rmtree(ckpts[0])
            
            # save model
            self.accelerator.save_model(self.world_model, step_ckpt_dir / f'world_model_plus_critic.pt')
            
            # save state
            if not save_agent_only:
                step_state_ckpt_dir = step_ckpt_dir / 'state'
                step_state_ckpt_dir.mkdir(exist_ok=False, parents=False)
                self.accelerator.save_state(output_dir=step_state_ckpt_dir)

    def load_checkpoint(self, path, component) -> None:
        def process_param_name(name: str) -> str:
            if name.startswith('module.'):
                name = name[7:]
            elif name.startswith('_orig_mod.module.'):
                name = name[17:]
            elif name.startswith('_orig_mod.'):
                return name
            else:
                pass
            return '_orig_mod.' + name
            
        if component == 'tokenizer':
            ckpt_token = torch.load(os.path.join(path, 'tokenizer.pt'), map_location=self.accelerator.device)
            tokenizer_dict = self.tokenizer.state_dict()
            
            for name, param in ckpt_token.items():
                if process_param_name(name) in tokenizer_dict:
                    tokenizer_dict[name] = param
            self.tokenizer.load_state_dict(tokenizer_dict, strict=False)
        
        elif component == 'world_model':
            # load world_model state_dict
            ckpt_world = torch.load(os.path.join(path, 'world_model.pt'), map_location=self.accelerator.device)
            world_model_dict = self.world_model.state_dict()
            
            for name, param in ckpt_world.items():
                if process_param_name(name) in world_model_dict:
                    world_model_dict[name] = param
            self.world_model.load_state_dict(world_model_dict, strict=False)
            # self.world_model.load_state_dict(ckpt_world)

            # load optimizer state_dict
            # ckpt_opt = torch.load(os.path.join(path, 'optimizer.pt'), map_location=self.accelerator.device)
            # self.optimizer_world_model.load_state_dict(ckpt_opt)

            # load epoch
            # self.start_epoch = int(path.split('/')[-1].split('_')[1]) + 1
        else:
            raise NotImplementedError(f"component {component} is not implemented.")
        
        if self.accelerator.is_local_main_process:
            print(f'Successfully loaded {component} from {path}.')

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.accelerator.device) for k in batch}

    def _gather_tensor(self, batch: Dict[str, float]) -> Dict[str, float]:
        # TODO: change to accelerator.gather
        res = defaultdict(float)
        for k, v in batch.items():
            tensor_v = torch.tensor(v, device=self.accelerator.device)
            gathered_tensor_v = self.accelerator.gather(tensor_v)
            mean_v = gathered_tensor_v.mean()
            res[k] = mean_v.item()
        return res
    
    def finish(self) -> None:
        if self.accelerator.is_local_main_process:
            wandb.finish()


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

@hydra_main(config_path="../config", config_name="trainer_all")
def get_hydra_config(cfg: DictConfig) -> DictConfig:
    return cfg

def main():
    accelerator = Accelerator()
    config = get_hydra_config()
    
    accelerator.print(f'{"Hydra Config":#^50}')
    if accelerator.is_local_main_process:
        pprint(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    accelerator.print('#' * 50)

    trainer = Trainer(config, accelerator)
    trainer.run()
    

if __name__ == '__main__':
    main()