import functools
import os
import pickle
import random
import re
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
from agent import Agent
from dataset import (
    AtariTrajectory, 
    AtariTrajInMemory,
    AtariTrajWithObsToken, 
    AtariTrajWithObsTokenInMemory,
    collate_fn,
)
from envs import AtariEnvWrapper, SingleProcessEnv
from game import AgentEnv, Game
from make_reconstructions import make_reconstructions_of_trajectories
from models.jowa_model import JOWAModel
from replay_buffer import ReplayBuffer
from utils import (
    capitalize_game_name,
    configure_optimizer,
    get_dtype,
    get_random_state,
    hydra_main,
    load_random_state,
    set_seed,
)

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        dist.init_process_group(
            backend="nccl", 
            timeout=timedelta(
                seconds=7200000
            ),  # avoid timeout when evaluating
        )
        local_rank = int(os.environ['LOCAL_RANK'])
        device_id = os.environ.get('DEVICE_ID')
        torch.cuda.set_device(local_rank if device_id is None else int(device_id))
        
        # ddp settings
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        self.rank = int(os.environ['RANK'])
        self.local_rank = local_rank
        self.is_main_process = self.rank == 0

        self.cfg = cfg
        self.global_training_step = 0

        # dirs 
        save_root_dir = Path(f'outputs/{time.strftime("%Y-%m-%d/%H-%M-%S")}')
        self.ckpt_dir = save_root_dir / 'checkpoints'
        self.media_dir = save_root_dir / 'media'
        self.reconstructions_dir = self.media_dir / 'reconstructed_imgs'
        self.video_dir = self.media_dir / 'game_records'

        # Initialize wandb and dirs
        if self.is_main_process:
            print(f'{"Hydra Config":#^50}')
            pprint(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
            print('#' * 50)
            print(f"Train {self.training_desc} with {self.world_size} GPUs.")
            
            save_root_dir.mkdir(exist_ok=False, parents=True)
            
            wandb.init(
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True,
                resume=True,
                dir=save_root_dir,
                **cfg.wandb
            )

            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=False, parents=False)
            self.video_dir.mkdir(exist_ok=False, parents=False)
            
        set_seed(cfg.common.seed)

        self.start_epoch = 1
        self.device = torch.device(cfg.common.device)
        self.dtype = get_dtype(cfg.training.dtype)

        # model        
        self.tokenizer = instantiate(cfg.tokenizer).to(self.device)
        self.tokenizer = torch.nn.parallel.DistributedDataParallel(
            self.tokenizer, 
            device_ids=[local_rank]
        )

        self.jowa_model = JOWAModel(
            obs_vocab_size=cfg.tokenizer.vocab_size,
            act_vocab_size=ATARI_NUM_ACTIONS,
            config_transformer=instantiate(cfg.transformer),
            config_critic_arch=cfg.critic_head,
            config_critic_train=cfg.training.action,
            device=self.device,
            name='JOWA_150M',
        ).to(self.device)
        self.jowa_model = torch.nn.parallel.DistributedDataParallel(
            self.jowa_model, 
            device_ids=[local_rank]
        )
        
        # count parameters
        if self.is_main_process:
            print(f'Training dtype: {self.dtype}, seed: {cfg.common.seed}')

            token_params_num = sum(
                p.numel() for p in self.tokenizer.parameters()
            ) / 10 ** 6
            print(f'{token_params_num:.2f}M params in tokenizer.')

            for jowa in self.all_jowas:
                jowa_params_num = sum(
                    p.numel() for p in jowa.parameters()
                ) / 10 ** 6
                print(f'{jowa_params_num:.2f}M params in transformer and heads.')
                print(
                    f'{token_params_num + jowa_params_num:.2f}M params in {str(jowa.module)}.'
                )

        # data
        # dataset for pretrain stage 1
        # self.train_dataset = AtariTrajectory(
        #     data_dir='dataset/downsampled/trajectory/data', 
        #     csv_dir='dataset/downsampled/segment/csv', 
        #     envs=cfg.common.envs,
        #     csv_suffix="_right_padding", 
        # )

        # dataset for pretrain stage 2
        # self.train_dataset = AtariTrajWithObsTokenInMemory(
        #     data_dir='dataset/downsampled/trajectory/token_data', 
        #     csv_dir='dataset/downsampled/segment/csv', 
        #     envs=cfg.common.envs,
        #     csv_suffix="_right_padding", 
        #     show_pbar=self.local_rank == 0,
        #     local_rank=self.local_rank,
        # )
        # # use obs-token dataset only when not training tokenizer
        # assert not self.get_config_in_this_stage(cfg.training.tokenizer).should
        
        # dataset for finetune
        self.train_dataset = AtariTrajInMemory(
            data_dir='dataset/finetune/expert/trajectory/data', 
            csv_dir='dataset/finetune/expert/segment/csv', 
            envs=cfg.common.envs, 
            csv_suffix="_tau_1_right_padding", 
        )

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset, 
            num_replicas=self.world_size, 
            rank=self.rank,
            seed=cfg.common.seed,
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=cfg.training.world.batch_size // self.world_size,
            collate_fn=collate_fn,
            num_workers=cfg.datasets.train.num_of_workers // self.local_world_size,
            sampler=self.train_sampler,
            pin_memory=True,
            prefetch_factor=8,
        )
        
        if self.use_imagination:
            self.imagine_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                seed=cfg.common.seed + 10000,
            )
            self.imagine_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_size=cfg.training.world.batch_size // self.world_size,
                collate_fn=collate_fn,
                num_workers=cfg.datasets.train.num_of_workers // self.local_world_size,
                sampler=self.imagine_sampler,
                pin_memory=True,
                prefetch_factor=2,
            )
            self.imagine_replay = ReplayBuffer(
                sequence_length=cfg.common.sequence_length, 
                capacity=len(self.train_dataset) * 10,
                obs_shape=(1, 84, 84),
                device=self.device,
            )

        self.optimizer_tokenizer, self.optimizer_jowa = self.reset_optimizers()
        
        if cfg.initialization.load_tokenizer is not None:
            self.load_checkpoint(
                cfg.initialization.load_tokenizer, 
                'tokenizer',
                cfg.initialization.load_optimizer_tokenizer,
            )

        if cfg.initialization.load_jowa is not None:
            self.load_checkpoint(
                cfg.initialization.load_jowa, 
                cfg.initialization.load_jowa_name,
                cfg.initialization.load_optimizer_jowa,
                cfg.initialization.load_start_epoch,
            )
            
            for jowa in self.all_jowas:
                jowa.module.hard_update_target_Q(wo_check=True)
        
        # evaluation
        if self.cfg.evaluation.should:
            h, w = 84, 84
            self.size = [h, 2 * w] if \
                self.cfg.evaluation.env.do_reconstruction else [h, w]
            self.env_token = torch.as_tensor(
                [GAME_NAMES.index(cfg.evaluation.env.env_name)], 
                dtype=torch.long, 
                device=self.device,
            )
            self.best_return = -np.inf
        
        dist.barrier()
    
    # if train more WM at once (such as 40M, 70M, 150M), you should modify the followings: 
    # 1. add the definitions of wm and optimizer in `__init__` function
    # 2. `all_jowas` function
    # 3. `all_jowa_optimizers` function
    # 4. `save` function
    # 5. `load` function
    @property
    def all_jowas(self):
        return (self.jowa_model,)

    @property
    def all_jowa_optimizers(self):
        return (self.optimizer_jowa,)

    @property
    def training_desc(self) -> str:
        desc = 'world'
        if self.train_critic:
            desc += '-action'
        if self.get_config_in_this_stage(self.cfg.training.tokenizer).should:
            desc = 'vqvae + ' + desc
        desc += ' model'
        return desc
    
    @property
    def train_critic(self) -> bool:
        return self.global_training_step > self.cfg.training.action.train_critic_after_n_steps

    @property
    def is_first_stage(self) -> bool:
        return not self.train_critic

    @property
    def use_imagination(self) -> bool:
        return self.train_critic and self.cfg.training.action.use_imagination
    
    def get_config_in_this_stage(self, cfg):
        if hasattr(cfg, 'first_stage') and hasattr(cfg, 'second_stage'):
            if self.is_first_stage:
                return cfg.first_stage
            else:
                return cfg.second_stage
        else:
            return cfg

    def reset_optimizers(self):
        opt_tokenizer = torch.optim.Adam(
            self.tokenizer.parameters(), 
            lr=self.cfg.training.tokenizer.learning_rate,
        )
        
        opt_jowa = configure_optimizer(
            self.jowa_model, 
            self.get_config_in_this_stage(self.cfg.training.world).learning_rate, 
            self.cfg.training.world.weight_decay, 
            self.cfg.training.action.learning_rate,
        )
        return opt_tokenizer, opt_jowa

    def run(self) -> None:
        keep_running = True

        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):
            if self.is_main_process:
                print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
                start_time = time.time()

            if self.cfg.training.should:
                self.train_sampler.set_epoch(epoch)
                if self.use_imagination:
                    self.imagine_sampler.set_epoch(epoch)

                keep_running = self.train(epoch)
                dist.barrier()

            metrics = {}

            if self.cfg.evaluation.should:
                eval_metrics = self.eval(epoch)
                dist.barrier()
                metrics.update(eval_metrics)
            
            if self.is_main_process:
                metrics.update(
                    {
                        'epoch': epoch, 
                        'duration': (time.time() - start_time) / 3600,
                    }
                )
                wandb.log(metrics)

            if not keep_running:
                break

        self.finish()

    def train(self, epoch: int) -> bool:
        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_jowa = self.cfg.training.action

        keep_running = True

        # record time
        start_time = time.time()
        load_batch_time = []
        training_time = []

        for batch_ in tqdm(
            zip(
                self.train_dataloader, 
                self.imagine_dataloader,
            ) if self.use_imagination else self.train_dataloader, 
            disable=not self.is_main_process, 
            desc=f"Train {self.training_desc}",
            file=sys.stdout,
        ):
            intermediate_losses = defaultdict(float)
            extra_logs = {}

            # record loading batch time
            end_time = time.time()
            load_batch_time.append(end_time - start_time)
            start_time = time.time()

            if self.use_imagination:
                batch_, ima_batch_ = batch_
                ima_batch = self._to_device(ima_batch_)
            batch = self._to_device(batch_)

            # train tokenizer
            should_train_token = self.get_config_in_this_stage(cfg_tokenizer).should
            extra_logs["tokenizer/train/should"] = int(should_train_token)
            if should_train_token:
                self.tokenizer.train()
                
                num_obs = batch['observations'].shape[0] * self.cfg.common.sequence_length
                tokenizer_bs = cfg_tokenizer.batch_size // self.world_size
                if num_obs <= tokenizer_bs:
                    random_idx = np.random.permutation(num_obs)
                else:
                    random_idx = np.random.randint(
                        0, num_obs, 
                        size=(tokenizer_bs,)
                    )
                
                # tokenizer training log
                extra_logs["tokenizer/train/batch_size"] = len(random_idx)
                extra_logs[
                    "tokenizer/train/learning_rate"
                ] = self.optimizer_tokenizer.param_groups[0]['lr']

                batch_tokenizer = {
                    'observations': rearrange(
                        batch['observations'], 'b t c h w -> (b t) 1 c h w'
                    )[random_idx]
                }

                with autocast(dtype=self.dtype):
                    losses = self.tokenizer.module.compute_loss(
                        batch_tokenizer
                    )
                    
                loss_total_step = losses.loss_total
                self.optimizer_tokenizer.zero_grad()
                loss_total_step.backward()

                # log losses
                for name, value in losses.intermediate_losses.items():
                    intermediate_losses[f"tokenizer/train/{name}"] += value
                intermediate_losses[
                    f"tokenizer/train/total_loss"] += loss_total_step.item()

                if cfg_tokenizer.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.tokenizer.parameters(), 
                        cfg_tokenizer.max_grad_norm,
                    )
                    extra_logs["tokenizer/train/grad_norm"] = grad_norm.item()

                self.optimizer_tokenizer.step()
                
            # train jowa
            for opt_jowa, jowa in zip(self.all_jowa_optimizers, self.all_jowas):

                # model-based data augmentation
                if self.use_imagination and \
                    self.global_training_step > cfg_jowa.imagine_after_n_steps:
                    
                    # switch status
                    self.tokenizer.eval()
                    for jowa in self.all_jowas:
                        jowa.eval()
                    
                    # change random seed for augmentation / imagination
                    # I am not sure whether is necessary
                    # cpu_rng_state, cuda_rng_state, numpy_state, python_state = get_random_state()
                    # set_seed(dist.get_rank() + epoch)
                    
                    with autocast(dtype=self.dtype):
                        imagine_batch, imagine_logs = jowa.module.imagine(
                            ima_batch,
                            self.tokenizer.module,
                            horizon=cfg_jowa.planning_horizon,
                            beam_width=cfg_jowa.planning_beam_width,
                            should_sample=False,
                        )
                    
                    extra_logs.update(imagine_logs)

                    # gather batch from all ranks
                    imagine_batch = self._gather_batch(imagine_batch)
                    avail_idxs = imagine_batch['envs'] != -1
                        
                    self.imagine_replay.add_batch(
                        imagine_batch['observations'][avail_idxs].detach().cpu().numpy(), 
                        imagine_batch['actions'][avail_idxs].detach().cpu().numpy(), 
                        imagine_batch['rewards'][avail_idxs].detach().cpu().numpy(), 
                        imagine_batch['ends'][avail_idxs].detach().cpu().numpy(), 
                        imagine_batch['mask_padding'][avail_idxs].detach().cpu().numpy(), 
                        imagine_batch['envs'][avail_idxs].detach().cpu().numpy(),
                    )
                    
                    # change seed for sampling
                    cpu_rng_state, cuda_rng_state, numpy_state, python_state = get_random_state()
                    set_seed(self.rank + epoch + np.random.randint(10000))

                    imagine_batch_for_training = self.imagine_replay.sample(
                        cfg_jowa.batch_size_in_imagination // self.world_size
                    )

                    load_random_state(cpu_rng_state, cuda_rng_state, numpy_state, python_state)
                    
                else:
                    imagine_batch_for_training = None

                # jowa training logs
                extra_logs[f"{str(jowa.module)}/train/real_bs"] = len(batch['ends'])
                extra_logs[
                    f"{str(jowa.module)}/train/imagined_bs"
                ] = 0 if imagine_batch_for_training is None else len(imagine_batch_for_training['ends'])
                extra_logs[f"{str(jowa.module)}/train/train_critic"] = int(self.train_critic)
                extra_logs[
                    f"{str(jowa.module)}/train/world_lr"
                ] = opt_jowa.param_groups[0]['lr']
                extra_logs[
                    f"{str(jowa.module)}/train/action_lr"
                ] = opt_jowa.param_groups[2]['lr']

                # switch status
                self.tokenizer.eval()
                for jowa in self.all_jowas:
                    jowa.train()
                
                # train jowa
                with autocast(dtype=self.dtype):
                    losses, info_logs = jowa.module.compute_loss(
                        real_batch=batch, 
                        imagine_batch=imagine_batch_for_training,
                        tokenizer=self.tokenizer.module, 
                        train_critic=self.train_critic, 
                        training=True,
                    )
                
                loss_total_step = losses.loss_total
                opt_jowa.zero_grad()
                loss_total_step.backward()

                for name, value in losses.intermediate_losses.items():
                    intermediate_losses[f"{str(jowa.module)}/train/{name}"] += value
                intermediate_losses[
                    f"{str(jowa.module)}/train/total_loss"] += loss_total_step.item()
                
                extra_logs.update(info_logs)

                if self.cfg.training.world.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        jowa.parameters(), 
                        self.cfg.training.world.max_grad_norm,
                    )
                    extra_logs[f"{str(jowa.module)}/train/grad_norm"] = grad_norm.item()

                opt_jowa.step()
                jowa.module.hard_update_target_Q()

            self.global_training_step += 1

            if self.global_training_step == self.cfg.training.action.train_critic_after_n_steps:
                self.optimizer_tokenizer, self.optimizer_jowa = self.reset_optimizers()

            # step frequency of evaluation
            if self.train_critic and \
                self.global_training_step % self.cfg.evaluation.action.step_frequency == 0:
                logs = self._eval_game(epoch, {"epoch": epoch})
                for log in logs:
                    extra_logs.update(log)

            # wandb logs
            if self.global_training_step % self.cfg.training.log_interval == 0:
                consume_time = {
                    'info/load_batch_time': np.mean(load_batch_time),
                    'info/training_time': np.mean(training_time)
                }
                load_batch_time.clear()
                training_time.clear()

                consume_time = self._gather_tensor(consume_time)
                intermediate_loss = self._gather_tensor(intermediate_losses)
                extra_log = self._gather_tensor(extra_logs)

                if self.is_main_process:
                    logs = {
                        'training_step': self.global_training_step, 
                        **consume_time,
                        **intermediate_loss, 
                        **extra_log, 
                    }
                    wandb.log(logs)

            # if using imagination batch for training, change the batch size of train_dataloader
            target_bs = (
                self.cfg.training.world.batch_size - cfg_jowa.batch_size_in_imagination
            ) // self.world_size
            if self.use_imagination and \
                self.global_training_step > cfg_jowa.imagine_after_n_steps and \
                    self.train_dataloader.batch_sampler.batch_size != target_bs:

                self.train_dataloader.batch_sampler.batch_size = target_bs
            
            end_time = time.time()
            training_time.append(end_time - start_time)
            start_time = time.time()

            # stop training
            if self.global_training_step >= self.cfg.common.steps:
                keep_running = False
                break
        
        dist.barrier()
        self.save_checkpoint(
            epoch, 
            save_agent_only=not self.cfg.common.do_checkpoint,
        )
        return keep_running
        
    @torch.no_grad()
    def eval(self, train_epoch: int) -> Dict:
        # switch to eval
        self.tokenizer.eval()
        for jowa in self.all_jowas:
            jowa.eval()
        
        cfg_eval = self.cfg.evaluation
        eval_metric = {}

        # play game
        if self.train_critic and train_epoch % cfg_eval.action.epoch_frequency == 0:
            logs = self._eval_game(
                train_epoch,
                {
                    "epoch": train_epoch,
                    "training_step": self.global_training_step,
                }
            )
            for log in logs:
                eval_metric.update(log)
        
        # reconstruct trajectory
        if cfg_eval.world.save_reconstructions and \
            type(self.train_dataset) in (AtariTrajectory, AtariTrajInMemory) and \
                self.is_main_process:
            train_batch = self.train_dataset.sample_batch(batch_num_samples=30)
            batch = self._to_device(train_batch)

            with autocast(dtype=self.dtype):
                for jowa in self.all_jowas:
                    make_reconstructions_of_trajectories(
                        batch, 
                        save_dir=self.reconstructions_dir, 
                        epoch=train_epoch, 
                        tokenizer=self.tokenizer.module, 
                        jowa=jowa.module,
                    )
                
        return eval_metric
    
    @torch.no_grad()
    def _eval_game(self, epoch: int, info: Dict = {}):
        logs = []
        cfg_eval = self.cfg.evaluation

        for jowa in self.all_jowas:
            buffer_size = cfg_eval.env.buffer_size
            
            # create env
            env_fn = AtariEnvWrapper(cfg_eval.env.env_name).create_env
            test_env = SingleProcessEnv(env_fn)
            
            agent = Agent(
                self.tokenizer.module, 
                jowa.module, 
                self.env_token, 
                self.dtype, 
                buffer_size, 
                self.device,
                should_plan=False,
            ).to(self.device)
            agent.eval()
            
            env = AgentEnv(
                agent, 
                test_env, 
                keymap_name='atari', 
                do_reconstruction=cfg_eval.env.do_reconstruction,
                verbose=False,
            )
            
            game = Game(
                env, 
                keymap_name='empty', 
                size=self.size, 
                fps=int(cfg_eval.env.fps), 
                verbose=bool(cfg_eval.env.header), 
                record_mode=bool(cfg_eval.env.save_mode),
                num_eval_episodes=int(cfg_eval.env.num_eval_episodes),
                save_in_rgb=False,
                record_dir=self.video_dir,
            )

            episode_info_collect = game.run(
                max_time=cfg_eval.env.max_time, 
            )
            episode_info_summary = {
                k: np.mean([i[k] for i in episode_info_collect]) \
                    for k, v in episode_info_collect[0].items() if isinstance(v, (int, float))
            }
            
            mean_return = torch.tensor(
                [episode_info_summary['return']], 
                dtype=torch.float32, 
                device=self.device,
            )
            ret_all = [torch.zeros_like(mean_return) for _ in range(self.world_size)]
            dist.all_gather(ret_all, mean_return)
            
            ret_all = torch.cat(ret_all, dim=0)
            mean_ret_all = ret_all.mean().item()

            if self.is_main_process and mean_ret_all >= self.best_return * 0.8:
                self.save_checkpoint(
                    epoch, 
                    save_agent_only=not self.cfg.common.do_checkpoint, 
                    best=True,
                    score=mean_ret_all,
                )

            if mean_ret_all > self.best_return:
                self.best_return = mean_ret_all

            log = {
                f"{str(jowa.module)}/eval/return": mean_ret_all,
                f"{str(jowa.module)}/eval/used_buffer_size": buffer_size,
            }
            log.update(info)
            logs.append(log)
        
        return logs

    def save_checkpoint(
        self, 
        epoch: int, 
        save_agent_only: bool, 
        best: bool = False, 
        score=None,
    ) -> None:
        def extract_score(filename):
            match = re.search(r'best_ckpt_score_(\d+?)_', filename)
            if match:
                return float(match.group(1))
            return float('-inf')

        if self.is_main_process:
            if best:
                assert score is not None
            
            # remove oldest dir
            ckpts = [f for f in self.ckpt_dir.glob('epoch_*') if f.name.startswith('epoch_')]
            ckpts.sort(key=lambda p: p.stat().st_mtime)
            if len(ckpts) > self.cfg.training.max_ckpts:
                shutil.rmtree(ckpts[0])

            # remove best dir with lowest score
            ckpts = [f for f in self.ckpt_dir.glob('best_ckpt_score_*')]
            ckpts.sort(key=lambda p: extract_score(p.name))
            if len(ckpts) > self.cfg.training.max_ckpts:
                shutil.rmtree(ckpts[0])
            
            # mkdir
            name_prefix = "best_ckpt_score_" + str(int(float(score))) + "_" if best else ""
            step_ckpt_dir = self.ckpt_dir / f'{name_prefix}epoch_{epoch}_step_{self.global_training_step}'
            step_ckpt_dir.mkdir(exist_ok=False, parents=False)
            
            torch.save(self.tokenizer.state_dict(), step_ckpt_dir / f'tokenizer.pt')
            for jowa in self.all_jowas:
                torch.save(jowa.state_dict(), step_ckpt_dir / f"{jowa.module}.pt")
            
            if not save_agent_only:
                torch.save(
                    self.optimizer_tokenizer.state_dict(), 
                    step_ckpt_dir / f'optimizer_tokenizer.pt'
                )
                for opt, opt_name in zip(
                    self.all_jowa_optimizers, 
                    [f'optimizer_{str(jowa.module)}' for jowa in self.all_jowas]
                ):
                    torch.save(opt.state_dict(), step_ckpt_dir / f'{opt_name}.pt')

    def load_checkpoint(
        self, 
        path, 
        component, 
        load_opt: bool = False, 
        load_start_epoch: bool = False,
    ) -> None:
        def process_param_name(name: str) -> str:
            if name.startswith('module.'):
                name = name[7:]
            elif name.startswith('_orig_mod.module.'):
                name = name[17:]
            elif name.startswith('_orig_mod.'):
                name = name[10:]
            else:
                pass
            return 'module.' + name

        if component == 'tokenizer':
            ckpt_token = torch.load(
                os.path.join(path, 'tokenizer.pt'), 
                map_location=self.device,
            )
            token_dict = self.tokenizer.state_dict()

            for name, param in ckpt_token.items():
                token_dict[process_param_name(name)] = param
            
            self.tokenizer.load_state_dict(token_dict)
            
            if load_opt:
                ckpt_opt_tokenizer = torch.load(
                    os.path.join(path, 'optimizer_tokenizer.pt'), 
                    map_location=self.device,
                )
                self.optimizer_tokenizer.load_state_dict(ckpt_opt_tokenizer)

        elif 'jowa' in component.lower():
            # modify if needs init of multiple jowas
            ckpt_jowa = torch.load(
                os.path.join(path, f'{component}.pt'), 
                map_location=self.device,
            )
            jowa_dict = self.jowa_model.state_dict()

            for name, param in ckpt_jowa.items():
                # if 'head_q' not in name:
                jowa_dict[process_param_name(name)] = param
            
            self.jowa_model.load_state_dict(jowa_dict)

            if load_opt:
                ckpt_opt_jowa = torch.load(
                    os.path.join(path, f'optimizer_{component}.pt'), 
                    map_location=self.device,
                )
                self.optimizer_jowa.load_state_dict(ckpt_opt_jowa)
            
            if load_start_epoch:
                self.start_epoch = int(path.split('_')[-3]) + 1
            
        else:
            raise IndexError(f"component {component} does not exists.")
        
        if self.is_main_process:
            print(f'Load {component} from {path}.')

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}

    def _gather_tensor(self, batch: Dict[str, float]) -> Dict[str, float]:
        res = defaultdict(float)
        for k, v in batch.items():
            tensor_v = torch.tensor(v, device=self.device)
            dist.all_reduce(tensor_v, op=dist.ReduceOp.SUM)
            mean_v = tensor_v / self.world_size
            res[k] = mean_v.item()
        return res

    def _gather_batch(self, batch):
        gathered_batches = {}
        for key, tensor in batch.items():
            gathered_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered_tensors, tensor)
            gathered_batches[key] = torch.cat(gathered_tensors, dim=0)
        return gathered_batches
        
    def finish(self) -> None:
        wandb.finish()
        dist.destroy_process_group()


# @hydra_main(config_path="../config", config_name="train_40M")
@hydra_main(config_path="../config", config_name="finetune_150M")
def get_hydra_config(cfg: DictConfig) -> DictConfig:
    return cfg


if __name__ == '__main__':
    config = get_hydra_config()
    trainer = Trainer(config)
    trainer.run()
