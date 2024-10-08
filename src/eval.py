import os
import random
import warnings
from pprint import pprint

import hydra
import numpy as np
from tqdm import tqdm
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from action_tokenizer import ATARI_NUM_ACTIONS, GAME_NAMES
from atari_env_wrapper import AtariEnvWrapper
from evaluation.collector import Collector
from evaluation.policy import Policy
from evaluation.vectorized_environment import SubprocVectorEnv
from make_reconstructions import *
from models.world_model import WorldModel

# from torchvision.transforms import ToTensor


warnings.filterwarnings("ignore")


class Evaluator:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.common.device)

        # initialize tokenizer
        self.tokenizer = instantiate(cfg.tokenizer)
        self.tokenizer.to(self.device)

        # initialize world model
        self.world_model = WorldModel(obs_vocab_size=cfg.tokenizer.vocab_size,
                                      act_vocab_size=ATARI_NUM_ACTIONS,
                                      config_transformer=instantiate(cfg.world_model),
                                      config_critic=instantiate(cfg.actor_critic),)
        self.world_model.to(self.device)
        
        self.load_checkpoint(cfg.initialization.load_tokenizer, 'tokenizer')
        self.load_checkpoint(cfg.initialization.load_world_model, 'world_model')

        self.tokenizer.eval()
        self.world_model.eval()
        
        # create atari envs
        # atari_envs = [AtariEnvWrapper(game_name).create_env for game_name in TRAIN_ENVS]
        # env_tokens = torch.as_tensor([GAME_NAMES.index(game_name) for game_name in TRAIN_ENVS], dtype=torch.long, device=self.device)
        # self.vectorEnvs = SubprocVectorEnv(atari_envs)

        # Debug...
        # test_idx = TRAIN_ENVS.index(cfg.common.game_name)
        print("#"*20 + f"Testing on {cfg.common.game_name}" + "#"*20)
        env_tokens = torch.as_tensor([GAME_NAMES.index(cfg.common.game_name)], dtype=torch.long, device=self.device)
        self.vectorEnvs = SubprocVectorEnv([AtariEnvWrapper(cfg.common.game_name).create_env])
                
        self.policy = Policy(
            self.tokenizer, self.world_model, 
            num_envs=len(self.vectorEnvs), 
            dtype=get_dtype(cfg.common.dtype),
            env_tokens=env_tokens,
            num_given_steps=cfg.common.num_given_steps
        )

    @torch.no_grad()
    def run(self):
        
        rews = np.zeros((len(self.vectorEnvs), self.cfg.common.num_seeds * self.cfg.common.eval_episodes_per_seed), dtype=float)
        lens = np.zeros((len(self.vectorEnvs), self.cfg.common.num_seeds * self.cfg.common.eval_episodes_per_seed), dtype=int)
        
        try:
            for seed in tqdm(range(self.cfg.common.num_seeds), 
                             desc=f"num_given_steps: {self.cfg.common.num_given_steps}"):
                set_seed(seed)
                self.vectorEnvs.seed([seed] * len(self.vectorEnvs))

                collector = Collector(self.policy, self.vectorEnvs, seed)
                result = collector.collect(n_episode=self.cfg.common.eval_episodes_per_seed)
                # pprint(result)

                rews[:, seed * self.cfg.common.eval_episodes_per_seed:(seed + 1) * self.cfg.common.eval_episodes_per_seed] = result['rews']
                lens[:, seed * self.cfg.common.eval_episodes_per_seed:(seed + 1) * self.cfg.common.eval_episodes_per_seed] = result['lens']
        
        except KeyboardInterrupt:
            rews = rews[:, :seed * self.cfg.common.eval_episodes_per_seed]
            lens = lens[:, :seed * self.cfg.common.eval_episodes_per_seed]

        eval_result = self.evaluate_episode(rews, lens)
        pprint(eval_result)
            
    def evaluate_episode(self, rews, lens):
        mean_rew = np.mean(rews, axis=1)
        std_rew = np.std(rews, axis=1)
        iqm_rew = calculate_iqm(rews)
        # compute the mean of top 3 rew
        top3_indices = np.argsort(rews, axis=1)[:, -3:]
        top3_values = np.take_along_axis(rews, top3_indices, axis=1)
        top3_mean = np.mean(top3_values, axis=1)

        mean_len = np.mean(lens, axis=1)
        std_len = np.std(lens, axis=1)
        iqm_len = calculate_iqm(lens)

        result = {
            'rew': {
                'mean': mean_rew,
                'std': std_rew,
                'iqm': iqm_rew,
                'top3': top3_mean
            },
            'len': {
                'mean': mean_len,
                'std': std_len,
                'iqm': iqm_len
            }
        }
        return result
    
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
            return name
            
        if component == 'tokenizer':
            ckpt_token = torch.load(os.path.join(path, 'tokenizer.pt'), map_location=self.device)
            tokenizer_dict = self.tokenizer.state_dict()
            
            for name, param in ckpt_token.items():
                if process_param_name(name) in tokenizer_dict:
                    tokenizer_dict[process_param_name(name)] = param
            self.tokenizer.load_state_dict(tokenizer_dict, strict=False)
        
        elif component == 'world_model':
            ckpt_world = torch.load(os.path.join(path, 'world_model.pt'), map_location=self.device)
            world_model_dict = self.world_model.state_dict()
            
            for name, param in ckpt_world.items():
                if process_param_name(name) in world_model_dict:
                    world_model_dict[process_param_name(name)] = param
            self.world_model.load_state_dict(world_model_dict, strict=False)
        else:
            raise NotImplementedError(f"component {component} is not implemented.")
        
        print(f'Successfully loaded {component} from {path}.')


def get_dtype(dtype: str):
    return torch.float16 if dtype == 'float16' else torch.bfloat16 if dtype == 'bfloat16' else torch.float32


def calculate_iqm(data, axis=1):
    q1 = np.percentile(data, 25, axis=axis)
    q3 = np.percentile(data, 75, axis=axis)
    
    mask = np.logical_and(data >= q1[..., np.newaxis], data <= q3[..., np.newaxis])
    iqm_data = np.ma.masked_array(data, ~mask)
    iqm = np.mean(iqm_data, axis=axis)
    
    return iqm.data


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    # tf.random.set_seed(seed)


@hydra.main(config_path="../config", config_name="eval")
def main(cfg: DictConfig):
    evaluator = Evaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()