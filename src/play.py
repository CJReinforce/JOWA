import os
import warnings
from pprint import pprint

import hydra
import numpy as np
import scipy.stats
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from action_tokenizer import ATARI_NUM_ACTIONS, GAME_NAMES
from agent import Agent
from envs import AtariEnvWrapper, SingleProcessEnv, MultiProcessEnv
from game import AgentEnv, Game
from models.jowa_model import JOWAModel
from utils import set_seed

warnings.filterwarnings("ignore")


@hydra.main(config_path="../config", config_name="eval")
def main(cfg: DictConfig):
    device = torch.device(cfg.common.device)
    env_fn = AtariEnvWrapper(cfg.common.game_name).create_env
    test_env = MultiProcessEnv(env_fn, cfg.common.num_envs)
    
    set_seed(cfg.common.seed)
    
    tokenizer = instantiate(cfg.tokenizer)
    jowa_model = JOWAModel(
        obs_vocab_size=cfg.tokenizer.vocab_size,
        act_vocab_size=ATARI_NUM_ACTIONS,
        config_transformer=instantiate(cfg.transformer),
        config_critic_arch=cfg.critic_head,
        config_critic_train=cfg.action,
        device=device,
    )
    
    agent = Agent(
        tokenizer, 
        jowa_model, 
        GAME_NAMES.index(cfg.common.game_name), 
        test_env.num_envs, 
        cfg.common.dtype, 
        cfg.common.num_given_steps, 
        device, 
        should_plan=cfg.common.use_planning,
        beam_width=cfg.common.beam_width,
        horizon=cfg.common.horizon,
        use_mean=cfg.common.use_mean,
        use_count=cfg.common.use_count,
        temperature=cfg.common.temperature,
        num_simulations=cfg.common.num_simulations,
    ).to(device)
    
    agent.load(
        cfg.initialization.path_to_checkpoint, 
        cfg.initialization.load_tokenizer, 
        cfg.initialization.load_jowa_model,
        cfg.initialization.tokenizer_name,
        cfg.initialization.jowa_model_name,
    )
    agent.eval()
    
    env = AgentEnv(
        agent, 
        test_env, 
        'atari', 
        do_reconstruction=cfg.common.do_reconstruction,
        verbose=cfg.common.verbose,
    )

    game = Game(
        env, 
        fps=cfg.common.fps, 
        record_mode=bool(cfg.common.save_mode),
        num_eval_episodes=cfg.common.num_eval_episodes,
        save_in_rgb=cfg.common.save_rgb_img
    )
    episode_info_collect = game.run(
        max_time=cfg.common.max_time, 
        max_steps=cfg.common.max_steps,
        name_prefix=f'{cfg.initialization.jowa_model_name}_play_{cfg.common.game_name}',
    )
    test_env.close()
    
    static_metric('clipped_return', episode_info_collect)
    static_metric('return', episode_info_collect)
        
        
def static_metric(name, episode_info_collect):
    metric = [info[name] for info in episode_info_collect]
    print()
    print(f"Original {name}s: {metric}")
    metric = np.array(metric).reshape(-1,1)
    
    print(f"Mean {name}: {np.mean(metric)}")
    print(f"Std {name}: {np.std(metric)}")
    print(f"Median {name}: {np.median(metric)}")
    print(f"IQM {name}: {scipy.stats.trim_mean(metric, proportiontocut=0.25, axis=None)}")


if __name__ == "__main__":
    main()