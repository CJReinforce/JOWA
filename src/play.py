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
from agent_expand_kv_cache import Agent
from atari_env_wrapper import AtariEnvWrapper
from envs import SingleProcessEnv
from game import AgentEnv, Game
from models.world_model_q_distributional import WorldModel
from utils import set_seed

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ["SDL_VIDEODRIVER"] = "dummy"


@hydra.main(config_path="../config", config_name="eval")
def main(cfg: DictConfig):
    print('#' * 50)
    info = {
        'game': cfg.common.game_name, 
        'num_given_steps': cfg.common.num_given_steps, 
        'use_planning': cfg.common.use_planning,
        'beam_width': cfg.common.beam_width, 
        'horizon': cfg.common.horizon,
        'device': cfg.common.device,
    }
    pprint(info)
    print('#' * 50)

    device = torch.device(cfg.common.device)
    env_fn = AtariEnvWrapper(cfg.common.game_name).create_env
    test_env = SingleProcessEnv(env_fn)
    
    set_seed(cfg.common.seed)

    h, w, _ = test_env.env.observation_space.shape
    size = [h, w]
    
    tokenizer = instantiate(cfg.tokenizer)
    world_model = WorldModel(
        obs_vocab_size=cfg.tokenizer.vocab_size,
        act_vocab_size=ATARI_NUM_ACTIONS,
        config_transformer=instantiate(cfg.transformer),
        config_critic=instantiate(cfg.critic_head),
        device=device,
    )
    
    env_token = torch.as_tensor(
        [GAME_NAMES.index(cfg.common.game_name)], 
        dtype=torch.long, 
        device=device
    )
    agent = Agent(
        tokenizer, 
        world_model, 
        env_token, 
        cfg.common.dtype, 
        cfg.common.num_given_steps, 
        device, 
        use_kv_cache=False, 
        should_plan=cfg.common.use_planning,
        beam_width=cfg.common.beam_width,
        horizon=cfg.common.horizon
    ).to(device)
    
    agent.load(
        cfg.initialization.path_to_checkpoint, 
        cfg.initialization.load_tokenizer, 
        cfg.initialization.load_world_model,
        cfg.initialization.tokenizer_name,
        cfg.initialization.world_model_name,
    )
    agent.eval()
    
    env = AgentEnv(
        agent, 
        test_env, 
        'atari', 
        do_reconstruction=cfg.common.do_reconstruction,
        verbose=cfg.common.verbose,
    )
    keymap = 'empty'
    if cfg.common.do_reconstruction:
        size[1] *= 2

    game = Game(
        env, 
        keymap_name=keymap, 
        size=size, 
        fps=cfg.common.fps, 
        verbose=bool(cfg.common.header), 
        record_mode=bool(cfg.common.save_mode),
        num_eval_episodes=cfg.common.num_eval_episodes,
    )
    episode_info_collect = game.run(
        max_time=cfg.common.max_time, 
        num_given_steps=cfg.common.num_given_steps, 
        max_steps=cfg.common.max_steps,
    )
    
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
