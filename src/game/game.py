from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Optional, Tuple, Union

import gym
import numpy as np
from PIL import Image

from envs import WorldModelEnv
from utils import make_video

from .agent_env import AgentEnv
from .keymap import get_keymap_and_action_names


class Game:
    def __init__(
        self, 
        env: Union[gym.Env, WorldModelEnv, AgentEnv], 
        keymap_name: str, 
        size: Tuple[int, int], 
        fps: int, 
        num_eval_episodes: int,
        verbose: bool = False, 
        record_mode: bool = False, 
        save_in_rgb: bool = False,
        record_dir: Path = None,
    ) -> None:
        self.env = env
        self.height, self.width = size
        self.fps = fps
        self.verbose = verbose
        self.record_mode = record_mode
        self.save_in_rgb = save_in_rgb
        self.keymap, self.action_names = get_keymap_and_action_names(
            keymap_name
        )
        self.num_eval_episodes = num_eval_episodes
        self.episodes = 0

        self.record_dir = Path('media') if record_dir is None else record_dir

    def run(
        self, 
        max_time: Optional[float] = None, 
        max_steps: Optional[float] = None,
        name_prefix: Optional[str] = None,
    ) -> None:
        if isinstance(self.env, gym.Env):
            _, info = self.env.reset(return_info=True)
            img = info['rgb']
        else:
            self.env.reset()
            img = self.env.render()

        episode_buffer = []
        segment_buffer = []
        episode_info_collect = []

        should_stop = False
        
        while not should_stop:
            _, reward, done, info = self.env.step()

            img = info['rgb'] if isinstance(self.env, gym.Env) else self.env.render()
            if self.record_mode:
                if self.save_in_rgb:
                    saved_img = self.env.env.env._env.environment.ale.getScreenRGB2()
                    episode_buffer.append(saved_img)
                else:
                    episode_buffer.append(np.array(img))

            if done or (
                max_time is not None and info['time'] >= max_time
            ) or (
                max_steps is not None and info['timestep'] >= max_steps
            ):
                self.episodes += 1
                pprint(info)
                episode_info_collect.append(info)
                self.env.reset()

                if self.record_mode:
                    name_prefix_ = name_prefix + "_" if name_prefix is not None else ""
                    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    
                    self.save_recording(
                        np.stack(episode_buffer),
                        f'{name_prefix}score_{info["return"]}_timestamp_{date}'
                    )
                    episode_buffer = []
                    
                if self.episodes >= self.num_eval_episodes:
                    should_stop = True

        return episode_info_collect

    def save_recording(self, frames: np.ndarray, name: str):
        self.record_dir.mkdir(exist_ok=True, parents=True)
        make_video(self.record_dir / f'{name}.mp4', fps=15, frames=frames)
        print(f'Saved recording {name}.')
