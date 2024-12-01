from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

from utils import make_video

from .agent_env import AgentEnv


class Game:
    def __init__(
        self, 
        env: AgentEnv, 
        fps: int, 
        num_eval_episodes: int,
        record_mode: bool = False, 
        save_in_rgb: bool = False,
        record_dir: Path = None,
        *args,
        **kwargs,
    ) -> None:
        self.env = env
        self.fps = fps
        self.record_mode = record_mode
        self.save_in_rgb = save_in_rgb
        self.num_eval_episodes = num_eval_episodes
        self.episodes = 0
        self.record_dir = Path('media') if record_dir is None else record_dir

    def run(
        self, 
        max_time: Optional[float] = None, 
        max_steps: Optional[float] = None,
        name_prefix: Optional[str] = None,
    ) -> None:
        name_prefix_ = "" if name_prefix is None else name_prefix + "_"
        img = self.env.reset()  # (B, H, W, C)

        episode_buffer = {i:[] for i in range(len(img))}
        episode_info_collect = []
        
        while True:
            _, _, done, info = self.env.step()
            
            if self.save_in_rgb:
                img = self.env.env.get_rgb_observation()  # (B, H, W, C)
            else:
                img = self.env.render()  # (B, H, W)
            
            if self.record_mode:
                for i in range(len(img)):
                    episode_buffer[i].append(img[i])

            if self.env.env.all_done or (
                max_time is not None and info['time'] >= max_time
            ) or (
                max_steps is not None and info['timestep'].max() >= max_steps
            ):
                self.episodes += 1
                pprint(info)
                episode_info_collect.append(info)

                self.env.reset()

                if self.record_mode:
                    for i in range(len(img)):
                        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                        self.save_recording(
                            np.stack(episode_buffer[i][:info['timestep'][i]]),
                            f'{name_prefix_}score_{info["return"][i]}_timestamp_{date}'
                        )
                    
                        episode_buffer[i] = []
                    
                if self.episodes >= self.num_eval_episodes:
                    break

        return episode_info_collect

    def save_recording(self, frames: np.ndarray, name: str):
        self.record_dir.mkdir(exist_ok=True, parents=True)
        make_video(self.record_dir / f'{name}.mp4', fps=15, frames=frames)
        print(f'Saved recording {name}.')