import time
from pprint import pprint
from typing import List, Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from agent import Agent
from envs import SingleProcessEnv, MultiProcessEnv
from .keymap import get_keymap_and_action_names


class AgentEnv:
    def __init__(
        self, 
        agent: Agent, 
        env: Union[SingleProcessEnv, MultiProcessEnv], 
        keymap_name: str, 
        do_reconstruction: bool,
        verbose: bool = True,
    ) -> None:
        self.agent = agent
        self.env = env
        _, self.action_names = get_keymap_and_action_names(keymap_name)
        self.do_reconstruction = do_reconstruction
        self.verbose = verbose
        self.obs = None
        self.start_time = None
        self._t = None
        self._return = None
        self._cliped_return = None

    def _to_tensor(self, obs: np.ndarray):
        return rearrange(
            torch.FloatTensor(obs).div(255), 
            'n h w c -> n c h w',
        ).to(self.agent.device)

    def _to_array(self, obs: torch.FloatTensor):
        return rearrange(obs, 'n c h w -> n h w c').mul(
            255).cpu().numpy().astype(np.uint8)

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self.obs = self._to_tensor(obs)
        
        self.start_time = time.time()

        num_envs = self.env.num_envs
        self.agent.reset(num_envs)
        self._t = np.zeros(num_envs, dtype=np.int32)
        self._return = np.zeros(num_envs)
        self._cliped_return = np.zeros(num_envs)
        return obs  # (B, H, W, C)

    def step(self, *args, **kwargs) -> torch.FloatTensor:
        with torch.no_grad():
            act_tensor = self.agent.act(self.obs)  # act_tensor.shape: (B,)
            act = act_tensor.cpu().numpy()
        
        obs, reward, done, _ = self.env.step(act)

        self.obs = self._to_tensor(obs)
        self.agent.update_memory(act_tensor.view(-1, 1))
        self._t += 1 * self.env.mask_dones
        self._return += reward * self.env.mask_dones
        self._cliped_return += np.sign(reward) * self.env.mask_dones
        info = {
            'timestep': self._t,
            'done': self.env.done_tracker,
            'action': act,
            'action_name': [self.action_names[a] for a in act], 
            'return': self._return,
            'clipped_return': self._cliped_return,
            'time': round(time.time() - self.start_time, 1),
        }
        if self.verbose and self._t.max() % 100 == 0:
            pprint(info)
        return obs, reward, done, info

    def render(self) -> np.ndarray:
        original_obs = self._to_array(self.obs)  # (B, H, W, C)
        
        if self.do_reconstruction:
            rec = torch.clamp(
                self.agent.tokenizer.encode_decode(
                    self.obs, 
                    should_preprocess=True,
                    should_postprocess=True,
                ), 0, 1,
            )
            arr = np.concatenate(
                (original_obs[..., 0], self._to_array(rec)[..., 0]), 
                axis=-1,
            )
        else:
            arr = original_obs[..., 0]
        
        return arr  # (B, H, W)