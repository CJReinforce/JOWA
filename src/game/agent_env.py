import time

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from agent_expand_kv_cache import Agent
from envs import SingleProcessEnv, WorldModelEnv
from game.keymap import get_keymap_and_action_names


class AgentEnv:
    def __init__(
        self, 
        agent: Agent, 
        env: SingleProcessEnv, 
        keymap_name: str, 
        do_reconstruction: bool,
        verbose: bool = True,
    ) -> None:
        assert isinstance(env, SingleProcessEnv) or isinstance(env, WorldModelEnv)
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
        assert isinstance(obs, np.ndarray) and obs.dtype == np.uint8
        return rearrange(torch.FloatTensor(obs).div(255), 'n h w c -> n c h w').to(self.agent.device)

    def _to_array(self, obs: torch.FloatTensor):
        assert obs.ndim == 4 and obs.size(0) == 1
        return obs[0].mul(255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    def reset(self):
        obs = self.env.reset()
        self.obs = self._to_tensor(obs) if isinstance(self.env, SingleProcessEnv) else obs
        self.start_time = time.time()
        self.agent.reset(1)
        self._t = 0
        self._return = 0
        self._cliped_return = 0
        return obs

    def step(self, *args, **kwargs) -> torch.FloatTensor:
        with torch.no_grad():
            act_tensor = self.agent.act(self.obs)
            act = act_tensor.cpu().numpy()
        obs, reward, done, _ = self.env.step(act)
        self.obs = self._to_tensor(obs) if isinstance(self.env, SingleProcessEnv) else obs
        self.agent.update_memory(act_tensor)
        self._t += 1
        self._return += reward[0]
        self._cliped_return += np.sign(reward[0])
        info = {
            'timestep': self._t,
            'action': self.action_names[act[0]],
            'return': self._return,
            'clipped_return': self._cliped_return,
            'time': round(time.time() - self.start_time, 1),
        }
        if self.verbose:
            print(info)
        return obs, reward, done, info

    def render(self) -> Image.Image:
        assert self.obs.size() == (1, 1, 84, 84)
        original_obs = self.env.env._env._pool_and_resize() if isinstance(self.env, SingleProcessEnv) else self._to_array(self.obs)
        
        if self.do_reconstruction:
            rec = torch.clamp(self.agent.tokenizer.encode_decode(self.obs, should_preprocess=True, should_postprocess=True), 0, 1)
            arr = np.concatenate((original_obs[..., 0], self._to_array(rec)[..., 0]), axis=1)
            # rec = self._to_array(resize(rec, original_obs.shape[:2], interpolation=InterpolationMode.NEAREST))
            # resized_obs = self._to_array(resize(self.obs, original_obs.shape[:2], interpolation=InterpolationMode.NEAREST))
            # arr = np.concatenate((original_obs, resized_obs, rec), axis=1)
        else:
            arr = original_obs[..., 0]
            
        img = Image.fromarray(arr)
        return img

