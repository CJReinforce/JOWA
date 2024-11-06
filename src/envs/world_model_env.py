import random
from typing import List, Optional, Union

import gym
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.distributions.categorical import Categorical

from action_tokenizer import (
    action_masks,
    ATARI_NUM_ACTIONS,
    GAME_NAMES,
    LIMITED_ACTION_TO_FULL_ACTION,
)
from models.tokenizer import Tokenizer
from utils import Batch


class WorldModelEnv:
    def __init__(
        self, 
        tokenizer: Tokenizer, 
        jowa: torch.nn.Module,
        device: Union[str, torch.device], 
        env_tokens: torch.LongTensor,
        env: Optional[gym.Env] = None,
    ) -> None:
        self.device = torch.device(device)
        self.jowa = jowa.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()
        
        self._num_obs_tokens = self.jowa.config_transformer.tokens_per_block - 1
        self.env_tokens, self.env = env_tokens.to(self.device), env
        self.keys_values_wm, self.obs_tokens, self.last_actions = [None] * 3

    @property
    def num_observations_tokens(self) -> int:
        return self._num_obs_tokens
    
    @property
    def epsilon(self):
        init_eps = self.jowa.config_critic_train.initial_epsilon
        final_eps = self.jowa.config_critic_train.final_epsilon
        decay_steps = self.jowa.config_critic_train.decay_steps
        return max(
            final_eps, 
            init_eps - (init_eps - final_eps) * self.jowa.training_steps / decay_steps,
        )
    
    @torch.no_grad()
    def choose_action(
        self, 
        critic: torch.FloatTensor, 
        mode: str = "epsilon-greedy", 
        epsilon: Optional[float] = None, 
        k: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Choose action using the given mode.
        """
        assert mode in ["epsilon-greedy", "greedy", "top-k", "logits"]
        if mode == "top-k":
            assert isinstance(k, int)
         
        if hasattr(self.jowa, 'atoms_support'):  # (num_q, B, L, #actions * atoms)
            critic = (
                rearrange(
                    critic[..., -1, :], 'q b (a z) -> q b a z', 
                    a=self.jowa.act_vocab_size,
                ).softmax(-1).mean(0) * self.jowa.atoms_support.view(1, 1, -1)
            ).sum(-1)
        else:  # (num_q, B, L or 1, #actions)
            critic = critic[..., -1, :].mean(0)
        
        # action mask according to env
        mask = action_masks.to(self.env_tokens.device)[self.env_tokens]
        critic = critic.masked_fill(~mask, -torch.inf)

        if mode == "logits":
            return critic.cpu()
        elif mode == "greedy":
            return critic.argmax(dim=-1).cpu()
        elif mode == "epsilon-greedy":
            actions = critic.argmax(dim=-1).cpu()

            epsilon = self.epsilon if epsilon is None else epsilon
            random_index = torch.rand(critic.size(0)) < epsilon
            
            if random_index.any():
                random_actions = []
                for index in torch.where(random_index)[0]:
                    action_set = LIMITED_ACTION_TO_FULL_ACTION[
                        GAME_NAMES[self.env_tokens[index]]]
                    random_action = action_set[torch.randint(len(action_set), (1,)).item()]
                    random_actions.append(random_action)
                
                actions[random_index] = torch.LongTensor(random_actions)
            
            return actions
        else:
            return torch.topk(critic, k, dim=-1).indices.cpu()  # (B, k)

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = torchvision.transforms.functional.to_tensor(
            self.env.reset()).to(self.device).unsqueeze(0)  # (1, C, H, W) in [0., 1.]
        return self.reset_from_initial_observations(obs)

    @torch.no_grad()
    def reset_from_initial_observations(
        self, 
        observations: Union[torch.FloatTensor, torch.LongTensor, Batch], 
        act_mode: str = "epsilon-greedy", 
        epsilon: Optional[float] = None, 
        k: Optional[int] = None,
    ) -> torch.FloatTensor:
        if isinstance(observations, dict):
            assert observations['observations'].size(1) == observations['actions'].size(1) + 1
            
            obs_tokens = self.tokenizer.encode(
                observations['observations'], 
                should_preprocess=True,
            ).tokens  # (B, L, K)
            num_observations_tokens = obs_tokens.size(-1)
            additional_col = torch.zeros(
                observations['actions'].size(0), 1, 
                dtype=observations['actions'].dtype, 
                device=observations['actions'].device,
            )
            actions = torch.cat((observations['actions'], additional_col), dim=1)
            act_tokens = rearrange(actions, 'b l -> b l 1')
            obs_tokens = rearrange(
                torch.cat((obs_tokens, act_tokens), dim=2), 
                'b l k1 -> b (l k1)',
            )[:, :-1]
        elif observations.ndim == 4:
            obs_tokens = self.tokenizer.encode(
                observations, 
                should_preprocess=True,
            ).tokens    # (B, C, H, W) -> (B, K)
            _, num_observations_tokens = obs_tokens.shape
        elif observations.ndim == 2:
            obs_tokens = observations
            num_observations_tokens = self.jowa.config_transformer.tokens_per_block - 1
        else:
            raise NotImplementedError
            
        if self.num_observations_tokens is None:
            self._num_obs_tokens = num_observations_tokens

        self.refresh_keys_values_with_initial_obs_tokens(
            obs_tokens, 
            act_mode=act_mode, 
            epsilon=epsilon, 
            k=k,
        )
        self.obs_tokens = obs_tokens[:, -self.num_observations_tokens:]

        return self.decode_obs_tokens()

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(
        self, 
        obs_tokens: torch.LongTensor, 
        act_mode: str = "epsilon-greedy", 
        epsilon: Optional[float] = None, 
        k: Optional[int] = None,
    ) -> torch.FloatTensor:
        n, num_observations_tokens = obs_tokens.shape
        self.keys_values_wm = self.jowa.transformer.generate_empty_keys_values(
            n=n, max_tokens=self.jowa.config_transformer.max_tokens)
        outputs_wm = self.jowa(
            obs_tokens, 
            self.env_tokens, 
            past_keys_values=self.keys_values_wm,
        )
        actions = self.choose_action(
            outputs_wm.logits_q, 
            mode=act_mode, 
            epsilon=epsilon, 
            k=k,
        )
        self.last_actions = actions
        return outputs_wm.output_sequence, actions

    @torch.no_grad()
    def step(
        self, 
        action: Optional[Union[int, np.ndarray, torch.LongTensor]] = None, 
        should_predict_next_obs: bool = True, 
        should_sample: bool = True, 
        act_mode: str = "epsilon-greedy", 
        epsilon: Optional[float] = None, 
        k: Optional[int] = None,
        kv_cache=None,
        jowa=None,
    ):
        kv_cache = kv_cache if kv_cache is not None else self.keys_values_wm
        jowa = jowa if jowa is not None else self.jowa
        assert kv_cache is not None and self.num_observations_tokens is not None

        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1
        output_sequence, obs_tokens = [], []

        if kv_cache.size + num_passes > self.jowa.config_transformer.max_tokens:
            raise ValueError

        if action is None:
            assert self.last_actions.ndim == 1
            action = self.last_actions
        
        token = action.clone().detach() if isinstance(action, torch.Tensor) else \
            torch.tensor(action, dtype=torch.long)
        token = token.reshape(-1, 1).to(self.device)  # (B, 1)
        jowa = jowa.to(self.device)

        for k in range(num_passes):  # assumption that there is only one action token.
            outputs_wm = jowa(token, self.env_tokens, past_keys_values=kv_cache)
            output_sequence.append(outputs_wm.output_sequence)

            if k == 0:
                if should_sample:
                    reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                    done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)
                else:
                    reward = Categorical(logits=outputs_wm.logits_rewards).mode.float().cpu().numpy().reshape(-1) - 1   # (B,)
                    done = Categorical(logits=outputs_wm.logits_ends).mode.cpu().numpy().astype(bool).reshape(-1)       # (B,)

            if k == self.num_observations_tokens:
                self.last_actions = self.choose_action(
                    outputs_wm.logits_q, 
                    mode=act_mode, 
                    epsilon=epsilon, 
                    k=k,
                )
            
            if k < self.num_observations_tokens:
                if should_sample:
                    token = Categorical(logits=outputs_wm.logits_observations).sample()
                else:
                    token = Categorical(logits=outputs_wm.logits_observations).mode
                obs_tokens.append(token)

        output_sequence = torch.cat(output_sequence, dim=1)   # (B, 1 + K, E)
        self.obs_tokens = torch.cat(obs_tokens, dim=1)        # (B, K)

        obs = self.decode_obs_tokens() if should_predict_next_obs else None
        return obs, reward, done, {
            'action': self.last_actions, 
            'obs_tokens': self.obs_tokens,
        }

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).to(torch.uint8).numpy()
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self, obs_tokens=None) -> List[Image.Image]:
        embedded_tokens = self.tokenizer.embedding(
            obs_tokens if obs_tokens is not None else self.obs_tokens)     # (B, K, E)
        z = rearrange(
            embedded_tokens, 
            'b (h w) e -> b e h w', 
            h=int(np.sqrt(self.num_observations_tokens))
        ).contiguous()
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        return torch.clamp(rec, 0, 1)

    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]
