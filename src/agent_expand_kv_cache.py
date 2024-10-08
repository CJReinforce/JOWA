import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.cuda.amp import autocast

from action_tokenizer import (
    ATARI_NUM_ACTIONS,
    GAME_NAMES,
    LIMITED_ACTION_TO_FULL_ACTION,
)
from envs.world_model_env import WorldModelEnv
from models.kv_caching import concate_kv, split_kv
from models.tokenizer import Tokenizer
from models.world_model_all_in_one import WorldModel


class Agent(nn.Module):
    def __init__(
        self, 
        tokenizer: Tokenizer, 
        world_model: WorldModel, 
        env_token: int, 
        dtype=torch.float32, 
        num_given_steps: int = 4,
        device: torch.device = torch.device('cuda'),
        use_kv_cache: bool = False,  # exist bugs using kv cache! 
        should_plan: bool = False,
        beam_width: int = 2,
        horizon: int = 2,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        
        self.dtype = get_dtype(dtype) if isinstance(dtype, str) else dtype
        self.device = device
        self.num_given_steps = num_given_steps
        self.use_kv_cache = use_kv_cache
        self.should_plan = should_plan
        # beam search
        self.beam_width = beam_width
        self.horizon = horizon
        
        self.env_token = torch.LongTensor([env_token]).to(self.device)
        self.action_mask = torch.tensor(
            [action in LIMITED_ACTION_TO_FULL_ACTION[GAME_NAMES[env_token]] for action in range(ATARI_NUM_ACTIONS)], 
            dtype=torch.bool, device=self.device
        )
        self.reset(1)
        
    def reset(self, n: int) -> None:
        if self.use_kv_cache:
            self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(
                n=n, max_tokens=self.num_given_steps * self.world_model.config_transformer.tokens_per_block,
            )
        else:
            self.token_buffer = torch.zeros(
                n, self.num_given_steps * self.world_model.config_transformer.tokens_per_block, 
                dtype=torch.long, device=self.device
            )
            self.used_token_idx = 0

    def load(
            self, 
            path_to_checkpoint: Path, 
            load_tokenizer: bool = True, 
            load_world_model: bool = True, 
            tokenizer_name: Optional[str] = None, 
            wm_name: Optional[str] = None,
        ) -> None:
        def reverse_process_param_name(name: str, ckpt) -> str:
            for prefix in ['_orig_mod.module.', '', 'module.', '_orig_mod.']:
                new_name = prefix + name
                if new_name in ckpt:
                    return new_name
                
            raise KeyError(f"{name} not found in ckpt.")

        tokenizer_name = 'tokenizer.pt' if tokenizer_name is None else tokenizer_name if tokenizer_name.endswith('.pt') else tokenizer_name + '.pt'
        wm_name = 'world_model.pt' if wm_name is None else wm_name if wm_name.endswith('.pt') else wm_name + '.pt' 

        if load_tokenizer:
            ckpt_token = torch.load(os.path.join(path_to_checkpoint, tokenizer_name), map_location=self.device)
            tokenizer_dict = self.tokenizer.state_dict()

            for name in tokenizer_dict.keys():
                tokenizer_dict[name] = ckpt_token[reverse_process_param_name(name, ckpt_token)]
    
            self.tokenizer.load_state_dict(tokenizer_dict)
        
        if load_world_model:
            ckpt_world = torch.load(os.path.join(path_to_checkpoint, wm_name), map_location=self.device)
            world_model_dict = self.world_model.state_dict()

            for name in world_model_dict.keys():
                world_model_dict[name] = ckpt_world[reverse_process_param_name(name, ckpt_world)]
            
            self.world_model.load_state_dict(world_model_dict)
        
        print(f"Successfully loaded {'tokenizer' if load_tokenizer else ''} {'world model' if load_world_model else ''} from {path_to_checkpoint}.")

    def eval(self):
        super().eval()
        self.tokenizer.eval()
        self.world_model.eval()
        return self

    def update_memory(self, input: torch.LongTensor) -> None:
        if not self.use_kv_cache:
            if input.ndim == 2:
                input = input[0]
            assert input.ndim == 1
            
            L_in = input.size(0)
            L_out = self.world_model.config_transformer.tokens_per_block
            
            if self.used_token_idx + L_in <= self.token_buffer.size(1):
                self.token_buffer[0, self.used_token_idx:self.used_token_idx + L_in] = input
                self.used_token_idx += L_in
            else:
                buffer_clone = self.token_buffer.clone()
                self.token_buffer[0, :self.used_token_idx - L_out] = buffer_clone[0, L_out:self.used_token_idx]
                self.token_buffer[0, self.used_token_idx - L_out:self.used_token_idx - L_out + L_in] = input
                self.used_token_idx += L_in - L_out
        else:
            if input.ndim == 1:
                input = input.unsqueeze(0)
            assert input.ndim == 2
            
            with autocast(dtype=self.dtype):
                self.world_model(input, self.env_token, past_keys_values=self.keys_values_wm)
    
    def choose_action(self, critic: torch.FloatTensor) -> torch.LongTensor:
        """
        Choose action using greedy policy.
        """
        if hasattr(self.world_model, 'atoms_support'):  # (num_q, B, L or 1, #actions * atoms)
            critic = (
                rearrange(
                    critic[..., -1, :], 'q b (a z) -> q b a z', a=self.world_model.act_vocab_size,
                ).softmax(-1).mean(0) * self.world_model.atoms_support.view(1, 1, -1)
            ).sum(-1)
        else:  # (num_q, B, L or 1, #actions)
            critic = critic[..., -1, :].mean(0)
        assert critic.ndim == 2  # (B, #actions)

        # action mask according to env
        critic = critic.masked_fill(~self.action_mask, -torch.inf)
        action = critic.argmax(dim=-1)
        return action
    
    @torch.no_grad()
    def act(self, obs: torch.FloatTensor) -> torch.LongTensor:
        with autocast(dtype=self.dtype):
            obs_tokens = self.tokenizer.encode(obs, should_preprocess=True).tokens
            
            if self.should_plan:
                assert not self.use_kv_cache
                self.update_memory(obs_tokens)
                act = self.plan()[0]
            elif not self.use_kv_cache:
                self.update_memory(obs_tokens)
                outputs_wm = self.world_model(self.token_buffer[:, :self.used_token_idx], self.env_token)
                act = self.choose_action(outputs_wm.logits_q)
            else:
                # TODO: fix bugs
                outputs_wm = self.world_model(obs_tokens, self.env_token, past_keys_values=self.keys_values_wm)
                act = self.choose_action(outputs_wm.logits_q)

        return act  # (1,)

    @torch.no_grad()
    def plan(self, should_sample: bool = False) -> torch.LongTensor:
        wm_env = WorldModelEnv(self.tokenizer, self.world_model, self.device, self.env_token)
        valid_actions = self.action_mask.sum().cpu().item()
        constrained_beam_width = min(valid_actions, self.beam_width)
        gamma = self.world_model.config_critic.gamma
        max_length = self.world_model.config_transformer.max_blocks
        assert (self.used_token_idx + 1) % self.world_model.config_transformer.tokens_per_block == 0

        num_given_steps = min(max_length - self.horizon, (self.used_token_idx + 1) // self.world_model.config_transformer.tokens_per_block)
        num_given_tokens = num_given_steps * self.world_model.config_transformer.tokens_per_block - 1
        _ = wm_env.reset_from_initial_observations(self.token_buffer[:, self.used_token_idx - num_given_tokens:self.used_token_idx], act_mode="logits")

        critic = wm_env.last_actions
        v_star = critic.max().item()
        root_node = StepNode(
            0., 
            v_star, 
            gamma, 
            None, 
            deepcopy(wm_env.keys_values_wm),
            None, 
            critic, 
            wm_env.decode_obs_tokens(self.token_buffer[
                :, 
                self.used_token_idx - self.world_model.config_transformer.tokens_per_block + 1:self.used_token_idx
            ]).squeeze()
        )

        nodes = [root_node]

        if self.horizon == 0:
            return critic.argmax(dim=-1)
        
        for _ in range(self.horizon):
            expanded_nodes = []
            all_nodes_top_k_actions = []
            all_nodes_cache = []

            for ori_node in nodes:
                top_k_actions = torch.topk(ori_node.critic, constrained_beam_width, dim=-1).indices.cpu()
                all_nodes_top_k_actions.append(top_k_actions.view(-1, 1))  # (constrained_beam_width, 1)
                all_nodes_cache.append(ori_node.kv_cache)

            all_nodes_top_k_actions = torch.cat(all_nodes_top_k_actions, dim=0)  # (constrained_beam_width * num_of_nodes, k)
            all_nodes_cache = concate_kv(all_nodes_cache, repeat_num=constrained_beam_width)  # TODO, must deepcopy
            
            wm_env.env_tokens = self.env_token.repeat(all_nodes_top_k_actions.size(0))
            all_obs, all_rew, _, _ = wm_env.step(all_nodes_top_k_actions, should_sample=should_sample, act_mode="logits", kv_cache=all_nodes_cache)
            all_critic = wm_env.last_actions  # (constrained_beam_width * num_of_nodes, num_of_actions)
            all_v_star = all_critic.max(dim=-1).values  # (constrained_beam_width * num_of_nodes, 1)
            all_nodes_cache_split = split_kv(all_nodes_cache)  # TODO, must deepcopy

            for i in range(all_rew.size):
                act, critic, v_star, rew, cache = map(lambda x: x[i], [all_nodes_top_k_actions, all_critic, all_v_star, all_rew, all_nodes_cache_split])
                node = StepNode(
                    rew, 
                    v_star, 
                    gamma, 
                    act, 
                    cache, 
                    nodes[i // constrained_beam_width], 
                    critic,
                    all_obs[i].squeeze()
                )
                expanded_nodes.append(node)
                
            expanded_nodes = sorted(expanded_nodes, key=lambda x: x.score, reverse=True)
            nodes = expanded_nodes[:self.beam_width]

        end_node = nodes[0]
        best_path = [end_node.action, end_node.obs]
        while end_node.prev.prev is not None:
            end_node = end_node.prev
            best_path.insert(0, end_node.obs)
            best_path.insert(0, end_node.action)
        
        return best_path


def get_dtype(dtype: str):
    return torch.float16 if dtype == 'float16' else torch.bfloat16 if dtype == 'bfloat16' else torch.float32


class StepNode:
    def __init__(
        self, 
        r: float, 
        v_star: float, 
        gamma: float, 
        action=None, 
        kv_cache=None, 
        prev: Optional['StepNode'] = None, 
        critic: Optional[torch.Tensor] = None,
        obs: Optional[torch.Tensor] = None,
    ) -> None:
        self.r = r
        self.v_star = v_star
        self.action = action
        self.kv_cache = kv_cache
        self.prev = prev
        self.critic = critic
        self.obs = obs

        prev_gamma = 1./gamma if self.prev is None else self.prev.gamma
        self.gamma = prev_gamma * gamma
    
    @property
    def step(self) -> int:
        return 0 if self.prev is None else self.prev.step + 1

    @property
    def sum_r(self) -> float:
        prev_sum_r = 0. if self.prev is None else self.prev.sum_r
        prev_gamma = 1. if self.prev is None else self.prev.gamma
        return prev_sum_r + prev_gamma * self.r

    @property
    def score(self) -> float:
        return self.sum_r + self.gamma * self.v_star