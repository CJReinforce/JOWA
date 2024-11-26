import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from einops import rearrange
from torch.cuda.amp import autocast

from MCTS import MCTS
from action_tokenizer import (
    action_masks,
    ATARI_NUM_ACTIONS,
    GAME_NAMES,
    LIMITED_ACTION_TO_FULL_ACTION,
)
from envs.world_model_env import WorldModelEnv
from models.kv_caching import concate_kv, split_kv
from models.tokenizer import Tokenizer
from models.jowa_model import JOWAModel
from utils import get_dtype, StepNode


class Agent(nn.Module):
    def __init__(
        self, 
        tokenizer: Tokenizer, 
        jowa_model: JOWAModel, 
        env_token: int, 
        dtype=torch.float32, 
        buffer_size: int = 8,
        device: torch.device = torch.device('cuda'),
        should_plan: Union[bool, str] = False,  # [False, 'beam_search', 'MCTS']
        # beam search
        beam_width: int = 2,
        horizon: int = 2,
        # MCTS
        num_simulations: Optional[int] = None,
        temperature: float = 1.0,
        use_mean=False,
        use_count=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.jowa_model = jowa_model
        
        self.dtype = get_dtype(dtype) if isinstance(dtype, str) else dtype
        self.device = device
        self.buffer_size = buffer_size
        assert should_plan in [False, 'beam_search', 'MCTS']
        self.should_plan = should_plan
        # beam search
        self.beam_width = beam_width
        self.horizon = horizon
        # MCTS
        self.num_simulations = beam_width ** 2 * (horizon - 1) + beam_width \
            if num_simulations is None else num_simulations
        self.temperature = temperature
        self.use_mean = use_mean
        self.use_count = use_count
        
        self.env_token = torch.LongTensor([env_token]).to(self.device)
        self.action_mask = action_masks.to(device)[env_token]
        self.reset(1)
        
    def reset(self, n: int) -> None:
        self.token_buffer = torch.zeros(
            n, 
            self.buffer_size * self.jowa_model.config_transformer.tokens_per_block, 
            dtype=torch.long, device=self.device
        )
        self.used_token_idx = 0

    def load(
            self, 
            path_to_checkpoint: Path, 
            load_tokenizer: bool = True, 
            load_jowa_model: bool = True, 
            tokenizer_name: Optional[str] = None, 
            model_name: Optional[str] = None,
        ) -> None:
        def process_param_name(name: str, ckpt) -> str:
            for prefix in ['_orig_mod.module.', '', 'module.', '_orig_mod.']:
                new_name = prefix + name
                if new_name in ckpt:
                    return new_name
            raise KeyError(f"{name} not found in ckpt.")

        tokenizer_name = 'tokenizer.pt' if tokenizer_name is None \
            else tokenizer_name if tokenizer_name.endswith('.pt') else tokenizer_name + '.pt'
        model_name = 'JOWA.pt' if model_name is None else model_name if \
            model_name.endswith('.pt') else model_name + '.pt' 

        if load_tokenizer:
            ckpt_token = torch.load(
                os.path.join(path_to_checkpoint, tokenizer_name), 
                map_location=self.device,
            )
            tokenizer_dict = self.tokenizer.state_dict()

            for name in tokenizer_dict.keys():
                tokenizer_dict[name] = ckpt_token[
                    process_param_name(name, ckpt_token)]
    
            self.tokenizer.load_state_dict(tokenizer_dict)
        
        if load_jowa_model:
            ckpt_world = torch.load(
                os.path.join(path_to_checkpoint, model_name), 
                map_location=self.device,
            )
            jowa_model_dict = self.jowa_model.state_dict()

            for name in jowa_model_dict.keys():
                jowa_model_dict[name] = ckpt_world[
                    process_param_name(name, ckpt_world)]
            
            self.jowa_model.load_state_dict(jowa_model_dict, strict=False)
        
        print(f"Load {'tokenizer' if load_tokenizer else ''} " + \
            f"{'jowa model' if load_jowa_model else ''} from {path_to_checkpoint}.")

    def eval(self):
        super().eval()
        self.tokenizer.eval()
        self.jowa_model.eval()
        return self

    def update_memory(self, input: torch.LongTensor) -> None:
        if input.ndim == 2:
            input = input[0]
        
        L_in = input.size(0)
        L_out = self.jowa_model.config_transformer.tokens_per_block
        
        if self.used_token_idx + L_in <= self.token_buffer.size(1):
            self.token_buffer[0, self.used_token_idx:self.used_token_idx + L_in] = input
            self.used_token_idx += L_in
        else:
            buffer_clone = self.token_buffer.clone()
            self.token_buffer[0, :self.used_token_idx - L_out] = buffer_clone[
                0, L_out:self.used_token_idx]
            self.token_buffer[
                0, self.used_token_idx - L_out:self.used_token_idx - L_out + L_in] = input
            self.used_token_idx += L_in - L_out
    
    def choose_action(self, critic: torch.FloatTensor) -> torch.LongTensor:
        """
        Choose action using greedy policy.
        """
        if hasattr(self.jowa_model, 'atoms_support'):  # (num_q, B, L or 1, #actions * atoms)
            critic = (
                rearrange(
                    critic[..., -1, :], 'q b (a z) -> q b a z', 
                    a=self.jowa_model.act_vocab_size,
                ).softmax(-1).mean(0) * self.jowa_model.atoms_support.view(1, 1, -1)
            ).sum(-1)
        else:  # (num_q, B, L or 1, #actions)
            critic = critic[..., -1, :].mean(0)

        # action mask according to env
        critic = critic.masked_fill(~self.action_mask, -torch.inf)
        action = critic.argmax(dim=-1)
        return action
    
    def policy_induced_by_Q(self, critic: torch.Tensor, temperature: float) -> torch.Tensor:
        assert critic.ndim <= 2
        return (critic / temperature).softmax(-1)

    def random_action(self) -> torch.LongTensor:
        true_indices = torch.nonzero(self.action_mask.squeeze())
        random_index = true_indices[torch.randint(0, true_indices.size(0), (1,)).item()]
        return random_index.reshape(-1,)
    
    @torch.no_grad()
    def act(self, obs: torch.FloatTensor) -> torch.LongTensor:
        with autocast(dtype=self.dtype):
            obs_tokens = self.tokenizer.encode(obs, should_preprocess=True).tokens
            self.update_memory(obs_tokens)

            if torch.rand(1).item() < 0.001:  # epsilon-eval
                act = self.random_action()
            elif self.should_plan == 'beam_search':
                act = self.plan_beam_search()
            elif self.should_plan == 'MCTS':
                act = self.plan_MCTS()
            else:
                outputs_wm = self.jowa_model(
                    self.token_buffer[:, :self.used_token_idx], 
                    self.env_token,
                )
                act = self.choose_action(outputs_wm.logits_q)

        return act  # (1,)

    @torch.no_grad()
    def plan_beam_search(self, should_sample: bool = False) -> torch.LongTensor:
        wm_env = WorldModelEnv(self.tokenizer, self.jowa_model, self.device, self.env_token)
        valid_actions = self.action_mask.sum().cpu().item()

        constrained_beam_width = min(valid_actions, self.beam_width)
        gamma = self.jowa_model.config_critic_train.gamma
        max_length = self.jowa_model.config_transformer.max_blocks
        assert (self.used_token_idx + 1) % self.jowa_model.config_transformer.tokens_per_block == 0

        # init dream
        num_given_steps = min(
            max_length - self.horizon, 
            (self.used_token_idx + 1) // self.jowa_model.config_transformer.tokens_per_block
        )
        num_given_tokens = num_given_steps * self.jowa_model.config_transformer.tokens_per_block - 1
        _ = wm_env.reset_from_initial_observations(
            self.token_buffer[:, self.used_token_idx - num_given_tokens:self.used_token_idx], 
            act_mode="logits",
        )

        # init root node
        critic = wm_env.last_actions
        v_star = critic.max().item()
        root_node = StepNode(
            r=0., 
            v_star=v_star, 
            gamma=gamma, 
            batch_idx=0,
            action=None, 
            kv_cache=deepcopy(wm_env.keys_values_wm),
            prev=None, 
            critic=critic, 
            obs=None,
        )

        nodes = [root_node]

        if self.horizon == 0:
            return critic.argmax(dim=-1)
        
        # planning in dream
        for _ in range(self.horizon):
            expanded_nodes = []
            all_nodes_top_k_actions = []
            all_nodes_cache = []

            for ori_node in nodes:
                top_k_actions = torch.topk(
                    ori_node.critic, 
                    constrained_beam_width, 
                    dim=-1,
                ).indices.cpu()
                all_nodes_top_k_actions.append(
                    top_k_actions.view(-1, 1)
                )  # (constrained_beam_width, 1)
                all_nodes_cache.append(ori_node.kv_cache)

            all_nodes_top_k_actions = torch.cat(
                all_nodes_top_k_actions, 
                dim=0,
            )  # (constrained_beam_width * num_of_nodes, 1)
            all_nodes_cache = concate_kv(all_nodes_cache, repeat_num=constrained_beam_width)
            
            wm_env.env_tokens = self.env_token.repeat(all_nodes_top_k_actions.size(0))
            _, all_rew, _, _ = wm_env.step(
                all_nodes_top_k_actions, 
                should_sample=should_sample, 
                act_mode="logits", 
                kv_cache=all_nodes_cache,
            )
            all_critic = wm_env.last_actions  # (constrained_beam_width * num_of_nodes, num_of_actions)
            all_v_star = all_critic.max(dim=-1).values  # (constrained_beam_width * num_of_nodes, 1)
            all_nodes_cache_split = split_kv(all_nodes_cache)

            for i in range(all_rew.size):
                act, critic, v_star, rew, cache = map(
                    lambda x: x[i], 
                    [
                        all_nodes_top_k_actions, 
                        all_critic, 
                        all_v_star, 
                        all_rew, 
                        all_nodes_cache_split
                    ]
                )
                node = StepNode(
                    r=rew, 
                    v_star=v_star, 
                    gamma=gamma, 
                    batch_idx=0,
                    action=act, 
                    kv_cache=cache, 
                    prev=nodes[i // constrained_beam_width], 
                    critic=critic,
                    obs=None,
                )
                expanded_nodes.append(node)
                
            expanded_nodes = sorted(expanded_nodes, key=lambda x: x.score, reverse=True)
            nodes = expanded_nodes[:self.beam_width]

        # optimal path
        end_node = nodes[0]
        while end_node.prev.prev is not None:
            end_node = end_node.prev
        
        return end_node.action
    
    @torch.no_grad()
    def plan_MCTS(self, should_sample: bool = False) -> torch.LongTensor:
        wm_env = WorldModelEnv(self.tokenizer, self.jowa_model, self.device, self.env_token)
        gamma = self.jowa_model.config_critic_train.gamma
        mcts = MCTS(discount=gamma)

        max_length = self.jowa_model.config_transformer.max_blocks
        assert (self.used_token_idx + 1) % self.jowa_model.config_transformer.tokens_per_block == 0

        # init dream
        num_given_steps = min(
            max_length - self.horizon, 
            (self.used_token_idx + 1) // self.jowa_model.config_transformer.tokens_per_block
        )
        num_given_tokens = num_given_steps * self.jowa_model.config_transformer.tokens_per_block - 1
        _ = wm_env.reset_from_initial_observations(
            self.token_buffer[:, self.used_token_idx - num_given_tokens:self.used_token_idx], 
            act_mode="logits",
        )

        mcts.generate_root_node()
        mcts.root.kv_cache = deepcopy(wm_env.keys_values_wm)

        critic = wm_env.last_actions
        policy = self.policy_induced_by_Q(critic, self.temperature)
        mcts.expand_the_children_of_the_root_node(policy, critic)

        for _ in range(self.num_simulations):
            history, search_path = mcts.initialize_history_node_searchpath_variable()
            parent = mcts.choose_node_to_expand_using_max_ucb_score(
                history, search_path, max_depth=self.horizon - 1)

            _, reward, _, _ = wm_env.step(
                torch.LongTensor([[history[-1]]]), 
                should_sample=should_sample, 
                act_mode="logits", 
                kv_cache=deepcopy(parent.kv_cache),
            )

            mcts.update_reward_and_kv_cache_for_the_chosen_node(
                reward.item(), deepcopy(wm_env.keys_values_wm))

            critic = wm_env.last_actions
            if self.use_mean:
                value = critic.mean().item()  # mean or max
            else:
                value = critic.max().item()
            policy = self.policy_induced_by_Q(critic, self.temperature)
            
            mcts.create_new_node_in_the_chosen_node_with_action_and_policy(
                policy, critic)
            mcts.back_propagate_and_update_min_max_bound(search_path, value)
            
        if self.use_count:
            action = torch.tensor([mcts.root.children[u].visit_count for u in mcts.root.children.keys()]).argmax(keepdim=True)
        else:
            action = torch.tensor([i.value() for i in mcts.root.children.values()]).argmax(keepdim=True)
        return action