from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from dataset import Batch
from envs.world_model_env import action_masks
from utils import Ensemble, LossWithIntermediateLosses, init_weights, mlp

from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_q: torch.FloatTensor


class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config_transformer: TransformerConfig, config_critic, name: str = 'world_model', *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config_transformer = config_transformer
        self.config_critic = config_critic
        self.transformer = Transformer(config_transformer)

        all_but_last_obs_tokens_pattern = torch.ones(config_transformer.tokens_per_block)
        all_but_last_obs_tokens_pattern[-2] = 0
        last_obs_tokens_pattern = 1 - all_but_last_obs_tokens_pattern
        act_tokens_pattern = torch.zeros(self.config_transformer.tokens_per_block)
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern

        self.pos_emb = nn.Embedding(config_transformer.max_tokens, config_transformer.embed_dim)
        self.task_emb = Embedder(
            max_blocks=config_transformer.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList(
                [nn.Embedding(config_transformer.max_tasks, config_transformer.embed_dim),
                 nn.Embedding(config_transformer.max_tasks, config_transformer.embed_dim)])
        )

        self.embedder = Embedder(
            max_blocks=config_transformer.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList(
                [nn.Embedding(act_vocab_size, config_transformer.embed_dim), 
                 nn.Embedding(obs_vocab_size, config_transformer.embed_dim)])
        )
        
        self.head_q = Head(
            max_blocks=config_transformer.max_blocks,
            block_mask=last_obs_tokens_pattern,
            head_module=Ensemble(
                [mlp(config_critic.latent_dim, 2*[config_critic.mlp_dim], act_vocab_size, 
                    dropout=config_critic.dropout) for _ in range(config_critic.num_q)]
            )
        )
        
        self.apply(init_weights)
        self.training_steps = 0
        self.name = name
        
    def __repr__(self) -> str:
        return self.name

    def forward(self, tokens: torch.LongTensor, tasks: torch.LongTensor, past_keys_values: Optional[KeysValues] = None) -> WorldModelOutput:

        assert tasks.size(0) == tokens.size(0)  # B == B
        num_steps = tokens.size(1)  # (B, T)
        assert num_steps <= self.config_transformer.max_tokens
        if past_keys_values is not None and past_keys_values.size + num_steps > past_keys_values[0]._k_cache._cache.shape[2]:
            # past_keys_values.shift(shifted_tokens=num_steps + 1)
            raise IndexError("Past keys and values are too short to accomodate the current tokens.")
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        sequences = self.embedder(tokens, num_steps, prev_steps) + \
            self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device)) + \
            self.task_emb(tasks.view(-1, 1).repeat(1, num_steps), num_steps, prev_steps)
            
        x = self.transformer(sequences, past_keys_values)
        logits_q = self.head_q(x, num_steps=num_steps, prev_steps=prev_steps)  # (num_q, B, L, #actions)
        
        return WorldModelOutput(x, logits_q)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, training: bool = True, **kwargs: Any) -> Tuple[LossWithIntermediateLosses, dict]:
        info_logs = {
            f'info/{str(self)}/reward_real': batch['rewards'].mean().item(),
            f'info/{str(self)}/done_real': batch['ends'].float().mean().item(),
        }

        with torch.no_grad():
            obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (B, L, K)
        act_tokens = rearrange(batch['actions'], 'b l -> b l 1')
        tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))
        outputs = self(tokens, batch['envs'])
        
        mask = action_masks.to(batch['ends'].device)[batch['envs']]
        distribution_a = rearrange(outputs.logits_q.masked_fill(
            ~mask.unsqueeze(1).unsqueeze(0).expand_as(outputs.logits_q), -torch.inf).softmax(dim=-1).mean(dim=0), 
            'b l a -> (b l) a'
        )
        label_a = batch['actions'].reshape(-1, 1)
        loss_a = - distribution_a.gather(1, label_a).log().mean()
        
        if training:
            self.training_steps += 1
        
        return LossWithIntermediateLosses(
            loss_act=loss_a,
        ), info_logs