import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from dataset import Batch
from envs.world_model_env import WorldModelEnv, action_masks
from utils import Ensemble, LossWithIntermediateLosses, ZeroEmbedding, init_weights, mlp

from .kv_caching import KeysValues, concate_kv, split_kv
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    logits_q: torch.FloatTensor
    logits_q_target: torch.FloatTensor


@dataclass
class ImagineOutput:
    observations: torch.ByteTensor
    actions: torch.LongTensor
    rewards: torch.FloatTensor
    ends: torch.BoolTensor
    mask_padding: torch.BoolTensor
    envs: torch.LongTensor


class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config_transformer: TransformerConfig, config_critic, device = "cuda", name: str = 'world_model') -> None:
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
        # self.task_emb = nn.Embedding(config_transformer.max_tasks, config_transformer.embed_dim)
        if config_critic.use_task_embed:
            self.task_emb = Embedder(
                max_blocks=config_transformer.max_blocks,
                block_masks=[act_tokens_pattern, obs_tokens_pattern],
                embedding_tables=nn.ModuleList(
                    [nn.Embedding(config_transformer.max_tasks, config_transformer.embed_dim),  # ZeroEmbedding(1, config_transformer.embed_dim),
                    nn.Embedding(config_transformer.max_tasks, config_transformer.embed_dim)])
            )
        else:
            self.task_emb = Embedder(
                max_blocks=config_transformer.max_blocks,
                block_masks=[act_tokens_pattern, obs_tokens_pattern],
                embedding_tables=nn.ModuleList(
                    [ZeroEmbedding(config_transformer.max_tasks, config_transformer.embed_dim),  # ZeroEmbedding(1, config_transformer.embed_dim),
                    ZeroEmbedding(config_transformer.max_tasks, config_transformer.embed_dim)])
            )

        self.embedder = Embedder(
            max_blocks=config_transformer.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList(
                [nn.Embedding(act_vocab_size, config_transformer.embed_dim), 
                 nn.Embedding(obs_vocab_size, config_transformer.embed_dim)])
        )

        self.head_observations = Head(
            max_blocks=config_transformer.max_blocks,
            block_mask=all_but_last_obs_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config_transformer.embed_dim, config_transformer.embed_dim),
                nn.ReLU(),
                nn.Linear(config_transformer.embed_dim, obs_vocab_size)
            )
        )

        self.head_rewards = Head(
            max_blocks=config_transformer.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config_transformer.embed_dim, config_transformer.embed_dim),
                nn.ReLU(),
                nn.Linear(config_transformer.embed_dim, 3)
            )
        )

        self.head_ends = Head(
            max_blocks=config_transformer.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config_transformer.embed_dim, config_transformer.embed_dim),
                nn.ReLU(),
                nn.Linear(config_transformer.embed_dim, 2)
            )
        )
        
        self.q_penalty_method = self.config_critic.q_penalty
        self.td_loss_method = self.config_critic.td_loss
        self.q_loss_backwards_wm = self.config_critic.q_loss_backwards_wm
        assert self.q_penalty_method in ['cql', 'combo', None]
        assert self.td_loss_method in ['c51', 'mse']

        q_output_dim = act_vocab_size * config_critic.num_atoms if self.td_loss_method == 'c51' else act_vocab_size
        self.head_q = Head(
            max_blocks=config_transformer.max_blocks,
            block_mask=last_obs_tokens_pattern,
            head_module=Ensemble([
                mlp(
                    config_critic.latent_dim, 
                    2*[config_critic.mlp_dim], 
                    q_output_dim, 
                    dropout=config_critic.dropout
                ) for _ in range(config_critic.num_q)
            ])
        )
        
        self.apply(init_weights)
        self.head_q_target = deepcopy(self.head_q).requires_grad_(False)
        
        # cql weight
        self.log_alpha = torch.tensor(
            np.log(self.config_critic.cql_weight),
            dtype=torch.float, device=device, # requires_grad=True,
        )

        if self.td_loss_method == 'c51':
            self.atoms_support = torch.linspace(
                config_critic.vmin, config_critic.vmax, config_critic.num_atoms,
            ).to(device)
        
        self.training_steps = 0
        self.wm_name = name

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self.head_q_target.train(False)
        return self
        
    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        with torch.no_grad():
            for p, p_target in zip(self.head_q.parameters(), self.head_q_target.parameters()):
                p_target.data.lerp_(p.data, self.config_critic.tau)
                
    def hard_update_target_Q(self):
        """
        Hard-update target Q-networks.
        """
        if self.training_steps % self.config_critic.target_update_frequency == 0:
            with torch.no_grad():
                self.head_q_target.load_state_dict(self.head_q.state_dict())

    def get_valid_alpha(self):
        return self.log_alpha.exp().clamp(min=0.0)
        
    def __repr__(self) -> str:
        return self.wm_name

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

        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)  # (B, L, 3)
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)  # (B, L, 2)
        logits_q = self.head_q(x if self.q_loss_backwards_wm else x.detach(), num_steps=num_steps, prev_steps=prev_steps)  # (num_q, B, L, #actions)
        with torch.no_grad():
            logits_q_target = self.head_q_target(x.detach(), num_steps=num_steps, prev_steps=prev_steps)  # (num_q, B, L, #actions)

        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends, logits_q, logits_q_target)

    def compute_loss(self, real_batch: Batch, tokenizer: Tokenizer, train_critic: bool = False, imagine_horizon: Optional[int] = None, training: bool = True, imagine_batch: Optional[Batch] = None, **kwargs: Any) -> Tuple[LossWithIntermediateLosses, dict]:

        real_batch_size = real_batch['observations'].size(0)
        info_logs = {
            f'info/{str(self)}/reward_real': real_batch['rewards'].mean().item(),
            f'info/{str(self)}/done_real': real_batch['ends'].float().mean().item(),
            f'info/{str(self)}/alpha': self.log_alpha.exp().item(),
        }
        additional_losses = {}

        # if train_critic and self.config_critic.use_imaginary_batch:
        #     # imagine
        #     imagine_batch, env_logs = self.imagine(real_batch if image_batch is None else image_batch, tokenizer, horizon=imagine_horizon)
        #     ## check batch size
        #     # imagine_batch_size = imagine_batch.observations.size(0)
        #     # assert real_batch_size == imagine_batch_size
            
        #     # mix real and imagine batch
        #     mix_batch = {k: torch.cat([real_batch[k], getattr(imagine_batch, k)], dim=0) for k in real_batch.keys()}

        #     info_logs.update({
        #         f'info/{str(self)}/reward_imagine': imagine_batch.rewards.mean().item(),
        #         f'info/{str(self)}/done_imagine': imagine_batch.ends.float().mean().item(),
        #         f'info/{str(self)}/epsilon': env_logs['epsilon'],
        #         f'info/{str(self)}/num_given_block': env_logs['num_given_block'],
        #     })
        # else:

        if imagine_batch is not None:
            mix_batch = {k: v if k=='observations' else torch.cat((v, imagine_batch[k]), dim=0) for k, v in real_batch.items()}
        else:
            mix_batch = real_batch
        mix_batch_size = mix_batch['observations'].size(0)

        with torch.no_grad():
            obs_tokens = tokenizer.encode(mix_batch['observations'], should_preprocess=True).tokens  # (B, L, K)
            if imagine_batch is not None:
                obs_tokens = torch.cat((obs_tokens, imagine_batch['observations']), dim=0)
        act_tokens = rearrange(mix_batch['actions'], 'b l -> b l 1')
        tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))
        outputs = self(tokens, mix_batch['envs'])
        
        # RL loss for head_q (and wm)
        if train_critic:
            # REM
            alpha = torch.rand(self.config_critic.num_q).to(tokens.device) if self.config_critic.use_rem else torch.ones(self.config_critic.num_q).to(tokens.device)
            alpha = alpha / torch.sum(alpha)

            # C51
            if self.td_loss_method == 'c51':
                # outputs.logits_q -> (num_q, B, L, actions * atoms)
                ensemble_logits_q_s_dist = rearrange(outputs.logits_q, 'q b l (a z) -> q b l a z', a=self.act_vocab_size)  # (num_q, B, L, actions, atoms)
                ensemble_logits_q_s_chosen_a_dist = ensemble_logits_q_s_dist.gather(
                    3, act_tokens.unsqueeze(0).unsqueeze(-1).expand(
                        ensemble_logits_q_s_dist.shape[0], -1, -1, -1, ensemble_logits_q_s_dist.shape[-1])
                ).squeeze(3)  # (num_q, B, L, atoms)
                q_s_chosen_a_dist = torch.sum(alpha.view(-1, 1, 1, 1) * ensemble_logits_q_s_chosen_a_dist.softmax(dim=-1), dim=0)  # (B, L, atoms)
                
                # compute mask for q
                B, T = q_s_chosen_a_dist.shape[:2]
                mix_batch_mask_padding = mix_batch['mask_padding'].int()
                first_ones = torch.argmax(mix_batch_mask_padding, dim=1)
                last_ones = T - 1 - torch.argmax(mix_batch_mask_padding.flip(dims=[1]), dim=1)
                # assert torch.all(first_ones & T-1-last_ones == 0)
                
                # not done state
                compute_q_mask = mix_batch['mask_padding'].clone()
                batch_indices = torch.arange(B)
                compute_q_mask[batch_indices, last_ones] = 0
                
                masked_q_s_chosen_a_dist = q_s_chosen_a_dist[compute_q_mask]  # (Bt, atoms)

                # done state
                ends = mix_batch['ends'].clone()  # (B, T)
                ends[~mix_batch['mask_padding']] = 0  # (B, T)
                # assert torch.all(ends.sum(-1) <= 1)
                ends = ends.to(bool)

                ended_q_s_chosen_a_dist = q_s_chosen_a_dist[ends]  # (num_end, atoms)
                num_end = ended_q_s_chosen_a_dist.shape[0]

                mixed_q_s_chosen_a_dist = torch.cat((masked_q_s_chosen_a_dist, ended_q_s_chosen_a_dist), dim=0)  # (Bt + num_end, atoms)
                
                with torch.no_grad():
                    # action mask
                    mask = action_masks.to(mix_batch['ends'].device)[mix_batch['envs']]
                    
                    ensemble_logits_q_next_s_dist = rearrange(outputs.logits_q_target, 'q b l (a z) -> q b l a z', a=self.act_vocab_size)  # (num_q, B, L, actions, atoms)
                    q_next_s_dist = torch.sum(alpha.view(-1, 1, 1, 1, 1) * ensemble_logits_q_next_s_dist.softmax(dim=-1), dim=0)  # (B, L, actions, atoms)
                    masked_q_next_s = (q_next_s_dist * self.atoms_support.view(1, 1, 1, -1)).sum(-1).masked_fill(
                        ~mask.unsqueeze(1).expand(-1, q_next_s_dist.shape[1], -1), -torch.inf)  # (B, L, actions)
                    a_star = torch.argmax(masked_q_next_s, dim=-1, keepdim=True)  # (B, L, 1)

                    q_next_s_argmax_a_dist = rearrange(
                        q_next_s_dist.gather(2, a_star.unsqueeze(-1).expand(-1, -1, -1, q_next_s_dist.shape[-1])).squeeze(2), 
                        'b l z -> (b l) z'
                    )  # (BL, atoms)

                    compute_q_target_mask = mix_batch['mask_padding'].clone()
                    compute_q_target_mask[batch_indices, first_ones] = 0

                    masked_q_next_s_argmax_a_dist = q_next_s_argmax_a_dist[compute_q_target_mask.view(-1)]  # (Bt, atoms)
                    num_dims = self.atoms_support.shape[0]  # 121
                    ended_q_next_s_argmax_a_dist = torch.ones(
                        (num_end, num_dims), 
                        dtype=masked_q_next_s_argmax_a_dist.dtype, 
                        device=masked_q_next_s_argmax_a_dist.device
                    ) * 1.0 / num_dims  # (num_end, atoms)
                    mixed_q_next_s_argmax_a_dist = torch.cat((masked_q_next_s_argmax_a_dist, ended_q_next_s_argmax_a_dist), dim=0)  # (Bt + num_end, atoms)

                    target_support = mix_batch['rewards'][compute_q_mask].unsqueeze(-1) + self.config_critic.gamma * self.atoms_support.unsqueeze(0)  # (Bt, atoms)
                    target_support_for_end = mix_batch['rewards'][ends].unsqueeze(-1) + 0.0 * self.atoms_support.unsqueeze(0)
                    mixed_target_support = torch.cat((target_support, target_support_for_end), dim=0)  # (Bt + num_end, atoms)
                    
                    # project distribution
                    v_min, v_max = self.atoms_support[0], self.atoms_support[-1]
                    delta_z = (v_max - v_min) / (num_dims - 1)
                    clipped_support = torch.clip(mixed_target_support, v_min, v_max)  # (Bt + num_end, atoms)

                    masked_bellman_target = (
                        1 - (clipped_support.unsqueeze(1) - self.atoms_support.view(1, -1, 1)).abs() / delta_z
                    ).clamp(0, 1) * mixed_q_next_s_argmax_a_dist.unsqueeze(1)
                    masked_bellman_target = masked_bellman_target.sum(-1)  # (Bt + num_end, atoms)
                
                loss_td_error = -(masked_bellman_target * torch.log(mixed_q_s_chosen_a_dist + 1e-8)).sum(-1).mean()
                additional_losses['loss_td_error'] = loss_td_error
                
                # CQL / COMBO loss
                if self.q_penalty_method is not None:
                    alpha = self.get_valid_alpha()

                    # negative Q
                    negative_sampling = torch.logsumexp(
                        (ensemble_logits_q_s_dist.softmax(dim=-1).mean(dim=0) * self.atoms_support.view(1, 1, 1, -1)).sum(dim=-1), 
                        dim=-1,
                    )
                    negative_sampling = negative_sampling[mix_batch['mask_padding']]
                    q_value_negative_sampling = negative_sampling.mean()
                    info_logs[f'info/{str(self)}/q_value_of_negative_sampling'] = q_value_negative_sampling.item()

                    # positive Q
                    dataset_expec = q_s_chosen_a_dist[:real_batch_size if self.q_penalty_method == 'combo' else mix_batch_size][real_batch['mask_padding']] @ self.atoms_support
                    q_value_positive_sampling = dataset_expec.mean()
                    info_logs[f'info/{str(self)}/q_value_of_positive_sampling'] = q_value_positive_sampling.item()

                    loss_cql_penalty = (q_value_negative_sampling - q_value_positive_sampling) * alpha.detach()
                    additional_losses['loss_cql_penalty'] = loss_cql_penalty
            
            # MSE
            else:
                ensemble_q = outputs.logits_q.gather(
                    -1, act_tokens.expand(self.config_critic.num_q, *act_tokens.shape)).squeeze(-1)  # (num_q, B, L)
                q = torch.sum(alpha.unsqueeze(-1).unsqueeze(-1) * ensemble_q, dim=0)  # (B, L)
                
                # compute mask for not done state
                B, T = q.shape
                mix_batch_mask_padding = mix_batch['mask_padding'].int()
                first_ones = torch.argmax(mix_batch_mask_padding, dim=1)
                last_ones = T - 1 - torch.argmax(mix_batch_mask_padding.flip(dims=[1]), dim=1)
                
                compute_q_mask = mix_batch['mask_padding'].clone()
                batch_indices = torch.arange(B)
                compute_q_mask[batch_indices, last_ones] = 0
                
                predict_q = q[compute_q_mask]  # (Bt,)

                # done state
                ends = mix_batch['ends'].clone()  # (B, T)
                ends[~mix_batch['mask_padding']] = 0  # (B, T)
                # assert torch.all(ends.sum(-1) <= 1)
                ends = ends.to(bool)

                predict_q_for_done = q[ends]  # (num_end,)
                mixed_predict_q = torch.cat((predict_q, predict_q_for_done), dim=0)  # (Bt + num_end,)
                
                with torch.no_grad():
                    # action mask
                    mask = action_masks.to(mix_batch['ends'].device)[mix_batch['envs']]
                    
                    target_v = torch.sum(alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * outputs.logits_q_target, dim=0)  # (B, L, #actions)
                    target_v = target_v.masked_fill(~mask.unsqueeze(1).expand_as(target_v), -torch.inf)
                    target_v = torch.max(target_v, dim=-1).values  # (B, L)
                    
                    compute_q_target_mask = mix_batch['mask_padding'].clone()
                    compute_q_target_mask[batch_indices, first_ones] = 0
                    
                    target_v = target_v[compute_q_target_mask]
                    target_q = mix_batch['rewards'][compute_q_mask] + self.config_critic.gamma * target_v  # (Bt,)
                    
                    target_q_for_done = mix_batch['rewards'][ends]  # (num_end,)
                    mixed_target_q = torch.cat((target_q, target_q_for_done), dim=0)  # (Bt + num_end,)
                
                loss_td_error = F.mse_loss(mixed_predict_q, mixed_target_q) * 0.5
                additional_losses['loss_td_error'] = loss_td_error
                    
                # CQL loss
                if self.q_penalty_method is not None:
                    alpha = self.get_valid_alpha()

                    # for mix batch
                    negative_sampling = torch.logsumexp(torch.sum(
                        alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * outputs.logits_q, dim=0), dim=-1)
                    negative_sampling = negative_sampling[mix_batch['mask_padding']]

                    q_value_negative_sampling = negative_sampling.mean()
                    info_logs[f'info/{str(self)}/q_value_of_negative_sampling'] = q_value_negative_sampling.item()
                    
                    if self.q_penalty_method == 'cql':
                        # for real batch
                        dataset_expec = q[:real_batch_size][real_batch['mask_padding']]
                        q_value_positive_sampling = dataset_expec.mean()
                        info_logs[f'info/{str(self)}/q_value_of_positive_sampling'] = q_value_positive_sampling.item()

                        loss_cql_penalty = (q_value_negative_sampling - q_value_positive_sampling) * alpha.detach()
                    else:
                        loss_cql_penalty = q_value_negative_sampling * alpha.detach()
                    
                    additional_losses['loss_cql_penalty'] = loss_cql_penalty

        # Supervised loss for dynamic model and predictors in world model
        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(
            obs_tokens[:real_batch_size], real_batch['rewards'], real_batch['ends'], real_batch['mask_padding'])
        logits_observations = rearrange(outputs.logits_observations[:real_batch_size, :-1], 'b t o -> (b t) o')
        
        supervised_weight = self.config_critic.supervised_weight if train_critic else 1.0
        loss_obs = F.cross_entropy(logits_observations, labels_observations) * supervised_weight
        loss_rewards = F.cross_entropy(rearrange(
            outputs.logits_rewards[:real_batch_size], 'b t e -> (b t) e'), labels_rewards) * supervised_weight
        loss_ends = F.cross_entropy(rearrange(
            outputs.logits_ends[:real_batch_size], 'b t e -> (b t) e'), labels_ends) * supervised_weight
        
        if training:
            self.training_steps += 1

        return LossWithIntermediateLosses(
            loss_obs=loss_obs, 
            loss_rewards=loss_rewards, 
            loss_ends=loss_ends, 
            **additional_losses,
        ), info_logs

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100), 'b t k -> b (t k)')[:, 1:]
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)
    
    @torch.no_grad()
    def imagine(self, batch: Batch, tokenizer: Tokenizer, horizon=None, beam_width=None, show_pbar: bool = False) -> Tuple[ImagineOutput, dict]:
        # # choose the num_given_block from 4 to 1
        # for num_given_block in range(10, 0, -1):
        #     if batch['mask_padding'][:, -num_given_block:].all():
        #         break
        policy_in_imagination = self.config_critic.policy_in_imagination
        assert policy_in_imagination in ['greedy', 'epsilon_greedy', 'planning']
        
        num_given_block = 8
        avail_idxs = batch['mask_padding'][:, -num_given_block:].all(1)
        avail_batch = {k: v[avail_idxs] for k, v in batch.items()}
        device = avail_batch['observations'].device
        
        if policy_in_imagination == 'planning':
            constrained_beam_width = beam_width
            assert beam_width is not None and horizon is not None
            
            obs_tokens = tokenizer.encode(avail_batch['observations'][:, -num_given_block + horizon:], should_preprocess=True).tokens  # (B, L, K)
            act_tokens = rearrange(avail_batch['actions'][:, -num_given_block + horizon:], 'b l -> b l 1')
            tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')[:, :-1]  # (B, L(K+1))

            wm_env = WorldModelEnv(tokenizer, self, device, avail_batch['envs'])
            gamma = self.config_critic.gamma
            _ = wm_env.reset_from_initial_observations(tokens, act_mode="logits")
            
            critic = wm_env.last_actions
            v_star = critic.max(1).values
            root_nodes_kv = split_kv(wm_env.keys_values_wm)
            
            group_nodes = {}
            for i in range(critic.size(0)):
                root_node = StepNode(
                    0., 
                    v_star[i].item(), 
                    gamma, 
                    i,
                    None, 
                    deepcopy(root_nodes_kv[i]),
                    None, 
                    critic[i], 
                    tokens[i, -self.config_transformer.tokens_per_block + 1:].detach().cpu()
                )
                group_nodes[i] = [root_node]
            
            for _ in range(horizon):
                expanded_group_nodes = {k:[] for k in group_nodes.keys()}
                all_group_nodes_top_k_actions = []
                all_group_nodes_cache = []
                
                for k, nodes in group_nodes.items():
                    all_nodes_top_k_actions = []
                    all_nodes_cache = []

                    for ori_node in nodes:
                        top_k_actions = torch.topk(ori_node.critic, constrained_beam_width, dim=-1).indices.cpu()
                        all_nodes_top_k_actions.append(top_k_actions.view(-1, 1))  # (constrained_beam_width, 1)
                        all_nodes_cache.append(ori_node.kv_cache)

                    all_nodes_top_k_actions = torch.cat(all_nodes_top_k_actions, dim=0)  # (constrained_beam_width * num_of_nodes, k)
                    all_nodes_cache = concate_kv(all_nodes_cache, repeat_num=constrained_beam_width)
                    
                    all_group_nodes_top_k_actions.append(all_nodes_top_k_actions)
                    all_group_nodes_cache.append(all_nodes_cache)
                
                all_group_nodes_top_k_actions = torch.cat(all_group_nodes_top_k_actions, dim=0)  # (bs * constrained_beam_width * num_of_nodes, k)
                all_group_nodes_cache = concate_kv(all_group_nodes_cache)
                
                wm_env.env_tokens = avail_batch['envs'].repeat_interleave(all_nodes_top_k_actions.size(0))
                _, all_rew, _, _ = wm_env.step(all_group_nodes_top_k_actions, should_sample=True, act_mode="logits", kv_cache=all_group_nodes_cache)
                all_critic = wm_env.last_actions  # (bs * constrained_beam_width * num_of_nodes, num_of_actions)
                all_v_star = all_critic.max(dim=-1).values  # (bs * constrained_beam_width * num_of_nodes, 1)
                all_group_nodes_cache_split = split_kv(all_group_nodes_cache)

                for i in range(all_rew.size):
                    batch_idx = i // all_nodes_top_k_actions.size(0)
                    act, critic, v_star, rew, cache, obs_token = map(
                        lambda x: x[i], 
                        [all_group_nodes_top_k_actions, 
                         all_critic, 
                         all_v_star, 
                         all_rew, 
                         all_group_nodes_cache_split,
                         wm_env.obs_tokens]
                    )
                    
                    node = StepNode(
                        rew, 
                        v_star, 
                        gamma, 
                        batch_idx,
                        act, 
                        cache, 
                        group_nodes[batch_idx][(i - batch_idx * all_nodes_top_k_actions.size(0)) // constrained_beam_width], 
                        critic,
                        obs_token.detach().cpu()
                    )
                    expanded_group_nodes[batch_idx].append(node)
                
                for k, nodes in expanded_group_nodes.items():
                    nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
                    group_nodes[k] = nodes[:constrained_beam_width]

            imagine_batch_obs = []
            imagine_batch_act = []
            imagine_batch_rew = []
            for k, nodes in group_nodes.items():
                end_node = nodes[0]
                imagine_obs = [end_node.obs]
                imagine_act = [end_node.action, end_node.action]  # dummy last action
                imagine_rew = [end_node.r, end_node.r]  # dummy last reward
                
                while end_node.prev.prev is not None:
                    end_node = end_node.prev
                    imagine_obs.insert(0, end_node.obs)
                    imagine_act.insert(0, end_node.action)
                    imagine_rew.insert(0, end_node.r)
                
                imagine_batch_obs.append(torch.stack(imagine_obs))
                imagine_batch_act.append(torch.cat(imagine_act))
                imagine_batch_rew.append(torch.FloatTensor(imagine_rew))
            
            imagine_batch_obs = torch.stack(imagine_batch_obs)
            imagine_batch_act = torch.stack(imagine_batch_act)
            imagine_batch_rew = torch.stack(imagine_batch_rew)
            
            imagine_batch = {}
            dummy_size = batch['observations'].size(0) - avail_batch['observations'].size(0)
            imagine_batch['observations'] = torch.cat((
                torch.cat(
                    (
                        obs_tokens, 
                        imagine_batch_obs.to(device),
                    ), 
                    dim=1,
                ), 
                torch.zeros(
                    dummy_size, avail_batch['observations'].size(1), obs_tokens.size(2), 
                    dtype=torch.long, 
                    device=device,
                )),
                dim=0
            )
            
            imagine_batch['actions'] = torch.cat((
                torch.cat(
                    (
                        avail_batch['actions'][:, -num_given_block + horizon + 1:], 
                        imagine_batch_act.to(device),
                    ), 
                    dim=1,
                ),
                torch.zeros(
                    dummy_size, avail_batch['actions'].size(1), 
                    dtype=torch.long, 
                    device=device,
                )),
                dim=0
            )
            
            imagine_batch['rewards'] = torch.cat((
                torch.cat(
                    (
                        avail_batch['rewards'][:, -num_given_block + horizon + 1:], 
                        imagine_batch_rew.to(device),
                    ), 
                    dim=1,
                ),
                torch.zeros(
                    dummy_size, avail_batch['rewards'].size(1), 
                    dtype=torch.float, 
                    device=device,
                )),
                dim=0
            )
            
            imagine_batch['ends'] = torch.zeros_like(
                imagine_batch['rewards'], 
                dtype=torch.bool, 
                device=device,
            )
            
            imagine_batch['envs'] = torch.cat(
                (
                    avail_batch['envs'], 
                    torch.ones(dummy_size, dtype=torch.long, device=device) * -1
                ),
                dim=0,
            )
            
            mask = torch.ones_like(
                imagine_batch['rewards'], 
                dtype=torch.bool, 
                device=device,
            )
            mask[:, num_given_block - horizon:] = False
            imagine_batch['mask_padding'] = mask

            return imagine_batch, {'synthetic amount': avail_idxs.sum().item()}
            
        else:
            if num_given_block == 1:
                initial_batch = avail_batch['observations'][:, -1]
            else:
                initial_batch = {
                    'observations': avail_batch['observations'][:, -num_given_block:],
                    'actions': avail_batch['actions'][:, -num_given_block:-1],
                    'rewards': avail_batch['rewards'][:, -num_given_block:-1],
                    'ends': avail_batch['ends'][:, -num_given_block:-1]
                }

            wm_env = WorldModelEnv(tokenizer, self, device, avail_batch['envs'])
            
            all_actions = []
            all_rewards = []
            all_ends = []
            all_observations = []

            _ = wm_env.reset_from_initial_observations(initial_batch)
            
            # record initial given block
            if num_given_block == 1:
                all_observations.append(initial_batch)
            else:
                for k in range(num_given_block):
                    all_observations.append(initial_batch['observations'][:, k])
                    if k != num_given_block - 1:
                        all_actions.append(initial_batch['actions'][:, k].cpu().reshape(-1, 1))
                        all_rewards.append(initial_batch['rewards'][:, k].cpu().reshape(-1, 1))
                        all_ends.append(initial_batch['ends'][:, k].cpu().reshape(-1, 1))
            # imagine
            for k in tqdm(range(horizon - num_given_block + 1), disable=not show_pbar, desc='Imagination', file=sys.stdout):
                all_actions.append(wm_env.last_actions.reshape(-1, 1))
                obs, reward, done, _ = wm_env.step(should_predict_next_obs=(k < horizon - num_given_block))
                all_rewards.append(torch.tensor(reward).reshape(-1, 1))
                all_ends.append(torch.tensor(done).reshape(-1, 1))
                if obs is not None:
                    all_observations.append(obs)

            ends = torch.cat(all_ends, dim=1)
            
            # create mask padding
            _, T = ends.shape
            mask_padding_ = torch.ones_like(ends).to(torch.bool)
            have_True_rows = ends.any(dim=1)
            
            if have_True_rows.any():
                first_one_indices = torch.argmax(ends.int()[have_True_rows], dim=1)
                indices_matrix = torch.arange(T).expand(have_True_rows.int().sum(), T)
                mask_padding_[have_True_rows] = indices_matrix <= first_one_indices.unsqueeze(1)
            
            wm_env_logs = {'epsilon': wm_env.epsilon, 'num_given_block': num_given_block}
            
            return ImagineOutput(
                observations=torch.stack(all_observations, dim=1).to(torch.float32),
                actions=torch.cat(all_actions, dim=1).to(device),
                rewards=torch.cat(all_rewards, dim=1).to(device),
                ends=ends.to(device, dtype=torch.long),
                mask_padding=mask_padding_.to(device),
                envs=avail_batch['envs']
            ), wm_env_logs
            

class StepNode:
    def __init__(
        self, 
        r: float, 
        v_star: float, 
        gamma: float, 
        batch_idx: int,
        action=None, 
        kv_cache=None, 
        prev: Optional['StepNode'] = None, 
        critic: Optional[torch.Tensor] = None,
        obs: Optional[torch.Tensor] = None,
    ) -> None:
        self.batch_idx = batch_idx
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