import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from action_tokenizer import action_masks
from envs.world_model_env import WorldModelEnv
from utils import (
    Batch,
    Ensemble,
    LossWithIntermediateLosses,
    StepNode,
    ZeroEmbedding,
    init_transformer_weights,
    mlp,
)

from .kv_caching import KeysValues, concate_kv, split_kv
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig


@dataclass
class JOWAOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    logits_q: torch.FloatTensor
    logits_q_target: torch.FloatTensor


class JOWAModel(nn.Module):
    def __init__(
        self, 
        obs_vocab_size: int, 
        act_vocab_size: int, 
        config_transformer: TransformerConfig, 
        config_critic_arch,
        config_critic_train, 
        device = "cuda", 
        name: str = 'JOWA',
    ) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config_transformer = config_transformer
        self.config_critic_train = config_critic_train
        self.num_q = config_critic_arch.num_q
        self.transformer = Transformer(config_transformer)

        all_but_last_obs_tokens_pattern = torch.ones(config_transformer.tokens_per_block)
        all_but_last_obs_tokens_pattern[-2] = 0
        last_obs_tokens_pattern = 1 - all_but_last_obs_tokens_pattern
        act_tokens_pattern = torch.zeros(self.config_transformer.tokens_per_block)
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern

        self.pos_emb = nn.Embedding(config_transformer.max_tokens, config_transformer.embed_dim)

        if config_critic_train.use_task_embed:
            self.task_emb = Embedder(
                max_blocks=config_transformer.max_blocks,
                block_masks=[act_tokens_pattern, obs_tokens_pattern],
                embedding_tables=nn.ModuleList(
                    [
                        nn.Embedding(config_transformer.max_tasks, config_transformer.embed_dim),
                        nn.Embedding(config_transformer.max_tasks, config_transformer.embed_dim)
                    ]
                )
            )
        else:
            self.task_emb = Embedder(
                max_blocks=config_transformer.max_blocks,
                block_masks=[act_tokens_pattern, obs_tokens_pattern],
                embedding_tables=nn.ModuleList(
                    [
                        ZeroEmbedding(config_transformer.max_tasks, config_transformer.embed_dim),
                        ZeroEmbedding(config_transformer.max_tasks, config_transformer.embed_dim)
                    ]
                )
            )

        self.embedder = Embedder(
            max_blocks=config_transformer.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList(
                [
                    nn.Embedding(act_vocab_size, config_transformer.embed_dim), 
                    nn.Embedding(obs_vocab_size, config_transformer.embed_dim)
                ]
            )
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
        
        self.q_penalty_method = config_critic_train.q_penalty
        self.td_loss_method = config_critic_train.td_loss
        self.q_loss_backwards_wm = bool(config_critic_train.q_loss_backwards_wm)
        assert self.q_penalty_method in ['cql', 'combo', None]
        assert self.td_loss_method in ['c51', 'mse']

        q_output_dim = act_vocab_size * config_critic_train.num_atoms if \
            self.td_loss_method == 'c51' else act_vocab_size
        self.head_q = Head(
            max_blocks=config_transformer.max_blocks,
            block_mask=last_obs_tokens_pattern,
            head_module=Ensemble([
                mlp(
                    config_transformer.embed_dim, 
                    [config_critic_arch.mlp_dim] * 2, 
                    q_output_dim, 
                    dropout=config_critic_arch.dropout
                ) for _ in range(config_critic_arch.num_q)
            ])
        )
        
        self.apply(init_transformer_weights)
        self.head_q_target = deepcopy(self.head_q).requires_grad_(False)
        
        # cql weight
        self.log_alpha = torch.tensor(
            config_critic_train.cql_weight,
            dtype=torch.float32, device=device,
        ).log()

        if self.td_loss_method == 'c51':
            self.atoms_support = torch.linspace(
                config_critic_train.vmin, 
                config_critic_train.vmax, 
                config_critic_train.num_atoms,
            ).to(device)
        
        self.training_steps = 0
        self.wm_name = name if 'jowa' in name.lower() else 'JOWA_' + name

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self.head_q_target.train(False)
        return self
    
    def _hard_update(self):
        with torch.no_grad():
            self.head_q_target.load_state_dict(self.head_q.state_dict())
          
    def hard_update_target_Q(self, wo_check=False):
        """
        Hard-update target Q-networks.
        """
        if wo_check:
            self._hard_update()
        elif self.training_steps % self.config_critic_train.target_update_frequency == 0:
            self._hard_update()

    def get_valid_alpha(self):
        return self.log_alpha.exp().clamp(min=0.0)
        
    def __repr__(self) -> str:
        return self.wm_name

    def forward(
        self, 
        tokens: torch.LongTensor, 
        tasks: torch.LongTensor, 
        past_keys_values: Optional[KeysValues] = None
    ) -> JOWAOutput:
        num_steps = tokens.size(1)  # (B, T)
        assert tasks.size(0) == tokens.size(0)  # B == B
        assert num_steps <= self.config_transformer.max_tokens
        
        if past_keys_values is not None and \
            past_keys_values.size + num_steps > past_keys_values[0]._k_cache._cache.shape[2]:
            raise IndexError("Past keys_values are too short to accomodate the current tokens.")
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        sequences = self.embedder(tokens, num_steps, prev_steps) + \
            self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device)) + \
            self.task_emb(tasks.view(-1, 1).repeat(1, num_steps), num_steps, prev_steps)
        
        x = self.transformer(sequences, past_keys_values)

        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)  # (B, L, 3)
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)  # (B, L, 2)
        logits_q = self.head_q(
            x if self.q_loss_backwards_wm else x.detach(), 
            num_steps=num_steps, 
            prev_steps=prev_steps,
        )  # (num_q, B, L, #actions)
        with torch.no_grad():
            logits_q_target = self.head_q_target(
                x.detach(), 
                num_steps=num_steps, 
                prev_steps=prev_steps,
            )  # (num_q, B, L, #actions)

        return JOWAOutput(
            x, 
            logits_observations, 
            logits_rewards, 
            logits_ends, 
            logits_q, 
            logits_q_target,
        )

    def compute_loss(
        self, 
        real_batch: Batch, 
        tokenizer: Tokenizer, 
        train_critic: bool = False, 
        training: bool = True, 
        imagine_batch: Optional[Batch] = None, 
        **kwargs: Any,
    ) -> Tuple[LossWithIntermediateLosses, Dict]:

        real_batch_size = real_batch['observations'].size(0)
        info_logs = {
            f'info/{str(self)}/reward_real': real_batch['rewards'].mean().item(),
            f'info/{str(self)}/done_real': real_batch['ends'].float().mean().item(),
            f'info/{str(self)}/alpha': self.log_alpha.exp().item(),
        }
        additional_losses = {}

        if imagine_batch is not None:
            info_logs.update({
                f'info/{str(self)}/reward_imagine': imagine_batch['rewards'].mean().item(),
                f'info/{str(self)}/done_imagine': imagine_batch['ends'].float().mean().item(),
            })

            mix_batch = {
                k: v \
                if k=='observations' and v.shape[1:] != imagine_batch[k].shape[1:] \
                else torch.cat(
                    (v, imagine_batch[k]), 
                    dim=0,
                ) for k, v in real_batch.items()
            }
        else:
            mix_batch = real_batch
        mix_batch_size = mix_batch['ends'].size(0)

        if mix_batch['observations'].ndim == 3:
            if mix_batch['observations'].size(0) == mix_batch_size:
                obs_tokens = mix_batch['observations']
            else:
                with torch.no_grad():
                    obs_tokens = tokenizer.encode(
                        imagine_batch['observations'], 
                        should_preprocess=True,
                    ).tokens  # (B, L, K)
                    
                    obs_tokens = torch.cat(
                        (
                            mix_batch['observations'],
                            obs_tokens, 
                        ), 
                        dim=0,
                    )
        else:
            with torch.no_grad():
                obs_tokens = tokenizer.encode(
                    mix_batch['observations'], 
                    should_preprocess=True,
                ).tokens  # (B, L, K)
                
                if mix_batch['observations'].size(0) != mix_batch_size:
                    obs_tokens = torch.cat(
                        (
                            obs_tokens, 
                            imagine_batch['observations']
                        ), 
                        dim=0,
                    )
        act_tokens = rearrange(mix_batch['actions'], 'b l -> b l 1')
        tokens = rearrange(
            torch.cat((obs_tokens, act_tokens), dim=2), 
            'b l k1 -> b (l k1)',
        )  # (B, L(K+1))
        outputs = self(tokens, mix_batch['envs'])
        
        # RL loss
        if train_critic:
            # REM
            alpha = torch.rand(self.num_q).to(tokens.device) if \
                self.config_critic_train.use_rem else \
                    torch.ones(self.num_q).to(tokens.device)
            alpha = alpha / torch.sum(alpha)

            # C51
            if self.td_loss_method == 'c51':
                # outputs.logits_q -> (num_q, B, L, actions * atoms)
                ensemble_logits_q_s_dist = rearrange(
                    outputs.logits_q, 
                    'q b l (a z) -> q b l a z', 
                    a=self.act_vocab_size,
                )  # (num_q, B, L, actions, atoms)
                ensemble_logits_q_s_chosen_a_dist = ensemble_logits_q_s_dist.gather(
                    3, 
                    act_tokens.unsqueeze(0).unsqueeze(-1).expand(
                        ensemble_logits_q_s_dist.shape[0], 
                        -1, -1, -1, 
                        ensemble_logits_q_s_dist.shape[-1],
                    )
                ).squeeze(3)  # (num_q, B, L, atoms)
                q_s_chosen_a_dist = torch.sum(
                    alpha.view(-1, 1, 1, 1) * ensemble_logits_q_s_chosen_a_dist.softmax(
                        dim=-1,
                    ),
                    dim=0,
                )  # (B, L, atoms)
                
                # compute mask for q
                B, T = q_s_chosen_a_dist.shape[:2]
                mix_batch_mask_padding = mix_batch['mask_padding'].int()
                first_ones = torch.argmax(mix_batch_mask_padding, dim=1)
                last_ones = T - 1 - torch.argmax(mix_batch_mask_padding.flip(dims=[1]), dim=1)
                
                # not done state
                compute_q_mask = mix_batch['mask_padding'].clone()
                batch_indices = torch.arange(B)
                compute_q_mask[batch_indices, last_ones] = 0
                
                masked_q_s_chosen_a_dist = q_s_chosen_a_dist[compute_q_mask]  # (Bt, atoms)

                # done state
                ends = mix_batch['ends'].clone()  # (B, T)
                ends[~mix_batch['mask_padding']] = 0  # (B, T)
                ends = ends.to(bool)

                ended_q_s_chosen_a_dist = q_s_chosen_a_dist[ends]  # (num_end, atoms)
                num_end = ended_q_s_chosen_a_dist.shape[0]

                mixed_q_s_chosen_a_dist = torch.cat(
                    (masked_q_s_chosen_a_dist, ended_q_s_chosen_a_dist), 
                    dim=0,
                )  # (Bt + num_end, atoms)
                
                with torch.no_grad():
                    # action mask
                    mask = action_masks.to(mix_batch['ends'].device)[mix_batch['envs']]
                    
                    ensemble_logits_q_next_s_dist = rearrange(
                        outputs.logits_q_target, 
                        'q b l (a z) -> q b l a z', 
                        a=self.act_vocab_size,
                    )  # (num_q, B, L, actions, atoms)
                    q_next_s_dist = torch.sum(
                        alpha.view(-1, 1, 1, 1, 1) * ensemble_logits_q_next_s_dist.softmax(
                            dim=-1,
                        ), 
                        dim=0,
                    )  # (B, L, actions, atoms)
                    masked_q_next_s = (
                        q_next_s_dist * self.atoms_support.view(1, 1, 1, -1)
                    ).sum(-1).masked_fill(
                        ~mask.unsqueeze(1).expand(-1, q_next_s_dist.shape[1], -1), 
                        -torch.inf,
                    )  # (B, L, actions)
                    a_star = torch.argmax(masked_q_next_s, dim=-1, keepdim=True)  # (B, L, 1)

                    q_next_s_argmax_a_dist = rearrange(
                        q_next_s_dist.gather(
                            2, 
                            a_star.unsqueeze(-1).expand(
                                -1, -1, -1, 
                                q_next_s_dist.shape[-1],
                            )
                        ).squeeze(2), 
                        'b l z -> (b l) z',
                    )  # (BL, atoms)

                    compute_q_target_mask = mix_batch['mask_padding'].clone()
                    compute_q_target_mask[batch_indices, first_ones] = 0

                    masked_q_next_s_argmax_a_dist = q_next_s_argmax_a_dist[
                        compute_q_target_mask.view(-1)
                    ]  # (Bt, atoms)
                    num_dims = self.atoms_support.shape[0]
                    ended_q_next_s_argmax_a_dist = torch.ones(
                        (num_end, num_dims), 
                        dtype=masked_q_next_s_argmax_a_dist.dtype, 
                        device=masked_q_next_s_argmax_a_dist.device
                    ) * 1.0 / num_dims  # fake, (num_end, atoms)
                    mixed_q_next_s_argmax_a_dist = torch.cat(
                        (masked_q_next_s_argmax_a_dist, ended_q_next_s_argmax_a_dist), 
                        dim=0,
                    )  # (Bt + num_end, atoms)

                    target_support = mix_batch['rewards'][compute_q_mask].unsqueeze(-1) + \
                        self.config_critic_train.gamma * self.atoms_support.unsqueeze(0)  # (Bt, atoms)
                    target_support_for_end = mix_batch['rewards'][ends].unsqueeze(-1) + \
                        0.0 * self.atoms_support.unsqueeze(0)
                    mixed_target_support = torch.cat(
                        (target_support, target_support_for_end), 
                        dim=0,
                    )  # (Bt + num_end, atoms)
                    
                    # project distribution
                    v_min, v_max = self.atoms_support[0], self.atoms_support[-1]
                    delta_z = (v_max - v_min) / (num_dims - 1)
                    clipped_support = torch.clip(
                        mixed_target_support, 
                        v_min, v_max,
                    )  # (Bt + num_end, atoms)

                    mixed_bellman_target = (
                        1 - (clipped_support.unsqueeze(1) - self.atoms_support.view(1, -1, 1)
                        ).abs() / delta_z
                    ).clamp(0, 1) * mixed_q_next_s_argmax_a_dist.unsqueeze(1)
                    mixed_bellman_target = mixed_bellman_target.sum(-1)  # (Bt + num_end, atoms)
                
                loss_td_error = -(
                    mixed_bellman_target * torch.log(mixed_q_s_chosen_a_dist + 1e-8)
                ).sum(-1).mean()
                additional_losses['loss_td_error'] = loss_td_error
                
                # CQL / COMBO loss
                if self.q_penalty_method is not None:
                    alpha = self.get_valid_alpha()

                    # negative Q
                    negative_sampling = torch.logsumexp(
                        (
                            ensemble_logits_q_s_dist.softmax(
                                dim=-1,
                            ).mean(dim=0) * self.atoms_support.view(
                                1, 1, 1, -1,
                            )
                        ).sum(dim=-1), 
                        dim=-1,
                    )
                    negative_sampling = negative_sampling[mix_batch['mask_padding']]
                    q_value_negative_samples = negative_sampling.mean()
                    info_logs[
                        f'info/{str(self)}/q_value_of_negative_samples'
                    ] = q_value_negative_samples.item()

                    # positive Q
                    positive_bs = real_batch_size if \
                        self.q_penalty_method == 'combo' else mix_batch_size
                    dataset_expec = q_s_chosen_a_dist[mix_batch['mask_padding']][
                        :positive_bs
                    ] @ self.atoms_support
                    q_value_positive_samples = dataset_expec.mean()
                    info_logs[
                        f'info/{str(self)}/q_value_of_positive_samples'
                    ] = q_value_positive_samples.item()

                    loss_cql_penalty = (
                        q_value_negative_samples - q_value_positive_samples
                    ) * alpha.detach()
                    additional_losses['loss_cql_penalty'] = loss_cql_penalty
            
            # MSE
            else:
                ensemble_q = outputs.logits_q.gather(
                    -1, 
                    act_tokens.expand(
                        self.num_q, 
                        *act_tokens.shape
                    )
                ).squeeze(-1)  # (num_q, B, L)
                q = torch.sum(alpha.view(-1, 1, 1) * ensemble_q, dim=0)  # (B, L)
                
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
                ends = ends.to(bool)

                predict_q_for_done = q[ends]  # (num_end,)
                mixed_predict_q = torch.cat(
                    (predict_q, predict_q_for_done), 
                    dim=0,
                )  # (Bt + num_end,)
                
                with torch.no_grad():
                    # action mask
                    mask = action_masks.to(mix_batch['ends'].device)[mix_batch['envs']]
                    
                    target_v = torch.sum(
                        alpha.view(-1, 1, 1, 1) * outputs.logits_q_target, 
                        dim=0,
                    )  # (B, L, #actions)
                    target_v = target_v.masked_fill(
                        ~mask.unsqueeze(1).expand_as(target_v), 
                        -torch.inf,
                    )
                    target_v = torch.max(target_v, dim=-1).values  # (B, L)
                    
                    compute_q_target_mask = mix_batch['mask_padding'].clone()
                    compute_q_target_mask[batch_indices, first_ones] = 0
                    
                    target_v = target_v[compute_q_target_mask]
                    target_q = mix_batch['rewards'][compute_q_mask] + \
                        self.config_critic_train.gamma * target_v  # (Bt,)
                    target_q_for_done = mix_batch['rewards'][ends]  # (num_end,)
                    mixed_target_q = torch.cat(
                        (target_q, target_q_for_done), 
                        dim=0,
                    )  # (Bt + num_end,)
                
                loss_td_error = F.mse_loss(mixed_predict_q, mixed_target_q) * 0.5
                additional_losses['loss_td_error'] = loss_td_error
                    
                # CQL / COMBO loss
                if self.q_penalty_method is not None:
                    alpha = self.get_valid_alpha()

                    # for mix batch
                    negative_sampling = torch.logsumexp(
                        torch.sum(
                            alpha.view(-1, 1, 1, 1) * outputs.logits_q, 
                            dim=0,
                        ), 
                        dim=-1,
                    )
                    negative_sampling = negative_sampling[mix_batch['mask_padding']]

                    q_value_negative_samples = negative_sampling.mean()
                    info_logs[
                        f'info/{str(self)}/q_value_of_negative_samples'
                    ] = q_value_negative_samples.item()
                    
                    # for real batch
                    positive_bs = real_batch_size if \
                        self.q_penalty_method == 'combo' else mix_batch_size
                    dataset_expec = q[mix_batch['mask_padding']][:positive_bs]
                    q_value_positive_samples = dataset_expec.mean()
                    info_logs[
                        f'info/{str(self)}/q_value_of_positive_samples'
                    ] = q_value_positive_samples.item()

                    loss_cql_penalty = (
                        q_value_negative_samples - q_value_positive_samples
                    ) * alpha.detach()
                    additional_losses['loss_cql_penalty'] = loss_cql_penalty

        # Supervised loss for world-part module
        labels_obses, labels_rewards, labels_ends = self.compute_labels_world_model(
            obs_tokens[:real_batch_size], 
            real_batch['rewards'], 
            real_batch['ends'], 
            real_batch['mask_padding'],
        )
        logits_obses = rearrange(
            outputs.logits_observations[:real_batch_size, :-1], 
            'b t o -> (b t) o',
        )
        
        supervised_weight = self.config_critic_train.supervised_weight if train_critic else 1.0
        loss_obs = F.cross_entropy(logits_obses, labels_obses) * supervised_weight
        loss_rewards = F.cross_entropy(
            rearrange(
                outputs.logits_rewards[:real_batch_size], 
                'b t e -> (b t) e',
            ), 
            labels_rewards,
        ) * supervised_weight
        loss_ends = F.cross_entropy(
            rearrange(
                outputs.logits_ends[:real_batch_size], 
                'b t e -> (b t) e',
            ), 
            labels_ends,
        ) * supervised_weight
        
        if training:
            self.training_steps += 1

        return LossWithIntermediateLosses(
            loss_obs=loss_obs, 
            loss_rewards=loss_rewards, 
            loss_ends=loss_ends, 
            **additional_losses,
        ), info_logs

    def compute_labels_world_model(
        self, 
        obs_tokens: torch.Tensor, 
        rewards: torch.Tensor, 
        ends: torch.Tensor, 
        mask_padding: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done

        mask_fill = torch.logical_not(mask_padding)
        labels_obses = rearrange(
            obs_tokens.masked_fill(
                mask_fill.unsqueeze(-1).expand_as(obs_tokens), 
                -100,
            ), 
            'b t k -> b (t k)',
        )[:, 1:]
        labels_rewards = (rewards.sign() + 1).masked_fill(
            mask_fill, -100
        ).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)

        return labels_obses.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)
    
    @torch.no_grad()
    def imagine(
        self, 
        batch: Batch, 
        tokenizer: Tokenizer, 
        horizon: int, 
        beam_width: Optional[int] = None, 
        should_sample: bool = True,
        show_pbar: bool = False
    ) -> Tuple[Dict, Dict]:
        policy_in_imagination = self.config_critic_train.policy_in_imagination
        assert policy_in_imagination in ['epsilon_greedy', 'planning']
        
        max_blocks = self.config_transformer.max_blocks
        avail_idxs = batch['mask_padding'][:, -max_blocks + horizon:].all(1)
        avail_batch = {k: v[avail_idxs] for k, v in batch.items()}
        device = avail_batch['ends'].device
        
        if policy_in_imagination == 'planning':
            assert beam_width is not None
            
            # init token
            obs_tokens = tokenizer.encode(
                avail_batch['observations'][:, -max_blocks + horizon:], 
                should_preprocess=True,
            ).tokens  # (B, L, K)
            act_tokens = rearrange(
                avail_batch['actions'][:, -max_blocks + horizon:], 
                'b l -> b l 1',
            )
            tokens = rearrange(
                torch.cat((obs_tokens, act_tokens), dim=2), 
                'b l k1 -> b (l k1)',
            )[:, :-1]  # (B, L(K+1))

            # init env
            wm_env = WorldModelEnv(tokenizer, self, device, avail_batch['envs'])
            gamma = self.config_critic_train.gamma
            _ = wm_env.reset_from_initial_observations(tokens, act_mode="logits")
            
            # init root node
            critic = wm_env.last_actions
            v_star = critic.max(1).values
            root_nodes_kv = split_kv(wm_env.keys_values_wm)
            
            group_nodes = {}
            for i in range(critic.size(0)):
                root_node = StepNode(
                    r=0., 
                    v_star=v_star[i].item(), 
                    gamma=gamma, 
                    batch_idx=i,
                    action=None, 
                    kv_cache=deepcopy(root_nodes_kv[i]),
                    prev=None, 
                    critic=critic[i], 
                    obs=tokens[i, -self.config_transformer.tokens_per_block + 1:]
                )
                group_nodes[i] = [root_node]
            
            # planning in the dream
            for _ in range(horizon):
                expanded_group_nodes = {k:[] for k in group_nodes.keys()}
                all_group_nodes_top_k_actions = []
                all_group_nodes_cache = []
                
                # build top k actions for batch
                for k, nodes in group_nodes.items():
                    all_nodes_top_k_actions = []
                    all_nodes_cache = []

                    for ori_node in nodes:
                        top_k_actions = torch.topk(ori_node.critic, beam_width, dim=-1).indices.cpu()
                        all_nodes_top_k_actions.append(top_k_actions.view(-1, 1))  # (beam_width, 1)
                        all_nodes_cache.append(ori_node.kv_cache)

                    all_nodes_top_k_actions = torch.cat(all_nodes_top_k_actions, dim=0)  # (beam_width * num_of_nodes, k)
                    all_nodes_cache = concate_kv(all_nodes_cache, repeat_num=beam_width)
                    
                    all_group_nodes_top_k_actions.append(all_nodes_top_k_actions)
                    all_group_nodes_cache.append(all_nodes_cache)
                
                all_group_nodes_top_k_actions = torch.cat(all_group_nodes_top_k_actions, dim=0)  # (bs * beam_width * num_of_nodes, k)
                all_group_nodes_cache = concate_kv(all_group_nodes_cache)
                
                # step in the dream
                wm_env.env_tokens = avail_batch['envs'].repeat_interleave(
                    all_nodes_top_k_actions.size(0)
                )
                _, all_rew, _, _ = wm_env.step(
                    all_group_nodes_top_k_actions, 
                    should_sample=should_sample, 
                    act_mode="logits", 
                    kv_cache=all_group_nodes_cache,
                )
                all_critic = wm_env.last_actions  # (bs * beam_width * num_of_nodes, num_of_actions)
                all_v_star = all_critic.max(dim=-1).values  # (bs * beam_width * num_of_nodes, 1)
                all_group_nodes_cache_split = split_kv(all_group_nodes_cache)

                # build new nodes
                for i in range(all_rew.size):
                    batch_idx = i // all_nodes_top_k_actions.size(0)
                    act, critic, v_star, rew, cache, obs_token = map(
                        lambda x: x[i], 
                        [
                            all_group_nodes_top_k_actions, 
                            all_critic, 
                            all_v_star,
                            all_rew,
                            all_group_nodes_cache_split,
                            wm_env.obs_tokens,
                        ]
                    )
                    
                    node = StepNode(
                        r=rew, 
                        v_star=v_star, 
                        gamma=gamma, 
                        batch_idx=batch_idx,
                        action=act, 
                        kv_cache=cache, 
                        prev=group_nodes[batch_idx][
                            (i - batch_idx * all_nodes_top_k_actions.size(0)) // beam_width
                        ], 
                        critic=critic,
                        obs=obs_token
                    )
                    expanded_group_nodes[batch_idx].append(node)
                
                # beam search, only retain K nodes
                for k, nodes in expanded_group_nodes.items():
                    nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
                    group_nodes[k] = nodes[:beam_width]

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
            dummy_size = batch['ends'].size(0) - avail_batch['ends'].size(0)
            imagine_batch['observations'] = torch.cat((
                torch.cat(
                    (
                        avail_batch['observations'][:, -max_blocks + horizon:], 
                        wm_env.decode_obs_tokens(obs_tokens=imagine_batch_obs),
                    ).mul(255).to(torch.uint8), 
                    dim=1,
                ), 
                torch.zeros(
                    dummy_size, *avail_batch['observations'].shape[1:], 
                    dtype=torch.uint8, 
                    device=device,
                )),
                dim=0,
            )
            
            imagine_batch['actions'] = torch.cat((
                torch.cat(
                    (
                        avail_batch['actions'][:, -max_blocks + horizon + 1:], 
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
                        avail_batch['rewards'][:, -max_blocks + horizon + 1:], 
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
            mask[:, max_blocks - horizon:] = False
            imagine_batch['mask_padding'] = mask

            return imagine_batch, {'synthetic amount': avail_idxs.sum().item()}
            
        else:
            initial_batch = {
                'observations': avail_batch['observations'][:, -max_blocks + horizon:],
                'actions': avail_batch['actions'][:, -max_blocks + horizon:-1],
                'rewards': avail_batch['rewards'][:, -max_blocks + horizon:-1],
                'ends': avail_batch['ends'][:, -max_blocks + horizon:-1]
            }

            wm_env = WorldModelEnv(tokenizer, self, device, avail_batch['envs'])
            
            all_actions = []
            all_rewards = []
            all_ends = []
            all_observations = []

            _ = wm_env.reset_from_initial_observations(initial_batch)
            
            # record initial given block
            for k in range(initial_batch['observations'].shape[1]):
                all_observations.append(initial_batch['observations'][:, k])
                if k != max_blocks - 1:
                    all_actions.append(initial_batch['actions'][:, k].cpu().reshape(-1, 1))
                    all_rewards.append(initial_batch['rewards'][:, k].cpu().reshape(-1, 1))
                    all_ends.append(initial_batch['ends'][:, k].cpu().reshape(-1, 1))
            
            # imagine
            for k in tqdm(
                range(horizon - max_blocks + 1), 
                disable=not show_pbar, 
                desc='Imagination', 
                file=sys.stdout
            ):
                all_actions.append(wm_env.last_actions.reshape(-1, 1))
                obs, reward, done, _ = wm_env.step(
                    should_predict_next_obs=(k < horizon - max_blocks), 
                    should_sample=should_sample,
                )
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

            # TODO: need padding if gathering
            return dict(
                observations=torch.stack(all_observations, dim=1).mul(255).to(torch.uint8),
                actions=torch.cat(all_actions, dim=1),
                rewards=torch.cat(all_rewards, dim=1),
                ends=ends.to(dtype=torch.long),
                mask_padding=mask_padding_,
                envs=avail_batch['envs'],
            ), {
                'epsilon': wm_env.epsilon, 
                'synthetic amount': avail_idxs.sum().item(),
            }