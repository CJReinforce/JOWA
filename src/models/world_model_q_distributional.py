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

from .kv_caching import KeysValues
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
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config_transformer: TransformerConfig, config_critic, device = "cuda") -> None:
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
        self.task_emb = Embedder(
            max_blocks=config_transformer.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList(
                [nn.Embedding(config_transformer.max_tasks, config_transformer.embed_dim),  # ZeroEmbedding(1, config_transformer.embed_dim),
                 nn.Embedding(config_transformer.max_tasks, config_transformer.embed_dim)])
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
        
        self.head_q = Head(
            max_blocks=config_transformer.max_blocks,
            block_mask=last_obs_tokens_pattern,
            head_module=Ensemble(
                [mlp(
                    config_critic.latent_dim, 
                    2*[config_critic.mlp_dim], 
                    act_vocab_size * config_critic.num_atoms, 
                    dropout=config_critic.dropout
                    ) for _ in range(config_critic.num_q)]
            )
        )
        
        self.apply(init_weights)
        self.head_q_target = deepcopy(self.head_q).requires_grad_(False)
        
        # cql weight
        self.log_alpha = torch.tensor(
            np.log(self.config_critic.cql_weight),
            dtype=torch.float, device=device, requires_grad=True
        )

        self.atoms_support = torch.linspace(
            config_critic.vmin, config_critic.vmax, config_critic.num_atoms,
        ).to(device)
        
        self.training_steps = 0

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
        return "world_model"

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
        logits_q = self.head_q(x, num_steps=num_steps, prev_steps=prev_steps)  # (num_q, B, L, #actions)
        with torch.no_grad():
            logits_q_target = self.head_q_target(x.detach(), num_steps=num_steps, prev_steps=prev_steps)  # (num_q, B, L, #actions)

        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends, logits_q, logits_q_target)

    def compute_loss(self, real_batch: Batch, tokenizer: Tokenizer, train_critic: bool = False, imagine_horizon: Optional[int] = None, training: bool = True, image_batch: Optional[Batch] = None, **kwargs: Any) -> Tuple[LossWithIntermediateLosses, dict]:

        real_batch_size = real_batch['observations'].size(0)
        info_logs = {
            'info/world_model/reward_real': real_batch['rewards'].mean().item(),
            'info/world_model/done_real': real_batch['ends'].float().mean().item(),
            'info/world_model/alpha': self.log_alpha.exp().item(),
        }

        if train_critic and self.config_critic.use_imaginary_batch:
            # imagine
            imagine_batch, env_logs = self.imagine(real_batch if image_batch is None else image_batch, tokenizer, horizon=imagine_horizon)
            # check batch size
            imagine_batch_size = imagine_batch.observations.size(0)
            assert real_batch_size == imagine_batch_size
            
            # mix real and imagine batch
            mix_batch = {k: torch.cat([real_batch[k], getattr(imagine_batch, k)], dim=0) for k in real_batch.keys()}

            info_logs.update({
                'info/world_model/reward_imagine': imagine_batch.rewards.mean().item(),
                'info/world_model/done_imagine': imagine_batch.ends.float().mean().item(),
                'info/world_model/epsilon': env_logs['epsilon'],
                'info/world_model/num_given_block': env_logs['num_given_block'],
            })
        else:
            mix_batch = real_batch

        with torch.no_grad():
            obs_tokens = tokenizer.encode(mix_batch['observations'], should_preprocess=True).tokens  # (B, L, K)
        act_tokens = rearrange(mix_batch['actions'], 'b l -> b l 1')
        tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))
        outputs = self(tokens, mix_batch['envs'])
        
        if train_critic:
            # RL loss for head_q
            # REM + C51 loss
            alpha = torch.rand(self.config_critic.num_q).to(tokens.device)
            alpha = alpha / torch.sum(alpha)

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
            assert torch.all(first_ones & T-1-last_ones == 0)
            
            # not done state
            compute_q_mask = mix_batch['mask_padding'].clone()
            batch_indices = torch.arange(B)
            compute_q_mask[batch_indices, last_ones] = 0
            
            masked_q_s_chosen_a_dist = q_s_chosen_a_dist[compute_q_mask]  # (Bt, atoms)

            # done state
            ends = mix_batch['ends'].clone()  # (B, T)
            ends[~mix_batch['mask_padding']] = 0  # (B, T)
            assert torch.all(ends.sum(-1) <= 1)
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

                # # TODO:
                # if num_end > 0:
                #     # import ipdb; ipdb.set_trace()
                #     ended_reward = mix_batch['rewards'][ends.to(bool)]  # (num_end,)
                #     ended_reward_index = ((ended_reward - v_min) / delta_z).to(torch.long)  # (num_end,)
                #     ended_bellman_target = torch.zeros_like(ended_q_s_chosen_a_dist)  # (num_end, atoms)
                #     ended_bellman_target[torch.arange(num_end), ended_reward_index] = 1.
            
            loss_td_error = -(masked_bellman_target * torch.log(mixed_q_s_chosen_a_dist + 1e-8)).sum(-1).mean()
            # batch_td_error = masked_bellman_target.shape[0]

            # if num_end > 0:
            #     loss_ended_td_error = -(ended_bellman_target * torch.log(ended_q_s_chosen_a_dist + 1e-8)).sum()
            #     loss_td_error = loss_td_error + loss_ended_td_error
            #     batch_td_error += num_end
            
            # loss_td_error = loss_td_error / batch_td_error
                
            # CQL loss
            # for real batch
            dataset_expec = q_s_chosen_a_dist[:real_batch_size][real_batch['mask_padding']] @ self.atoms_support
            # for mix batch
            negative_sampling = torch.logsumexp(
                (ensemble_logits_q_s_dist.softmax(dim=-1).mean(dim=0) * self.atoms_support.view(1, 1, 1, -1)).sum(dim=-1), 
                dim=-1,
            )
            negative_sampling = negative_sampling[mix_batch['mask_padding']]

            alpha = self.get_valid_alpha()
            q_value_negative_sampling = negative_sampling.mean()
            q_value_positive_sampling = dataset_expec.mean()
            info_logs.update({
                'info/world_model/q_value_of_negative_sampling': q_value_negative_sampling.item(),
                'info/world_model/q_value_of_positive_sampling': q_value_positive_sampling.item(),
            })
            loss_cql_penalty = q_value_negative_sampling - q_value_positive_sampling

            loss_alpha = - alpha * (loss_cql_penalty.detach() - self.config_critic.target_cql_penalty_value)
            loss_cql_penalty = loss_cql_penalty * alpha.detach()

        # Supervised loss for dynamic model and predictors in world model
        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(
            obs_tokens[:real_batch_size], real_batch['rewards'], real_batch['ends'], real_batch['mask_padding'])
        logits_observations = rearrange(outputs.logits_observations[:real_batch_size, :-1], 'b t o -> (b t) o')
        
        loss_obs = F.cross_entropy(logits_observations, labels_observations) * self.config_critic.supervised_weight
        loss_rewards = F.cross_entropy(rearrange(
            outputs.logits_rewards[:real_batch_size], 'b t e -> (b t) e'), labels_rewards) * self.config_critic.supervised_weight
        loss_ends = F.cross_entropy(rearrange(
            outputs.logits_ends[:real_batch_size], 'b t e -> (b t) e'), labels_ends) * self.config_critic.supervised_weight
        
        if training:
            self.training_steps += 1

        if train_critic:
            return LossWithIntermediateLosses(
                loss_obs=loss_obs, 
                loss_rewards=loss_rewards, 
                loss_ends=loss_ends, 
                loss_td_error=loss_td_error, 
                loss_cql_penalty=loss_cql_penalty,
                loss_alpha=loss_alpha
            ), info_logs
        else:
            return LossWithIntermediateLosses(
                loss_obs=loss_obs, 
                loss_rewards=loss_rewards, 
                loss_ends=loss_ends
            ), info_logs

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100), 'b t k -> b (t k)')[:, 1:]
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)
    
    @torch.no_grad()
    def imagine(self, batch: Batch, tokenizer: Tokenizer, horizon: int, show_pbar: bool = False) -> Tuple[ImagineOutput, dict]:
        # choose the num_given_block from 4 to 1
        for num_given_block in range(10, 0, -1):
            if batch['mask_padding'][:, -num_given_block:].all():
                break
        assert batch['mask_padding'][:, -num_given_block:].all()  # must sample from ends
        
        if num_given_block == 1:
            initial_batch = batch['observations'][:, -1]
        else:
            initial_batch = {
                'observations': batch['observations'][:, -num_given_block:],
                'actions': batch['actions'][:, -num_given_block:-1],
                'rewards': batch['rewards'][:, -num_given_block:-1],
                'ends': batch['ends'][:, -num_given_block:-1]
            }

        device = batch['observations'].device
        wm_env = WorldModelEnv(tokenizer, self, device, batch['envs'])
        
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
            envs=batch['envs'],
        ), wm_env_logs