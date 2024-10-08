import math

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCausalAttention(nn.Module):
    def __init__(
        self,
        h_dim,
        tokens_per_block,
        max_blocks,
        n_heads,
        drop_p,
        att_mask=None,
    ):
        super().__init__()

        self.n_heads = n_heads
        max_T = tokens_per_block * max_blocks

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        if att_mask is not None:
            mask = att_mask
        else:
            ones = torch.ones((max_T, max_T))
            mask = torch.tril(ones)
            num_timesteps = max_blocks
            num_non_obs_tokens = 3  # R, a, r
            num_obs_tokens = tokens_per_block - num_non_obs_tokens
            diag = [
                np.ones((num_obs_tokens, num_obs_tokens)) if i % 2 == 0 else np.zeros(
                    (num_non_obs_tokens, num_non_obs_tokens))
                for i in range(num_timesteps * 2)
            ]
            block_diag = torch.from_numpy(scipy.linalg.block_diag(*diag))
            mask = torch.logical_or(mask, block_diag).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape  # batch size, seq length, h_dim * n_heads

        N, D = (
            self.n_heads,
            C // self.n_heads,
        )  # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        # causal mask applied to weights
        #print(f"shape of weights: {weights.shape}, shape of mask: {self.mask.shape}, T: {T}")
        weights = weights.masked_fill(
            self.mask[..., :T, :T] == 0, float("-inf")
        )
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(
        self,
        h_dim,
        tokens_per_block,
        max_blocks,
        n_heads,
        drop_p,
        att_mask=None,
    ):
        super().__init__()
        self.attention = MaskedCausalAttention(
            h_dim,
            tokens_per_block,
            max_blocks,
            n_heads,
            drop_p,
            att_mask=att_mask,
        )
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        #print(f"shape of x: {x.shape}, shape of attention: {self.attention(x).shape}")
        x = x + self.attention(x)  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        obs_vocab_size,
        act_vocab_size,
        h_dim,
        tokens_per_block,
        max_blocks,
        transformer_layers,
        n_heads,
        drop_p,
        name,
        num_bin=121,
    ):
        super().__init__()

        self.h_dim = h_dim
        self.num_bin = num_bin
        self.name = name

        ### transformer blocks
        blocks = [
            Block(
                h_dim,
                tokens_per_block,
                max_blocks,
                n_heads,
                drop_p,
            )
            for _ in range(transformer_layers)
        ]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_timestep = nn.Embedding(max_blocks, h_dim)
        self.embed_rtg = nn.Embedding(num_bin, h_dim)
        self.embed_state = nn.Embedding(obs_vocab_size, h_dim)
        self.embed_reward = nn.Embedding(3, h_dim)
        self.embed_action = nn.Embedding(act_vocab_size, h_dim)

        ### prediction heads
        self.predict_rtg = nn.Linear(h_dim, int(num_bin))
        self.predict_action = nn.Linear(h_dim, act_vocab_size)
        self.predict_reward = nn.Linear(h_dim, 3)

    def __repr__(self):
        return self.name
    
    def forward(self, timesteps, states, actions, returns_to_go, rewards):
        # timesteps: relative timestep
        # states: B, T, Patchs
        # actions: B, T
        # returns_to_go: B, T
        # rewards: B, T
        B, T, _ = states.shape

        returns_to_go = encode_return(returns_to_go.float(), num_bin=self.num_bin)  # check torch.LongTensor
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        rewards_embeddings = self.embed_reward(rewards) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        # TODO: use slicer to handle h!!!
        h = (
            torch.stack(
                (
                    state_embeddings,
                    returns_embeddings,
                    action_embeddings,
                    rewards_embeddings,
                ),
                dim=1,
            )
            .permute(0, 2, 1, 3)
            .reshape(B, 4 * T, self.h_dim)
        )

        # transformer and prediction
        h = self.transformer(h)

        h = h.reshape(B, T, 4, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_rtg(h[:, 0])  # predict next rtg given s
        action_preds = self.predict_action(
            h[:, 1]
        )  # predict action given s, R
        reward_preds = self.predict_reward(
            h[:, 2]
        )  # predict reward given s, R, a

        return action_preds, return_preds, reward_preds


# a version that does not use reward at all
class ElasticDecisionTransformer(
    DecisionTransformer
):
    def __init__(
        self,
        state_dim,
        act_dim,
        n_blocks,
        h_dim,
        context_len,
        n_heads,
        drop_p,
        env_name,
        max_timestep=4096,
        num_bin=120,
        dt_mask=False,
        rtg_scale=1000,
        num_inputs=3,
        real_rtg=False,
        is_continuous=True, # True for continuous action
    ):
        super().__init__(
            state_dim,
            act_dim,
            n_blocks,
            h_dim,
            context_len,
            n_heads,
            drop_p,
            env_name,
            max_timestep=max_timestep,
            num_bin=num_bin,
            dt_mask=dt_mask,
            rtg_scale=rtg_scale,
        )

        # return, state, action
        self.num_inputs = num_inputs
        self.is_continuous = is_continuous
        input_seq_len = num_inputs * context_len
        blocks = [
            Block(
                h_dim,
                input_seq_len,
                n_heads,
                drop_p,
                mgdt=True,
                dt_mask=dt_mask,
                num_inputs=num_inputs,
                real_rtg=real_rtg,
            )
            for _ in range(n_blocks)
        ]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # # discrete actions
        if not self.is_continuous:
            self.embed_action = torch.nn.Embedding(18, h_dim)
        else:
            self.embed_action = torch.nn.Linear(act_dim, h_dim)

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, int(num_bin))
        self.predict_rtg2 = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim + act_dim, state_dim)
        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(h_dim, act_dim)]
                + ([nn.Tanh()] if is_continuous else [])
            )
        )
        self.predict_reward = torch.nn.Linear(h_dim, 1)

    def forward(
        self, timesteps, states, actions, returns_to_go, *args, **kwargs
    ):
        B, T, _ = states.shape
        returns_to_go = returns_to_go.float()
        returns_to_go = (
            encode_return(
                self.env_name, returns_to_go, num_bin=self.num_bin, rtg_scale=self.rtg_scale
            )
            - self.num_bin / 2
        ) / (self.num_bin / 2)
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = (
            torch.stack(
                (
                    state_embeddings,
                    returns_embeddings,
                    action_embeddings,
                    # rewards_embeddings,
                ),
                dim=1,
            )
            .permute(0, 2, 1, 3)
            .reshape(B, self.num_inputs * T, self.h_dim)
        )

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)
        h = h.reshape(B, T, self.num_inputs, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_rtg(h[:, 0])  # predict next rtg given s
        return_preds2 = self.predict_rtg2(
            h[:, 0]
        )  # predict next rtg with implicit loss
        action_preds = self.predict_action(
            h[:, 1]
        )  # predict action given s, R
        state_preds = self.predict_state(torch.cat((h[:, 1], action_preds), 2))
        reward_preds = self.predict_reward(
            h[:, 2]
        )  # predict reward given s, R, a

        return (
            state_preds,
            action_preds,
            return_preds,
            return_preds2,
            reward_preds,
        )



def encode_return(ret, num_bin=121, rtg_scale=1.0):
    ret_max = 100
    ret_min = -20
    ret_max /= rtg_scale
    ret_min /= rtg_scale
    interval = (ret_max - ret_min) / (num_bin-1)
    ret = torch.clip(ret, ret_min, ret_max)
    return (ret - ret_min) // interval