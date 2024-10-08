from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.cuda.amp import autocast

from .action_tokenizer import (
    ATARI_NUM_ACTIONS,
    GAME_NAMES,
    LIMITED_ACTION_TO_FULL_ACTION,
)
from .batch import Batch


class Policy(ABC, nn.Module):
    """The base class for any RL policy.

    Tianshou aims to modularize RL algorithms. It comes into several classes of
    policies in Tianshou. All of the policy classes must inherit
    :class:`~tianshou.policy.BasePolicy`.

    A policy class typically has the following parts:

    * :meth:`~tianshou.policy.BasePolicy.__init__`: initialize the policy, including \
        coping the target network and so on;
    * :meth:`~tianshou.policy.BasePolicy.forward`: compute action with given \
        observation;

    Most of the policy needs a neural network to predict the action and an
    optimizer to optimize the policy. The rules of self-defined networks are:

    1. Input: observation "obs" (may be a ``numpy.ndarray``, a ``torch.Tensor``, a \
    dict or any others), hidden state "state" (for RNN usage), and other information \
    "info" provided by the environment.
    2. Output: some "logits", the next hidden state "state", and the intermediate \
    result during policy forwarding procedure "policy". The "logits" could be a tuple \
    instead of a ``torch.Tensor``. It depends on how the policy process the network \
    output. For example, in PPO, the return of the network might be \
    ``(mu, sigma), state`` for Gaussian policy. The "policy" can be a Batch of \
    torch.Tensor or other things, which will be stored in the replay buffer, and can \
    be accessed in the policy update process (e.g. in "policy.learn()", the \
    "batch.policy" is what you need).

    Since :class:`~tianshou.policy.BasePolicy` inherits ``torch.nn.Module``, you can
    use :class:`~tianshou.policy.BasePolicy` almost the same as ``torch.nn.Module``,
    for instance, loading and saving the model:
    ::

        torch.save(policy.state_dict(), "policy.pth")
        policy.load_state_dict(torch.load("policy.pth"))
    """

    def __init__(
        self,
        tokenizer: nn.Module,
        world_model: nn.Module,
        num_envs: int,
        dtype = torch.float32,
        env_tokens: Optional[torch.LongTensor] = None,
        num_given_steps: int = 4,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        
        self.n = num_envs
        self.token_buffer = None
        # self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(
        #     n=self.n, max_tokens=self.world_model.config_transformer.max_tokens)
        self.device = self.tokenizer.encoder.conv_in.weight.device
        self.env_tokens = env_tokens if env_tokens is not None else \
            torch.arange(self.n, dtype=torch.long, device=self.device)
        
        self.dtype = dtype
        self.num_given_steps = num_given_steps
        self.action_masks = None
        self.create_action_masks()
        
    def reset_keys_values_wm(self) -> None:
        self.token_buffer = None
        # self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(
        #     n=self.n, max_tokens=self.world_model.config_transformer.max_tokens)
    
    def create_action_masks(self) -> None:
        action_masks = []

        for env in GAME_NAMES:
            action_mask_for_each_env = []
            for action in range(ATARI_NUM_ACTIONS):
                action_mask_for_each_env.append(action in LIMITED_ACTION_TO_FULL_ACTION[env])
            action_masks.append(action_mask_for_each_env)
            
        self.action_masks = torch.tensor(action_masks, dtype=torch.bool, device=self.device)
        
    @torch.no_grad()
    def choose_action(self, critic: torch.FloatTensor) -> np.ndarray:
        """
        Choose action using epsilon-greedy policy.
        """
        if critic.ndim == 4:  # (num_q, B, L or 1, #actions)
            critic = critic[:, :, -1].mean(0)
        assert critic.ndim == 2
        
        # action mask according to env
        mask = self.action_masks[self.env_tokens]
        critic = critic.masked_fill(~mask, -torch.inf)
        actions = critic.argmax(dim=-1).cpu().numpy()
        return actions
    
    def map_action(self, act: np.ndarray) -> np.ndarray:
        return act

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which MUST have the following keys:

            * ``act`` an numpy.ndarray or a torch.Tensor, the action over \
                given batch data.
            * ``state`` a dict, an numpy.ndarray or a torch.Tensor, the \
                internal state of the policy, ``None`` as default.

        Other keys are user-defined. It depends on the algorithm. For example,
        ::

            # some code
            return Batch(logits=..., act=..., state=None, dist=...)

        The keyword ``policy`` is reserved and the corresponding data will be
        stored into the replay buffer. For instance,
        ::

            # some code
            return Batch(..., policy=Batch(log_prob=dist.log_prob(act)))
            # and in the sampled data batch, you can directly use
            # batch.policy.log_prob to get your data.

        .. note::

            In continuous action space, you should do another step "map_action" to get
            the real action:
            ::

                act = policy(batch).act  # doesn't map to the target action range
                act = policy.map_action(act, batch)
        """
        obs = torch.as_tensor(rearrange(batch.obs, '... H W C -> ... C H W')).to(self.device) / 255.0
        
        with autocast(dtype=self.dtype):
            obs_tokens = self.tokenizer.encode(obs, should_preprocess=True).tokens
            
            if self.token_buffer is None:
                self.token_buffer = obs_tokens.clone()
            else:
                act = torch.as_tensor(batch.act, dtype=torch.long, device=self.device).reshape(-1, 1)
                self.token_buffer = torch.cat((self.token_buffer, act, obs_tokens), dim=1)
                
            if self.token_buffer.size(1) > self.num_given_steps * self.world_model.config_transformer.tokens_per_block:  # self.world_model.config_transformer.max_tokens:
                self.token_buffer = self.token_buffer[:, self.world_model.config_transformer.tokens_per_block:]
            
            # if self.keys_values_wm.size == 0:
            #     input_tokens = obs_tokens
            # else:
            #     act = torch.as_tensor(batch.act, dtype=torch.long, device=self.device).reshape(-1, 1)
            #     input_tokens = torch.cat((act, obs_tokens), dim=1)
                
            outputs_wm = self.world_model(self.token_buffer, self.env_tokens)  # , past_keys_values=self.keys_values_wm)

        act = self.choose_action(outputs_wm.logits_q)
        return Batch(act=act, state=state)