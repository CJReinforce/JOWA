import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np
import torch
from tqdm import tqdm

from .batch import Batch
from .policy import Policy
from .replay_buffer import VectorReplayBuffer
from .vectorized_environment import BaseVectorEnv, SubprocVectorEnv


class Collector(object):
    """Collector enables the policy to interact with different types of envs with \
    exact number of steps or episodes.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class.
        If set to None, it will not store the data. Default to None.
    :param function preprocess_fn: a function called before the data has been added to
        the buffer, see issue #42 and :ref:`preprocess_fn`. Default to None.
    :param bool exploration_noise: determine whether the action needs to be modified
        with corresponding policy's exploration noise. If so, "policy.
        exploration_noise(act, batch)" will be called automatically to add the
        exploration noise into action. Default to False.

    The "preprocess_fn" is a function called before the data has been added to the
    buffer with batch format. It will receive only "obs" and "env_id" when the
    collector resets the environment, and will receive six keys "obs_next", "rew",
    "done", "info", "policy" and "env_id" in a normal env step. It returns either a
    dict or a :class:`~tianshou.data.Batch` with the modified keys and values. Examples
    are in "test/base/test_collector.py".

    .. note::

        Please make sure the given environment has a time limitation if using n_episode
        collect option.
    """

    def __init__(
        self,
        policy: Policy,
        env: Union[gym.Env, BaseVectorEnv],
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if isinstance(env, gym.Env) and not hasattr(env, "__len__"):
            warnings.warn("Single environment detected, wrap to SubprocVectorEnv.")
            env = SubprocVectorEnv([lambda: env])
        self.env = env
        self.env_num = len(env)
        self._assign_buffer()
        self.policy = policy
        self._action_space = env.action_space
        self._seed = seed
        # avoid creating attribute outside __init__
        self.reset()

    def _assign_buffer(self) -> None:
        """Check if the buffer matches the constraint."""
        self.buffer = VectorReplayBuffer(self.env_num, self.env_num)

    def reset(self, keep_statistics: bool = False) -> None:
        """Reset all related variables in the collector."""
        # use empty Batch for "state" so that self.data supports slicing
        # convert empty Batch to None when passing data to policy
        self.data = Batch(
            obs={}, act={}, rew={}, done={}, obs_next={}, info={}, policy={}
        )
        self.reset_env()
        self.reset_buffer()
        if not keep_statistics:
            self.reset_stat()

    def reset_stat(self) -> None:
        """Reset the statistic variables."""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """Reset the data buffer."""
        self.buffer.reset(keep_statistics=keep_statistics)

    def reset_env(self) -> None:
        """Reset all of the environments."""
        obs = self.env.reset()
        self.data.obs = obs

    def _reset_state(self, id: Union[int, List[int]]) -> None:
        """Reset the hidden state: self.data.state[id]."""
        if hasattr(self.data.policy, "hidden_state"):
            state = self.data.policy.hidden_state  # it is a reference
            if isinstance(state, torch.Tensor):
                state[id].zero_()
            elif isinstance(state, np.ndarray):
                state[id] = None if state.dtype == object else 0
            elif isinstance(state, Batch):
                state.empty_(id)

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
    ) -> Dict[str, Any]:
        """Collect a specified number of step or episode.

        To ensure unbiased sampling result with n_episode option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        :param int n_step: how many steps you want to collect.
        :param int n_episode: how many episodes you want to collect.
        :param bool random: whether to use random policy for collecting data. Default
            to False.
        :param float render: the sleep time between rendering consecutive frames.
            Default to None (no rendering).
        :param bool no_grad: whether to retain gradient in policy.forward(). Default to
            True (no gradient retaining).

        .. note::

            One and only one collection number specification is permitted, either
            ``n_step`` or ``n_episode``.

        :return: A dict including the following keys

            * ``n/ep`` collected number of episodes.
            * ``n/st`` collected number of steps.
            * ``rews`` array of episode reward over collected episodes.
            * ``lens`` array of episode length over collected episodes.
            * ``idxs`` array of episode start index in buffer over collected episodes.
        """
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        assert n_step is None and n_episode is not None and n_episode > 0, "Please use n_episode > 0"
        start_time = time.time()

        step_count = np.zeros(self.env_num, np.int64)
        # episode_count = np.zeros(self.env_num, np.int64)
        episode_rews = np.zeros((self.env_num, n_episode), np.float64)
        episode_lens = np.zeros((self.env_num, n_episode), np.int64)
        # episode_start_indices = np.zeros((self.env_num, n_episode), np.int64)

        for episode in tqdm(range(n_episode), desc=f"Seed: {self._seed}. Collecting episodes", disable=True):
            self.reset(keep_statistics=True)
            ready_env_ids = np.arange(self.env_num)
            self.data = self.data[:self.env_num]
            self.policy.reset_keys_values_wm()
            # n_episode is set for per env
            
            assert len(self.data) == len(ready_env_ids)
            
            while True:
                # restore the state: if the last state is None, it won't store
                last_state = self.data.policy.pop("hidden_state", None)

                # get the next action
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = result.act
                assert isinstance(act, np.ndarray)
                self.data.update(policy=policy, act=act)
                
                # get bounded and remapped actions first (not saved into buffer)
                action_remap = self.policy.map_action(self.data.act)
                # step in env
                result = self.env.step(action_remap[ready_env_ids], ready_env_ids)
                obs_next, rew, done, info = result

                # Any other better methods?
                inplace_change_value_of_batch(self.data, ready_env_ids, 'obs_next', obs_next)
                inplace_change_value_of_batch(self.data, ready_env_ids, 'rew', rew)
                inplace_change_value_of_batch(self.data, ready_env_ids, 'done', done)
                inplace_change_value_of_batch(self.data, ready_env_ids, 'info', info)

                if render:
                    self.env.render()
                    if render > 0 and not np.isclose(render, 0):
                        time.sleep(render)

                # add data into the buffer
                ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                    self.data[ready_env_ids], buffer_ids=ready_env_ids
                )

                # collect statistics
                step_count += len(ready_env_ids)

                if np.any(done):
                    env_ind_local = np.where(done)[0]
                    env_ind_global = ready_env_ids[env_ind_local]
                    episode_lens[env_ind_global, episode] = ep_len[env_ind_local]
                    episode_rews[env_ind_global, episode] = ep_rew[env_ind_local]
                    # episode_start_indices[env_ind_global, episode] = ep_idx[env_ind_local]
                    self.data.obs_next[env_ind_global] = self.data.obs[env_ind_global]  # fake obs

                    # remove surplus env id from ready_env_ids
                    # to avoid bias in selecting environments
                    if n_episode:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local] = False
                        ready_env_ids = ready_env_ids[mask]
                        # self.data = self.data[mask]
                        
                if np.all(self.data.done):
                    break

                self.data.obs = self.data.obs_next

        # generate statistics
        episode_count = n_episode * self.env_num
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)
        
        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": episode_rews,
            "lens": episode_lens,
            # "idxs": episode_start_indices,
        }
        

def inplace_change_value_of_batch(data, index, key, value):
    ori_data = data.__dict__[key]
    
    if isinstance(ori_data, Batch) and ori_data.is_empty():
        data.__dict__[key] = value
    else:
        data.__dict__[key][index] = value
        
    return data