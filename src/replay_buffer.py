import numpy as np
import torch


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, sequence_length, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device
        self.obs_shape = obs_shape

        obs_type = int if len(obs_shape) == 1 else np.uint8
        self.obses = np.empty((capacity, sequence_length, *obs_shape), dtype=obs_type)
        self.actions = np.empty((capacity, sequence_length), dtype=int)
        self.rewards = np.empty((capacity, sequence_length), dtype=np.float32)
        self.dones = np.empty((capacity, sequence_length), dtype=bool)
        self.mask_paddings = np.empty((capacity, sequence_length), dtype=bool)
        self.envs = np.empty((capacity,), dtype=int)
        
        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, done, mask_padding, env):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.dones[self.idx], done)
        np.copyto(self.mask_paddings[self.idx], mask_padding)
        np.copyto(self.envs[self.idx], env)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def add_batch(self, obs, action, reward, done, mask_padding, env):
        batch_size = obs.shape[0]
        next_index = self.idx + batch_size
                
        if next_index >= self.capacity:
            self.full = True
            maximum_index = self.capacity - self.idx
            np.copyto(self.obses[self.idx:self.capacity], obs[:maximum_index])
            np.copyto(self.actions[self.idx:self.capacity], action[:maximum_index])
            np.copyto(self.rewards[self.idx:self.capacity], reward[:maximum_index])
            np.copyto(self.dones[self.idx:self.capacity], done[:maximum_index])
            np.copyto(self.mask_paddings[self.idx:self.capacity], mask_padding[:maximum_index])
            np.copyto(self.envs[self.idx:self.capacity], env[:maximum_index])
            remain = batch_size - maximum_index
            if remain > 0:
                np.copyto(self.obses[0:remain], obs[maximum_index:])
                np.copyto(self.actions[0:remain], action[maximum_index:])
                np.copyto(self.rewards[0:remain], reward[maximum_index:])
                np.copyto(self.dones[0:remain], done[maximum_index:])
                np.copyto(self.mask_paddings[0:remain], mask_padding[maximum_index:])
                np.copyto(self.envs[0:remain], env[maximum_index:])
            self.idx = remain
        
        else:
            np.copyto(self.obses[self.idx:next_index], obs)
            np.copyto(self.actions[self.idx:next_index], action)
            np.copyto(self.rewards[self.idx:next_index], reward)
            np.copyto(self.dones[self.idx:next_index], done)
            np.copyto(self.mask_paddings[self.idx:next_index], mask_padding)
            np.copyto(self.envs[self.idx:next_index], env)
            self.idx = next_index
         
    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obs_type = torch.long if len(self.obs_shape) == 1 else torch.uint8
        obses = torch.as_tensor(self.obses[idxs], device=self.device, dtype=obs_type)
        actions = torch.as_tensor(self.actions[idxs], device=self.device, dtype=torch.long)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(self.dones[idxs], device=self.device, dtype=torch.long)
        mask_paddings = torch.as_tensor(self.mask_paddings[idxs], device=self.device, dtype=torch.bool)
        envs = torch.as_tensor(self.envs[idxs], device=self.device, dtype=torch.long)

        return dict(
            observations=obses,
            actions=actions,
            rewards=rewards,
            ends=dones,
            mask_padding=mask_paddings,
            envs=envs,
        )