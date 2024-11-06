from functools import partial
from typing import Tuple

import numpy as np
import torch


def create_empty_matrix(n, num_heads, max_tokens, embed_dim, device):
    return torch.empty(n, num_heads, max_tokens, embed_dim // num_heads, device=device)


class Cache:
    def __init__(
        self, 
        num_samples: int, 
        num_heads: int, 
        max_tokens: int, 
        embed_dim: int, 
        device: torch.device,
    ) -> None:
        assert embed_dim % num_heads == 0
        
        self._n, self._cache, self._size = num_samples, None, None
        self._reset = partial(
            create_empty_matrix, 
            num_heads=num_heads, 
            max_tokens=max_tokens, 
            embed_dim=embed_dim, 
            device=device
        )  # (B, nh, T, hs)
        self.reset()

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        n, num_heads, _, head_dim = self._cache.shape
        
        return n, num_heads, self._size, head_dim

    def reset(self) -> None:
        self._cache = self._reset(self._n)
        self._size = 0

    def prune(self, mask: np.ndarray) -> None:
        assert mask.ndim == 1 and mask.shape[0] == self.shape[0]
        
        self._cache = self._cache[mask]
        self._n = self._cache.shape[0]

    def get(self) -> torch.Tensor:
        return self._cache[:, :, :self._size, :]

    def update(self, x: torch.Tensor) -> None:
        assert (x.ndim == self._cache.ndim) and all(
            [x.size(i) == self._cache.size(i) for i in (0, 1, 3)])
        assert self._size + x.size(2) <= self._cache.shape[2]
        
        self._cache = AssignWithoutInplaceCheck.apply(
            self._cache, 
            x, 
            2, 
            self._size, 
            self._size + x.size(2),
        )
        self._size += x.size(2)


class KVCache:
    def __init__(
        self, 
        n: int, 
        num_heads: int, 
        max_tokens: int, 
        embed_dim: int, 
        device: torch.device
    ) -> None:
        self._k_cache = Cache(n, num_heads, max_tokens, embed_dim, device)
        self._v_cache = Cache(n, num_heads, max_tokens, embed_dim, device)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return self._k_cache.shape

    def reset(self) -> None:
        self._k_cache.reset()
        self._v_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        self._k_cache.prune(mask)
        self._v_cache.prune(mask)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._k_cache.get(), self._v_cache.get()

    def update(self, k: torch.Tensor, v: torch.Tensor):
        self._k_cache.update(k)
        self._v_cache.update(v)


class KeysValues:
    def __init__(
        self, 
        n: int, 
        num_heads: int, 
        max_tokens: int, 
        embed_dim: int, 
        num_layers: int, 
        device: torch.device
    ) -> None:
        self._keys_values = tuple(
            [KVCache(n, num_heads, max_tokens, embed_dim, device) for _ in range(num_layers)]
        )

    def __getitem__(self, key: int) -> KVCache:
        return self._keys_values[key]

    def __len__(self):
        return len(self._keys_values)

    @property
    def size(self):
        return self._keys_values[0].shape[2]

    def reset(self) -> None:
        for kv_cache in self._keys_values:
            kv_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        for kv_cache in self._keys_values:
            kv_cache.prune(mask)


class AssignWithoutInplaceCheck(torch.autograd.Function):
    """
    Inspired from : https://discuss.pytorch.org/t/disable-in-place-correctness-version-check-any-other-workaround/90738/4
    Warning : do not use it to overwrite a slice twice.
    """

    @staticmethod
    def get_slice(dim: int, start: int, stop: int) -> Tuple[slice]:
        return tuple([slice(None), ] * dim + [slice(start, stop)])

    @staticmethod
    def forward(
        ctx, 
        input: torch.Tensor, 
        value: torch.Tensor, 
        dim: int, 
        start: int, 
        stop: int,
    ) -> torch.Tensor:
        ctx.dim = dim
        ctx.start = start
        ctx.stop = stop
        input.data[AssignWithoutInplaceCheck.get_slice(dim, start, stop)] = value
        return input

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor]:
        return grad_out, grad_out[
            AssignWithoutInplaceCheck.get_slice(ctx.dim, ctx.start, ctx.stop)
        ], None, None, None


def concate_kv(kvs, repeat_num: int = 1):
    k_cache_example = kvs[0][0]._k_cache._cache
    n, num_heads, max_tokens, head_dim = k_cache_example.shape
    embed_dim = head_dim * num_heads
    device = k_cache_example.device
    num_layers = len(kvs[0])
    size = kvs[0][0]._k_cache._size

    result_kv = KeysValues(
        n * repeat_num * len(kvs), 
        num_heads, 
        max_tokens, 
        embed_dim, 
        num_layers, 
        device,
    )
    
    for i in range(num_layers):
        all_k_cache = []
        all_v_cache = []
        
        for kv in kvs:
            all_k_cache.append(kv[i]._k_cache._cache.repeat(repeat_num, 1, 1, 1))
            all_v_cache.append(kv[i]._v_cache._cache.repeat(repeat_num, 1, 1, 1))
        
        all_k_cache = torch.cat(all_k_cache, dim=0)
        all_v_cache = torch.cat(all_v_cache, dim=0)
        
        result_kv[i]._k_cache._cache = AssignWithoutInplaceCheck.apply(
            result_kv[i]._k_cache._cache, 
            all_k_cache[:, :, :size, :], 
            2, 0, size,
        )
        result_kv[i]._k_cache._size = size
        
        result_kv[i]._v_cache._cache = AssignWithoutInplaceCheck.apply(
            result_kv[i]._v_cache._cache,
            all_v_cache[:, :, :size, :], 
            2, 0, size,
        )
        result_kv[i]._v_cache._size = size

    return result_kv


def split_kv(kv):
    k_cache_example = kv[0]._k_cache._cache
    n, num_heads, max_tokens, head_dim = k_cache_example.shape
    embed_dim = head_dim * num_heads
    device = k_cache_example.device
    num_layers = len(kv)
    size = kv[0]._k_cache._size

    result_kvs = []

    for i in range(n):
        splited_kv = KeysValues(1, num_heads, max_tokens, embed_dim, num_layers, device)
        
        for j in range(num_layers):
            splited_kv[j]._k_cache._cache = AssignWithoutInplaceCheck.apply(
                splited_kv[j]._k_cache._cache, 
                kv[j]._k_cache._cache[[i], :, :size, :], 
                2, 0, size,
            )
            splited_kv[j]._k_cache._size = size
            
            splited_kv[j]._v_cache._cache = AssignWithoutInplaceCheck.apply(
                splited_kv[j]._v_cache._cache, 
                kv[j]._v_cache._cache[[i], :, :size, :], 
                2, 0, size,
            )
            splited_kv[j]._v_cache._size = size            
        result_kvs.append(splited_kv)
    
    return result_kvs