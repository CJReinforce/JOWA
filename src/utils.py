import functools
import os
import random
from typing import Dict, Optional

import cv2
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import combine_state_for_ensemble

Batch = Dict[str, torch.Tensor]

def configure_optimizer(model, learning_rate, weight_decay, critic_lr, *blacklist_module_names):
    """Credits to https://github.com/karpathy/minGPT"""
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    head_q = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if any([fpn.startswith(module_name) for module_name in blacklist_module_names]):
                no_decay.add(fpn)
            elif 'bias' in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif '.head_q' in fpn:
                head_q.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = (decay | head_q) & no_decay
    union_params = decay | head_q | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))], 
            "weight_decay": weight_decay, "lr": learning_rate
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))], 
            "weight_decay": 0.0, "lr": learning_rate
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(head_q))], 
            "weight_decay": weight_decay, "lr": critic_lr
        },
    ]
    optimizer = torch.optim.AdamW(optim_groups)
    return optimizer


def init_transformer_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def init_tokenizer_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def get_dtype(dtype: str):
    if dtype == 'float16':
        return torch.float16
    elif dtype == 'bfloat16':
        return torch.bfloat16
    else:
        return torch.float32


# to get hydra config
def hydra_main(*args, **kw):
    main = hydra.main(*args, **kw)
    def main_decorator(f):
        returned_values = []
        @functools.wraps(f)
        def f_wrapper(*args, **kw):
            ret = f(*args, **kw)
            returned_values.append(ret)
            return ret
        wrapped = main(f_wrapper)
        @functools.wraps(wrapped)
        def main_wrapper(*args, **kw):
            wrapped(*args, **kw)
            return returned_values[0] if len(returned_values) == 1 else returned_values
        return main_wrapper
    return main_decorator


def capitalize_game_name(game):
    if game[0].islower():
        game = game.replace('-', '_')
        return ''.join([g.capitalize() for g in game.split('_')])
    else:
        return game


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def get_random_state():
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    return cpu_rng_state, cuda_rng_state, numpy_state, python_state


def load_random_state(cpu_rng_state, cuda_rng_state, numpy_state, python_state):
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
    np.random.set_state(numpy_state)
    random.setstate(python_state)


def compute_lambda_returns(rewards, values, ends, gamma, lambda_):
    assert rewards.ndim == 2 or (rewards.ndim == 3 and rewards.size(2) == 1)
    assert rewards.shape == ends.shape == values.shape, f"{rewards.shape}, {values.shape}, {ends.shape}"  # (B, T, 1)
    t = rewards.size(1)
    lambda_returns = torch.empty_like(values)
    lambda_returns[:, -1] = values[:, -1]
    lambda_returns[:, :-1] = rewards[:, :-1] + ends[:, :-1].logical_not() * gamma * (1 - lambda_) * values[:, 1:]

    last = values[:, -1]
    for i in list(range(t - 1))[::-1]:
        lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, i]

    return lambda_returns


class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


def make_video(fname, fps, frames):
    assert frames.ndim == 3 or (frames.ndim == 4 and frames.shape[3] == 3)
    
    t, h, w = frames.shape[:3]
    
    video = cv2.VideoWriter(str(fname), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    for frame in frames:
        if frames.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = frame[:, :, ::-1]
        video.write(frame)
    
    video.release()


class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		modules = nn.ModuleList(modules)
		fn, params, _ = combine_state_for_ensemble(modules)
		self.vmap = torch.vmap(fn, in_dims=(0, 0, None), randomness='different', **kwargs)
		self.params = nn.ParameterList([nn.Parameter(p) for p in params])
		self._repr = str(modules)

	def forward(self, *args, **kwargs):
		return self.vmap([p for p in self.params], (), *args, **kwargs)

	def __repr__(self):
		return 'Vectorized ' + self._repr


class ZeroEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight.requires_grad = False
        self.weight.zero_()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            x.size(0), 
            x.size(1), 
            self.embedding_dim, 
            device=x.device, 
            dtype=self.weight.dtype
        )
    
    
class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=nn.Mish(inplace=True), **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))
	
	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	mlp.append(
        NormedLinear(
            dims[-2], dims[-1], act=act
        ) if act else nn.Linear(dims[-2], dims[-1])
    )
	return nn.Sequential(*mlp)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""
	
	def __init__(self, simnorm_dim):
		super().__init__()
		self.dim = simnorm_dim
	
	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)
		
	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


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


class ShiftAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        origin_dtype = x.dtype
        x = x.to(torch.float32)
        n, _, h, w = x.size()
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        result = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        result = result.to(origin_dtype)
        return result