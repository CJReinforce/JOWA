"""
Credits to https://github.com/CompVis/taming-transformers
"""

import time
from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from utils import (
    Batch, 
    LossWithIntermediateLosses, 
    init_tokenizer_weights, 
    ShiftAug,
)

from .lpips import LPIPS
from .nets import Decoder, Encoder


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor


class Tokenizer(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int, 
        encoder: Encoder, 
        decoder: Decoder, 
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.aug = ShiftAug()

        # network
        self.encoder = encoder
        self.pre_quant_conv = torch.nn.Conv2d(
            encoder.config.z_channels, 
            embed_dim, 
            1,
        )
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.post_quant_conv = torch.nn.Conv2d(
            embed_dim, 
            decoder.config.z_channels, 
            1,
        )
        self.decoder = decoder
        
        # init
        self.encoder.apply(init_tokenizer_weights)
        self.decoder.apply(init_tokenizer_weights)
        self.embedding.weight.data.uniform_(
            -1.0 / vocab_size,
            1.0 / vocab_size,
        )
        self.lpips = LPIPS().eval()

    def __repr__(self) -> str:
        return "tokenizer"

    def forward(
        self, 
        x: torch.Tensor, 
        should_preprocess: bool = False, 
        should_postprocess: bool = False, 
        freeze_encoder: bool = False,
    ) -> Tuple[torch.Tensor]:
        if not freeze_encoder:
            outputs = self.encode(x, should_preprocess)
            decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        else:
            decoder_input = self.encode(x, should_preprocess).z_quantized
        
        reconstructions = self.decode(decoder_input, should_postprocess)
        
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(
        self, 
        batch: Batch, 
        do_aug: bool = False, 
        freeze_encoder: bool = False, 
        **kwargs: Any,
    ) -> LossWithIntermediateLosses:
        origin_obs = rearrange(
            batch['observations'], 
            'b t c h w -> (b t) c h w',
        )
        origin_obs = self.aug(origin_obs) if do_aug else origin_obs

        observations = self.preprocess_input(origin_obs)
        z, z_quantized, reconstructions = self(
            observations, 
            should_preprocess=False, 
            should_postprocess=False, 
            freeze_encoder=freeze_encoder,
        )

        reconstruction_loss = torch.abs(observations - reconstructions).mean()
        perceptual_loss = torch.mean(self.lpips(observations, reconstructions))
        
        if not freeze_encoder:
            # Codebook loss. Notes:
            # - beta position is different from taming and identical to original VQVAE paper
            # - VQVAE uses 0.25 by default
            beta = 1.0
            commitment_loss = (z.detach() - z_quantized).pow(2).mean() + \
                beta * (z - z_quantized.detach()).pow(2).mean()
            
            return LossWithIntermediateLosses(
                commitment_loss=commitment_loss, 
                reconstruction_loss=reconstruction_loss, 
                perceptual_loss=perceptual_loss,
            )
        else:
            return LossWithIntermediateLosses(
                reconstruction_loss=reconstruction_loss, 
                perceptual_loss=perceptual_loss,
            )

    def encode(
        self, 
        x: torch.Tensor, 
        should_preprocess: bool = False,
    ) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)

        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)
        z = self.pre_quant_conv(z)
        b, e, h, w = z.shape
        z_flattened = rearrange(z, 'b e h w -> (b h w) e')
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
                2 * torch.matmul(z_flattened, self.embedding.weight.t())

        tokens = dist_to_embeddings.argmin(dim=-1)
            
        z_q = rearrange(
            self.embedding(tokens), 
            '(b h w) e -> b e h w', 
            b=b, e=e, h=h, w=w,
        ).contiguous()

        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        z_q = self.post_quant_conv(z_q)
        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        
        return rec

    @torch.no_grad()
    def encode_decode(
        self, 
        x: torch.Tensor, 
        should_preprocess: bool = False, 
        should_postprocess: bool = False,
    ) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)