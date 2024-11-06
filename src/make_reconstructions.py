import sys

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm


@torch.no_grad()
def make_reconstructions_from_batch(
    batch, 
    save_dir, 
    epoch, 
    tokenizer, 
):
    original_frames = tensor_to_np_frames(
        rearrange(batch['observations'], 'b t c h w -> b t h w c')
    )
    rec_frames = generate_reconstructions_with_tokenizer(
        batch, 
        tokenizer, 
    )

    res = np.concatenate((original_frames, rec_frames), axis=-2)
    res = rearrange(res, 'b t h w c -> (b t) h w c')
    res = np.squeeze(res, axis=-1)  # due to gray scale
    for i, image in enumerate(res):
        img = Image.fromarray(image)
        img.save(save_dir / f'epoch_{epoch:03d}_t_{i:03d}.png')


def insert_separators(imgs, separator_width=1, separator_color=255):
    b, t, h, w, c = imgs.shape
    separator = np.full((h, separator_width, c), separator_color, dtype=imgs.dtype)
    imgs_with_separators = []
    for batch in imgs:
        batch_with_separators = [batch[0]]
        for img in batch[1:]:
            batch_with_separators.append(separator)
            batch_with_separators.append(img)
        imgs_with_separators.append(np.concatenate(batch_with_separators, axis=1))
    return np.stack(imgs_with_separators)


@torch.no_grad()
def make_reconstructions_of_trajectories(
    batch, 
    save_dir, 
    epoch, 
    tokenizer, 
    jowa, 
    separator_width=2,
):
    b, t = batch['observations'].shape[:2]
    original_frames = tensor_to_np_frames(
        rearrange(batch['observations'], 'b t c h w -> b t h w c')
    )
    original_concate_frames = insert_separators(
        original_frames, 
        separator_width=separator_width, 
        separator_color=255,
    )
    tokenizer_rec_frames = generate_reconstructions_with_tokenizer(
        batch, 
        tokenizer, 
    )
    tokenizer_concate_rec_frames = insert_separators(
        tokenizer_rec_frames, 
        separator_width=separator_width, 
        separator_color=255,
    )
    
    # teacher-forcing regression
    temp = tokenizer.encode(batch['observations'], should_preprocess=True)
    h, w = temp.z_quantized.shape[-2:]
    obs_tokens = temp.tokens  # (B, L, K)
    act_tokens = rearrange(batch['actions'], 'b l -> b l 1')
    tokens = rearrange(
        torch.cat((obs_tokens, act_tokens), dim=2), 
        'b l k1 -> b (l k1)',
    )  # (B, L(K+1))
    outputs = jowa(tokens, batch['envs'])
    
    def get_concate_rec_frames_from_outputs(outputs):
        logits_observations = outputs.logits_observations[:, :-1]
        tokens_observations = logits_observations.argmax(dim=-1)
        tokens_observations = torch.cat(
            (obs_tokens[:, 0, 0].unsqueeze(1), tokens_observations), 
            dim=1,
        )  # (B, t*h*w)
        embeddings_observations = tokenizer.embedding(tokens_observations.flatten())  # (B*t*h*w, E)
        e = embeddings_observations.size(1)
        z_q = rearrange(
            embeddings_observations, 
            '(b t h w) e -> (b t) e h w', 
            b=b, t=t, e=e, h=h, w=w,
        ).contiguous()
        output_frames = torch.clamp(
            tokenizer.decode(z_q, should_postprocess=True), 0, 1,
        )
        output_frames = rearrange(
            output_frames, 
            '(b t) c h w -> b t h w c', 
            b=b, t=t,
        )
        rec_frames = tensor_to_np_frames(output_frames)
        
        return insert_separators(
            rec_frames, 
            separator_width=separator_width, 
            separator_color=255,
        )
    
    teacher_forcing_regression_concate_rec_frames = get_concate_rec_frames_from_outputs(
        outputs
    )
    
    # auto-regression
    num_given_blocks = 4
    given_blocks_tokens = tokens[:, :num_given_blocks*(h * w + 1)]
    for step in tqdm(
        range(num_given_blocks * (h * w + 1), t * (h * w + 1)), 
        disable=True, 
        desc='auto-regression', 
        file=sys.stdout,
    ):
        if (step+1) % (h*w+1) == 0:
            given_blocks_tokens = torch.cat(
                (given_blocks_tokens, act_tokens[:, step // (h*w+1)]), 
                dim=1,
            )
        else:
            outputs = jowa(given_blocks_tokens, batch['envs'])
            logits_observations = outputs.logits_observations[:, -1]
            tokens_observations = logits_observations.argmax(dim=-1)
            given_blocks_tokens = torch.cat(
                (given_blocks_tokens, tokens_observations.unsqueeze(1)), 
                dim=1,
            )

    outputs = jowa(given_blocks_tokens, batch['envs'])
    auto_regression_concate_rec_frames = get_concate_rec_frames_from_outputs(outputs)
    
    # save
    separator = np.full(
        (b, separator_width, *original_concate_frames.shape[2:]), 
        255, 
        dtype=original_concate_frames.dtype,
    )
    res = np.concatenate(
        (
            original_concate_frames, separator, 
            tokenizer_concate_rec_frames, separator, 
            teacher_forcing_regression_concate_rec_frames, separator,
            auto_regression_concate_rec_frames,
        ), 
        axis=1,
    )
    res = np.squeeze(res, axis=-1)  # due to gray scale
    
    for i, image in enumerate(res):
        img = Image.fromarray(image)
        img.save(save_dir / f'{str(jowa)}_epoch_{epoch:03d}_t_{i:03d}.png')


def tensor_to_np_frames(inputs):
    check_float_btw_0_1(inputs)
    return inputs.to(float).mul(255).cpu().numpy().astype(np.uint8)


def check_float_btw_0_1(inputs):
    assert inputs.is_floating_point() and (inputs >= 0).all() and (inputs <= 1).all()


@torch.no_grad()
def generate_reconstructions_with_tokenizer(batch, tokenizer):
    inputs = rearrange(batch['observations'], 'b t c h w -> (b t) c h w')
    outputs = reconstruct_through_tokenizer(inputs, tokenizer)
    b, t, _, _, _ = batch['observations'].size()
    outputs = rearrange(outputs, '(b t) c h w -> b t h w c', b=b, t=t)
    rec_frames = tensor_to_np_frames(outputs)
    return rec_frames


@torch.no_grad()
def reconstruct_through_tokenizer(inputs, tokenizer):
    check_float_btw_0_1(inputs)
    reconstructions = tokenizer.encode_decode(
        inputs, 
        should_preprocess=True, 
        should_postprocess=True,
    )
    return torch.clamp(reconstructions, 0, 1)
