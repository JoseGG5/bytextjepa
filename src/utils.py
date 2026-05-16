from typing import Union

import yaml
from transformers import ModernBertConfig, ModernBertModel
import datasets
from datasets import load_dataset
import torch
import torch.nn.functional as F


def load_cfg(path: str) -> dict:
    """Loads the config"""
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def load_hf_dataset(cfg: dict) -> datasets.arrow_dataset.Dataset:
    """ Loads a dataset from HF datasets """
    dataset = load_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["version"],
        split=cfg["dataset"]["split"]
    )
    return dataset


# TODO: This works for 1D tokens, but code is confusing as it is in some parts generic and expecting 2D.
# Maybe should refactor a bit but it works so its fine
def pad_tokens(
        x: torch.tensor,
        output_attn_mask: bool,
        max_length: int,
        pad_token_id: int
        ) -> Union[torch.tensor, tuple[torch.tensor, torch.tensor]]:
    """ Pads a sequence of tokens with zeros.
        
        Currently is very specific to the part of padding crops and it is not used in the part of padding in tokenization.
        It could be done via allowing the function to receive a list of str"""

    device = "cuda" if torch.cuda.is_available else "cpu"

    # set attn_mask as a tensor of ones because this will recieve raw tokens after the crops (no padding has been done)
    attn_mask = torch.ones(size=x.size(), device=device)

    if x.size(-1) > max_length:
        x = x[..., :max_length]

    elif x.size(-1) < max_length:
        n_tokens_pad = max_length - x.size(-1)
        
        previous_size = x.size(-1)
        x = F.pad(
            input=x,
            pad=(0, n_tokens_pad),
            value=pad_token_id
        )
        attn_mask = torch.ones(size=x.size())
        attn_mask[previous_size:] = 0

    if output_attn_mask:
        return x, attn_mask

    return x


def mean_pooling(x: torch.Tensor, attn_mask: torch.Tensor):

    # x -> (B, T, D)
    # attn_mask -> (B, T)

    # (B, T) -> (B, T, 1)
    input_mask_expanded = attn_mask.unsqueeze(-1)

    # (B, T, 1) -> (B, T, D)
    input_mask_expanded = input_mask_expanded.expand(
        x.size()
    ).float()

    # Set padding embeddings to 0
    # (B, T, D)
    masked_embeddings = (
        x * input_mask_expanded
    )

    # Sum over token dimension
    # (B, T, D) -> (B, D)
    sum_embeddings = torch.sum(
        masked_embeddings,
        dim=1
    )

    # Count valid tokens
    # (B, T, D) -> (B, D)
    sum_mask = input_mask_expanded.sum(dim=1)

    # Avoid division by zero
    sum_mask = torch.clamp(sum_mask, min=1e-9)

    # Compute mean pooled embeddings
    # (B, D)
    mean_embeddings = (
        sum_embeddings / sum_mask
    )

    return mean_embeddings




