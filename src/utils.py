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


def init_encoder(cfg: dict) -> ModernBertModel:
    """ Initializes a random ModernBERT model with the
    specified config"""
    cfg = ModernBertConfig(**cfg["model"])
    return ModernBertModel(cfg)


def load_hf_dataset(cfg: dict) -> datasets.arrow_dataset.Dataset:
    """ Loads a dataset from HF datasets """
    dataset = load_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["version"],
        split=cfg["dataset"]["split"]
    )
    return dataset


def pad_tokens(
        x: torch.tensor,
        output_attn_mask: bool,
        max_length: int,
        pad_token_id: int
        ) -> Union[torch.tensor, tuple[torch.tensor, torch.tensor]]:
    """ Pads a sequence of tokens with zeros.
        
        Currently is very specific to the part of padding crops and it is not used in the part of padding in tokenization.
        It could be done via allowing the function to receive a list of str"""

    # set attn_mask as a tensor of ones because this will recieve raw tokens after the crops (no padding has been done)
    attn_mask = torch.ones(size=x.size())

    if x.size(-1) > max_length:
        x = x[..., :max_length]

    elif x.size(-1) < max_length:
        n_tokens_pad = max_length - x.size(-1)
        x = F.pad(
            input=x,
            pad=(0, n_tokens_pad),
            value=pad_token_id
        )
        attn_mask = torch.ones(size=x.size())
        attn_mask[max_length:] = 0

    if output_attn_mask:
        return x, attn_mask

    return x