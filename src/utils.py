import os
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


def validate_cfg(cfg: dict) -> None:
    """Validates the length-related configuration for crop construction."""
    max_position_embeddings = cfg["model"]["max_position_embeddings"]
    global_max_length = cfg["aug"]["global_max_length"]
    local_max_length = cfg["aug"]["local_max_length"]
    record_max_bytes = cfg["data"]["record_max_bytes"]
    mlm_max_length = cfg.get("mlm", {}).get("max_length")
    max_steps = cfg.get("exp", {}).get("max_steps")

    if max_position_embeddings <= 0:
        raise ValueError("model.max_position_embeddings must be greater than 0")
    if global_max_length <= 0:
        raise ValueError("aug.global_max_length must be greater than 0")
    if local_max_length <= 0:
        raise ValueError("aug.local_max_length must be greater than 0")
    if global_max_length > max_position_embeddings:
        raise ValueError("aug.global_max_length cannot be greater than model.max_position_embeddings")
    if local_max_length > max_position_embeddings:
        raise ValueError("aug.local_max_length cannot be greater than model.max_position_embeddings")
    if local_max_length > global_max_length:
        raise ValueError("aug.local_max_length cannot be greater than aug.global_max_length")
    if record_max_bytes is not None and record_max_bytes <= 0:
        raise ValueError("data.record_max_bytes must be greater than 0 or null")
    if mlm_max_length is not None:
        if mlm_max_length <= 0:
            raise ValueError("mlm.max_length must be greater than 0")
        if mlm_max_length > max_position_embeddings:
            raise ValueError("mlm.max_length cannot be greater than model.max_position_embeddings")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("exp.max_steps must be greater than 0 or null")


def load_hf_dataset(cfg: dict) -> datasets.arrow_dataset.Dataset:
    """ Loads a dataset from HF datasets """
    dataset = load_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["version"],
        split=cfg["dataset"]["split"]
    )
    return dataset


def load_mixture_dataset(cfg: dict) -> datasets.arrow_dataset.Dataset:
    """Loads and concatenates several sources into one Dataset with a 'text' column."""
    pieces = []
    for source in cfg["dataset"]["mixture"]:
        if source["name"] == "bookcorpus_local":
            # Dataset.from_pandas would dill-pickle the whole in-memory table just to
            # compute a fingerprint, which OOMs on a ~5GB dataframe. from_csv streams
            # through Arrow's CSV reader instead.
            ds = datasets.Dataset.from_csv("data/BookCorpus3.csv")
        else:
            ds = load_dataset(
                source["name"],
                source.get("version"),
                split=cfg["dataset"]["split"],
                revision=source.get("revision"),
            )

        text_col = "text" if "text" in ds.column_names else ds.column_names[0]
        ds = ds.select_columns([text_col])
        if text_col != "text":
            ds = ds.rename_column(text_col, "text")
        ds = ds.filter(lambda x: x["text"] is not None and len(x["text"].strip()) >= 64)
        n = min(source["target_rows"], len(ds))
        pieces.append(ds.shuffle(seed=13).select(range(n)))

    combined = datasets.concatenate_datasets(pieces)
    return combined.shuffle(seed=13)


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

    device = x.device

    # set attn_mask as a tensor of ones because this will recieve raw tokens after the crops (no padding has been done)
    attn_mask = torch.ones(size=x.size(), device=device, dtype=torch.long)

    if x.size(-1) > max_length:
        x = x[..., :max_length]
        attn_mask = attn_mask[..., :max_length]

    elif x.size(-1) < max_length:
        n_tokens_pad = max_length - x.size(-1)
        
        previous_size = x.size(-1)
        x = F.pad(
            input=x,
            pad=(0, n_tokens_pad),
            value=pad_token_id
        )
        attn_mask = torch.ones(size=x.size(), device=device, dtype=torch.long)
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


def pool_encoder_output(
    encoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run an encoder and pool its token states with the correct attention mask.

    Some backbones, such as the CNN byte encoder, shrink the sequence length before
    the transformer. In that case we must reduce the attention mask before pooling.
    """

    hidden = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state

    if hasattr(encoder, "_reduce_attention_mask"):
        pooled_mask = encoder._reduce_attention_mask(attention_mask)
    else:
        pooled_mask = attention_mask

    return mean_pooling(hidden, pooled_mask)


def get_exp_name():
    """ Useful function to ensure a convention in experiment naming """
    all_exps = os.listdir("results")

    if len(all_exps) == 0:  # handle first exp
        return "exp0"

    numbers = []
    for exp in all_exps:
        numbers.append(int(exp[3:]))  # all exps will be named like expX

    return f"exp{max(numbers)+1}"

