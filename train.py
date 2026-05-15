import os
from pprint import pprint

import torch
import lejepa
from torch.utils.data import DataLoader

from src.utils import load_cfg, init_encoder
from src.data.dataset import TextDataset
from src.data.byte_tokenizer import BaselineTokenizer
from src.aug.augmentations import Augmentations
from src.model.model import ByteModernBertEncoder


# TODO: think about where tokenization is handled (create padding utils than can be reusable and padding logic is not attached to tokenizers)
# this way: inside the tokenizer we call padding utils, after that we perform augmentations and then we apply padding logic to each view

if __name__ == "__main__":
    cfg = load_cfg("cfg.yml")
    encoder = init_encoder(cfg=cfg)

    if cfg["tokenizer"]["type"] == "baseline":
        tokenizer = BaselineTokenizer(cfg=cfg)
    else:
        raise ValueError(f"{cfg['tokenizer']['type']} is currently not implemented")

    augmenter = Augmentations(cfg=cfg)

    text_dataset = TextDataset(cfg=cfg, tokenizer=tokenizer, augmenter=augmenter)
    dataloader = DataLoader(dataset=text_dataset, batch_size=2, shuffle=True)
    first_batch = next(iter(dataloader))

    encoder = ByteModernBertEncoder(cfg=cfg)

    with torch.no_grad():
        global_input_ids=first_batch["global_crops"]
        global_attn_mask=first_batch["global_masks"]
        local_input_ids=first_batch["local_crops"]
        local_attn_mask=first_batch["local_masks"]

        z = encoder(
             global_input_ids=global_input_ids,
             global_attn_mask=global_attn_mask,
             local_input_ids=local_input_ids,
             local_attn_mask=local_attn_mask,
             )
        

