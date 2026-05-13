import os
from pprint import pprint

import torch
import lejepa

from src.utils import load_cfg, init_encoder
from src.data.dataset import TextDataset
from src.data.byte_tokenizer import BaselineTokenizer
from src.aug.augmentations import Augmentations


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

    pprint(text_dataset[0])

