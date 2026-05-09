import os

import torch
import lejepa

from src.utils import load_cfg, init_encoder
from src.data.dataset import TextDataset
from src.data.byte_tokenizer import *

if __name__ == "__main__":
    cfg = load_cfg("cfg.yml")
    encoder = init_encoder(cfg=cfg)

    if cfg["tokenizer"]["type"] == "baseline":
        tokenizer = BaselineTokenizer(cfg=cfg)
    else:
        raise ValueError(f"{cfg['tokenizer']['type']} is currently not implemented")

    text_dataset = TextDataset(cfg=cfg, tokenizer=tokenizer)

    print(text_dataset[0])

