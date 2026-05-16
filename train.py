import os

import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import wandb

from src.utils import load_cfg, init_encoder
from src.data.dataset import TextDataset
from src.data.byte_tokenizer import BaselineTokenizer
from src.aug.augmentations import Augmentations
from src.model.model import ByteModernBertEncoder

# TODO: Add W&B logging
# TODO: Create checkpoint saving method
# TODO: Implement train pipeline
# TODO: Decide optimizer
# TODO: Run pipeline on just one example to verify everything is fine (loss should go down to 0)
# TODO: Add cuda support

if __name__ == "__main__":

    load_dotenv()

    # load master cfg
    cfg = load_cfg("cfg.yml")

    # setup w&b
    run = wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        config=cfg,
    )

    # setup tokenizer
    if cfg["tokenizer"]["type"] == "baseline":
        tokenizer = BaselineTokenizer(cfg=cfg)
    else:
        raise ValueError(f"{cfg['tokenizer']['type']} is currently not implemented")

    # setup data modules
    augmenter = Augmentations(cfg=cfg)
    text_dataset = TextDataset(cfg=cfg, tokenizer=tokenizer, augmenter=augmenter, dev=True)
    dataloader = DataLoader(dataset=text_dataset, batch_size=2, shuffle=True)
    
    # setup encoder
    encoder = ByteModernBertEncoder(cfg=cfg).to(cfg["exp"]["device"])

    

