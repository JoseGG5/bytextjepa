

import torch
from torch.utils.data import DataLoader

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
        

