import os

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dotenv import load_dotenv
import wandb

from src.utils import load_cfg, get_exp_name
from src.data.dataset import TextDataset
from src.data.byte_tokenizer import BaselineTokenizer
from src.aug.augmentations import Augmentations
from src.model.model import ByteModernBertEncoder
from src.loss.full_loss import FullLoss

# TODO: Add W&B logging
# TODO: Implement train pipeline
# TODO: Run pipeline on just one example to verify everything is fine (loss should go down to 0)

if __name__ == "__main__":

    load_dotenv()

    # handle path to save results
    os.makedirs("results", exist_ok=True)
    name_exp = get_exp_name()
    save_ckp_path = f"results/{name_exp}"
    os.makedirs(save_ckp_path)    

    # load master cfg
    cfg = load_cfg("cfg.yml")
    requested_device = cfg["exp"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    # setup w&b
    run = wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        config=cfg,
        name=name_exp
    )

    # setup tokenizer
    if cfg["tokenizer"]["type"] == "baseline":
        tokenizer = BaselineTokenizer(cfg=cfg)
    else:
        raise ValueError(f"{cfg['tokenizer']['type']} is currently not implemented")

    # setup data modules
    augmenter = Augmentations(cfg=cfg)
    text_dataset = TextDataset(
        cfg=cfg,
        tokenizer=tokenizer,
        augmenter=augmenter
        )
    dataloader = DataLoader(
        dataset=text_dataset,
        batch_size=cfg["tokenizer"]["batch_size"],
        shuffle=True
        )
    
    # setup encoder
    encoder = ByteModernBertEncoder(cfg=cfg).to(device)
    encoder.train()

    # setup the optim

    """layernorms and biases should not have weight decay because biases 
    are really small (do not contribute to overfitting) and pushing norm layers
    params to 0 lowers representational capabilities """
    decay_params = []
    no_decay_params = []

    for name, param in encoder.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim < 2 or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if cfg["exp"]["optim"] == "adamw":
        optimizer = AdamW(
            [
                {
                    "params": decay_params,
                    "weight_decay": float(cfg["exp"]["wd"]),
                },
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                },
            ],
            lr=float(cfg["exp"]["lr"]),
            betas=(
                float(cfg["exp"]["beta0"]),
                float(cfg["exp"]["beta1"])
                ),
            eps=float(cfg["exp"]["eps"]),
        )
    else:
        raise ValueError(f"Optim {cfg['exp']['optim']} not implemented in codebase")

    # setup loss function
    loss_fn = FullLoss(
        num_global_views=cfg["loss"]["num_global_views"],
        num_points=cfg["loss"]["num_points"],
        num_slices=cfg["loss"]["num_slices"],
        ld=float(cfg["loss"]["ld"]),
    ).to(device)

    # start train pipe
    for epoch in range(cfg["exp"]["epochs"]):
        for step, batch in enumerate(dataloader):
            
            batch = {  # just to move to gpu
                k: v.to(device)  
                for k, v in batch.items()
            }

            z = encoder(
                global_input_ids=batch["global_crops"],
                global_attn_mask=batch["global_masks"],
                local_input_ids=batch["local_crops"],
                local_attn_mask=batch["local_masks"],
            )

            loss = loss_fn(z=z)

            optimizer.zero_grad()
            loss["loss"].backward()
            optimizer.step()

            if step % 10 == 0:  # log to w&b
                wandb.log({
                    "loss": loss["loss"].item(),
                    "pred_loss": loss["pred_loss"].item(),
                    "sigreg_loss": loss["sigreg_loss"].item(),
                    "epoch": epoch,
                    "step": step,
                })
            
            if step % 50 == 0:  # save checks
                torch.save(
                    {
                        "epoch": epoch,
                        "step": step,
                        "model_state_dict": encoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "cfg": cfg,
                        "loss": loss["loss"].item(),
                    },
                    os.path.join(save_ckp_path, f"epoch_{epoch}_step_{step}.pt"),
                )
