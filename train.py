import os

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from dotenv import load_dotenv
import wandb

from src.utils import load_cfg, get_exp_name, validate_cfg
from src.data.dataset import TextDataset
from src.data.byte_tokenizer import BaselineTokenizer
from src.aug.augmentations import Augmentations
from src.model.model import ByteModernBertEncoder
from src.loss.full_loss import FullLoss


if __name__ == "__main__":

    load_dotenv()    

    # load master cfg
    cfg = load_cfg("cfg.yml")
    validate_cfg(cfg)
    requested_device = cfg["exp"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    # handle path to save results
    os.makedirs("results", exist_ok=True)
    name_exp = get_exp_name()
    save_ckp_path = f"results/{name_exp}"
    os.makedirs(save_ckp_path)

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

    # LR scheduler: short warmup followed by cosine decay.
    total_steps = max(1, len(dataloader) * cfg["exp"]["epochs"])
    warmup_steps = max(1, int(float(cfg['optim']['warmup_steps']) * total_steps))
    cosine_steps = max(1, total_steps - warmup_steps)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=float(cfg['optim']['start_factor']),
        end_factor=float(cfg['optim']['end_factor']),
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # setup loss function
    loss_fn = FullLoss(
        num_global_views=cfg["loss"]["num_global_views"],
        num_points=cfg["loss"]["num_points"],
        num_slices=cfg["loss"]["num_slices"],
        ld=float(cfg["loss"]["ld"]),
    ).to(device)

    # start train pipe
    step = 0
    for epoch in range(cfg["exp"]["epochs"]):
        for batch in dataloader:
            
            batch = {  # just to move to gpu
                k: v.to(device)  
                for k, v in batch.items()
            }

            # if bf16 is set compute the forward with bf16
            if cfg["exp"]["use_bf16"]:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                    z = encoder(
                        global_input_ids=batch["global_crops"],
                        global_attn_mask=batch["global_masks"],
                        local_input_ids=batch["local_crops"],
                        local_attn_mask=batch["local_masks"],
                    )
            else:  # else high precision
                z = encoder(
                        global_input_ids=batch["global_crops"],
                        global_attn_mask=batch["global_masks"],
                        local_input_ids=batch["local_crops"],
                        local_attn_mask=batch["local_masks"],
                    )

            # loss always in high precision
            loss = loss_fn(z=z)

            optimizer.zero_grad()
            loss["loss"].backward()

            # get grad norms to log
            total_norm_sq = 0.0
            for param in encoder.parameters():
                if param.grad is not None:
                    param_norm = param.grad.detach().data.norm(2)
                    total_norm_sq += param_norm.item() ** 2
            grad_norm = total_norm_sq ** 0.5

            optimizer.step()
            scheduler.step()

            if step % cfg["exp"]["log_every_n_step"] == 0:  # log to w&b
                wandb.log(
                    {
                        "loss": loss["loss"].item(),
                        "pred_loss": loss["pred_loss"].item(),
                        "sigreg_loss": loss["sigreg_loss"].item(),
                        "step": step,
                        "grad_norm": grad_norm,
                        "lr": scheduler.get_last_lr()[0],
                        "step": step,
                    }
                )
                
            if step % cfg["exp"]["save_every_n_step"] == 0:  # save checks
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": encoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "cfg": cfg,
                        "loss": loss["loss"].item(),
                    },
                    os.path.join(save_ckp_path, f"epoch_{epoch}_step_{step}.pt"),
                )
            
            step += 1
