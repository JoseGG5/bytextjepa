
import torch
from torch import nn

from transformers import ModernBertConfig, ModernBertModel

from src.utils import load_cfg, mean_pooling

class ByteModernBertEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        self.encoder = ModernBertModel(
            ModernBertConfig(**cfg["model"])
        )


    def forward(
            self,
            global_input_ids: torch.Tensor,
            global_attn_mask: torch.Tensor,
            local_input_ids: torch.Tensor,
            local_attn_mask: torch.Tensor
            ):

        B, Vg, Tg = global_input_ids.shape
        _, Vl, Tl = local_input_ids.shape

        flat_global_ids = global_input_ids.view(B * Vg, Tg)
        flat_global_mask = global_attn_mask.view(B * Vg, Tg)
        flat_local_ids = local_input_ids.view(B * Vl, Tl)
        flat_local_mask = local_attn_mask.view(B * Vl, Tl)

        global_hidden = self.encoder(
            input_ids=flat_global_ids,
            attention_mask=flat_global_mask,
        ).last_hidden_state  # [B * Vg, Tg, D]

        local_hidden = self.encoder(
            input_ids=flat_local_ids,
            attention_mask=flat_local_mask,
        ).last_hidden_state  # [B * Vl, Tl, D]

        _, _, D = global_hidden.shape

        z_global = mean_pooling(global_hidden, flat_global_mask)  # [B * Vg, D]
        z_local = mean_pooling(local_hidden, flat_local_mask)     # [B * Vl, D]

        z_global = z_global.view(B, Vg, D)                       # [B, Vg, D]
        z_local = z_local.view(B, Vl, D)                         # [B, Vl, D]
        z = torch.cat([z_global, z_local], dim=1)                # [B, V, D]

        return z
        
        

if __name__ == "__main__":
    
    cfg = load_cfg("cfg.yml")
    encoder = ByteModernBertEncoder(cfg=cfg)
    

