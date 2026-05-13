
import torch
from torch import nn

from transformers import ModernBertConfig, ModernBertModel

from src.utils import load_cfg

class ByteModernBertEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        self.encoder = ModernBertModel(
            ModernBertConfig(**cfg["model"])
        )


    def forward(self, x: torch.Tensor):
        
        pass
        
        

if __name__ == "__main__":
    
    cfg = load_cfg("cfg.yml")
    encoder = ByteModernBertEncoder(cfg=cfg)

