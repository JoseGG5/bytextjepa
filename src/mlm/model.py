import torch
from torch import nn
from transformers import ModernBertConfig, ModernBertModel


class ByteModernBertMlm(nn.Module):
    """Byte-level MLM wrapper around ModernBERT."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.encoder = ModernBertModel(ModernBertConfig(**cfg["model"]))
        self.lm_head = nn.Linear(cfg["model"]["hidden_size"], cfg["model"]["vocab_size"])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        return self.lm_head(hidden)
