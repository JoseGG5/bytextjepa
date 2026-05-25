import torch
from torch import nn


class MlmLoss(nn.Module):
    """Cross-entropy over masked positions only."""

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        vocab_size = logits.size(-1)
        loss = self.criterion(logits.view(-1, vocab_size), labels.view(-1))

        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            masked_positions = labels != -100
            if masked_positions.any():
                accuracy = (predictions[masked_positions] == labels[masked_positions]).float().mean()
            else:
                accuracy = torch.tensor(0.0, device=logits.device)

        return {
            "loss": loss,
            "masked_token_accuracy": accuracy,
        }
