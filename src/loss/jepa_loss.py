import torch
from torch import nn

""" This loss is just computing the mean (B, D) of the first global views
    and then computes the squared distance from mu to every other view """

class JepaLoss(nn.Module):
    def __init__(self, num_global_views: int):
        super().__init__()
        self.num_global_views = num_global_views

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, V, D]
        if z.ndim != 3:
            raise ValueError(f"Expected z to have shape [B, V, D], got {tuple(z.shape)}")
        if self.num_global_views > z.size(1):
            raise ValueError(
                f"num_global_views={self.num_global_views} cannot be greater than the "
                f"number of views V={z.size(1)}"
            )

        global_z = z[:, :self.num_global_views, :]   # [B, Vg, D]
        mu = global_z.mean(dim=1)                    # [B, D]
        loss = ((mu[:, None, :] - z) ** 2).mean()
        return loss




