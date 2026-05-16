import torch
from torch import nn
import lejepa

from src.loss.jepa_loss import JepaLoss

""" Full loss that combines jepa loss (view alignment) + SIGReg """

class FullLoss(nn.Module):
    def __init__(
        self,
        num_global_views: int,
        num_points: int,
        num_slices: int,
        ld: float
    ):
        super().__init__()
        self.jepa_loss = JepaLoss(num_global_views=num_global_views)
        self.sigreg_loss = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=lejepa.univariate.EppsPulley(n_points=num_points),
            num_slices=num_slices,
        )
        self.ld = ld

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if z.ndim != 3:
            raise ValueError(f"Expected z to have shape [B, V, D], got {tuple(z.shape)}")

        pred_loss = self.jepa_loss(z)

        # sigreg loss is computed for each view and then meaned
        per_view_sigreg = []
        for view_idx in range(z.size(1)):
            per_view_sigreg.append(self.sigreg_loss(z[:, view_idx, :]))
        sigreg_loss = torch.stack(per_view_sigreg).mean()

        loss = (1.0 - self.ld) * pred_loss + self.ld * sigreg_loss

        return {
            "loss": loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
        }
