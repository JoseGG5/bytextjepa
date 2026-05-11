
import torch
from torch import nn


""" Global views: should crop between 70% and 100% of the original text
    Local views: should crop 10% and 35% of the original text """

class Augmentations(nn.Module):
    """ This class handles how different views of the same
        text are created for LeJEPA"""
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        
    
    def _get_idxs(self, length_real_patches: int, mode: str) -> tuple[float, int, int]:
        """ Takes the real length of the sentence and computes the indexes to
            crop the text for a global view """
        
        # get the pct of the global crop
        if mode == "global":
            pct = torch.randint(
                self.cfg["aug"]["global_margin_min"],
                self.cfg["aug"]["global_margin_max"],
                (1,)
                ).item() * 0.01
        elif mode == "local":
            pct = torch.randint(
                self.cfg["aug"]["local_margin_min"],
                self.cfg["aug"]["local_margin_max"],
                (1,)
                ).item() * 0.01
        else:
            raise ValueError(f"Mode {mode} not implemented. Choose between global or local")

        # max min index we can select to respect the percentage
        max_idx_down = int(length_real_patches - (pct * length_real_patches))
        
        # we can select from zero to max_idx_down our first idx
        idx_down = torch.randint(0, max_idx_down, (1,)).item()
        
        # now select idx_up
        idx_up = idx_down + int(pct * length_real_patches)
        
        return pct, idx_down, idx_up


    # TODO: Note that all global crops should have the same size and all local crops also.
    # They should be processed independently, output a tensor of shape [B, G_or_L_V, D] and before the loss those should be concatenated on a single tensor [B, V, D] 
    def forward(self, x: dict):
        
        tokens = x["input_ids"]  # B, T
        attention_mask = x["attention_mask"]  # B, T

        num_global_views = self.cfg["loss"]["num_global_views"]
        num_local_views = self.cfg["loss"]["num_local_views"]

        if len(attention_mask.shape) == 2:
            real_length = torch.where(attention_mask == 1)[1].size(0)
        elif len(attention_mask.shape) == 1:
            real_length = torch.where(attention_mask == 1)[0].size(0)
        else:
            raise ValueError(f"There is some kind of bug. Check shape {real_length.shape} from tensor attention_mask")

        # extract crops
        global_views = []
        local_views = []
        for _ in range(num_global_views):
            _, idx_down, idx_up = self._get_idxs(real_length, "global")
            global_views.append(tokens[idx_down:idx_up + 1])

        for _ in range(num_local_views):
            _, idx_down, idx_up = self._get_idxs(real_length, "local")
            local_views.append(tokens[idx_down:idx_up + 1])

        return torch.tensor()