
import torch
from torch import nn

from src.utils import pad_tokens


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
        
        # get the pct of the crop depending of the mode
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


    # They should be processed independently, output a tensor of shape [B, G_or_L_V, D] and before the loss those should be concatenated on a single tensor [B, V, D] 
    def forward(self, x: dict):
        """ Computes local and global crops of the text, pads them and return them 
            with their corresponding attn mask"""
        tokens = x["input_ids"]  # B, T
        attention_mask = x["attention_mask"]  # B, T

        num_global_views = self.cfg["loss"]["num_global_views"]
        num_local_views = self.cfg["loss"]["num_local_views"]

        # ensure shape is B, T
        if (len(tokens.shape) == 1):
            tokens = tokens.unsqueeze(0)
        if (len(attention_mask.shape) == 1):
            attention_mask = attention_mask.unsqueeze(0)

        # get real length of tokens (excluding pad tokens)
        real_length = torch.where(attention_mask == 1)[1].size(0)

        # extract crops
        global_views = []
        local_views = []
        global_attn_masks = []
        local_attn_masks = []
        for _ in range(num_global_views):
            # get idxs and crop
            _, idx_down, idx_up = self._get_idxs(real_length, "global")
            crop = tokens[:, idx_down:idx_up + 1]

            # pad it and return attn masks
            padded_crop, attn_mask_crop = pad_tokens(
                x=crop,
                output_attn_mask=True,
                max_length=self.cfg["model"]["max_position_embeddings"],
                pad_token_id=self.cfg["model"]["pad_token_id"]
                )
            global_views.append(padded_crop)
            global_attn_masks.append(attn_mask_crop)

        for _ in range(num_local_views):
            # get idxs and crop
            _, idx_down, idx_up = self._get_idxs(real_length, "local")
            crop = tokens[:, idx_down:idx_up + 1]
            
            # pad it and return attn masks
            padded_crop, attn_mask_crop = pad_tokens(
                x=crop,
                output_attn_mask=True,
                max_length=self.cfg["model"]["max_position_embeddings"],
                pad_token_id=self.cfg["model"]["pad_token_id"]
                )
            local_views.append(padded_crop)
            local_attn_masks.append(attn_mask_crop)

        # convert to tensor
        global_views = torch.stack(global_views).transpose(0, 1)  # B, Gv, T
        local_views = torch.stack(local_views).transpose(0, 1)  # B, Lv, T
        global_attn_masks = torch.stack(global_attn_masks).transpose(0, 1)  # B, Gv, T
        local_attn_masks = torch.stack(local_attn_masks).transpose(0, 1)  # B, Lv, T

        return global_views, local_views, global_attn_masks, local_attn_masks