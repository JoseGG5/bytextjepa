
import torch
from torch import nn

from src.utils import pad_tokens


""" Every view (global or local) for a given record is cropped from a single shared
    anchor window instead of being sampled independently from the whole record. This
    guarantees that all views of a sample actually share content: global views crop
    70-100% of the anchor (so any two global views overlap heavily) and local views
    crop 10-35% of that same anchor (so locals always fall inside the region the
    globals are drawn from), matching the multi-crop assumption the LeJEPA loss
    relies on. The anchor window itself is capped at global_max_length, and every
    crop length is capped at its view's max_length, so pad_tokens never has to
    silently truncate a crop (which used to make the margin config a no-op). """

class Augmentations(nn.Module):
    """ This class handles how different views of the same
        text are created for LeJEPA"""
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def _sample_anchor_start(self, real_length: int, anchor_len: int) -> int:
        """ Picks where the shared anchor window starts inside the record """
        max_start = real_length - anchor_len
        if max_start <= 0:
            return 0
        return torch.randint(0, max_start + 1, (1,)).item()

    def _sample_crop(self, anchor_len: int, pct_min: int, pct_max: int, cap_len: int) -> tuple[int, int]:
        """ Samples a crop (offset, length) inside the anchor window. The crop length is
            a percentage of the anchor window, capped so it never exceeds cap_len (the
            padding budget for this kind of view) - this is what keeps the margin config
            meaningful instead of being overridden by pad_tokens truncating afterwards. """
        pct = torch.randint(pct_min, pct_max, (1,)).item() * 0.01
        length = max(1, min(int(pct * anchor_len), cap_len, anchor_len))
        max_offset = anchor_len - length
        offset = torch.randint(0, max_offset + 1, (1,)).item()
        return offset, length

    # They should be processed independently, output a tensor of shape [G_or_L_V, D] and before the loss those should be concatenated on a single tensor [B, V, D]
    def forward(self, x: dict):
        """ Computes local and global crops of the text, pads them and return them
            with their corresponding attn mask"""
        tokens = x["input_ids"]  # T
        attention_mask = x["attention_mask"]  # T

        num_global_views = self.cfg["loss"]["num_global_views"]
        num_local_views = self.cfg["loss"]["num_local_views"]
        global_max_length = self.cfg["aug"]["global_max_length"]
        local_max_length = self.cfg["aug"]["local_max_length"]
        pad_token_id = self.cfg["model"]["pad_token_id"]

        # get real length of tokens (excluding pad tokens)
        real_length = torch.where(attention_mask == 1)[0].size(0)

        # every view of this record is cropped from the same anchor window
        anchor_len = min(global_max_length, real_length)
        anchor_start = self._sample_anchor_start(real_length, anchor_len)
        anchor = tokens[anchor_start:anchor_start + anchor_len]

        # extract crops
        global_views = []
        local_views = []
        global_attn_masks = []
        local_attn_masks = []
        for _ in range(num_global_views):
            offset, length = self._sample_crop(
                anchor_len,
                self.cfg["aug"]["global_margin_min"],
                self.cfg["aug"]["global_margin_max"],
                cap_len=global_max_length,
            )
            crop = anchor[offset:offset + length]

            # pad it and return attn masks
            padded_crop, attn_mask_crop = pad_tokens(
                x=crop,
                output_attn_mask=True,
                max_length=global_max_length,
                pad_token_id=pad_token_id
                )

            global_views.append(padded_crop)
            global_attn_masks.append(attn_mask_crop)

        for _ in range(num_local_views):
            offset, length = self._sample_crop(
                anchor_len,
                self.cfg["aug"]["local_margin_min"],
                self.cfg["aug"]["local_margin_max"],
                cap_len=local_max_length,
            )
            crop = anchor[offset:offset + length]

            # pad it and return attn masks
            padded_crop, attn_mask_crop = pad_tokens(
                x=crop,
                output_attn_mask=True,
                max_length=local_max_length,
                pad_token_id=pad_token_id
                )
            local_views.append(padded_crop)
            local_attn_masks.append(attn_mask_crop)

        # convert to tensor
        global_views = torch.stack(global_views)
        local_views = torch.stack(local_views)
        global_attn_masks = torch.stack(global_attn_masks)
        local_attn_masks = torch.stack(local_attn_masks)

        return global_views, local_views, global_attn_masks, local_attn_masks