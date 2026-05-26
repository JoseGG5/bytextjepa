import random

import torch
from torch.utils.data import Dataset

from src.data.base_tokenizer import Tokenizer
from src.utils import load_hf_dataset, pad_tokens


class MlmDataset(Dataset):
    """Dataset that builds byte-level MLM examples from raw text records."""

    def __init__(self, cfg: dict, tokenizer: Tokenizer):
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = cfg["dataset"]
        self.tokenizer = tokenizer
        self.mask_prob = float(cfg["mlm"]["mask_prob"])
        self.replace_mask_prob = float(cfg["mlm"]["replace_mask_prob"])
        self.replace_random_prob = float(cfg["mlm"]["replace_random_prob"])
        self.max_length = int(cfg["mlm"]["max_length"])
        self.pad_token_id = int(cfg["model"]["pad_token_id"])
        self.mask_token_id = int(cfg["model"]["mask_token_id"])
        self.byte_vocab_size = 256

        data = load_hf_dataset(cfg=cfg)
        data = data.filter(lambda x: x["text"] is not None and len(x["text"].strip()) >= 64)

        if self.dataset_cfg["dev"]:
            data = data.select(range(1))

        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def _mask_input_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked_input_ids = input_ids.clone()
        labels = torch.full_like(input_ids, fill_value=-100)

        valid_positions = torch.where(attention_mask == 1)[0]
        if valid_positions.numel() == 0:
            return masked_input_ids, labels

        num_to_mask = max(1, int(valid_positions.numel() * self.mask_prob))
        shuffled = valid_positions[torch.randperm(valid_positions.numel())]
        mask_positions = shuffled[:num_to_mask]

        labels[mask_positions] = input_ids[mask_positions]

        for pos in mask_positions.tolist():
            draw = random.random()
            if draw < self.replace_mask_prob:
                masked_input_ids[pos] = self.mask_token_id
            elif draw < self.replace_mask_prob + self.replace_random_prob:
                masked_input_ids[pos] = random.randint(0, self.byte_vocab_size - 1)

        return masked_input_ids, labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.data[idx]["text"]
        tokenized = self.tokenizer.tokenize(text)

        input_ids, attention_mask = pad_tokens(
            x=tokenized["input_ids"],
            output_attn_mask=True,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
        )
        masked_input_ids, labels = self._mask_input_ids(input_ids, attention_mask)

        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
