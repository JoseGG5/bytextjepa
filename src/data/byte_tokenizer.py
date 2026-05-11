
import torch

from src.data.base_tokenizer import Tokenizer

class BaselineTokenizer(Tokenizer):
    """ This tokenizer simply encodes to bytes and handles padding and truncation.
        It does not handle chunking strategies like learned chunk or fixed size chunk"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def tokenize(self, text: list[str]) -> dict:

        # handle single input instead of a list of inputs
        if type(text) == str:
            text = [text]

        # encode to bytes
        batch_bytes = [list(record.encode("utf-8")) for record in text]

        # handle truncation and padding
        batch_bytes_curated = []
        attention_masks = []
        for record in batch_bytes:
            if len(record) >= self.cfg["model"]["max_position_embeddings"]:  # truncate
                truncated_record = record[:self.cfg["model"]["max_position_embeddings"]]
                batch_bytes_curated.append(truncated_record)
                attention_masks.append([1] * self.cfg["model"]["max_position_embeddings"])
            else:
                padded_record = record + [self.cfg["model"]["pad_token_id"]] * (self.cfg["model"]["max_position_embeddings"] - len(record)) 
                batch_bytes_curated.append(padded_record)
                attention_masks.append([1] * len(record) + [0] * (self.cfg["model"]["max_position_embeddings"] - len(record)))

        # convert to tensor
        input_ids = torch.tensor(batch_bytes_curated, dtype=torch.long)  # [B, T]
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)  # [B, T]

        tokenized_text = {
            "input_ids": input_ids,
            "attention_mask": attention_masks
        }

        return tokenized_text

    def detokenize(self, tokenized_text: dict) -> list[str]:
        
        input_ids = tokenized_text["input_ids"]
        attention_masks = tokenized_text["attention_mask"]

        decoded_text = []
        for ids_row, mask_row in zip(input_ids, attention_masks):
            valid_ids = ids_row[mask_row.bool()].tolist()
            decoded_text.append(bytes(valid_ids).decode("utf-8"))

        return decoded_text
