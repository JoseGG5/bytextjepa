
import torch

from src.data.base_tokenizer import Tokenizer

class BaselineTokenizer(Tokenizer):
    """ This tokenizer simply encodes to bytes and handles padding and truncation.
        It does not handle chunking strategies like learned chunk or fixed size chunk"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg


    def tokenize(self, text: str) -> dict:
        byte_text = list(text.encode("utf-8"))

        if len(byte_text) >= self.cfg["model"]["max_position_embeddings"]:  # truncate
            record = byte_text[:self.cfg["model"]["max_position_embeddings"]]
            attention_mask = [1] * self.cfg["model"]["max_position_embeddings"]
        else:  # pad
            record = byte_text + [self.cfg["model"]["pad_token_id"]] * (self.cfg["model"]["max_position_embeddings"] - len(byte_text)) 
            attention_mask = [1] * len(byte_text) + [0] * (self.cfg["model"]["max_position_embeddings"] - len(byte_text))

        # convert to tensor
        input_ids = torch.tensor(record, dtype=torch.long)  # [T]
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)  # [T]

        tokenized_text = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        return tokenized_text


    def detokenize(self, tokenized_text: dict) -> list[str]:
        """ Detokenizes the tokenized text """
        
        input_ids = tokenized_text["input_ids"]
        attention_mask = tokenized_text["attention_mask"]

        valid_ids = input_ids[attention_mask.bool()].tolist()

        return bytes(valid_ids).decode("utf-8")

