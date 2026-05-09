from torch.utils.data import Dataset

from src.utils import load_hf_dataset


class TextDataset(Dataset):
    """ Wrapper of a HF dataset for torch training """
    def __init__(self, cfg: dict, tokenizer):
        super().__init__()
        self.cfg = cfg["dataset"]
        self.data = load_hf_dataset(cfg=cfg)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]

        input = {
            "text": text,
            "idx": idx,
        }

        if self.tokenizer:
            out_tokenizer = self.tokenizer.tokenize(text)
            input["input_ids"] = out_tokenizer["input_ids"]
            input["attention_mask"] = out_tokenizer["attention_mask"]
        
        return input

        



        
    

