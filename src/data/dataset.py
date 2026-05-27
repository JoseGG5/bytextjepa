import pandas as pd
from torch.utils.data import Dataset

from src.utils import load_hf_dataset
from src.data.base_tokenizer import Tokenizer
from src.aug.augmentations import Augmentations

# TODO: Make a method to see training crops

class TextDataset(Dataset):
    """ Wrapper of a HF dataset for torch training """
    def __init__(
            self,
            cfg: dict,
            tokenizer: Tokenizer,
            augmenter: Augmentations
            ):
        super().__init__()
        self.cfg = cfg["dataset"]
        self.tokenizer = tokenizer
        self.augmenter = augmenter

        if self.cfg["name"] == "IIC/ClinText-SP":
            data = load_hf_dataset(cfg=cfg)

        elif self.cfg["name"] == "bookcorpus":
            data = pd.read_csv("data/BookCorpus3.csv")
            if "text" not in data.columns:
                first_column = data.columns[0]
                data = data.rename(columns={first_column: "text"})
            data = data[data["text"].notna() & (data["text"].str.strip().str.len() >= 64)]
            data = data.reset_index(drop=True)

        else:
            raise ValueError("Dataset not found. Choose between: bookcorpus or IIC/ClinText-SP")

        """After inspecting with the SQL console in HF there are empty 
        or really short records. Given that n_bytes ~= n_chars we can safely
        filter at the data size and not at the tokenized data size
        """
        if self.cfg["name"] != "bookcorpus":
            data = data.filter(lambda x: x["text"] is not None and len(x["text"].strip()) >= 64)  # only works for hf atasets

        if cfg["dataset"]["dev"]:  # to check we can get the loss to 0
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:1].reset_index(drop=True)
            else:
                data = data.select(range(1))

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(self.data, pd.DataFrame):
            sample = self.data.iloc[idx]
        else:
            sample = self.data[idx]
        text = sample["text"]

        input = {
            "idx": idx,
        }

        # tokenize the data and store ids and masks
        out_tokenizer = self.tokenizer.tokenize(text)
        input["input_ids"] = out_tokenizer["input_ids"]
        input["attention_mask"] = out_tokenizer["attention_mask"]

        # augment (basically create the crops and pad them) and store results
        global_crops, local_crops, global_masks, local_masks = self.augmenter(input)
        input["global_crops"] = global_crops
        input["local_crops"] = local_crops
        input["global_masks"] = global_masks
        input["local_masks"] = local_masks

        return {
            "idx": idx,
            "global_crops": global_crops,
            "local_crops": local_crops,
            "global_masks": global_masks,
            "local_masks": local_masks,
        }

    def visualize_crops(self, idx: int = 0, max_chars: int = 200) -> None:
        """Print the source text and the decoded crops for quick inspection."""
        if isinstance(self.data, pd.DataFrame):
            sample = self.data.iloc[idx]
        else:
            sample = self.data[idx]

        text = sample["text"]
        tokenized = self.tokenizer.tokenize(text)
        input_data = {
            "idx": idx,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
        global_crops, local_crops, global_masks, local_masks = self.augmenter(input_data)

        print(f"Sample {idx}")
        print(f"Original: {text[:max_chars]}")
        print()

        for crop_idx, (crop, mask) in enumerate(zip(global_crops, global_masks), start=1):
            decoded = self.tokenizer.detokenize(
                {"input_ids": crop, "attention_mask": mask}
            )
            print(f"Global {crop_idx}: {decoded[:max_chars]}")

        print()

        for crop_idx, (crop, mask) in enumerate(zip(local_crops, local_masks), start=1):
            decoded = self.tokenizer.detokenize(
                {"input_ids": crop, "attention_mask": mask}
            )
            print(f"Local {crop_idx}: {decoded[:max_chars]}")

        



        
    
