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
        data = load_hf_dataset(cfg=cfg)
        
        """After inspecting with the SQL console in HF there are empty 
        or really short records. Given that n_bytes ~= n_chars we can safely
        filter at the data size and not at the tokenized data size
        """
        data = data.filter(lambda x: x["text"] is not None and len(x["text"].strip()) >= 64)

        if cfg["dataset"]["dev"]:  # to check we can get the loss to 0
            data = data.select(range(1))

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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

        



        
    

