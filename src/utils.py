
import yaml
from transformers import ModernBertConfig, ModernBertModel
import datasets
from datasets import load_dataset


def load_cfg(path: str) -> dict:
    """Loads the config"""
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def init_encoder(cfg: dict) -> ModernBertModel:
    """ Initializes a random ModernBERT model with the
    specified config"""
    cfg = ModernBertConfig(**cfg["model"])
    return ModernBertModel(cfg)


def load_hf_dataset(cfg: dict) -> datasets.arrow_dataset.Dataset:
    """ Loads a dataset from HF datasets """
    dataset = load_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["version"],
        split=cfg["dataset"]["split"]
    )
    return dataset