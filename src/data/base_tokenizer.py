from abc import ABC, abstractmethod

import torch

class Tokenizer(ABC):
    """ Abstract base class for different types of tokenizers"""
    @abstractmethod
    def tokenize(self, text: list[str]) -> dict:
        pass

    @abstractmethod
    def detokenize(self, tokens: torch.tensor) -> list[str]:
        pass