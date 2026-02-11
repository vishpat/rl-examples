import requests
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from typing import Tuple


def test_encode_decode():
    text = "Hello, world! This is a test."
    tokenizer = CharTokenizer(text)
    print(f"tokenizer: {tokenizer.vocab_size}")
    text = text.lower()
    print(f"text: {text}")
    encoded = tokenizer.encode(text)
    print(f"encoded: {encoded} {encoded.shape}")
    decoded = tokenizer.decode(encoded)
    print(f"decoded: {decoded}")
    assert decoded == text, f"decoded: {decoded} != text: {text}"
    print("test_encode_decode passed")


def download_shakespeare():
    """Download the tiny Shakespeare dataset"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    if not os.path.exists("shakespeare.txt"):
        print("Downloading Shakespeare dataset...")
        response = requests.get(url)
        with open("shakespeare.txt", "w") as f:
            f.write(response.text)

    with open("shakespeare.txt", "r") as f:
        text = f.read()

    return text


class CharTokenizer:
    """Simple character-level tokenizer"""

    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)

    def decode(self, indices: torch.Tensor) -> str:
        return "".join([self.idx_to_char[int(i)] for i in indices])


class ShakespeareDataset(Dataset):
    """Dataset for autoregressive language modeling"""

    def __init__(self, text, tokenizer, block_size=128, train_test_split=0.9):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = tokenizer.encode(text)

        size = len(self.data)
        train_size = int(size * train_test_split)
        self.train_data = torch.tensor(self.data[:train_size], dtype=torch.long)
        self.test_data = torch.tensor(self.data[train_size:], dtype=torch.long)

    def get_test_data(self, batch_size=64) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randperm(len(self.test_data))[:batch_size]
        x = torch.stack([self.test_data[i : i + self.block_size] for i in indices])
        y = torch.stack(
            [self.test_data[i + 1 : i + self.block_size + 1] for i in indices]
        )
        return x, y

    def get_train_data(self, batch_size=64) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randperm(len(self.train_data))[:batch_size]
        x = torch.stack([self.train_data[i : i + self.block_size] for i in indices])
        y = torch.stack(
            [self.train_data[i + 1 : i + self.block_size + 1] for i in indices]
        )
        return x, y


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, y=None):
        logits = self.embed(x)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        print(f"logits: {logits.shape} {logits} y: {y.shape} {y}")
        if y is not None:
            y = y.view(B * T)
            loss = F.cross_entropy(logits, y, ignore_index=-1)
            return logits, loss
        return logits, None


if __name__ == "__main__":
    tokenizer = CharTokenizer(download_shakespeare())
    vocab_size = tokenizer.vocab_size
    dataset = ShakespeareDataset(download_shakespeare(), tokenizer)
    model = BigramLanguageModel(vocab_size)
    x, y = dataset.get_train_data(batch_size=64)
    print(f"x: {x.shape} {x} y: {y.shape} {y}")
    logits, loss = model(x, y)
    print(f"loss: {loss}")
    print(f"logits: {logits.shape} y: {y.shape}")
