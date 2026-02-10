import requests
import os
import torch
from torch.utils.data import Dataset


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

    def __init__(self, text, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = tokenizer.encode(text)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Input sequence and target (shifted by 1)
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


if __name__ == "__main__":
    test_encode_decode()
    print("All tests passed")
