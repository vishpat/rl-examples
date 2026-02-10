import requests
import os
import torch
from torch.utils.data import Dataset

def download_shakespeare():
    """Download the tiny Shakespeare dataset"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    if not os.path.exists('shakespeare.txt'):
        print("Downloading Shakespeare dataset...")
        response = requests.get(url)
        with open('shakespeare.txt', 'w') as f:
            f.write(response.text)
    
    with open('shakespeare.txt', 'r') as f:
        text = f.read()
    
    return text

class CharTokenizer:
    """Simple character-level tokenizer"""
    
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])

class ShakespeareDataset(Dataset):
    """Dataset for autoregressive language modeling"""
    
    def __init__(self, text, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Input sequence and target (shifted by 1)
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

if __name__ == "__main__":
    text = download_shakespeare()
    print(f"text: {text} {len(text)}")
    tokenizer = CharTokenizer(text)
    print(f"tokenizer: {tokenizer.vocab_size}")
    dataset = ShakespeareDataset(text, tokenizer)
    print(f"dataset: {len(dataset)}")
    for x, y in dataset:
        print(f"x: {x} {x.shape}")
        print(f"y: {y} {y.shape}")
        break