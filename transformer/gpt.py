import requests
import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
        self.data = tokenizer.encode(text).to(device)

        size = len(self.data)
        train_size = int(size * train_test_split)
        self.train_data = self.data[:train_size]
        self.test_data = self.data[train_size:]

    def get_test_data(self, batch_size=64) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randperm(len(self.test_data) - self.block_size)[:batch_size]
        x = torch.stack([self.test_data[i : i + self.block_size] for i in indices])
        y = torch.stack(
            [self.test_data[i + 1 : i + self.block_size + 1] for i in indices]
        )
        return x, y

    def get_train_data(self, batch_size=64) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randperm(len(self.train_data) - self.block_size)[:batch_size]
        x = torch.stack([self.train_data[i : i + self.block_size] for i in indices])
        y = torch.stack(
            [self.train_data[i + 1 : i + self.block_size + 1] for i in indices]
        )
        return x, y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, block_size=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(block_size, d_model).to(device)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        # Apply the log-space calculation for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, block_size, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # Add positional encoding to input embedding
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, head_size, d_model, decoder=True):
        super().__init__()
        self.head_size = head_size
        self.decoder = decoder
        self.query = nn.Linear(d_model, head_size).to(device)
        self.key = nn.Linear(d_model, head_size).to(device)
        self.value = nn.Linear(d_model, head_size).to(device)

    def forward(self, x):
        B, T, C = x.shape
        query = self.query(x)
        key = self.key(x)
        weights = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
        triu_mask = torch.triu(torch.ones(T, T), diagonal=1).to(device)
        if self.decoder:
            weights = weights.masked_fill(triu_mask == 1, float("-inf")).to(device)
        weights = F.softmax(weights, dim=-1)
        value = self.value(x)
        output = weights @ value
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, d_model):
        super().__init__()
        self.heads = nn.ModuleList(
            [Attention(head_size, d_model) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.proj(torch.cat([head(x) for head in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=16,
        num_heads=2,
        num_layers=2,
        dropout=0.1,
        block_size=128,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, d_model).to(device)
        self.pos_encoding = PositionalEncoding(d_model, block_size, dropout)
        self.lm_head = nn.Linear(d_model, vocab_size).to(device)
        self.single_head_attention = Attention(block_size, d_model)

    def forward(self, x, y=None):
        embed = self.embed(x)
        embed = self.pos_encoding(embed)
        logits = self.single_head_attention(embed)
        loss = None
        if y is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            y = y.view(B * T)
            loss = F.cross_entropy(logits, y, ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, context, max_new_tokens=50):
        for _ in range(max_new_tokens):
            logits, loss = self(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)
        return context


@torch.no_grad()
def estimate_loss(model, dataset, eval_iters=10):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = dataset.get_test_data(batch_size=64)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


if __name__ == "__main__":
    tokenizer = CharTokenizer(download_shakespeare())
    vocab_size = tokenizer.vocab_size
    print(f"vocab_size: {vocab_size}")
    dataset = ShakespeareDataset(download_shakespeare(), tokenizer)
    model = GPTLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(10000):
        x, y = dataset.get_train_data(batch_size=64)
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 1000 == 0:
            loss = estimate_loss(model, dataset, eval_iters=10)
            print(f"Epoch {epoch}, Loss: {loss}")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(context, max_new_tokens=100)[0].tolist()))
