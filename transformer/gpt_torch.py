import requests
import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from typing import Tuple
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(f"Using device: {device}")

BATCH_SIZE = 128
SEQ_LEN = 256
N_HEADS = 8
D_MODEL = 256


def test_encode_decode():
    text = "Hello, world! This is a test."
    tokenizer = CharTokenizer(text)
    logger.debug(f"tokenizer: {tokenizer.vocab_size}")
    text = text.lower()
    logger.debug(f"text: {text}")
    encoded = tokenizer.encode(text)
    logger.debug(f"encoded: {encoded} {encoded.shape}")
    decoded = tokenizer.decode(encoded)
    logger.debug(f"decoded: {decoded}")
    assert decoded == text, f"decoded: {decoded} != text: {text}"
    logger.debug("test_encode_decode passed")


def download_shakespeare():
    """Download the tiny Shakespeare dataset"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    if not os.path.exists("shakespeare.txt"):
        logger.debug("Downloading Shakespeare dataset...")
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

    def __init__(self, text, tokenizer, block_size=SEQ_LEN, train_test_split=0.9):
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
    def __init__(self, d_model, block_size=SEQ_LEN, dropout=0.1):
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


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=D_MODEL,
        num_heads=N_HEADS,
        num_layers=2,
        dropout=0.1,
        block_size=SEQ_LEN,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, d_model).to(device)
        self.pos_encoding = PositionalEncoding(d_model, block_size, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            device=device,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model).to(device)
        self.lm_head = nn.Linear(d_model, vocab_size).to(device)
        self.register_buffer(
            "triu_mask",
            torch.triu(
                torch.full((block_size, block_size), float("-inf")), diagonal=1
            ).to(device),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.blocks.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.lm_head.weight)

    def forward(self, x, y=None):
        embed = self.embed(x)
        embed = self.pos_encoding(embed)
        x = self.decoder(
            tgt=self.triu_mask[x.size(1), x.size(1)],
            memory=torch.zeros(x.size(0), x.size(1), self.d_model).to(device),
        )
        x = self.norm(x)
        logits = self.lm_head(x)
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
        X, Y = dataset.get_test_data(batch_size=BATCH_SIZE)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


if __name__ == "__main__":
    tokenizer = CharTokenizer(download_shakespeare())
    vocab_size = tokenizer.vocab_size
    logger.debug(f"vocab_size: {vocab_size}")
    dataset = ShakespeareDataset(download_shakespeare(), tokenizer)
    model = GPTLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(10000):
        x, y = dataset.get_train_data(batch_size=BATCH_SIZE)
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 1000 == 0:
            loss = estimate_loss(model, dataset, eval_iters=10)
            logger.info(f"Epoch {epoch}, Loss: {loss}")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    logger.info(
        tokenizer.decode(
            model.generate(context, max_new_tokens=SEQ_LEN - 1)[0].tolist()
        )
    )
