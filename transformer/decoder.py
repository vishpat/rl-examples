import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import  TransformerDecoderLayer, TransformerDecoder
import math
import requests
import os

# ============================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================

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

# ============================================
# 2. TRANSFORMER DECODER ARCHITECTURE
# ============================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DecoderLM(nn.Module):
    """Decoder-only language model: embedding + positional encoding + TransformerDecoder + LM head."""

    def __init__(self, vocab_size, d_model, decoder_layer, num_layers, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _causal_mask(self, sz, device):
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, x, y=None):
        # x: (batch_size, seq_len) token indices
        # y: (batch_size, seq_len) target token indices (for training)
        N, T = x.shape
        # Embed and add positional encoding: (N, T, d_model)
        h = self.pos_enc(self.embed(x))
        # TransformerDecoder expects (seq_len, batch_size, d_model)
        h = h.transpose(0, 1)  # (T, N, d_model)
        causal_mask = self._causal_mask(T, x.device)
        out = self.decoder(h, h, tgt_mask=causal_mask, memory_mask=None)  # (T, N, d_model)
        out = out.transpose(0, 1)  # (N, T, d_model)
        logits = self.lm_head(out)  # (N, T, vocab_size)
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
            return logits, loss
        return logits, None

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=200, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            T = input_ids.size(1)
            h = self.pos_enc(self.embed(input_ids))
            h = h.transpose(0, 1)
            causal_mask = self._causal_mask(T, input_ids.device)
            out = self.decoder(h, h, tgt_mask=causal_mask, memory_mask=None)
            logits = self.lm_head(out[-1:])  # (1, N, vocab_size)
            logits = logits.squeeze(0) / max(temperature, 1e-8)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (N, 1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids


# ============================================
# 3. TRAINING LOOP
# ============================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=3e-4,
    device='cuda',
    grad_clip=1.0
):
    """Training loop with validation"""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs * len(train_loader)
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Validation
        val_loss = evaluate(model, val_loader, device)
        avg_train_loss = total_loss / len(train_loader)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Perplexity: {math.exp(val_loss):.2f}\n")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    return model

@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

# ============================================
# 4. MAIN EXECUTION
# ============================================

def main():
    # Hyperparameters
    BLOCK_SIZE = 128      # Context length
    BATCH_SIZE = 64
    D_MODEL = 256         # Embedding dimension
    N_HEADS = 8           # Attention heads
    N_LAYERS = 6          # Transformer blocks
    D_FF = 1024           # Feed-forward dimension
    DROPOUT = 0.1
    EPOCHS = 5
    LEARNING_RATE = 3e-4
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    text = download_shakespeare()
    print(f"Dataset size: {len(text):,} characters")
    
    # Create tokenizer
    tokenizer = CharTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
#    # Train/validation split
#    n = int(0.9 * len(text))
#    train_text = text[:n]
#    val_text = text[n:]
#    
#    # Create datasets
#    train_dataset = ShakespeareDataset(train_text, tokenizer, BLOCK_SIZE)
#    val_dataset = ShakespeareDataset(val_text, tokenizer, BLOCK_SIZE)
#    
#    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
#    
#    print(f"Training samples: {len(train_dataset):,}")
#    print(f"Validation samples: {len(val_dataset):,}")
#
    decoder_layer = TransformerDecoderLayer(
        d_model=D_MODEL,
        nhead=N_HEADS,
        dim_feedforward=D_FF,
        dropout=DROPOUT
    )

    model = DecoderLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        decoder_layer=decoder_layer,
        num_layers=N_LAYERS,
        dropout=DROPOUT,
        max_len=BLOCK_SIZE,
    )
    
    # Train
#    model = train_model(
#        model,
#        train_loader,
#        val_loader,
#        epochs=EPOCHS,
#        lr=LEARNING_RATE,
#        device=device
#    )
    
    model.load_state_dict(torch.load('best_model.pt'))    
    model.to(device)
    model.eval()

    # Generate sample text
    print("\n" + "="*50)
    print("GENERATED TEXT SAMPLES")
    print("="*50)
    
    prompts = ["ROMEO:", "To be or", "The king"]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        
        # Encode prompt
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        print(f"input_ids: {input_ids.shape}")
        
        # Generate
        output_ids = model.generate(
            input_ids,
            max_new_tokens=200,
            temperature=0.8,
            top_k=40
        )
        
        # Decode and print
        generated_text = tokenizer.decode(output_ids[0].tolist())
        print(generated_text)
        print()

if __name__ == "__main__":
    main()
