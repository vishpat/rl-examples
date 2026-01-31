"""
Simple Transformer Model for Text Classification
Author: LLM Engineer
Dataset: AG News (4-class news classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import math
from collections import Counter
from typing import List, Tuple
import time

# Set random seed for reproducibility
torch.manual_seed(42)

# ============================================================================
# PART 1: TRANSFORMER COMPONENTS
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings using sinusoidal functions.
    This allows the model to understand token order in sequences.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    Allows the model to jointly attend to information from different 
    representation subspaces at different positions.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query, key, value: Tensors of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        # Shape: (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask (for padding tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # Shape: (batch_size, num_heads, seq_len, d_k)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply final linear projection
        # Shape: (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Applies two linear transformations with a ReLU activation in between.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.
    Consists of Multi-Head Attention + Feed-Forward with residual connections
    and layer normalization.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class TransformerClassifier(nn.Module):
    """
    Complete Transformer model for text classification.
    Uses an encoder-only architecture with a classification head.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        num_classes: int = 4,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Stack of transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of token indices, shape (batch_size, seq_len)
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Create padding mask
        # Shape: (batch_size, 1, 1, seq_len) for broadcasting
        mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Embed tokens and add positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Global average pooling over sequence dimension
        # (ignoring padding tokens)
        mask_expanded = (mask.squeeze(1).squeeze(1)).unsqueeze(-1).float()
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


# ============================================================================
# PART 2: DATA PROCESSING
# ============================================================================

class Vocabulary:
    """Simple vocabulary class for token-to-index mapping."""
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.word_freq = Counter()
    
    def build(self, texts: List[str]):
        """Build vocabulary from list of texts."""
        for text in texts:
            tokens = self.tokenize(text)
            self.word_freq.update(tokens)
        
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"Vocabulary size: {len(self.word2idx)}")
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()
    
    def encode(self, text: str, max_len: int = 256) -> List[int]:
        """Convert text to list of token indices."""
        tokens = self.tokenize(text)[:max_len]
        return [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]
    
    def __len__(self):
        return len(self.word2idx)


class NewsDataset(Dataset):
    """PyTorch Dataset for AG News."""
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        encoded = self.vocab.encode(text, self.max_len)
        
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """Collate function for DataLoader with padding."""
    texts, labels = zip(*batch)
    
    # Pad sequences to same length
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    
    return texts_padded, labels


def load_ag_news():
    """
    Load AG News dataset.
    If torchtext is available, use it; otherwise, use a synthetic dataset for demo.
    """
    try:
        from torchtext.datasets import AG_NEWS
        from torchtext.data.utils import get_tokenizer
        
        print("Loading AG News dataset...")
        
        train_iter = AG_NEWS(split='train')
        test_iter = AG_NEWS(split='test')
        
        train_texts, train_labels = [], []
        test_texts, test_labels = [], []
        
        for label, text in train_iter:
            train_texts.append(text)
            train_labels.append(label - 1)  # Convert to 0-indexed
        
        for label, text in test_iter:
            test_texts.append(text)
            test_labels.append(label - 1)
        
        print(f"Training samples: {len(train_texts)}")
        print(f"Test samples: {len(test_texts)}")
        
        return train_texts, train_labels, test_texts, test_labels
        
    except ImportError:
        print("torchtext not available. Creating synthetic dataset for demonstration...")
        return create_synthetic_dataset()


def create_synthetic_dataset():
    """Create a synthetic news classification dataset for demonstration."""
    import random
    random.seed(42)
    
    # Sample templates for each category
    templates = {
        0: [  # World
            "The president of {} announced new diplomatic relations with {}",
            "International summit in {} discusses global {} policy",
            "United Nations reports on {} situation in {}",
            "{} government implements new {} regulations",
        ],
        1: [  # Sports
            "{} wins championship against {} in thrilling final",
            "Star player {} signs contract with {}",
            "{} Olympics sees record-breaking {} performance",
            "{} team advances to playoffs after defeating {}",
        ],
        2: [  # Business
            "{} company reports {} percent growth in quarterly earnings",
            "Stock market {} following {} economic news",
            "{} announces merger with {} valued at billions",
            "Tech giant {} launches new {} product line",
        ],
        3: [  # Science/Technology
            "Scientists discover new {} species in {}",
            "Research breakthrough in {} technology announced by {}",
            "{} launches new satellite for {} research",
            "AI system {} achieves breakthrough in {}",
        ],
    }
    
    # Words to fill in templates
    fillers = {
        "countries": ["France", "Japan", "Brazil", "Germany", "India", "Canada", "Australia"],
        "topics": ["climate", "trade", "security", "health", "education", "technology"],
        "teams": ["Lakers", "Warriors", "Patriots", "Yankees", "United", "City"],
        "sports": ["basketball", "football", "tennis", "soccer", "baseball"],
        "companies": ["Apple", "Google", "Microsoft", "Amazon", "Tesla", "Meta"],
        "numbers": ["5", "10", "15", "20", "25", "30"],
        "tech_terms": ["quantum", "neural", "blockchain", "genetic", "robotic"],
    }
    
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    
    # Generate training data
    for _ in range(8000):
        label = random.randint(0, 3)
        template = random.choice(templates[label])
        
        # Fill template with random words
        text = template
        for _ in range(text.count("{}")):
            filler_type = random.choice(list(fillers.keys()))
            filler = random.choice(fillers[filler_type])
            text = text.replace("{}", filler, 1)
        
        train_texts.append(text)
        train_labels.append(label)
    
    # Generate test data
    for _ in range(2000):
        label = random.randint(0, 3)
        template = random.choice(templates[label])
        
        text = template
        for _ in range(text.count("{}")):
            filler_type = random.choice(list(fillers.keys()))
            filler = random.choice(fillers[filler_type])
            text = text.replace("{}", filler, 1)
        
        test_texts.append(text)
        test_labels.append(label)
    
    print(f"Synthetic training samples: {len(train_texts)}")
    print(f"Synthetic test samples: {len(test_texts)}")
    
    return train_texts, train_labels, test_texts, test_labels


# ============================================================================
# PART 3: TRAINING AND EVALUATION
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (texts, labels) in enumerate(dataloader):
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(texts)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {100. * correct / total:.2f}%")
    
    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def predict(model, text: str, vocab: Vocabulary, device, class_names: List[str]):
    """Make prediction on a single text."""
    model.eval()
    
    with torch.no_grad():
        encoded = vocab.encode(text)
        input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)
        
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        predicted_class = outputs.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence, probabilities[0].cpu().numpy()


# ============================================================================
# PART 4: MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("SIMPLE TRANSFORMER MODEL FOR TEXT CLASSIFICATION")
    print("=" * 70)
    
    # Hyperparameters
    config = {
        "d_model": 256,          # Embedding dimension
        "num_heads": 8,          # Number of attention heads
        "num_layers": 4,         # Number of transformer layers
        "d_ff": 512,             # Feed-forward dimension
        "dropout": 0.1,
        "max_len": 256,          # Maximum sequence length
        "batch_size": 64,
        "learning_rate": 1e-4,
        "num_epochs": 5,
        "min_freq": 2,           # Minimum word frequency for vocabulary
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    train_texts, train_labels, test_texts, test_labels = load_ag_news()
    
    class_names = ["World", "Sports", "Business", "Sci/Tech"]
    
    # Show class distribution
    print("\nClass distribution (training):")
    from collections import Counter
    label_counts = Counter(train_labels)
    for label, count in sorted(label_counts.items()):
        print(f"  {class_names[label]}: {count}")
    
    # Build vocabulary
    print("\n" + "=" * 70)
    print("BUILDING VOCABULARY")
    print("=" * 70)
    
    vocab = Vocabulary(min_freq=config["min_freq"])
    vocab.build(train_texts)
    
    # Create datasets and dataloaders
    print("\n" + "=" * 70)
    print("CREATING DATALOADERS")
    print("=" * 70)
    
    train_dataset = NewsDataset(train_texts, train_labels, vocab, config["max_len"])
    test_dataset = NewsDataset(test_texts, test_labels, vocab, config["max_len"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    
    model = TransformerClassifier(
        vocab_size=len(vocab),
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        num_classes=len(class_names),
        max_len=config["max_len"],
        dropout=config["dropout"],
        pad_idx=0
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"]
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    best_accuracy = 0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 50)
        
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  ✓ New best model saved! (Accuracy: {best_accuracy:.2f}%)")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    # Load best model
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"\nBest Model Test Accuracy: {test_acc:.2f}%")
    
    # Example predictions
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)
    
    test_examples = [
        "The president announced new economic policies today",
        "The team won the championship game in overtime",
        "Stock prices rose following the earnings report",
        "Scientists discovered a new species of deep sea fish",
        "International talks continue regarding trade agreements",
        "The quarterback threw four touchdowns in the victory",
    ]
    
    print("\n")
    for text in test_examples:
        predicted_class, confidence, probs = predict(model, text, vocab, device, class_names)
        print(f"Text: \"{text}\"")
        print(f"  Prediction: {predicted_class} (Confidence: {confidence:.2%})")
        print(f"  All probabilities: {dict(zip(class_names, [f'{p:.2%}' for p in probs]))}")
        print()
    
    # Training history summary
    print("\n" + "=" * 70)
    print("TRAINING HISTORY")
    print("=" * 70)
    
    print("\nEpoch | Train Loss | Train Acc | Test Loss | Test Acc")
    print("-" * 55)
    for i in range(len(history["train_loss"])):
        print(f"  {i+1}   |   {history['train_loss'][i]:.4f}   |  {history['train_acc'][i]:.2f}%  |  {history['test_loss'][i]:.4f}   | {history['test_acc'][i]:.2f}%")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    return model, vocab, history


if __name__ == "__main__":
    model, vocab, history = main()
