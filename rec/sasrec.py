import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SASRec(nn.Module):
    def __init__(
        self, num_items, max_len, hidden_dim, num_heads, num_layers, dropout=0.1
    ):
        super(SASRec, self).__init__()
        self.item_emb = nn.Embedding(num_items, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(hidden_dim, num_items)

    def forward(self, x):
        x = self.item_emb(x)
        x = x + self.pos_emb(torch.arange(0, x.size(1)).float().to(device))
        x = self.layer_norm(x)
        x = self.dropout(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        logits = self.fc(x)
        return logits
