import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SASRec(nn.Module):
    def __init__(
        self, num_items: int, max_len: int, d_model: int, num_heads: int, num_layers: int, dropout: float = 0.1
    ):
        super(SASRec, self).__init__()
        self.item_emb = nn.Embedding(num_items, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(d_model, num_items)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.item_emb(x)
        pos = self.pos_emb(torch.arange(0, x.size(1)).long().to(device))
        x = x + pos
        x = self.layer_norm(x)
        x = self.dropout(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        logits = self.fc(x)
        return logits

if __name__ == "__main__":
    num_items = 100
    max_len = 50
    d_model = 128
    num_heads = 8
    num_layers = 2
    dropout = 0.1
    model = SASRec(num_items=num_items, max_len=max_len, d_model=d_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout).to(device)
    print(model)

    x = torch.randint(0, num_items, (1, max_len)).to(device)
    print(x.shape)
    print(model(x).shape)