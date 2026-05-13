import torch
import torch.nn as nn


class ExprEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dim = embed_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len)
        embedded = self.embedding(token_ids)          # (batch, seq_len, embed_dim)

        # Mask PAD tokens (id == 0) before mean pooling
        mask = (token_ids != 0).float().unsqueeze(-1) # (batch, seq_len, 1)
        sum_embed = (embedded * mask).sum(dim=1)       # (batch, embed_dim)
        count = mask.sum(dim=1).clamp(min=1.0)        # (batch, 1) — avoid div by zero
        z_e = sum_embed / count                        # (batch, embed_dim)
        return z_e
