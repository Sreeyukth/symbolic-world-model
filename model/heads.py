import torch
import torch.nn as nn


class ModelHeads(nn.Module):
    def __init__(self, hidden_size: int, embed_dim: int, n_ops: int):
        super().__init__()
        self.latent_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim),
        )
        self.op_predictor = nn.Linear(hidden_size, n_ops)

    def forward(self, r_t: torch.Tensor) -> dict[str, torch.Tensor]:
        # r_t: (batch, hidden_size)
        return {
            "z_pred":    self.latent_predictor(r_t),  # (batch, embed_dim)
            "op_logits": self.op_predictor(r_t),       # (batch, n_ops)
        }
