import torch
import torch.nn as nn


class GRUTransition(nn.Module):
    def __init__(self, embed_dim: int, op_embed_dim: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim + op_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(
        self,
        z_e: torch.Tensor,       # (batch, embed_dim)
        z_a: torch.Tensor,       # (batch, op_embed_dim)
        h_prev: torch.Tensor,    # (num_layers, batch, hidden_size)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z_e, z_a], dim=-1).unsqueeze(1)  # (batch, 1, embed_dim+op_embed_dim)
        _, h_new = self.gru(x, h_prev)                   # h_new: (num_layers, batch, hidden_size)
        r_t = h_new[-1]                                  # last layer: (batch, hidden_size)
        return h_new, r_t
