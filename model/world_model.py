import torch
import torch.nn as nn

from model.encoder import ExprEncoder
from model.transition import GRUTransition
from model.heads import ModelHeads


class SymbolicWorldModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        op_embed_dim: int,
        hidden_size: int,
        num_layers: int,
        n_ops: int,
    ):
        super().__init__()
        self.expr_encoder  = ExprEncoder(vocab_size, embed_dim)
        self.op_embedding  = nn.Embedding(n_ops, op_embed_dim)
        self.transition    = GRUTransition(embed_dim, op_embed_dim, hidden_size, num_layers)
        self.heads         = ModelHeads(hidden_size, embed_dim, n_ops)
        self.num_layers    = num_layers
        self.hidden_size   = hidden_size

    def forward(
        self,
        expr_tokens: torch.Tensor,  # (batch, seq_len)
        op_id: torch.Tensor,        # (batch,)
        h_prev: torch.Tensor,       # (num_layers, batch, hidden_size)
    ) -> dict[str, torch.Tensor]:
        z_e = self.expr_encoder(expr_tokens)   # (batch, embed_dim)
        z_a = self.op_embedding(op_id)         # (batch, op_embed_dim)
        h_new, r_t = self.transition(z_e, z_a, h_prev)
        outputs = self.heads(r_t)
        outputs["h_new"] = h_new               # (num_layers, batch, hidden_size)
        outputs["r_t"]   = r_t                 # (batch, hidden_size)
        return outputs

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
