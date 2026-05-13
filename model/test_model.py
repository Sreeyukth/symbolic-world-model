"""
Smoke test for SymbolicWorldModel.
Run from project root: python model/test_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import config
from data.tokenizer import Tokenizer
from model.world_model import SymbolicWorldModel

# --- setup ---
tok = Tokenizer()  # sets config.VOCAB_SIZE

BATCH = 4
model = SymbolicWorldModel(
    vocab_size  = config.VOCAB_SIZE,
    embed_dim   = config.EMBED_DIM,
    op_embed_dim= config.OP_EMBED_DIM,
    hidden_size = config.GRU_HIDDEN,
    num_layers  = config.GRU_LAYERS,
    n_ops       = config.N_OPS,
)
model.eval()

device = torch.device(config.DEVICE)
model.to(device)

# --- dummy inputs ---
expr_tokens = torch.randint(0, config.VOCAB_SIZE, (BATCH, config.MAX_SEQ_LEN)).to(device)
op_id       = torch.randint(0, config.N_OPS, (BATCH,)).to(device)
h_prev      = model.init_hidden(BATCH, device)

# --- forward pass ---
with torch.no_grad():
    out = model(expr_tokens, op_id, h_prev)

# --- print shapes ---
print("\n=== SymbolicWorldModel forward pass ===")
print(f"  expr_tokens : {tuple(expr_tokens.shape)}")
print(f"  op_id       : {tuple(op_id.shape)}")
print(f"  h_prev      : {tuple(h_prev.shape)}")
print(f"  --- outputs ---")
print(f"  z_pred      : {tuple(out['z_pred'].shape)}")
print(f"  op_logits   : {tuple(out['op_logits'].shape)}")
print(f"  h_new       : {tuple(out['h_new'].shape)}")
print(f"  r_t         : {tuple(out['r_t'].shape)}")

# --- shape assertions ---
assert out["z_pred"].shape    == (BATCH, config.EMBED_DIM),                        f"z_pred shape wrong: {out['z_pred'].shape}"
assert out["op_logits"].shape == (BATCH, config.N_OPS),                            f"op_logits shape wrong: {out['op_logits'].shape}"
assert out["h_new"].shape     == (config.GRU_LAYERS, BATCH, config.GRU_HIDDEN),    f"h_new shape wrong: {out['h_new'].shape}"
assert out["r_t"].shape       == (BATCH, config.GRU_HIDDEN),                       f"r_t shape wrong: {out['r_t'].shape}"

print("\nAll shape assertions passed.")
print(f"Device: {device}")
