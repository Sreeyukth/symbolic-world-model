"""
Training loop for SymbolicWorldModel.
Run from project root: python train.py

Prints live loss to terminal at each epoch.
Saves: checkpoints/model.pt, training_curves.png
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import config
from data.tokenizer import Tokenizer
from model.world_model import SymbolicWorldModel

OP_MAP = {"EXPAND": 0, "FACTOR": 1, "DIFF": 2, "SIMPLIFY": 3}
MAX_STEPS = 6  # max trajectory length (pad all trajectories to this)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    def __init__(self, path: str, tokenizer: Tokenizer):
        with open(path) as f:
            raw = json.load(f)

        pad = torch.zeros(config.MAX_SEQ_LEN, dtype=torch.long)
        self.samples = []

        for traj in raw:
            steps  = traj["steps"]          # new format: {"chain_type": ..., "steps": [...]}
            length = len(steps)
            expr_seq, op_seq, next_expr_seq = [], [], []

            for step in steps:
                expr_seq.append(tokenizer.encode(step["expr"]))
                op_seq.append(OP_MAP[step["op"]])
                next_expr_seq.append(tokenizer.encode(step["next_expr"]))

            # Pad to MAX_STEPS so all trajectories stack cleanly
            while len(expr_seq) < MAX_STEPS:
                expr_seq.append(pad.clone())
                op_seq.append(0)
                next_expr_seq.append(pad.clone())

            self.samples.append((
                torch.stack(expr_seq),                          # (MAX_STEPS, seq_len)
                torch.tensor(op_seq,  dtype=torch.long),        # (MAX_STEPS,)
                torch.stack(next_expr_seq),                     # (MAX_STEPS, seq_len)
                torch.tensor(length,  dtype=torch.long),        # scalar
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    device = torch.device(config.DEVICE)
    print(f"Device : {device}")

    # tokenizer (also sets config.VOCAB_SIZE)
    tok = Tokenizer()

    print("Loading dataset...")
    dataset = TrajectoryDataset("data/trajectories.json", tok)
    loader  = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    print(f"Dataset : {len(dataset):,} trajectories | {len(loader)} batches/epoch\n")

    model = SymbolicWorldModel(
        vocab_size   = config.VOCAB_SIZE,
        embed_dim    = config.EMBED_DIM,
        op_embed_dim = config.OP_EMBED_DIM,
        hidden_size  = config.GRU_HIDDEN,
        num_layers   = config.GRU_LAYERS,
        n_ops        = config.N_OPS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    loss_history      = []
    expr_loss_history = []
    op_loss_history   = []

    print(f"\n{'─'*75}")
    print(f"  {'Epoch':>5}  {'Total Loss':>10}  {'Expr Loss':>10}  {'Op Loss':>9}  {'LR':>8}  {'Time':>7}")
    print(f"{'─'*75}")

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = total_expr = total_op = 0.0
        n_batches = 0
        t0 = time.time()

        for expr_seq, op_seq, next_expr_seq, lengths in loader:
            expr_seq      = expr_seq.to(device)       # (B, MAX_STEPS, seq_len)
            op_seq        = op_seq.to(device)          # (B, MAX_STEPS)
            next_expr_seq = next_expr_seq.to(device)   # (B, MAX_STEPS, seq_len)
            lengths       = lengths.to(device)         # (B,)

            B = expr_seq.size(0)
            h = model.init_hidden(B, device)

            batch_loss = torch.tensor(0.0, device=device)
            batch_expr = 0.0
            batch_op   = 0.0

            for t in range(MAX_STEPS):
                active = lengths > t            # (B,) which trajectories are still running
                if not active.any():
                    break

                out = model(expr_seq[:, t], op_seq[:, t], h)
                h   = out["h_new"]              # carry hidden state across steps (full BPTT)

                # --- expr loss: MSE between predicted latent and actual encoded next expr ---
                next_t   = next_expr_seq[:, t, :]                        # (B, seq_len)
                with torch.no_grad():
                    z_e_next = model.expr_encoder(next_t)                # (B, embed_dim)
                expr_loss = F.mse_loss(out["z_pred"][active], z_e_next[active])

                # --- op loss: predict next op; skip at last step of each trajectory ---
                has_next = active & (lengths > t + 1)
                if has_next.any():
                    op_loss = F.cross_entropy(
                        out["op_logits"][has_next],
                        op_seq[has_next, t + 1],
                    )
                else:
                    op_loss = torch.tensor(0.0, device=device)

                batch_loss = batch_loss + expr_loss + op_loss
                batch_expr += expr_loss.item()
                batch_op   += op_loss.item()

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            total_expr += batch_expr
            total_op   += batch_op
            n_batches  += 1

        elapsed  = time.time() - t0
        avg_loss = total_loss / n_batches
        avg_expr = total_expr / n_batches
        avg_op   = total_op   / n_batches

        loss_history.append(avg_loss)
        expr_loss_history.append(avg_expr)
        op_loss_history.append(avg_op)

        current_lr = scheduler.get_last_lr()[0]
        print(f"  {epoch+1:>5d}  {avg_loss:>10.4f}  {avg_expr:>10.4f}  {avg_op:>9.4f}  {current_lr:>8.2e}  {elapsed:>6.1f}s")
        scheduler.step()

    print(f"{'─'*75}")

    # --- save checkpoint ---
    Path("checkpoints").mkdir(exist_ok=True)
    ckpt_path = "checkpoints/model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nCheckpoint saved  →  {ckpt_path}")

    # --- training curves ---
    epochs = range(1, config.EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, loss_history,      label="Total Loss", linewidth=2)
    ax.plot(epochs, expr_loss_history, label="Expr Loss",  linestyle="--", linewidth=1.5)
    ax.plot(epochs, op_loss_history,   label="Op Loss",    linestyle=":",  linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Symbolic World Model — Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = "training_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training curves saved  →  {plot_path}")


if __name__ == "__main__":
    train()
