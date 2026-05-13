"""
Latent space probing for SymbolicWorldModel.
Run from project root: python probe.py

Experiments:
  1. PCA visualization of GRU hidden states (3 plots)
  2. Linear probe accuracy — chain type, op, step index
  3. Trajectory drift — L2 distance from step-0 hidden state

All results saved to probe_results/
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              r2_score, mean_absolute_error)
from sklearn.preprocessing import LabelEncoder

import torch

import config
from data.tokenizer import Tokenizer
from model.world_model import SymbolicWorldModel

RESULTS_DIR = Path("probe_results")

OP_MAP = {"EXPAND": 0, "FACTOR": 1, "DIFF": 2, "SIMPLIFY": 3}

CHAIN_COLORS = {
    "simplification_chain": "#e41a1c",
    "calculus_chain":        "#377eb8",
    "expand_diff_chain":     "#4daf4a",
    "factor_expand_cycle":   "#984ea3",
    "repeated_diff":         "#ff7f00",
    "expand_chain":          "#a65628",
}

OP_COLORS = {
    "EXPAND":   "#e41a1c",
    "FACTOR":   "#377eb8",
    "DIFF":     "#4daf4a",
    "SIMPLIFY": "#984ea3",
}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_model_and_data():
    tok    = Tokenizer()
    device = torch.device(config.DEVICE)

    model = SymbolicWorldModel(
        vocab_size   = config.VOCAB_SIZE,
        embed_dim    = config.EMBED_DIM,
        op_embed_dim = config.OP_EMBED_DIM,
        hidden_size  = config.GRU_HIDDEN,
        num_layers   = config.GRU_LAYERS,
        n_ops        = config.N_OPS,
    )
    model.load_state_dict(torch.load("checkpoints/model.pt", map_location=device,
                                     weights_only=True))
    model.to(device)
    model.eval()
    print(f"Model loaded  →  checkpoints/model.pt  |  device: {device}")

    with open("data/trajectories.json") as f:
        trajectories = json.load(f)
    print(f"Trajectories  →  {len(trajectories):,} loaded")
    return tok, model, device, trajectories


def collect_hidden_states(tok, model, device, trajectories):
    """Run all trajectories through the frozen model and collect h_t at each step."""
    records       = []   # flat list — one entry per (traj, step)
    traj_by_chain = defaultdict(list)  # chain_type -> [(traj_id, [(step_idx, h_t)])]

    print(f"\nCollecting hidden states from {len(trajectories):,} trajectories...")
    with torch.no_grad():
        for traj_id, traj in enumerate(trajectories):
            chain_type = traj["chain_type"]
            steps      = traj["steps"]
            h          = model.init_hidden(1, device)
            traj_steps = []

            for step_idx, step in enumerate(steps):
                expr_tokens = tok.encode(step["expr"]).unsqueeze(0).to(device)
                op_id       = torch.tensor([OP_MAP[step["op"]]], device=device)

                out = model(expr_tokens, op_id, h)
                h   = out["h_new"]
                h_t = out["r_t"].squeeze(0).cpu().numpy()  # (128,)

                records.append({
                    "h_t":        h_t,
                    "chain_type": chain_type,
                    "op":         step["op"],
                    "step_index": step_idx,
                    "traj_id":    traj_id,
                })
                traj_steps.append((step_idx, h_t))

            traj_by_chain[chain_type].append((traj_id, traj_steps))

            if (traj_id + 1) % 2_000 == 0:
                print(f"  {traj_id+1:,} / {len(trajectories):,} processed "
                      f"({len(records):,} vectors collected)")

    print(f"  Done — {len(records):,} hidden state vectors total.")
    return records, traj_by_chain


# ---------------------------------------------------------------------------
# Experiment 1 — PCA Visualization
# ---------------------------------------------------------------------------

def experiment_pca(records):
    print("\n--- Experiment 1: PCA Visualization ---")
    H = np.stack([r["h_t"] for r in records])   # (N, 128)

    pca  = PCA(n_components=2)
    H_2d = pca.fit_transform(H)                  # (N, 2)
    ev   = pca.explained_variance_ratio_
    print(f"  Variance explained: PC1={ev[0]:.3f}, PC2={ev[1]:.3f}, "
          f"total={ev.sum():.3f}")

    chain_types  = [r["chain_type"] for r in records]
    ops          = [r["op"]         for r in records]
    step_indices = np.array([r["step_index"] for r in records])

    # --- Plot 1: color by chain_type ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for cname, color in CHAIN_COLORS.items():
        mask = np.array([ct == cname for ct in chain_types])
        ax.scatter(H_2d[mask, 0], H_2d[mask, 1],
                   c=color, label=cname.replace("_", " "),
                   s=4, alpha=0.3, rasterized=True)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("GRU hidden states colored by chain type")
    ax.legend(markerscale=3, loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pca_chain.png", dpi=150)
    plt.close()
    print("  Saved pca_chain.png")

    # --- Plot 2: color by op ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for op_name, color in OP_COLORS.items():
        mask = np.array([o == op_name for o in ops])
        ax.scatter(H_2d[mask, 0], H_2d[mask, 1],
                   c=color, label=op_name,
                   s=4, alpha=0.3, rasterized=True)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("GRU hidden states colored by operation")
    ax.legend(markerscale=3, loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pca_op.png", dpi=150)
    plt.close()
    print("  Saved pca_op.png")

    # --- Plot 3: color by step_index ---
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(H_2d[:, 0], H_2d[:, 1],
                    c=step_indices, cmap="viridis",
                    s=4, alpha=0.3, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Step index")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("GRU hidden states colored by trajectory step")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pca_step.png", dpi=150)
    plt.close()
    print("  Saved pca_step.png")


# ---------------------------------------------------------------------------
# Experiment 2 — Linear Probe Accuracy
# ---------------------------------------------------------------------------

def experiment_linear_probe(records):
    print("\n--- Experiment 2: Linear Probes ---")
    H           = np.stack([r["h_t"]        for r in records])
    chain_labels = [r["chain_type"] for r in records]
    op_labels    = [r["op"]         for r in records]
    step_labels  = np.array([r["step_index"] for r in records], dtype=float)

    lines = [
        "================================",
        "LINEAR PROBE RESULTS",
        "================================",
    ]

    # --- Task 1: predict chain_type ---
    le_chain = LabelEncoder()
    y_chain  = le_chain.fit_transform(chain_labels)
    Xtr, Xte, ytr, yte = train_test_split(H, y_chain, test_size=0.2,
                                           random_state=42, stratify=y_chain)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(Xtr, ytr)
    y_pred    = clf.predict(Xte)
    acc_chain = accuracy_score(yte, y_pred) * 100
    cr_chain  = classification_report(yte, y_pred,
                                       target_names=le_chain.classes_, digits=3)
    print(f"  Chain type accuracy : {acc_chain:.1f}%  (random baseline: 16.7%)")
    lines += [
        "",
        "Chain type prediction:",
        f"  Accuracy : {acc_chain:.1f}%  (random baseline: 16.7%)",
        "",
        cr_chain,
    ]

    # --- Task 2: predict op ---
    le_op = LabelEncoder()
    y_op  = le_op.fit_transform(op_labels)
    Xtr, Xte, ytr, yte = train_test_split(H, y_op, test_size=0.2,
                                           random_state=42, stratify=y_op)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xte)
    acc_op = accuracy_score(yte, y_pred) * 100
    cr_op  = classification_report(yte, y_pred,
                                    target_names=le_op.classes_, digits=3)
    print(f"  Op accuracy         : {acc_op:.1f}%  (random baseline: 25.0%)")
    lines += [
        "Operation prediction:",
        f"  Accuracy : {acc_op:.1f}%  (random baseline: 25.0%)",
        "",
        cr_op,
    ]

    # --- Task 3: predict step_index ---
    Xtr, Xte, ytr, yte = train_test_split(H, step_labels, test_size=0.2,
                                           random_state=42)
    reg   = Ridge(alpha=1.0)
    reg.fit(Xtr, ytr)
    y_pred = reg.predict(Xte)
    r2    = r2_score(yte, y_pred)
    mae   = mean_absolute_error(yte, y_pred)
    print(f"  Step index R²       : {r2:.3f}   MAE: {mae:.3f}  (baseline R²: 0.0)")
    lines += [
        "Step index regression:",
        f"  R²  : {r2:.3f}  (random baseline: 0.0)",
        f"  MAE : {mae:.3f} steps",
        "",
    ]

    out_path = RESULTS_DIR / "linear_probe_report.txt"
    out_path.write_text("\n".join(lines))
    print(f"  Saved linear_probe_report.txt")


# ---------------------------------------------------------------------------
# Experiment 3 — Trajectory Drift
# ---------------------------------------------------------------------------

def experiment_trajectory_drift(traj_by_chain):
    print("\n--- Experiment 3: Trajectory Drift ---")
    MAX_SAMPLE = 50

    fig, ax = plt.subplots(figsize=(10, 7))

    for chain_name, color in CHAIN_COLORS.items():
        trajs  = traj_by_chain[chain_name]
        sample = trajs[:MAX_SAMPLE]

        # Bucket h_t vectors by step index
        step_vecs = defaultdict(list)
        for _, steps in sample:
            for step_idx, h_t in steps:
                step_vecs[step_idx].append(h_t)

        present = sorted(step_vecs.keys())
        if 0 not in step_vecs:
            continue

        mean_h = {s: np.mean(step_vecs[s], axis=0) for s in present}
        h0     = mean_h[0]

        drift_x = present
        drift_y = [float(np.linalg.norm(mean_h[s] - h0)) for s in present]

        ax.plot(drift_x, drift_y,
                color=color,
                label=chain_name.replace("_", " "),
                marker="o", linewidth=2, markersize=5)

    ax.set_xlabel("Step index")
    ax.set_ylabel("L2 distance from step-0 mean hidden state")
    ax.set_title("Hidden state drift across trajectory steps")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "trajectory_drift.png", dpi=150)
    plt.close()
    print("  Saved trajectory_drift.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)

    tok, model, device, trajectories = load_model_and_data()
    records, traj_by_chain = collect_hidden_states(tok, model, device, trajectories)

    experiment_pca(records)
    experiment_linear_probe(records)
    experiment_trajectory_drift(traj_by_chain)

    print(f"\nAll results saved to {RESULTS_DIR}/")
    print("Files:")
    for f in sorted(RESULTS_DIR.iterdir()):
        print(f"  {f.name}")
