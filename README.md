# Symbolic World Model — A Recurrent Architecture for Procedural Mathematical Reasoning (v0)

> **Actively developed research project — paper in progress**

---

## Overview

This is a lightweight recurrent world model that treats symbolic mathematical expressions as world states and mathematical operations (expand, factor, differentiate, simplify) as state transitions. The core idea is to study whether structured latent state dynamics and recurrent memory can serve as a foundation for procedural reasoning in AI systems. This is not a calculator or LLM competitor — it is a controlled research testbed for studying how memory and reasoning emerge in world-model architectures, using symbolic math because ground truth is always verifiable.

---

## Key Results (v0)

| Metric | Result | Random Baseline |
|--------|--------|----------------|
| Op prediction accuracy (linear probe) | **100%** | 25.0% |
| Chain type accuracy (linear probe) | **88.2%** | 16.7% |
| Step index R² (Ridge regression) | **0.892** | 0.0 |
| Step index MAE | **0.345 steps** | — |

GRU hidden states form spatially distinct clusters per reasoning chain type — verified via PCA on ~40,000 collected hidden state vectors across 10,000 trajectories.

---

## Architecture

```
expr_string
     │
     ▼
[Expression Encoder]
  Embedding(vocab=34, dim=64) + PAD-masked mean pool
  → z_e ∈ R^64

     │ concat with
     ▼
[Operation Embedding]
  Embedding(4 ops, dim=32)
  → z_a ∈ R^32

     │ [z_e ‖ z_a] ∈ R^96
     ▼
[GRU Transition Model]
  2 layers, 128 hidden units
  → h_t ∈ R^128  (the "world state")

     ├──▶ [Latent Predictor Head]
     │      Linear(128,128) → ReLU → Linear(128,64)
     │      → z_pred ∈ R^64  (predicted next expression latent)
     │      Loss: MSE(z_pred, encoder(next_expr).detach())
     │
     └──▶ [Op Predictor Head]
            Linear(128, 4)
            → op_logits ∈ R^4  (predicted next operation)
            Loss: cross-entropy
```

- **213K parameters total**
- Trains in ~14 minutes on Apple M2 (MPS, float32)
- Vocabulary: 34 tokens (operators, variable x, integers −10 to 10, special tokens)

---

## Project Structure

```
symbolic-world-model/
├── CLAUDE.md               — project documentation and changelog
├── README.md               — this file
├── LICENSE                 — MIT license
├── config.py               — all hyperparameters in one place
├── train.py                — training loop (MPS-aware, foreground only)
├── probe.py                — latent space probing and PCA visualization
├── requirements.txt        — pinned dependencies
│
├── data/
│   ├── generator.py        — SymPy trajectory generator (structured op chains)
│   ├── tokenizer.py        — prefix (Polish notation) tokenizer
│   └── trajectories.json   — 10,000 generated trajectories (gitignored)
│
├── model/
│   ├── encoder.py          — ExprEncoder: embedding + mean pool → z_e
│   ├── transition.py       — GRUTransition: GRU carrying world state h_t
│   ├── heads.py            — LatentPredictorHead + OpPredictorHead
│   ├── world_model.py      — SymbolicWorldModel: full forward pass wrapper
│   └── test_model.py       — shape smoke test for forward pass
│
├── checkpoints/
│   └── model.pt            — trained model weights
│
└── probe_results/
    ├── pca_chain.png        — PCA colored by chain type
    ├── pca_op.png           — PCA colored by operation
    ├── pca_step.png         — PCA colored by trajectory step
    ├── trajectory_drift.png — L2 drift of hidden state per chain
    └── linear_probe_report.txt — full probe accuracy report
```

---

## How to Run

```bash
# 1. Setup
python -m venv worl_model_env
source worl_model_env/bin/activate
pip install -r requirements.txt

# 2. Generate structured trajectory data
python data/generator.py

# 3. Train the world model (runs in foreground — watch live loss)
python train.py

# 4. Probe the latent space
python probe.py
```

All outputs are saved automatically: checkpoint to `checkpoints/model.pt`, training curve to `training_curves.png`, probe results to `probe_results/`.

---

## Research Questions Being Explored

- Do recurrent hidden states spontaneously organize by reasoning type without explicit supervision on chain identity?
- Can symbolic math environments serve as compact, fully verifiable testbeds for studying procedural memory in neural networks?
- How does structured vs random trajectory design affect what information the model chooses to store in its hidden state?

---

## Status & Roadmap

**v0 — complete.** GRU baseline trained, latent space probed, linear probe results verified.

| Version | Focus | Status |
|---------|-------|--------|
| v0 | GRU world model on polynomial chains | ✅ Done |
| v1 | Replace GRU with Mamba (SSM) transition model | Planned |
| v2 | Stochastic latent states (VAE-style world model) | Planned |
| v3 | Multi-step imagination rollout without supervision | Planned |


---

## Author

**Sreeyukth Shankar**
Final-year B.Tech Computer Science, Christ University Bangalore

