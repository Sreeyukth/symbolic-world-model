"""
Generates structured symbolic math trajectories using SymPy.
Each trajectory follows one mathematically meaningful operation chain.
Saves 10,000 trajectories to data/trajectories.json.

Trajectory format:
{
    "chain_type": "calculus_chain",
    "steps": [
        {"expr": "...", "op": "DIFF", "next_expr": "..."},
        ...
    ]
}
"""

import json
import random
from pathlib import Path

import sympy as sp
from sympy import expand, factor, diff, simplify, Symbol

x = Symbol("x")

# Structured operation chains — each represents a meaningful math workflow
CHAINS = {
    "simplification_chain": ["EXPAND", "SIMPLIFY", "FACTOR"],
    "calculus_chain":        ["DIFF",   "SIMPLIFY", "DIFF"],
    "expand_diff_chain":     ["EXPAND", "DIFF",     "SIMPLIFY"],
    "factor_expand_cycle":   ["FACTOR", "EXPAND",   "SIMPLIFY"],
    "repeated_diff":         ["DIFF",   "DIFF",     "SIMPLIFY"],
    "expand_chain":          ["EXPAND", "EXPAND",   "SIMPLIFY"],
}
CHAIN_NAMES = list(CHAINS.keys())

DEGREE_RANGE = (1, 4)
COEFF_RANGE  = list(range(-5, 6))
COEFF_RANGE.remove(0)

TRAJ_LEN_RANGE = (3, 6)
N_TRAJECTORIES = 10_000
OUTPUT_PATH    = Path(__file__).parent / "trajectories.json"


def random_polynomial() -> sp.Expr:
    degree = random.randint(*DEGREE_RANGE)
    coeffs = [random.choice(COEFF_RANGE) for _ in range(degree + 1)]
    return sum(c * x**i for i, c in enumerate(coeffs))


def apply_op(expr: sp.Expr, op: str) -> sp.Expr:
    if op == "EXPAND":
        return expand(expr)
    if op == "FACTOR":
        return factor(expr)
    if op == "DIFF":
        return diff(expr, x)
    if op == "SIMPLIFY":
        return simplify(expr)
    raise ValueError(f"Unknown op: {op}")


def build_trajectory(length: int) -> dict | None:
    chain_name = random.choice(CHAIN_NAMES)
    op_sequence = CHAINS[chain_name]

    # Trim or cycle the chain to the requested length
    ops = [op_sequence[i % len(op_sequence)] for i in range(length)]

    expr = random_polynomial()
    steps = []

    for op in ops:
        next_expr = apply_op(expr, op)
        # Keep degenerate (no-change) steps — they are part of the chain structure.
        # Only discard if the starting expression itself is degenerate (constant 0).
        steps.append({
            "expr":      str(expr),
            "op":        op,
            "next_expr": str(next_expr),
        })
        expr = next_expr

    return {"chain_type": chain_name, "steps": steps}


def generate(n: int = N_TRAJECTORIES, seed: int = 42) -> list:
    random.seed(seed)
    trajectories = []
    attempts     = 0
    max_attempts = 100_000

    while len(trajectories) < n and attempts < max_attempts:
        attempts += 1
        length = random.randint(*TRAJ_LEN_RANGE)
        traj   = build_trajectory(length)
        if traj is not None:
            trajectories.append(traj)

        if len(trajectories) % 5_000 == 0 and len(trajectories) > 0:
            print(f"  {len(trajectories):,} / {n:,} trajectories generated "
                  f"({attempts:,} attempts)")

    if len(trajectories) < n:
        print(f"Warning: only generated {len(trajectories):,} trajectories "
              f"after {attempts:,} attempts.")

    return trajectories


if __name__ == "__main__":
    print(f"Generating {N_TRAJECTORIES:,} structured trajectories...")
    data = generate()

    # Print chain type distribution
    from collections import Counter
    counts = Counter(t["chain_type"] for t in data)
    print("\nChain type distribution:")
    for name, count in sorted(counts.items()):
        print(f"  {name:<25} {count:>6,}  ({count/len(data)*100:.1f}%)")

    print(f"\nDone. Saving to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(data):,} trajectories to {OUTPUT_PATH}.")
