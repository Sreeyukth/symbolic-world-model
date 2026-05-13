"""
Prefix (Polish notation) tokenizer for SymPy expressions.
Converts expr strings to fixed-length padded integer tensors and back.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sympy as sp
from sympy import Symbol, Integer, Add, Mul, Pow, Rational
import torch

# Special tokens
PAD = "<PAD>"
UNK = "<UNK>"
SOS = "<SOS>"
EOS = "<EOS>"

# Operator token names
_OPERATORS = ["+", "-", "*", "**", "/", "sin", "cos", "log"]
_VARIABLES = ["x"]
_INTEGERS = [str(i) for i in range(-10, 11)]  # -10 to 10
_SPECIAL = [PAD, UNK, SOS, EOS]

_VOCAB_TOKENS = _SPECIAL + _OPERATORS + _VARIABLES + _INTEGERS

MAX_SEQ_LEN = 32
x = Symbol("x")


def _expr_to_prefix(expr) -> list[str]:
    """Recursively convert a SymPy expression to prefix token list."""
    if isinstance(expr, Integer):
        val = int(expr)
        return [str(val) if -10 <= val <= 10 else UNK]

    if isinstance(expr, Rational):
        # Represent as (numerator / denominator) in prefix: / num den
        return ["/"] + _expr_to_prefix(expr.p) + _expr_to_prefix(expr.q)

    if isinstance(expr, Symbol):
        return [str(expr)]

    if isinstance(expr, Add):
        args = expr.args
        # left-fold: + a (+ b c) for 3 terms
        result = _expr_to_prefix(args[-1])
        for arg in reversed(args[:-1]):
            result = ["+"] + _expr_to_prefix(arg) + result
        return result

    if isinstance(expr, Mul):
        args = expr.args
        result = _expr_to_prefix(args[-1])
        for arg in reversed(args[:-1]):
            result = ["*"] + _expr_to_prefix(arg) + result
        return result

    if isinstance(expr, Pow):
        base, exp = expr.args
        return ["**"] + _expr_to_prefix(base) + _expr_to_prefix(exp)

    # SymPy function nodes (sin, cos, log, etc.)
    name = type(expr).__name__.lower()
    if len(expr.args) == 1:
        return [name] + _expr_to_prefix(expr.args[0])

    # Fallback: unknown structure
    return [UNK]


class Tokenizer:
    def __init__(self):
        self.vocab = _VOCAB_TOKENS
        self.token2id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id2token = {i: tok for i, tok in enumerate(self.vocab)}
        self.pad_id = self.token2id[PAD]
        self.unk_id = self.token2id[UNK]
        self.sos_id = self.token2id[SOS]
        self.eos_id = self.token2id[EOS]
        self.max_len = MAX_SEQ_LEN

        # Write vocab size back to config
        import config
        config.VOCAB_SIZE = len(self.vocab)

        print(f"Tokenizer initialized. Vocab size: {len(self.vocab)}")

    def encode(self, expr_str: str) -> torch.Tensor:
        """Convert an expression string to a padded integer tensor of length MAX_SEQ_LEN."""
        try:
            expr = sp.sympify(expr_str)
            tokens = [SOS] + _expr_to_prefix(expr) + [EOS]
        except Exception:
            tokens = [SOS, UNK, EOS]

        ids = [self.token2id.get(t, self.unk_id) for t in tokens]

        # Truncate if needed (leaves EOS intact by truncating interior)
        if len(ids) > self.max_len:
            ids = ids[:self.max_len - 1] + [self.eos_id]

        # Pad to max_len
        ids += [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Convert a padded integer tensor back to a token string."""
        tokens = []
        for idx in token_ids.tolist():
            tok = self.id2token.get(idx, UNK)
            if tok == EOS:
                break
            if tok in (SOS, PAD):
                continue
            tokens.append(tok)
        return " ".join(tokens)


if __name__ == "__main__":
    tok = Tokenizer()
    test_exprs = ["x**2 + 3*x - 5", "2*x**3 - x + 1", "x*(x + 2)", "-4"]
    for expr_str in test_exprs:
        ids = tok.encode(expr_str)
        decoded = tok.decode(ids)
        print(f"  expr : {expr_str}")
        print(f"  ids  : {ids[:10].tolist()}...")
        print(f"  back : {decoded}")
        print()
