#!/usr/bin/env python3
"""Demo: linguistic chirality index on a tiny synthetic corpus."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from linguistic_chirality import (
    corpus_tau,
    random_tree_null_model,
    permutation_p_value,
)
from utils import signed_chirality_index

class FakeSentence:
    """Minimal conllu-like sentence for demonstration."""
    def __init__(self, tokens):
        self.tokens = tokens

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

def make_token(idx, head, form="x"):
    return {"id": idx, "head": head, "form": form, "upos": "NOUN"}

def main():
    print("=" * 60)
    print("Linguistic Chirality Index Demo")
    print("=" * 60)

    # Construct two simple dependency trees.
    # Tree 1: root=1, dependents 2 and 3 on the right.
    sent1 = FakeSentence([
        make_token(1, 0), make_token(2, 1), make_token(3, 1),
    ])

    # Tree 2: root=2, dependents 1 (left) and 3 (right) -> left-handed crossing.
    sent2 = FakeSentence([
        make_token(1, 2), make_token(2, 0), make_token(3, 2),
    ])

    mean_tau, std_tau, n = corpus_tau([sent1, sent2])
    print(f"Mean tau = {mean_tau:.4f}, std = {std_tau:.4f}, n = {n}")

    # Null model.
    null_taus = random_tree_null_model(n_nodes=3, n_samples=1000)
    print(f"Null model: mean = {np.mean(null_taus):.4f}, std = {np.std(null_taus):.4f}")

    # Permutation p-value.
    p = permutation_p_value(mean_tau, n_nodes=3, n_permutations=1000)
    print(f"Permutation p-value = {p:.4f}")

    # Utility demonstration.
    print("\nSigned chirality index examples:")
    print(f"  (60, 40) -> {signed_chirality_index(60, 40):.3f}")
    print(f"  (40, 60) -> {signed_chirality_index(40, 60):.3f}")

if __name__ == "__main__":
    main()
