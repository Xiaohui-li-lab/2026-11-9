"""
Linguistic chirality index based on Universal Dependencies trees.

For each sentence, a dependency tree is built. The signed tree-chirality
index tau quantifies a left-handed helical bias in the tree structure.
"""

import networkx as nx
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

def build_dependency_tree(sentence) -> nx.DiGraph:
    """
    Build a directed dependency tree from a conllu Sentence object.

    Parameters
    ----------
    sentence : conllu.models.Sentence
        Parsed sentence from the conllu library.

    Returns
    -------
    networkx.DiGraph
        Edges point from head to dependent.
    """
    G = nx.DiGraph()
    token_list = list(sentence)

    for token in token_list:
        if isinstance(token["id"], int):
            G.add_node(token["id"], form=token["form"], upos=token["upos"])

    for token in token_list:
        if isinstance(token["id"], int):
            head = token["head"]
            if head is not None and head != 0:
                G.add_edge(head, token["id"], deprel=token["deprel"])

    return G

def tree_chirality_index(tree: nx.DiGraph, root: int = 1) -> float:
    """
    Compute a signed chirality index for a dependency tree.

    We use a simple proxy: for each ordered pair of siblings (a, b) where
    a < b are dependents of the same head, count a left branch if a is to the
    left of the head and b to the right (a 'left-handed crossing'). This is
    intentionally a first-order estimator; more sophisticated geometric
    embeddings can be substituted.

    A more principled estimator: project the tree onto a line (token order),
    compute the signed area swept by edges when read left-to-right.

    Parameters
    ----------
    tree : networkx.DiGraph
    root : int
        Root node id.

    Returns
    -------
    float
        Signed chirality index in [-1, 1].
    """
    if tree.number_of_nodes() < 3:
        return 0.0

    # Group dependents by head.
    siblings = defaultdict(list)
    for head, dep in tree.edges():
        siblings[head].append(dep)

    left_count = 0
    right_count = 0

    for head, deps in siblings.items():
        if len(deps) < 2:
            continue
        sorted_deps = sorted(deps)
        for i in range(len(sorted_deps)):
            for j in range(i + 1, len(sorted_deps)):
                a, b = sorted_deps[i], sorted_deps[j]
                if a < head < b:
                    left_count += 1
                elif b < head < a:
                    right_count += 1

    total = left_count + right_count
    if total == 0:
        return 0.0
    return float(left_count - right_count) / float(total)

def sentence_tau(sentence) -> Optional[float]:
    """Compute tau for a single conllu sentence."""
    try:
        tree = build_dependency_tree(sentence)
        if tree.number_of_nodes() == 0:
            return None
        return tree_chirality_index(tree)
    except Exception:
        return None

def corpus_tau(
    sentences: List, min_length: int = 3
) -> Tuple[float, float, int]:
    """
    Compute mean linguistic chirality index across a corpus.

    Parameters
    ----------
    sentences : list of conllu.models.Sentence
    min_length : int
        Minimum number of tokens per sentence.

    Returns
    -------
    tuple
        (mean_tau, std_tau, n_sentences)
    """
    values = []
    for sent in sentences:
        if len(sent) < min_length:
            continue
        tau = sentence_tau(sent)
        if tau is not None:
            values.append(tau)

    values = np.array(values)
    if len(values) == 0:
        return 0.0, 0.0, 0
    return float(np.mean(values)), float(np.std(values, ddof=1)), len(values)

def random_tree_null_model(
    n_nodes: int, n_samples: int = 1000, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate random rooted trees with n_nodes and compute their tau values.

    Used for permutation testing.
    """
    if rng is None:
        rng = np.random.default_rng()

    taus = np.empty(n_samples)
    for i in range(n_samples):
        # Random recursive tree.
        G = nx.DiGraph()
        G.add_node(1)
        for new_node in range(2, n_nodes + 1):
            parent = rng.integers(1, new_node)
            G.add_edge(int(parent), new_node)
        taus[i] = tree_chirality_index(G, root=1)

    return taus

def permutation_p_value(
    observed_tau: float, n_nodes: int, n_permutations: int = 1000
) -> float:
    """
    Estimate a one-sided p-value for observed tau against random trees.
    """
    null_taus = random_tree_null_model(n_nodes, n_samples=n_permutations)
    if observed_tau >= 0:
        return np.mean(null_taus >= observed_tau)
    else:
        return np.mean(null_taus <= observed_tau)
