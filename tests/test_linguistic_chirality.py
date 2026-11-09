import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import networkx as nx
from linguistic_chirality import tree_chirality_index, random_tree_null_model

def test_symmetric_tree_zero():
    G = nx.DiGraph()
    G.add_node(1)
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    assert abs(tree_chirality_index(G, root=1)) < 1e-12

def test_left_bias_positive():
    # root=2, dependents 1 (left) and 3 (right).
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3])
    G.add_edge(2, 1)
    G.add_edge(2, 3)
    tau = tree_chirality_index(G, root=2)
    assert tau > 0

def test_null_model_range():
    taus = random_tree_null_model(n_nodes=5, n_samples=100, random_state=42)
    assert np.all((taus >= -1) & (taus <= 1))
