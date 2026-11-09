import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from utils import signed_chirality_index, bootstrap_significance

def test_signed_chirality():
    assert signed_chirality_index(100, 0) == 1.0
    assert signed_chirality_index(0, 100) == -1.0
    assert signed_chirality_index(50, 50) == 0.0
    assert signed_chirality_index(0, 0) == 0.0

def test_bootstrap():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=0.5, scale=0.1, size=100)
    mean, se, ci = bootstrap_significance(data, n_bootstrap=500, random_state=0)
    assert ci[0] < mean < ci[1]
    assert se > 0
