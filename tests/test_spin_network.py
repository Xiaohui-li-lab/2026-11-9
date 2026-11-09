import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from spin_network import ChiralSpinNetwork, benchmark_ising_critical_temperature

def test_order_parameter_range():
    model = ChiralSpinNetwork(N=10, J=1.0, h=0.0, kT=1.0, random_state=0)
    sigma = model.order_parameter()
    assert -1.0 <= sigma <= 1.0

def test_ising_critical_temperature():
    beta_c = benchmark_ising_critical_temperature()
    assert abs(beta_c - np.log(1 + np.sqrt(2)) / 2) < 1e-12

def test_run_produces_series():
    model = ChiralSpinNetwork(N=10, J=1.0, h=0.1, kT=1.0, random_state=1)
    result = model.run(n_sweeps=1000, sample_interval=50, burn_in=100, show_progress=False)
    assert len(result.sigma_series) > 0
    assert len(result.times) == len(result.sigma_series)
