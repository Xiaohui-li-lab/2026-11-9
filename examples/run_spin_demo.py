#!/usr/bin/env python3
"""Demo: chiral spin-network simulation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from spin_network import ChiralSpinNetwork, benchmark_ising_critical_temperature

def main():
    print("=" * 60)
    print("Chiral Spin-Network Demo")
    print("=" * 60)

    model = ChiralSpinNetwork(N=50, J=1.2, h=1e-3, kT=1.0, random_state=42)
    result = model.run(
        n_sweeps=50_000,
        sample_interval=50,
        burn_in=5_000,
        show_progress=True,
    )

    print(f"Final order parameter sigma = {result.final_sigma:.4f}")
    print(f"Binder cumulant U_L = {result.binder_cumulant:.4f}")
    print(f"Mean |sigma| after burn-in = {np.mean(np.abs(result.sigma_series)):.4f}")

    # Plot time series.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.times, result.sigma_series, lw=0.8)
    ax.axhline(0, color="black", ls="--", lw=0.5)
    ax.set_xlabel("Monte Carlo sweep")
    ax.set_ylabel(r"Chiral order parameter $\sigma$")
    ax.set_title("Chiral spin-network evolution")
    fig.tight_layout()
    fig.savefig("figures/spin_evolution.png", dpi=150)
    print("Saved figure to figures/spin_evolution.png")

    # Benchmark: critical temperature of 2D Ising.
    beta_c = benchmark_ising_critical_temperature()
    T_c = 1.0 / beta_c
    print(f"Exact 2D Ising critical temperature T_c = {T_c:.6f}")

    # Quick critical-temperature scan.
    print("\nScanning temperatures near T_c...")
    temps = np.linspace(1.8, 2.6, 9)
    _, mean_abs = model.estimate_critical_temperature(
        temperatures=temps, N=30, n_sweeps=10_000, h=0.0
    )
    for T, m in zip(temps, mean_abs):
        print(f"  T = {T:.3f}, <|sigma|> = {m:.4f}")

if __name__ == "__main__":
    main()
