#!/usr/bin/env python3
"""Demo: CMB chirality estimator on toy data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from cmb_chirality import (
    generate_isotropic_cmb_alm,
    chirality_statistic_Sl,
    monte_carlo_null_distribution,
    false_positive_rate,
    inject_chiral_signal,
)

def main():
    print("=" * 60)
    print("CMB Chirality Estimator Demo")
    print("=" * 60)

    lmax = 30
    rng = np.random.default_rng(42)

    # Isotropic realization.
    alm_E, alm_B = generate_isotropic_cmb_alm(lmax, rng=rng)
    ells, Sl_iso = chirality_statistic_Sl(alm_E, alm_B)

    # Inject a small chiral signal.
    _, alm_B_chiral = inject_chiral_signal(alm_E, alm_B, amplitude=0.3, l_peak=10, width=4)
    _, Sl_chiral = chirality_statistic_Sl(alm_E, alm_B_chiral)

    # Null distribution.
    ells_mc, Sl_all, null_max = monte_carlo_null_distribution(
        lmax=lmax, n_simulations=500, lmin=2, rng=np.random.default_rng(123)
    )

    fpr = false_positive_rate(Sl_chiral, null_max)
    print(f"False-positive rate for chiral realization: {fpr:.4f}")

    # Plot.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ells, Sl_iso, label="Isotropic", alpha=0.7)
    ax.plot(ells, Sl_chiral, label="Injected chiral signal", alpha=0.8)
    ax.axhline(0, color="black", ls="--", lw=0.5)

    # 95% confidence band from null.
    low = np.percentile(Sl_all, 2.5, axis=0)
    high = np.percentile(Sl_all, 97.5, axis=0)
    ax.fill_between(ells_mc, low, high, color="gray", alpha=0.2, label="95% isotropic band")

    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"$S_\ell$")
    ax.set_title("CMB chirality statistic")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/cmb_chirality_demo.png", dpi=150)
    print("Saved figure to figures/cmb_chirality_demo.png")

if __name__ == "__main__":
    main()
