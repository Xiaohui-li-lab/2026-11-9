"""
CMB chirality estimator.

This module implements a simplified chirality statistic S_l based on the
correlation between E-mode and B-mode spherical-harmonic coefficients.

For full-sky analysis with real data, use HEALPix (healpy) and the Planck
public maps. The functions here provide the reference algorithm and can be
adapted to standard CMB pipelines.
"""

import numpy as np
from typing import Optional, Tuple

def generate_isotropic_cmb_alm(
    lmax: int, rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Gaussian isotropic E and B mode alm coefficients.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    rng : np.random.Generator or None

    Returns
    -------
    tuple
        (alm_E, alm_B) complex arrays of shape (lmax+1, 2*lmax+1).
    """
    if rng is None:
        rng = np.random.default_rng()

    alm_E = np.zeros((lmax + 1, 2 * lmax + 1), dtype=np.complex128)
    alm_B = np.zeros((lmax + 1, 2 * lmax + 1), dtype=np.complex128)

    for l in range(2, lmax + 1):
        for m in range(-l, l + 1):
            idx_m = m + lmax
            sigma = 1.0 / (l * (l + 1))  # toy power spectrum
            alm_E[l, idx_m] = rng.normal(0, sigma) + 1j * rng.normal(0, sigma)
            alm_B[l, idx_m] = rng.normal(0, sigma) + 1j * rng.normal(0, sigma)

    return alm_E, alm_B

def chirality_statistic_Sl(
    alm_E: np.ndarray, alm_B: np.ndarray, lmin: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the chirality statistic S_l = mean_m( (a_lm^E)* a_lm^B ) / sigma_l.

    Parameters
    ----------
    alm_E, alm_B : np.ndarray
        Complex spherical-harmonic coefficients indexed as [l, m+lmax].
    lmin : int
        Minimum multipole to include.

    Returns
    -------
    tuple
        (ells, Sl) arrays.
    """
    lmax = alm_E.shape[0] - 1
    ells = np.arange(lmin, lmax + 1)
    Sl = np.empty_like(ells, dtype=np.float64)

    for idx, l in enumerate(ells):
        m_idx_start = lmax - l
        m_idx_end = lmax + l + 1
        product = np.conj(alm_E[l, m_idx_start:m_idx_end]) * alm_B[
            l, m_idx_start:m_idx_end
        ]
        mean_val = np.mean(product)
        # Use sample standard deviation over m as noise estimate.
        sigma_l = np.std(product, ddof=1)
        if sigma_l == 0:
            sigma_l = 1.0
        Sl[idx] = np.real(mean_val) / sigma_l

    return ells, Sl

def monte_carlo_null_distribution(
    lmax: int,
    n_simulations: int = 1000,
    lmin: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an isotropic Monte Carlo null distribution for max |S_l|.

    Returns
    -------
    tuple
        (ells, Sl_all, max_abs_Sl_null) where Sl_all has shape
        (n_simulations, n_l).
    """
    if rng is None:
        rng = np.random.default_rng()

    ells = np.arange(lmin, lmax + 1)
    Sl_all = np.empty((n_simulations, len(ells)))

    for i in range(n_simulations):
        alm_E, alm_B = generate_isotropic_cmb_alm(lmax, rng=rng)
        _, Sl = chirality_statistic_Sl(alm_E, alm_B, lmin=lmin)
        Sl_all[i, :] = Sl

    max_abs_Sl_null = np.max(np.abs(Sl_all), axis=1)

    return ells, Sl_all, max_abs_Sl_null

def false_positive_rate(
    observed_Sl: np.ndarray, null_max_abs: np.ndarray
) -> float:
    """
    Compute the false-positive rate (one-sided) for the observed max |S_l|.
    """
    observed_max = np.max(np.abs(observed_Sl))
    return np.mean(null_max_abs >= observed_max)

def inject_chiral_signal(
    alm_E: np.ndarray,
    alm_B: np.ndarray,
    amplitude: float = 0.05,
    l_peak: int = 10,
    width: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject a toy chiral signal: B_l = B_l + amplitude * E_l * phase(l).
    Used only for validation and algorithm tests.
    """
    lmax = alm_E.shape[0] - 1
    alm_B_out = alm_B.copy()

    for l in range(2, lmax + 1):
        weight = amplitude * np.exp(-((l - l_peak) ** 2) / (2 * width ** 2))
        m_idx_start = lmax - l
        m_idx_end = lmax + l + 1
        alm_B_out[l, m_idx_start:m_idx_end] += (
            weight * 1j * alm_E[l, m_idx_start:m_idx_end]
        )

    return alm_E, alm_B_out
