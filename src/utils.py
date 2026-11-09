"""
Common utilities for chirality computations.
"""

import numpy as np

def signed_chirality_index(n_left: int, n_right: int) -> float:
    """
    Compute the signed chiral order parameter.

    sigma = (N_L - N_R) / (N_L + N_R)

    Parameters
    ----------
    n_left, n_right : int
        Counts of left/right chiral modes.

    Returns
    -------
    float
        Chirality index in [-1, 1]. Returns 0 if total count is zero.
    """
    total = n_left + n_right
    if total == 0:
        return 0.0
    return float(n_left - n_right) / float(total)

def bootstrap_significance(values, n_bootstrap=1000, random_state=None):
    """
    Estimate the standard error of a statistic by bootstrapping.

    Parameters
    ----------
    values : array-like
        Sample values.
    n_bootstrap : int
        Number of bootstrap resamples.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    tuple
        (mean, std_error, 95% confidence interval)
    """
    rng = np.random.default_rng(random_state)
    values = np.asarray(values)
    n = len(values)
    boot_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    mean = np.mean(values)
    std_err = np.std(boot_means, ddof=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return mean, std_err, (ci_low, ci_high)
