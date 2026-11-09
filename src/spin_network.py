"""
Chiral spin-network simulation using a Metropolis algorithm.

The model is an Ising-like spin system with two chiral states (L=+1, R=-1).
A small external field h biases the system toward the left-handed state, and
same-chiral neighbors attract (J > 0).
"""

import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class SpinNetworkResult:
    """Container for spin-network simulation results."""
    lattice: np.ndarray
    sigma_series: np.ndarray
    times: np.ndarray
    binder_cumulant: Optional[float] = None
    final_sigma: Optional[float] = None
    final_energy: Optional[float] = None

class ChiralSpinNetwork:
    """
    2D square-lattice chiral spin network.

    Attributes
    ----------
    N : int
        Linear lattice size (total spins = N*N).
    J : float
        Same-chirality coupling strength (>0).
    h : float
        External left-handed bias field.
    kT : float
        Temperature in energy units.
    """

    def __init__(
        self,
        N: int = 100,
        J: float = 1.2,
        h: float = 1e-3,
        kT: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.N = N
        self.J = J
        self.h = h
        self.kT = kT
        self.rng = np.random.default_rng(random_state)

        # Initialize with a small left-handed bias consistent with h.
        p_left = 0.5 + 0.5 * np.tanh(h / kT)
        self.lattice = self.rng.choice(
            [1, -1], size=(N, N), p=[p_left, 1.0 - p_left]
        ).astype(np.int8)

    def _energy_delta(self, i: int, j: int) -> float:
        """
        Energy change for flipping spin at (i, j).

        E = -J * sum_{nn} delta(chi, chi_nn) - h * chi
          = -J/2 * sum_{nn} chi*chi_nn - h*chi   (up to constants)
        """
        s = self.lattice[i, j]
        neighbors = (
            self.lattice[(i + 1) % self.N, j]
            + self.lattice[(i - 1) % self.N, j]
            + self.lattice[i, (j + 1) % self.N]
            + self.lattice[i, (j - 1) % self.N]
        )
        # Delta E = 2 * s * (J * neighbors + h)
        return 2.0 * s * (self.J * neighbors + self.h)

    def step(self) -> None:
        """Perform one Metropolis sweep (N*N single-spin flip attempts)."""
        for _ in range(self.N * self.N):
            i = self.rng.integers(0, self.N)
            j = self.rng.integers(0, self.N)
            dE = self._energy_delta(i, j)
            if dE <= 0 or self.rng.random() < np.exp(-dE / self.kT):
                self.lattice[i, j] *= -1

    def order_parameter(self) -> float:
        """Compute the chiral order parameter sigma."""
        n_left = np.count_nonzero(self.lattice == 1)
        n_right = np.count_nonzero(self.lattice == -1)
        total = n_left + n_right
        if total == 0:
            return 0.0
        return float(n_left - n_right) / float(total)

    def total_energy(self) -> float:
        """Compute the total energy of the current configuration."""
        energy = 0.0
        for i in range(self.N):
            for j in range(self.N):
                s = self.lattice[i, j]
                neighbors = (
                    self.lattice[(i + 1) % self.N, j]
                    + self.lattice[(i - 1) % self.N, j]
                    + self.lattice[i, (j + 1) % self.N]
                    + self.lattice[i, (j - 1) % self.N]
                )
                energy += -0.5 * self.J * s * neighbors - self.h * s
        return energy

    def binder_cumulant(self) -> float:
        """
        Compute the Binder cumulant U_L = 1 - <m^4> / (3 <m^2>^2)
        for the current configuration.
        """
        m = float(np.mean(self.lattice))
        m2 = m * m
        m4 = m2 * m2
        if m2 < 1e-12:
            return 0.0
        return 1.0 - m4 / (3.0 * m2 * m2)

    def run(
        self,
        n_sweeps: int = 100_000,
        sample_interval: int = 100,
        burn_in: Optional[int] = None,
        show_progress: bool = True,
    ) -> SpinNetworkResult:
        """
        Run the Metropolis simulation.

        Parameters
        ----------
        n_sweeps : int
            Total number of Metropolis sweeps.
        sample_interval : int
            Measure order parameter every `sample_interval` sweeps.
        burn_in : int or None
            Number of initial sweeps to discard. Defaults to 10% of n_sweeps.
        show_progress : bool
            Whether to show a tqdm progress bar.

        Returns
        -------
        SpinNetworkResult
        """
        if burn_in is None:
            burn_in = n_sweeps // 10

        iterator = range(n_sweeps)
        if show_progress:
            iterator = tqdm(iterator, desc="Spin-network MC")

        sigma_list: List[float] = []
        time_list: List[int] = []

        for sweep in iterator:
            self.step()
            if sweep >= burn_in and sweep % sample_interval == 0:
                sigma_list.append(self.order_parameter())
                time_list.append(sweep)

        sigma_series = np.array(sigma_list)
        times = np.array(time_list)

        return SpinNetworkResult(
            lattice=self.lattice.copy(),
            sigma_series=sigma_series,
            times=times,
            binder_cumulant=self.binder_cumulant(),
            final_sigma=self.order_parameter(),
            final_energy=self.total_energy(),
        )

def benchmark_ising_critical_temperature() -> float:
    """
    Return the exact critical inverse temperature of the 2D Ising model.

    beta_c = ln(1 + sqrt(2)) / 2
    """
    return np.log(1.0 + np.sqrt(2.0)) / 2.0

def estimate_critical_temperature(
    temperatures: np.ndarray,
    N: int = 50,
    n_sweeps: int = 20_000,
    h: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the absolute value of the order parameter as a function of
    temperature. For h=0 the peak of the susceptibility locates T_c.

    Returns
    -------
    tuple of (temperatures, mean_abs_sigma)
    """
    mean_abs_sigma = np.empty_like(temperatures)

    for idx, T in enumerate(temperatures):
        model = ChiralSpinNetwork(N=N, J=1.0, h=h, kT=T, random_state=42 + idx)
        result = model.run(
            n_sweeps=n_sweeps,
            sample_interval=50,
            burn_in=n_sweeps // 5,
            show_progress=False,
        )
        mean_abs_sigma[idx] = np.mean(np.abs(result.sigma_series))

    return temperatures, mean_abs_sigma
