from psytest.adftest import adfuller_stat, ADFtest
from psytest.utils.functions import (
    r0_default,
    index_combinations,
    parallel_apply,
    simulate_random_walks,
)
from collections.abc import Callable
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from numba import njit, prange

KMAX = 0
NREPS = 1000
TEST_SIZE: NDArray[np.float64] = np.array([0.1, 0.05, 0.01])


# Sup ADF Test
@njit(parallel=True)
def sadfuller_stat(y: np.ndarray, r0: int, kmax: int) -> np.float64:
    nobs: int = len(y)
    stat: np.float64 = np.float64(-np.inf)
    for r in prange(r0, nobs + 1):
        stat = max(stat, adfuller_stat(y[:r], kmax))
    return stat


@njit(parallel=True)
def sadfuller_asymp_dist(w: np.ndarray, r0: int) -> np.float64:
    nobs: int = len(w)
    stat: float = 0.0
    for r2 in prange(r0, nobs):
        for r1 in prange(r0, r2 - r0):
            rw: int = r2 - r1
            num: float = 1 / 2 * rw * (w[r2] ** 2 - w[r1] ** 2 - rw) - sum(w[r1:r2]) * (
                w[r2] - w[r1]
            )
            den: float = (
                rw**0.5 * (rw * sum(w[r1:r2] ** 2) - (sum(w[r1:r2])) ** 2) ** 0.5
            )
            stat = max(stat, num / den)
    return np.float64(stat)


def sadfuller_dist(nobs: int, nreps: int, r0: int) -> np.ndarray:
    random_walks: np.ndarray = simulate_random_walks(nreps, nobs)
    return np.apply_along_axis(sadfuller_asymp_dist, 1, random_walks, r0)


class SADFtest(ADFtest):
    def __init__(self, y: np.ndarray, r0: int | None = None) -> None:
        super().__init__(y, sadfuller_stat, sadfuller_dist)
        self.r0 = r0 or r0_default(len(y))

    def teststat(self, r0: int | None):
        """
        Calculates the test_statistics for the hypothesis test.

        Args:
            r0 (int | None, optional): The initial period to start evaluating the test. Defaults to None.

        Returns:
            np.float64: The test statistics.
        """
        return super().teststat(r0=r0 or r0_default(self.nobs))

    def critval(self, nreps: int = NREPS) -> np.ndarray:
        """
        Calculates the critical values for the hypothesis test.

        Args:
            nreps (int, optional): Number of simulations to perform for the Monte Carlo. Defaults to 1000.

        Returns:
            np.ndarray: Critical values, on the order of the test sizes.
        """
        return super().critval(nreps=nreps)
