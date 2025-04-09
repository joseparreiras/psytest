from psytest.hyptest import HypothesisTest
from psytest.adftest import adfuller_stat
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


class SADFuller(HypothesisTest):
    def __init__(self, y: np.ndarray) -> None:
        super().__init__(y, sadfuller_stat, simulate_random
        self.y: np.ndarray = y
        self.nobs: int = len(y)
    def __init__(self, y: np.ndarray, kmax: int = KMAX) -> None:
        self.y: np.ndarray = y
        self.nobs: int = len(y)

    def teststat(self, r0: int | None = None) -> np.float64:
        if not hasattr(self, "_teststat"):
            r0 = r0 or r0_default(self.nobs)
            self._teststat: np.float64 = sadfuller_stat(self.y, r0, KMAX)
            self._r0: int = r0
        return self._teststat
    
    def critval(self, alpha_list: np.ndarray = TEST_SIZE, nreps: int = 1000) -> np.ndarray:
        if not hasattr(self, "_simdist") or len(self._simdist) != nreps:
            self._simdist: np.ndarray = simulate_random_walks(self.nobs, nreps)
        return np.quantile(self._simdist, alpha_list)
