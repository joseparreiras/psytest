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

KMAX = 0
# Functions for the different ADF tests ----------------------------------------


# Sup ADF Test


@njit(parallel=True)
def sadfuller_stat(y: np.ndarray, r0: int, kmax: int) -> np.float64:
    nobs: int = len(y)
    stat: np.float64 = np.float64(-np.inf)
    for r in prange(r0, nobs + 1):
        stat = max(stat, adfuller_stat(y[:r], kmax))
    return stat


# Generalized Sup ADF Test
@njit(parallel=True)
def gsadfuller_stat(y: np.ndarray, r0: int, kmax: int) -> np.float64:
    nobs: int = len(y)
    stat = np.float64(-np.inf)
    for r2 in prange(r0, nobs + 1):
        for r1 in prange(0, r2 - r0):
            stat = max(stat, adfuller_stat(y[r1:r2], kmax))
    return stat


@njit(parallel=True)
def sim_gsadf_stat(y: np.ndarray, r0: int) -> np.float64:
    nobs: int = len(y)
    stat: float = 0.0
    for r2 in prange(r0, nobs):
        for r1 in prange(r0, r2 - r0):
            rw: int = r2 - r1
            num: float = 1 / 2 * rw * (y[r2] ** 2 - y[r1] ** 2 - rw) - sum(y[r1:r2]) * (
                y[r2] - y[r1]
            )
            den: float = (
                rw**0.5 * (rw * sum(y[r1:r2] ** 2) - (sum(y[r1:r2])) ** 2) ** 0.5
            )
            stat = max(stat, num / den)
    return np.float64(stat)


# Backward sup ADF Test
@njit(parallel = True)
def bsadfuller_stat(y: np.ndarray, r0: int, kmax: int) -> np.float64:
    nobs: int = len(y)
    r2_grid: np.arange(r0, nobs + 1)
    bsadf_stat: np.ndarray = np.repeat(-np.inf, len(r2_grid))
    

# Final class for the GSADF test -----------------------------------------------


class GSADFuller:
    def __init__(self, y: np.ndarray, kmax: int = KMAX):
        self.y: np.ndarray = y
        self.nobs: int = len(y)
        self.kmax: int = kmax

    def teststat(self, r0: int | None = None) -> np.float64:
        r0 = r0 or r0_default(self.nobs)
        if not hasattr(self, "_teststat"):
            self._teststat: np.float64 = gsadfuller_stat(self.y, r0, self.kmax)
            self._r0: int = r0
        return self._teststat

    def critval(
        self, alpha_list: np.ndarray = np.array([0.1, 0.05, 0.01]), nreps: int = 1000
    ) -> np.ndarray:
        if not hasattr(self, "_critval"):
            random_walks: np.ndarray = simulate_random_walks(nreps, self.nobs)
            adf_dist: np.ndarray = np.apply_along_axis(
                sim_gsadf_stat, 1, random_walks, self._r0
            )
            self._nreps: int = nreps
            self._alpha: np.ndarray = alpha_list
            self._critval: np.ndarray = np.quantile(adf_dist, 1 - alpha_list)
        return self._critval
