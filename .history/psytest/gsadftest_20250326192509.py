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

KMAX = 0
# Functions for the different ADF tests ----------------------------------------


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
def gsadfuller_asymp_dist(y: np.ndarray, r0: int) -> np.float64:
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


def gsadftest_dist(nobs: int, nreps: int, r0: int) -> np.ndarray:
    random_walks: np.ndarray = simulate_random_walks(nreps, nobs)
    return np.apply_along_axis(gsadfuller_asymp_dist, 1, random_walks, r0)


# Final class for the GSADF test -----------------------------------------------


class GSADFtest(ADFtest):
    def __init__(self, y: np.ndarray, r0: int | None = None):
        super().__init__(y, gsadfuller_stat, gsadftest_dist)
        self.r0: int = r0 or r0_default(len(y))
