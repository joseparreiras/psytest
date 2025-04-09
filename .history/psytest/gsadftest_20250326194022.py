from psytest.adftest import adfuller_stat, ADFtest, NumArray
from psytest.utils.functions import (
    r0_default,
    simulate_random_walks,
)
import numpy as np
from numpy import float64, apply_along_axis
from numba import njit, prange


# Generalized Sup ADF Test
@njit(parallel=True)
def gsadfuller_stat(y: NumArray, r0: int, kmax: int) -> float64:
    nobs: int = len(y)
    stat = float64(-np.inf)
    for r2 in prange(r0, nobs + 1):
        for r1 in prange(0, r2 - r0):
            stat = max(stat, adfuller_stat(y[r1:r2], kmax))
    return stat


@njit(parallel=True)
def gsadfuller_asymp_dist(y: NumArray, r0: int) -> float64:
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
    return float64(stat)


def gsadftest_dist(nobs: int, nreps: int, r0: int) -> NumArray:
    random_walks: NumArray = simulate_random_walks(nreps, nobs)
    return np.apply_along_axis(gsadfuller_asymp_dist, 1, random_walks, r0)


# Final class for the GSADF test ------------------------------


class GSADFtest(ADFtest):
    def __init__(self, y: NumArray, r0: int | None = None) -> None:
        """
        Generalized Sup ADF test class.

        Args:
            y (NumArray): Time series data
            r0 (int | None, optional): Initial period to start the test. Defaults to None.
        """
        super().__init__(y, gsadfuller_stat, gsadftest_dist)
        self.r0: int = r0 or r0_default(len(y))
