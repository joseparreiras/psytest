from psytest.adftest import adfuller_stat, NumArray, KMAX, TEST_SIZE, NREPS, corridor_adfuller_stat
from psytest.utils.functions import (
    r0_default,
    simulate_random_walks,
)
from numpy.typing import NDArray
from numpy import (
    float64,
    inf,
    apply_along_axis,
    quantile,
    arange,
    repeat,
    int64,
    array,
    repeat,
)
from numba import njit, prange
from typing import Any


@njit(parallel=True)
def calc_bsadfuller_stat(y: NumArray, r0: int, r2: int, kmax: int) -> float64:
    """
    Calculates the test statistics for the Backward Sup Augmented Dickey-Fuller test for a given value of `r2`.

    Args:
        y (NumArray): The time series data.
        r0 (int): Initial period to start the test.
        r2 (int): The final period to end the test.
        kmax (int): Maximum lag to use in the test.

    Returns:
        float64: The test statistics.
    """
    stat: float64 = float64(-inf)
    for r1 in prange(0, r2 - r0 + 1):
        stat = max(stat, corridor_adfuller_stat(y, r1, r2, kmax))
    return stat


# Backward Sup ADF Test
def basdfuller_stat(y: NumArray, r0: int, r2_grid: NumArray, kmax: int) -> tuple[NDArray[int64], NumArray]:
    """
    Calculates the test statistics for the Backward Sup Augmented Dickey-Fuller test.

    Args:
        y (NumArray): The time series data.
        r0 (int): Initial period to start the test.
        kmax (int): Maximum lag to use in the test.

    Returns:
        NumArray: The test statistics.
    """
    nobs: int = len(y)
    nr2: int = len(r2_grid)
    stat: NumArray = repeat(-inf, nr2)
    for i in prange(nr2):
        r2: int = r2_grid[i]
        stat[i] = calc_bsadfuller_stat(y, r0, r2, kmax)

    return (r2_grid, stat)

def bsadfuller_asymp_dist(w: NumArray, r0: int, r2_grid: NDArray[int64]):
    