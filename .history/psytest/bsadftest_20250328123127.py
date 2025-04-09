from psytest.adftest import rolling_adfuller_stat
from numpy.typing import NDArray
from numpy import float64, inf, empty, int64, repeat
from numba import njit, prange


@njit(parallel=True)
def calc_bsadfuller_stat(y: NDArray[float64], r0: int, r2: int, kmax: int) -> float64:
    """
    Calculates the test statistics for the Backward Sup Augmented Dickey-Fuller test for a given value of `r2`.

    Args:
        y (NDArray[float64]): The time series data.
        r0 (int): Initial period to start the test.
        r2 (int): The final period to end the test.
        kmax (int): Maximum lag to use in the test.

    Returns:
        float64: The test statistics.
    """
    stat: float64 = float64(-inf)
    for r1 in prange(0, r2 - r0 + 1):
        stat = max(stat, rolling_adfuller_stat(y, r1, r2, kmax))
    return stat


# Backward Sup ADF Test
@njit(parallel=True)
def bsadfuller_stat(y: NDArray[float64], r0: int, kmax: int) -> NDArray[float64]:
    """
    Calculates the test statistics for the Backward Sup Augmented Dickey-Fuller test.

    Args:
        y (NDArray[float64]): The time series data.
        r0 (int): Initial period to start the test.
        r2_grid (NDArray[float64]): The grid of final periods to end the test.
        kmax (int): Maximum lag to use in the test.

    Returns:
        NDArray[float64]: The test statistics.
    """
    nobs: int = len(y)
    r2_size: int = nobs - r0 + 1
    r2r1zip: NDArray[int64] = __combinations_r2r1__(r2_size)
    stat: NDArray[float64] = repeat(-inf, r2_size)
    for i in prange(len(r2r1zip)):
        r2: int = r2r1zip[i, 0]
        r1: int = r2r1zip[i, 1]
        if r1 <= r2 - r0:
            stat[i] = max(stat[i], rolling_adfuller_stat(y, r1, r2, kmax))

    for i in prange(nr2):
        r2: int = r2_grid[i]
        stat[i] = calc_bsadfuller_stat(y, r0, r2, kmax)

    return stat


@njit(parallel=True)
def __combinations_r2r1__(rmax) -> NDArray[int64]:
    total: int64 = rmax**2
    result: NDArray[int64] = empty((total, 2), dtype=int64)
    idx = 0
    for r2 in prange(0, rmax + 1):
        for r1 in range(0, rmax + 1):
            result[idx, 0] = r2
            result[idx, 1] = r1
            idx += 1
    return result
