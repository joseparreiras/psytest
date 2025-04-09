from psytest.adftest import corridor_adfuller_stat
from numpy.typing import NDArray
from numpy import float64, inf, empty
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
        stat = max(stat, corridor_adfuller_stat(y, r1, r2, kmax))
    return stat


# Backward Sup ADF Test
@njit(parallel=True)
def bsadfuller_stat(
    y: NDArray[float64], r0: int, r2_grid: NDArray[float64], kmax: int
) -> NDArray[float64]:
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
    nr2: int = len(r2_grid)
    stat: NDArray[float64] = empty(nr2)
    for i in prange(nr2):
        r2: int = r2_grid[i]
        stat[i] = calc_bsadfuller_stat(y, r0, r2, kmax)

    return stat
