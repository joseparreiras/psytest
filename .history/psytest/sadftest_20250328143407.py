from psytest.adftest import (
    rolling_adfuller_stat,
    rolling_adfuller_cdf,
)
from psytest.utils.functions import random_walk
from psytest.utils.constants import KMAX
from numpy.typing import NDArray
from numpy import float64, inf, zeros, repeat, int64, arange
from numba import njit, prange


# Sup ADF Test
@njit(parallel=True)
def sadfuller_stat(y: NDArray[float64], r0: int, kmax: int) -> float64:
    """
    Calculates the test statistics for the Sup Augmented Dickey-Fuller test.

    Args:
        y (NDArray[float64]): The time series data.
        r0 (int): Initial period to start the test.
        kmax (int): Maximum lag to use in the test.

    Returns:
        float64: The test statistics.
    """
    nobs: int = len(y)
    stat: float64 = float64(-inf)
    for r in prange(r0, nobs + 1):
        stat = max(stat, rolling_adfuller_stat(y, 0, r, kmax))
    return stat


def sadfuller_dist(nobs: int64, nreps: int64, r0: int64) -> NDArray[float64]:
    # Simulate random walks
    rw: NDArray[float64] = random_walk(int(nreps), int(nobs))
    # Make tuple of (j, r2, r1) combinations
    iterator: NDArray[int64] = __jr2r1_combinations__(nobs, nreps, r0)
    # Calculate statistic for each j
    stat: tuple[NDArray[int64], NDArray[float64]] = (
        __sadfuller_dist_from_random_walks__(rw, iterator, r0)
    )
    return stat


@njit(parallel=False)
def __jr2r1_combinations__(nobs: int64, nreps: int64, r0: int64) -> NDArray[int64]:
    total: int64 = nreps * (nobs - r0) * (nobs - r0)
    result: NDArray[int64] = zeros((total, 3), dtype=int64)

    idx: int = 0
    for n in range(0, nreps):
        for r1 in range(nobs - r0 + 1):
            for r2 in range(r0, nobs + 1):
                result[idx, 0] = n
                result[idx, 1] = r1
                result[idx, 2] = r2
                idx += 1

    return result


@njit(parallel=True)
def __sadfuller_dist_from_random_walks__(
    random_walks: NDArray[float64], iterable: NDArray[int64], r0: int64
) -> tuple[NDArray[int64], NDArray[float64]]:
    nreps: int = len(random_walks)
    nobs: int = len(random_walks[0])
    stats: NDArray[float64] = repeat(-inf, nreps * (nobs - r0))
    stats = stats.reshape((nreps, nobs - r0))
    niter: int = len(iterable)
    for i in prange(niter):
        next_iter: NDArray[int64] = iterable[i]
        j: int64 = next_iter[0]
        r1: int64 = next_iter[1]
        r2: int64 = next_iter[2]
        if r1 <= r2 - r0:
            stats[j, r2 - r0] = max(
                stats[j, r2 - r0], rolling_adfuller_cdf(random_walks[j], r1, r2)
            )
    return arange(nobs - r0), stats


# Backward Sup ADF Test


def bsadf_stat(y: NDArray[float64], r0: int, r2: int, kmax: int) -> float64:
    """
    Calculates the Backward Sup Augmented Dickey-Fuller statistics.

    Args:
        y: (NDArray[float64]): The time series data.
        r0 (int): Initial period to start the test.
        r2 (int): Final period for the test. Where to evaluate the BSADF.
        kmax (int): Maximum lag to use in the test.

    Returns:
        float64: The test statistic at `r2`
    """
    stat: float64 = float64(-inf)
    for r1 in prange(r2 - r0 + 1):
        stat = max(stat, rolling_adfuller_stat(y, r1, r2, kmax))
    return stat


def bsadf_stat_all_series(
    y: NDArray[float64], r0: int, r2_grid: NDArray[int64], kmax: int
) -> NDArray[int64]:
    """
    Calculates the Backward Sup Augmented Dickey-Fuller statistics optimized for a grid of `r2`.

    Args:
        y: (NDArray[float64]): The time series data.
        r0 (int): Initial period to start the test
        r2_grid (NDArray[int64]): The grid to evaluate the test stat.
        kmax (int): Maximum lag to use in the test.

    Returns:
        NDArray[int64]: The test statistic for each element in `r2_grid`
    """
    nobs: int = len(y)
    r2r1_grid: NDArray[int64] = __r2r1_combinations__(nobs, r0)
    stat: NDArray[float64] = repeat(float64(-inf), nobs-r0+1)


@njit(parallel=True)
def __r2r1_combinations__(nobs: int64, r0: int64) -> NDArray[int64]:
    """
    Calculates all the combinations of (r1, r2) for the calcuation of the BSADF test. This function does NOT filter for r1 <= r2 - r0.

    Args:
        nobs (int64): Number of observations in the sample (maximum value of r2)
        r0 (int64): Minimum window size (minimum value for r2)

    Returns:
        NDArray[int64]: Array of the combinations of (r1, r2)
    """
    total: int64 = (nobs - r0) ** 2
    result: NDArray[int64] = zeros(shape=(total, 2), dtype=int64)
    idx: int = 0
    for r1 in prange(nobs - r0 + 1):
        for r2 in range(r0, nobs + 1):
            result[idx, 0] = r1
            result[idx, 1] = r2
            idx += 1

    return result
