from psytest.adftest import (
    adfuller_stat,
    NumArray,
    KMAX,
    TEST_SIZE,
    NREPS,
    corridor_adfuller_stat,
    corridor_adfuller_cdf,
)
from psytest.utils.functions import r0_default, random_walk
from numpy.typing import NDArray
from numpy import float64, inf, quantile, zeros, repeat, int64
from numba import njit, prange


# Sup ADF Test
@njit(parallel=True)
def sadfuller_stat(y: NumArray, r0: int, kmax: int) -> float64:
    """
    Calculates the test statistics for the Sup Augmented Dickey-Fuller test.

    Args:
        y (NumArray): The time series data.
        r0 (int): Initial period to start the test.
        kmax (int): Maximum lag to use in the test.

    Returns:
        float64: The test statistics.
    """
    nobs: int = len(y)
    stat: float64 = float64(-inf)
    for r in prange(r0, nobs + 1):
        stat = max(stat, corridor_adfuller_stat(y, 0, r, kmax))
        stat = max(stat, adfuller_stat(y[:r], kmax))
    return stat


def sadfuller_dist(nobs: int64, nreps: int64, r0: int64) -> NDArray[float64]:
    rw: NDArray[float64] = random_walk(int(nreps), int(nobs))
    iterator: NDArray[int64] = __make_iterator__(nobs, nreps, r0)
    stat = __sadfuller_dist_from_random_walks__(rw, iterator, r0)
    return stat


@njit(parallel=True)
def __sadfuller_dist_from_random_walks__(
    random_walks: NDArray[float64], iterable: NDArray[int64], r0: int64
) -> NDArray[float64]:
    nreps: int = len(random_walks)
    stats: NumArray = repeat(-inf, nreps)
    niter: int = len(iterable)
    for i in prange(niter):
        next_iter: NDArray[int64] = iterable[i]
        j: int64 = next_iter[0]
        r1: int64 = next_iter[1]
        r2: int64 = next_iter[2]
        if r1 <= r2 - r0:
            stats[j] = max(stats[j], corridor_adfuller_cdf(random_walks[j], r1, r2))
    return stats


@njit(parallel=False)
def __make_iterator__(nobs: int64, nreps: int64, r0: int64) -> NDArray[int64]:
    total: int64 = nreps * (nobs - r0 + 1) * (nobs - r0 + 1)
    result: NDArray[int64] = zeros((total, 3), dtype=int64)

    idx: int = 0
    for n in range(0, nreps):
        for r1 in range(r0, nobs + 1):
            for r2 in range(r0, nobs + 1):
                result[idx, 0] = n
                result[idx, 1] = r1
                result[idx, 2] = r2
                idx += 1

    return result