from psytest.adftest import (
    rolling_adfuller_stat,
    rolling_adfuller_cdf,
)
from psytest.utils.functions import random_walk
from psytest.utils.constants import KMAX
from numpy.typing import NDArray
from numpy import (
    float64,
    inf,
    zeros,
    repeat,
    int64,
    arange,
    empty,
    array,
    quantile,
    floor,
)
from numba import njit, prange


# Sup ADF Test
@njit(parallel=True)
def sadfuller_stat(y: NDArray[float64], r0: float, kmax: int) -> float:
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
    stat: float = -inf
    for r in prange(r0, nobs + 1):
        stat = max(stat, rolling_adfuller_stat(y, 0, r, kmax))
    return stat


@njit(parallel=True)
def __sadfuller_dist_from_random_walks__(
    random_walks: NDArray[float64], nreps: int, nobs: int, r0: float
) -> NDArray[float64]:
    stats: NDArray[float64] = repeat(-inf, nreps * (nobs - r0))
    stats = stats.reshape((nreps, nobs - r0))
    for j in prange(nreps):
        for r2 in range(r0, nobs):
            for r1 in range(r2 - r0):
                stats[j, r2 - r0 - 1] = stats[j, r2 - r0] = max(
                    stats[j, r2 - r0 - 1], rolling_adfuller_cdf(random_walks[j], r1, r2)
                )
    return stats


# Backward Sup ADF Test


@njit(parallel=True)
def bsadf_stat(y: NDArray[float64], r0: float, r2: float, kmax: int) -> float:
    stat: float = -inf
    for r1 in prange(r2 - r0 + 1):
        stat = max(stat, rolling_adfuller_stat(y, r1, r2, kmax))
    return stat


@njit(parallel=True)
def bsadf_stat_all_series(
    y: NDArray[float64], r0: float, kmax: int, rstep: float = 0.01
) -> NDArray[float64]:
    r1r2_grid: NDArray[float64] = __r1r2_combinations__(r0, rstep)
    ntups: int = len(r1r2_grid)
    nstat: int = int(floor((1 - r0) / rstep))
    stat: NDArray[float64] = repeat(-inf, nstat)
    for i in prange(ntups):
        r1: int = r1r2_grid[i][0]
        r2: int = r1r2_grid[i][1]
        i: int = int(floor((r2 - r1 - r0) / rstep))
        stat[i] = max(stat[i], rolling_adfuller_stat(y, r1, r2, kmax))
    return stat


@njit(parallel=False)
def __r1r2_combinations__(r0: float, rstep: float) -> NDArray[float64]:
    total: int = int(floor((1 - r0 + rstep) * (1 - r0) / 2 / (rstep) ** 2))
    result: NDArray[float64] = empty(shape=(total, 2), dtype=float64)
    idx: int = 0
    for r2 in arange(r0, 1 + rstep, rstep):
        for r1 in arange(0, r2 - r0 + rstep, rstep):
            result[idx, 0] = r1
            result[idx, 1] = r2
            idx += 1
    return result


# @njit(parallel=False)
# def __r1r2_combinations__(nobs: int, r0: float) -> NDArray[int64]:
#     total: int = (nobs - r0) ** 2
#     result: NDArray[int64] = zeros(shape=(total, 2), dtype=int64)
#     idx: int = 0
#     for r1 in range(nobs - r0):
#         for r2 in range(r0, nobs):
#             result[idx, 0] = r1
#             result[idx, 1] = r2
#             idx += 1

#     return result


def bsadfuller_critval(
    nobs: int,
    nreps: int,
    r0: float,
    testsize: NDArray[float64] = array([0.10, 0.05, 0.01]),
) -> NDArray[float64]:
    rw: NDArray[float64] = random_walk(nreps, nobs)
    sadf_dist: NDArray[float64] = __sadfuller_dist_from_random_walks__(
        rw, nreps, nobs, r0
    )
    critval: NDArray[float64] = quantile(sadf_dist, testsize, axis=0)
    return critval
