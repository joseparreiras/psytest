from psytest.adftest import adfuller_stat, ADFtest, KMAX, NumArray, NREPS
from numpy import float64, zeros, sum as npsum
from numpy.random import normal
from numba import njit, prange
import numpy as np


def corridor_adfuller_test(
    y: NumArray, index_start: int = 0, index_end: int | None = None, kmax: int = KMAX
) -> float64:
    return adfuller_stat(y[index_start:index_end], kmax=kmax)


def corridor_adfuller_cdf(wiener: NumArray, r1: int, r2: int) -> float64:
    w2: float64 = wiener[r2]
    w1: float64 = wiener[r1]
    w_sum: float64 = npsum(wiener[r1:r2])
    w_sumsq: float64 = npsum(wiener[r1:r2] ** 2)
    rw: int = r2 - r1
    return 1 / 2 * rw * (w2**2 - w1**1 - rw) - w_sum * (w2 - w1) / (
        rw**0.5 * (rw * w_sumsq - w_sum**2) ** 0.5
    )


def corridor_adfuller_adist(
    nobs: int, nreps: int = NREPS, index_start: int = 0, index_end: int | None = None
) -> NumArray:
    if index_end is None:
        index_end = nobs
    rw: int = index_end - index_start
    dist: NumArray = np.zeros(nreps)
    for j in prange(nreps):
        white_noise: list[float] = [0] + [normal() for _ in prange(nobs - 1)]
        wiener: NumArray = np.array([sum(white_noise[:i]) for i in prange(nobs)])
        dist[j] = corridor_adfuller_cdf(wiener, index_start, index_end)

    return dist

def sum_nonumba(x):
    return npsum(x)

@njit
def sum_numba(x):
    return npsum(x)

x = np.arange(1000)

%timeit sum_nonumba(x)
%timeit sum_numba(x)