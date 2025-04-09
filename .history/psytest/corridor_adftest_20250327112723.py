from adftest import adfuller_stat, ADFtest, KMAX, NumArray, NREPS
from numpy import float64, zeros
from numpy.random import normal
from numba import njit, prange
import numpy as np
import random


def corridor_adfuller_test(
    y: NumArray, index_start: int = 0, index_end: int | None = None, kmax: int = KMAX
) -> float64:
    if index_end is None:
        index_end = len(y)
    stat: float64 = adfuller_stat(y[index_start:index_end], kmax=kmax)


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
        dist[j] = (
            1 / 2 * rw * (wiener[index_end] ** 2 - wiener[index_start] ** 2 - rw)
            - sum(wiener[index_start:index_end])
            * (wiener[index_end] - wiener[index_start])
        ) / (
            rw**0.5
            * (
                rw * sum(wiener[index_start:index_end] ** 2)
                - (sum(wiener[index_start:index_end])) ** 2
            )
            ** 0.5
        )

    return dist
