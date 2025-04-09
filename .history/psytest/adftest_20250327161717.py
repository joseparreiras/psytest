"""
Implementation of the Augmented Dickey-Fuller test for unit roots.
"""

import numpy as np
from numba import njit, prange
from typing import Any
from numpy import (
    float64,
    quantile,
    array,
    sum,
    square,
    diag,
    diff,
    empty,
    ones,
    sqrt,
    int64,
)
from numpy.linalg import inv
from numpy.typing import NDArray
from numba import njit, prange
from collections.abc import Callable
from typing import TypeAlias, ParamSpec, Concatenate
from psytest.utils.functions import random_walk

from psytest.utils.functions import random_walk
from numpy import float64, zeros, sum as npsum
from numpy.random import normal
from numba import njit, prange
import numpy as np


# Type aliases
NumArray: TypeAlias = NDArray[float64]
Param = ParamSpec("Param")
TStatFunc: TypeAlias = Callable[Concatenate[NumArray, int, Param], float64]
SimDistFunc: TypeAlias = Callable[Concatenate[int, int, int, Param], NumArray]

# Global variables
TEST_SIZE: NumArray = array([0.1, 0.05, 0.01])
NREPS: int = 1000
KMAX: int = 0


@njit(parallel=False)
def adfuller_stat(y: NumArray, kmax: int) -> float64:
    """
    Calculates the test statistics for the Augmented Dickey-Fuller test.

    Args:
        y (NumArray): The time series data.
        kmax (int): Maximum lag to use in the test.

    Returns:
        float64: The test statistics.
    """
    nobs: int = len(y)
    y_diff: NumArray = diff(y)
    X: NumArray = empty((nobs - kmax - 1, 2 + kmax))
    X[:, 0] = ones(nobs - kmax - 1)  # Constant
    for k in prange(1, kmax + 1):
        X[:, k] = y_diff[kmax - k : -k]
    X[:, -1] = y[kmax:-1]
    y_diff = y_diff[kmax:]
    beta: NumArray = inv(X.T @ X) @ X.T @ y_diff
    fit: NumArray = X @ beta
    resid: NumArray = y_diff - fit
    ssr: float64 = sum(square(resid))
    sigma_sq_hat: float64 = ssr / (nobs - kmax - 2)

    coef: float64 = beta[-1]
    coef_var: float64 = sigma_sq_hat * diag(np.linalg.inv(X.T @ X))[-1]
    return coef / sqrt(coef_var)


def adfuller_dist(nobs: int, nreps: int, kmax: int) -> NumArray:
    """
    Simulates tha asymptotic distribution of the Augmented Dickey-Fuller test.

    Args:
        nobs (int): Number of observations in the time series.
        nreps (int): Number of simulations to perform.
        kmax (int): Maximum lag to use in the test.

    Returns:
        NumArray: The simulated distribution of the test statistics.
    """
    random_walks: NumArray = random_walk(nreps, nobs)
    adf_dist: NumArray = empty(nreps)
    for i in prange(nreps):
        y: NumArray = random_walks[i]
        adf_stat: float64 = adfuller_stat(y, kmax)
        adf_dist[i] = adf_stat
    return adf_dist


# Corridor ADF Test ------------------------------------------------------------

@njit
def corridor_adfuller_stat(
    y: NumArray, r1: int = 0, r2: int | None = None, kmax: int = KMAX
) -> float64:
    if r2 is None:
        r2 = len(y)
    return adfuller_stat(y[r1:r2], kmax=kmax)


@njit
def corridor_adfuller_cdf(wiener: NumArray, r1: int64, r2: int64) -> float64:
    w2: float64 = wiener[r2]
    w1: float64 = wiener[r1]
    w_sum: float64 = npsum(wiener[r1:r2])
    w_sumsq: float64 = npsum(wiener[r1:r2] ** 2)
    rw: int64 = r2 - r1
    return 1 / 2 * rw * (w2**2 - w1**1 - rw) - w_sum * (w2 - w1) / (
        rw**0.5 * (rw * w_sumsq - w_sum**2) ** 0.5
    )


