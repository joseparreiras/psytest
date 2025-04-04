"""
Implementation of the Augmented Dickey-Fuller test for unit roots.
"""

from numba import njit, prange
from numpy import (
    float64,
    sum,
    square,
    diag,
    diff,
    empty,
    ones,
    sqrt,
    int64,
    sum as npsum,
)
from numpy.linalg import inv
from numpy.typing import NDArray
from numba import njit, prange
from psytest.utils.functions import random_walk
from psytest.utils.constants import KMAX


@njit(parallel=False)
def adfuller_stat(y: NDArray[float64], kmax: int) -> float:
    """
    Calculates the test statistics for the Augmented Dickey-Fuller test.

    Args:
        y (NDArray[float64]): The time series data.
        kmax (int): Maximum lag to use in the test.

    Returns:
        float64: The test statistics.
    """
    nobs: int = len(y)
    y_diff: NDArray[float64] = diff(y)
    X: NDArray[float64] = empty((nobs - kmax - 1, 2 + kmax))
    X[:, 0] = ones(nobs - kmax - 1)  # Constant
    for k in prange(1, kmax + 1):
        X[:, k] = y_diff[kmax - k : -k]
    X[:, -1] = y[kmax:-1]
    y_diff = y_diff[kmax:]
    beta: NDArray[float64] = inv(X.T @ X) @ X.T @ y_diff
    fit: NDArray[float64] = X @ beta
    resid: NDArray[float64] = y_diff - fit
    ssr: float = sum(square(resid))
    sigma_sq_hat: float = ssr / (nobs - kmax - 2)

    coef: float = beta[-1]
    coef_var: float = sigma_sq_hat * diag(inv(X.T @ X))[-1]
    return coef / sqrt(coef_var)


def adfuller_dist(nobs: int, nreps: int, kmax: int) -> NDArray[float64]:
    """
    Simulates tha asymptotic distribution of the Augmented Dickey-Fuller test.

    Args:
        nobs (int): Number of observations in the time series.
        nreps (int): Number of simulations to perform.
        kmax (int): Maximum lag to use in the test.

    Returns:
        NDArray[float64]: The simulated distribution of the test statistics.
    """
    random_walks: NDArray[float64] = random_walk(nreps, nobs)
    adf_dist: NDArray[float64] = empty(nreps)
    for i in prange(nreps):
        y: NDArray[float64] = random_walks[i]
        adf_stat: float = adfuller_stat(y, kmax)
        adf_dist[i] = adf_stat
    return adf_dist


# Rolling ADF TEST


@njit
def rolling_adfuller_stat(
    y: NDArray[float64], r1: int = 0, r2: int | None = None, kmax: int = KMAX
) -> float:
    if r2 is None:
        r2 = len(y)
    return adfuller_stat(y[r1:r2], kmax=kmax)


@njit
def rolling_adfuller_cdf(wiener: NDArray[float64], i1: int, i2: int) -> float:
    w2: float = wiener[i2]
    w1: float = wiener[i1]
    nobs: int = len(wiener)
    dt: float = 1 / nobs
    w_sum: float = float(dt * npsum(wiener[i1:i2]))
    w_sumsq: float = float(dt * npsum(wiener[i1:i2] ** 2))
    rw: float = (i2 - i1) / len(wiener)
    return (1 / 2 * rw * (w2**2 - w1**1 - rw) - w_sum * (w2 - w1)) / (
        rw**0.5 * (rw * w_sumsq - w_sum**2) ** 0.5
    )
