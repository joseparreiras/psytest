"""psytest.adftest
=================

This module contains the functions related to the Augmented Dickey-Fuller test for unit roots. The functions allow us to calculate both the test statistic and the asymptotic distribution of the test statistic. Additionally, it provides functions for rolling ADF tests and cumulative distribution functions based on a Wiener process.
"""

from numba import njit, prange
from numpy import float64, sum, square, diag, diff, empty, ones, sqrt, sum as npsum
from numpy.linalg import inv
from numpy.typing import NDArray
from .utils.functions import random_walk


@njit(parallel=False)
def adfuller_fit(y: NDArray[float64], kmax: int) -> NDArray[float64]:
    """Calculates the fitted values of the Augmented Dickey-Fuller regression.

    Parameters
    ----------
    y : NDArray[float64]
        The time series data.
    kmax : int
        Maximum lag to use in the test.

    Returns
    -------
    fit: NDArray[float64]
        The fitted values of the regression.
    """
    nobs: int = len(y)
    y_diff: NDArray[float64] = diff(y)
    X: NDArray[float64] = empty((nobs - kmax - 1, 2 + kmax))
    X[:, 0] = ones(nobs - kmax - 1)
    for lag in range(1, kmax + 1):
        X[:, lag] = y_diff[kmax - lag : -lag]
    X[:, -1] = y[kmax:-1]
    y_diff = y_diff[kmax:]
    beta: NDArray[float64] = inv(X.T @ X) @ X.T @ y_diff
    fit: NDArray[float64] = X @ beta
    return fit


@njit(parallel=False)
def adfuller_stat(y: NDArray[float64], kmax: int) -> float:
    """Calculates the test statistic for the Augmented Dickey-Fuller.

    Parameters
    ----------
    y : NDArray[float64]
        The time series data.
    kmax : int
        Maximum lag to use in the test.

    Returns
    -------
    tstat: float
        The test statistic.

    Notes
    -----
    The Augmented Dickey-Fuller test statistic is calculated as:
    .. math::

        \\text{ADF} = \\frac{\\hat{\\beta}}{\\sqrt{\\widehat{\\mathrm{Var}}(\\hat{\\beta})}}

    where :math:`\\hat{\\beta}` is the coefficient of :math:`y_{t-1}` and :math:`\\widehat{\\mathrm{Var}}(\\hat{\\beta})` is its estimated variance in the regression:F
    .. math::

        \\Delta y_{t} = \\alpha + \\beta y_{t-1} + \\sum_{k=2}^{kmax} \\gamma_k \\Delta y_{t-k} + \\epsilon_t
    """
    nobs: int = len(y)
    y_diff: NDArray[float64] = diff(y)
    X: NDArray[float64] = empty((nobs - kmax - 1, 2 + kmax))
    X[:, 0] = ones(nobs - kmax - 1)
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
    """Simulates the asymptotic distribution of the Augmented Dickey-Fuller test statistic.

    Parameters
    ----------
    nobs : int
        Number of observations in the time series.
    nreps : int
        Number of simulations to perform.
    kmax : int
        Maximum lag to use in the test.

    Returns
    -------
    distribution: NDArray[float64]
        The simulated distribution of the test statistics.
    """
    random_walks: NDArray[float64] = random_walk(nreps, nobs)
    adf_dist: NDArray[float64] = empty(nreps)
    for i in prange(nreps):
        y: NDArray[float64] = random_walks[i]
        adf_stat: float = adfuller_stat(y, kmax)
        adf_dist[i] = adf_stat
    return adf_dist


@njit
def rolling_adfuller_stat(
    y: NDArray[float64], kmax: int, r1: float = 0, r2: float = 1.0
) -> float:
    """Calculates the Augmented Dickey-Fuller test statistic for a window of the time series.

    Parameters
    ----------
    y : NDArray[float64]
        Values of the time series.
    r1 : float, optional
        Start index. Defaults to 0.
    r2 : float, optional
        End index. Defaults to 1.
    kmax : int, optional
        Maximum lag to use in the test.

    Returns
    -------
    teststat: float
        Value of the test statistic.

    Notes
    -----
    The rolling ADF test statistic is calculated as:
    .. math::

        \\text{ADF}(r_1, r_2) = \\frac{\\hat{\\beta}}{\\sqrt{\\widehat{\\mathrm{Var}}(\\hat{\\beta})}}

    where :math:`\\hat{\\beta}` is the coefficient of :math:`y_{t-1}` and :math:`\\widehat{\\mathrm{Var}}(\\hat{\\beta})` is its estimated variance in the regression:
    .. math::

        \\Delta y_{t} = \\alpha + \\beta y_{t-1} + \\sum_{k=2}^{kmax} \\gamma_k \\Delta y_{t-k} + \\epsilon_t

    Raises
    ------
    ValueError
        If :paramref:`r1` or :paramref:`r2` are not in the range [0, 1] or if :paramref:`r1` is greater than :paramref:`r2`.
    """
    if not 0 <= r1 <= 1 or not 0 <= r2 <= 1:
        raise ValueError("r1 and r2 should be in the range [0, 1]")
    if r1 > r2:
        raise ValueError("r1 should be less than r2")
    nobs: int = len(y)
    i1: int = int(r1 * nobs)
    i2: int = int(r2 * nobs)
    return adfuller_stat(y[i1:i2], kmax=kmax)


@njit
def rolling_adfuller_cdf(wiener: NDArray[float64], r1: float, r2: float) -> float:
    """Calculates the cumulative asymptotic distribution of the Augmented Dickey-Fuller test statistic based on a Wiener process.

    Parameters
    ----------
    wiener : NDArray[float64]
        Values of the Wiener process.
    r1 : float
        Start index.
    r2 : float
        End index.

    Returns
    -------
    cdf: float
        Value of the cumulative distribution function.

    Raises
    ------
    ValueError
        If :paramref:`r1` or :paramref:`r2` are not in the range [0, 1] or if :paramref:`r1` is greater than :paramref:`r2`.
    """
    if not 0 <= r1 <= 1 or not 0 <= r2 <= 1:
        raise ValueError("`r1` and `r2` should be in the range [0, 1]")
    if r1 > r2:
        raise ValueError("`r1` should be less than `r2`")
    nobs: int = len(wiener)
    i1: int = int(r1 * nobs)
    i2: int = int(r2 * nobs)
    w2: float = wiener[i2]
    w1: float = wiener[i1]
    dt: float = 1 / nobs
    w_sum: float = float(dt * npsum(wiener[i1:i2]))
    w_sumsq: float = float(dt * npsum(wiener[i1:i2] ** 2))
    rw: float = float(i2 - i1)
    return (1 / 2 * rw * (w2**2 - w1**1 - rw) - w_sum * (w2 - w1)) / (
        rw**0.5 * (rw * w_sumsq - w_sum**2) ** 0.5
    )
