"""
Implementation of the Augmented Dickey-Fuller test for unit roots.
"""

from psytest.hyptest import HypothesisTest
import numpy as np
from numba import njit, prange


@njit(parallel=False)
def adfuller_stat(y: np.ndarray, kmax: int) -> np.float64:
    nobs: int = len(y)
    y_diff: np.ndarray = np.diff(y)
    X: np.ndarray = np.zeros((nobs - kmax - 1, 2 + kmax))
    X[:, 0] = np.ones(nobs - kmax - 1)  # Constant
    for k in prange(1, kmax + 1):
        X[:, k] = y_diff[kmax - k : -k]
    X[:, -1] = y[kmax:-1]
    y_diff = y_diff[kmax:]
    beta: np.ndarray = np.linalg.inv(X.T @ X) @ X.T @ y_diff
    fit: np.ndarray = X @ beta
    resid: np.ndarray = y_diff - fit
    ssr: np.float64 = np.sum(np.square(resid))
    sigma_sq_hat: np.float64 = ssr / (nobs - kmax - 2)

    coef: np.float64 = beta[-1]
    coef_var: np.float64 = sigma_sq_hat * np.diag(np.linalg.inv(X.T @ X))[-1]
    return coef / np.sqrt(coef_var)


def adfuller_dist(nobs: int, kmax: int, nreps: int = 1000) -> np.ndarray:
    random_walks: np.ndarray = np.cumsum(np.random.normal(size=(nreps, nobs)), axis=1)
    adf_dist: np.ndarray = np.zeros(nreps)
    for i in prange(nreps):
        y: np.ndarray = random_walks[i]
        adf_stat: np.float64 = adfuller_stat(y, kmax)
        adf_dist[i] = adf_stat
    return adf_dist


class ADFuller(HypothesisTest):
    def __init__(self, y: np.ndarray):
        super().__init__(y, adfuller_stat, adfuller_dist)