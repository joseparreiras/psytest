"""
Attempt to replicate the Sup-ADF test of Phillips, Shi & Yu (2015)
Optimized version using Numba and performance benchmarking
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable, Callable
from numba import jit, njit, prange
import timeit


@njit(parallel=True)
def adfuller(y: np.ndarray, kmax: int) -> np.float64:
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
        adf_stat: np.float64 = adfuller(y, kmax)
        adf_dist[i] = adf_stat
    return adf_dist


class ADFuller:
    def __init__(self, y: np.ndarray):
        self.y: np.ndarray = y
        self.nobs: int = len(y)

    def teststat(self, kmax: int = 1) -> np.float64:
        if not hasattr(self, "_teststat"):
            self._teststat: np.float64 = adfuller(self.y, kmax=kmax)
            self._kmax: int = kmax
        return self._teststat

    def critval(self, alpha_list: list[float] = [0.1, 0.05, 0.01]) -> np.ndarray:
        adf_dist: np.ndarray = adfuller_dist(self.nobs, self._kmax)
        return np.quantile(adf_dist, alpha_list)
