from psytest.adftest import adfuller_dist, adfuller_stat
from psytest.utils.functions import r0_default, index_combinations, parallel_apply
from collections.abc import Callable
import numpy as np
from numba import njit, prange

KMAX = 0


def sadfuller_stat(y: np.ndarray, r0: int) -> np.float64:
    nobs: int = len(y)
    func: Callable = lambda r: adfuller_stat(y[:r], KMAX)
    adf_stats: list[float] = parallel_apply(func, range(r0, nobs + 1))
    return np.max(adf_stats)


def gsadfuller_stat(y: np.ndarray, r0: int) -> np.float64:
    nobs: int = len(y)
    func: Callable = lambda x: adfuller_stat(y[x[0] : x[1]], KMAX)
    iterable: list[tuple[int, int]] = index_combinations(r0, nobs)
    adf_stats: list[float] = parallel_apply(func, iterable)
    return np.max(adf_stats)

