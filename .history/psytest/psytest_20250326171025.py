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



def simulate_random_walks(nreps: int, nobs: int) -> np.ndarray:
    return np.cumsum(np.random.normal(size=(nreps, nobs)), axis=1)

def simulate_random_walks_numba(nreps: int, nobs: int) -> np.ndarray:
    random_walks: np.ndarray = np.zeros((nreps, nobs))
    for i in prange(nreps):
        for t in prange(1, nobs):
            random_walks[i, t] = random_walks[i, t - 1] + np.random.normal()
    return random_walks

def gsadfuller_dist(nobs: int, r0: int, nreps: int = 1000) -> np.ndarray: 
    
