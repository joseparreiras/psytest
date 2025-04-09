from psytest.adftest import adfuller_dist, adfuller_stat
from psytest.utils.functions import r0_default, index_combinations, parallel_apply
import numpy as np
from numba import njit, prange

KMAX = 0


@njit(parallel=True)
def sadfuller(y: np.ndarray, r0: int) -> np.float64:
    nobs: int = len(y)
    func: Callable = lambda r: adfuller_stat(y[:r], KMAX)
    adf_stats: np.ndarray = np.zeros(nobs - r0 + 1)
    adf_stats
    for r in prange(r0, nobs + 1):
        adf_stats[r - r0] = adfuller_stat(y[:r], KMAX)
    return max(adf_stats)


