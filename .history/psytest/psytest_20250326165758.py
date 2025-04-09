from .adftest import adfuller_dist, adfuller_stat
from .utils.functions import r0_default, index_combinations, parallel_apply
import numpy as np
from numba import njit, prange

KMAX = 0

@njit(parallel = True)
def sadfuller(y: np.ndarray, r0: int | None) -> np.float64:
    nobs: int = len(y)
    r0 = r0 or r0_default(nobs)
    adf_stats: np.ndarray = np.zeros(nobs-r0)
    for r in prange(r0, nobs+1):
        