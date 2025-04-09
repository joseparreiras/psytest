from psytest.adftest import adfuller_stat
from psytest.utils.functions import (
    r0_default,
    index_combinations,
    parallel_apply,
    simulate_random_walks,
)
from collections.abc import Callable
import numpy as np
from numba import njit, prange

KMAX = 0


# Sup ADF Test
@njit(parallel=True)
def sadfuller_stat(y: np.ndarray, r0: int, kmax: int) -> np.float64:
    nobs: int = len(y)
    stat: np.float64 = np.float64(-np.inf)
    for r in prange(r0, nobs + 1):
        stat = max(stat, adfuller_stat(y[:r], kmax))
    return stat


def sadfuller_dist(nobs: int, r0: int, nreps: int = 1000) -> np.ndarray:
    wiener: np.ndarray = simulate_random_walks(nreps, nobs)
