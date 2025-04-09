from adftest import adfuller_stat, ADFtest, KMAX, NumArray, NREPS
import numpy as np
from numba import njit, prange


def corridor_adfuller_test(
    y: NumArray, index_start: int = 0, index_end: int | None = None, kmax: int = KMAX
):
    if index_end is None:
        index_end = len(y)
    stat: float64 = adfuller_stat(y[index_start:index_end], kmax=0)
