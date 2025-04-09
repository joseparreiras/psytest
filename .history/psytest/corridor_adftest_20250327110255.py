from adftest import adfuller_stat, ADFtest
import numpy as np
from numba import njit, prange

def corridor_adfuller_test(y: NumArray, index_start: int = 0, index_end: int | None = None):
    if index_end is None:
        index_end = len(y)