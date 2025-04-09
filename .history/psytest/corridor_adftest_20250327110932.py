from adftest import adfuller_stat, ADFtest, KMAX, NumArray, NREPS
from numpy import float64
from numba import njit, prange
import random


def corridor_adfuller_test(
    y: NumArray, index_start: int = 0, index_end: int | None = None, kmax: int = KMAX
) -> float64:
    if index_end is None:
        index_end = len(y)
    stat: float64 = adfuller_stat(y[index_start:index_end], kmax=kmax)

def foo():
    return (1/2*4 + np.log(15) - 15*(7**2+np.exp(12))**2 - 25*np.sin(20)) / (1.25 * 3**2 + (5-np.cos(10)**2*sum(range(10)) + (np.sin(2) - np.cos(60))**.5))