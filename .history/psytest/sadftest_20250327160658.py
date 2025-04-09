from psytest.adftest import (
    adfuller_stat,
    NumArray,
    KMAX,
    TEST_SIZE,
    NREPS,
    corridor_adfuller_cdf,
)
from psytest.utils.functions import r0_default, random_walk
from numpy.typing import NDArray
from numpy import float64, inf, apply_along_axis, quantile, zeros, repeat, int64
from numba import njit, prange
from typing import Any
from itertools import product
from collections.abc import Iterable, Generator


# Sup ADF Test
@njit(parallel=True)
def sadfuller_stat(y: NumArray, r0: int, kmax: int) -> float64:
    """
    Calculates the test statistics for the Sup Augmented Dickey-Fuller test.

    Args:
        y (NumArray): The time series data.
        r0 (int): Initial period to start the test.
        kmax (int): Maximum lag to use in the test.

    Returns:
        float64: The test statistics.
    """
    nobs: int = len(y)
    stat: float64 = float64(-inf)
    for r in prange(r0, nobs + 1):
        stat = max(stat, adfuller_stat(y[:r], kmax))
    return stat


@njit
def sadfuller_dist(nobs: int64, nreps: int64, r0: int64) -> NDArray[float64]:
    random_walks: NDArray[float64] = random_walk(int(nreps), int(nobs))
    iterator: NDArray[int64] = __make_iterator__(nreps, r0, nobs) # type: ignore
    stat = __sadfuller_dist_from_random_walks__(random_walks, iterator, r0)
    return stat


def __sadfuller_dist_from_random_walks__(
    random_walks: NDArray[float64], iterable: NDArray[int64], r0: int64
) -> NDArray[float64]:
    nreps: int = len(random_walks)
    stats: NumArray = repeat(-inf, nreps)
    niter: int = len(iterable)
    for i in prange(niter):
        next_iter: NDArray[int64] = iterable[i]
        j: int64 = next_iter[0]
        r1: int64 = next_iter[1]
        r2: int64 = next_iter[2]
        if r1 <= r2 - r0:
            stats[j] = max(stats[j], corridor_adfuller_cdf(random_walks[j], r1, r2))
    return stats


@njit(parallel=False)
def __make_iterator__(nreps: int64, r0: int64, nobs: int64) -> NDArray[int64]:
    total: int = nreps * (nobs - r0 + 1) * (nobs - r0 + 1)
    result: NDArray[int64] = zeros((total, 3), dtype=int64)

    idx: int = 0
    for n in range(1, nreps + 1):
        for r1 in range(r0, nobs + 1):
            for r2 in range(r0, nobs + 1):
                result[idx, 0] = n
                result[idx, 1] = r1
                result[idx, 2] = r2
                idx += 1

    return result


# class SADFtest(ADFtest):
#     def __init__(self, y: NumArray, kmax: int = KMAX, r0: int | None = None) -> None:
#         """
#         Sup ADF test class.

#         Args:
#             y (np.ndarray): Time series data.
#             r0 (int | None, optional): Initial period to start the test. Defaults to None.
#         """
#         super().__init__(y=y, kmax=kmax)
#         self.r0 = r0 or r0_default(len(y))

#     def teststat(self) -> float64:
#         """
#         Calculates the test statistics for the Sup Augmented Dickey-Fuller test.

#         Returns:
#             float64: The test statistics.
#         """
#         if not hasattr(self, "__teststat__"):
#             self.__teststat__: float64 = sadfuller_stat(
#                 y=self.y, r0=self.r0, kmax=self.kmax
#             )
#         return self.__teststat__

#     def critval(
#         self, test_size: NumArray = TEST_SIZE, nreps: int = NREPS
#     ) -> NDArray[float64]:
#         """
#         Calculates the critical values for the hypothesis test.

#         Args:
#             test_size (NDArray[float64]): The test sizes (alpha) to calculate the critical values.
#             nreps (int, optional): Number of simulations to perform for the Monte Carlo. Defaults to 1000.

#         Returns:
#             NDArray[float64]: Critical values, on the order of the test sizes.
#         """
#         if not hasattr(self, "__simdist__") or len(self.__simdist__) != nreps:
#             self.__simdist__: NumArray = sadfuller_dist(
#                 nobs=self.nobs, nreps=nreps, kmax=self.kmax, r0=self.r0
#             )
#             self.critval_kwargs: dict[str, Any] = {
#                 "test_size": test_size,
#                 "nreps": nreps,
#             }
#         return quantile(self.__simdist__, test_size)
