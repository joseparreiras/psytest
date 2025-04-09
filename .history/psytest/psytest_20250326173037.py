from psytest.adftest import adfuller_dist, adfuller_stat
from psytest.utils.functions import r0_default, index_combinations, parallel_apply
from collections.abc import Callable
import numpy as np
from numba import njit, prange

KMAX = 0


def sadfuller_stat(y: np.ndarray, r0: int) -> float:
    nobs: int = len(y)
    func: Callable = lambda r: adfuller_stat(y[:r], KMAX)
    adf_stats: list[float] = parallel_apply(func, range(r0, nobs + 1))
    return np.max(adf_stats)


def gsadfuller_stat(y: np.ndarray, r0: int) -> float:
    nobs: int = len(y)
    func: Callable = lambda x: adfuller_stat(y[x[0] : x[1]], KMAX)
    iterable: list[tuple[int, int]] = index_combinations(r0, nobs)
    adf_stats: list[float] = parallel_apply(func, iterable)
    return np.max(adf_stats)


def simulate_random_walks(nreps: int, nobs: int) -> np.ndarray:
    return np.cumsum(np.random.normal(size=(nreps, nobs)), axis=1)


@njit(parallel=True)
def sim_gsadf_stat(y: np.ndarray, r0: int) -> float:
    nobs: int = len(y)
    stat: float = 0.0
    for r2 in prange(r0, nobs):
        for r1 in prange(r0, r2 - r0):
            rw: int = r2 - r1
            num: float = 1 / 2 * rw * (y[r2] ** 2 - y[r1] ** 2 - rw) - sum(y[r1:r2]) * (
                y[r2] - y[r1]
            )
            den: float = (
                rw**0.5 * (rw * sum(y[r1:r2] ** 2) - (sum(y[r1:r2])) ** 2) ** 0.5
            )
            stat = max(stat, num / den)
    return stat


class GSADFuller:
    def __init__(self, y: np.ndarray, kmax: int = KMAX):
        self.y: np.ndarray = y
        self.nobs: int = len(y)
        self.kmax: int = kmax

    def teststat(self, r0: int | None = None) -> float:
        r0 = r0 or r0_default(self.nobs)
        if not hasattr(self, "_teststat"):
            self._teststat: float = gsadfuller_stat(self.y, r0)
            self._r0: int = r0
        return self._teststat

    def critval(
        self, alpha_list: Iterable[float] = np.array([0.1, 0.05, 0.01]), nreps: int = 1000
    ) -> dict[float, float]:
        if not hasattr(self, "_critval"):
            random_walks: np.ndarray = simulate_random_walks(nreps, self.nobs)
            adf_dist: np.ndarray = np.apply_along_axis(
                sim_gsadf_stat, 1, random_walks, self._r0
            )
            self._critval: dict[float, float] = dict(
                zip(alpha_list, np.quantile(adf_dist, [1 - q for q in alpha_list]))
            )
            self._nreps: int = nreps
        return self._critval
