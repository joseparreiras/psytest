from numpy.typing import NDArray
from numpy import float64, quantile
from psytest.utils.functions import r0_default
from psytest.sadftest import bsadf_stat_all_series
from collections.abc import Callable


class PSYBubbles:
    def __init__(self, y: NDArray[float64], r0: int | None = None, kmax: int = 0):
        self.y: NDArray[float64] = y
        self.nobs: int = len(y)
        self.r0: int = r0 or r0_default(self.nobs)
        self.kmax: int = kmax

    def bsadf(self, force: bool = False) -> dict[int, float]:
        if force or not hasattr(self, "teststat"):
            # Calculate if forced or nonexistant
            r2_grid, teststat = bsadf_stat_all_series(self.y, self.r0, self.kmax)
            self.teststat: dict[int, float] = dict(zip(r2_grid, teststat))

        return self.teststat

    def critval(
        self,
        nreps: int = 2_000,
        force: bool = False,
        test_size: list[float] = [0.10, 0.05, 0.01],
    ) -> NDArray[float64]:
        if force or not hasattr(self, "critval"):
            sadf_dist: NDArray[float64]
            get_cv: Callable[NDArray[float64], NDArray[float64]] = lambda x: quantile(x, [1-q for q in test_size])

        return self.critval
