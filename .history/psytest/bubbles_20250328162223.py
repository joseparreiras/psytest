from numpy.typing import NDArray
from numpy import float64, quantile, apply_along_axis
from psytest.utils.functions import r0_default, minlength_default
from psytest.sadftest import bsadf_stat_all_series, sadfuller_dist
from collections.abc import Callable


class PSYBubbles:
    def __init__(self, y: NDArray[float64], r0: int | None = None, kmax: int = 0, minlength: int | None = None, delta: float | None = None) -> None:
        self.y: NDArray[float64] = y
        self.nobs: int = len(y)
        self.r0: int = r0 or r0_default(self.nobs)
        self.kmax: int = kmax
        if not (minlength and delta):
            raise ValueError("Either `minlength` or `delta` must be provided")
        self.minlength: int = minlength or minlength_default(self.nobs, delta)
        self.delta = delta

    def bsadf(self, force: bool = False) -> dict[int, float]:
        if force or not hasattr(self, "_teststat"):
            # Calculate if forced or nonexistant
            r2_grid, teststat = bsadf_stat_all_series(self.y, self.r0, self.kmax)
            self._teststat: dict[int, float] = dict(zip(r2_grid, teststat))

        return self._teststat

    def critval(
        self,
        nreps: int = 2_000,
        force: bool = False,
        test_size: list[float] = [0.10, 0.05, 0.01],
    ) -> dict[int, NDArray[float64]]:
        if (
            force
            or not hasattr(self, "_critval")
            or getattr(self, "_nreps", None) != nreps
            or getattr(self, "_testsize", None) != test_size
        ):
            r2_grid, dist = sadfuller_dist(self.nobs, nreps, self.r0)
            get_cv: Callable[..., NDArray[float64]] = lambda x: quantile(
                x, [1 - q for q in test_size]
            )
            cval: NDArray[float64] = apply_along_axis(get_cv, 0, dist).T
            self._critval: dict[int, NDArray[float64]] = dict(zip(r2_grid, cval))
            self._nreps: int = nreps
            self._testsize: list[float] = test_size

        return self._critval

    def find_bubbles(self, alpha: float):
        stat: dict[int, float] = self.bsadf()
        cval: dict[int, NDArray[float]] = self.critval()
        alpha_index: int = self._testsize.index(alpha)
        bubble_bool: list[bool] =  [stat[i] > cval[i][alpha_index] for i in stat.keys()]
        bubble_index: list = []
        for i in stat.keys():
            
        

def get_start_end(bubble)