from numpy.typing import NDArray, ArrayLike
from numpy import float64, quantile, apply_along_axis, bool_, array
from psytest.utils.functions import r0_default, minlength_default
from psytest.sadftest import bsadf_stat_all_series, sadfuller_dist
from collections.abc import Callable, Generator
from typing import Generic, TypeVar, Any


class PSYBubbles:
    def __init__(
        self,
        y: NDArray[float64],
        r0: int | None = None,
        kmax: int = 0,
        minlength: int | None = None,
        delta: float | None = None,
    ) -> None:
        self.y: NDArray[float64] = y
        self.nobs: int = len(y)
        self.index: NDArray | None = None
        self.r0: int = r0 or r0_default(self.nobs)
        self.kmax: int = kmax
        if minlength:
            self.minlength: int = minlength
            self.delta: float | None = None
        elif delta:
            self.minlength: int = minlength_default(self.nobs, delta)
            self.delta: float | None = delta
        else:
            raise ValueError("Either `minlength` or `delta` must be provided")

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
            q_array = array(test_size)
            cval: NDArray[float64] = quantile(dist, 1 - q_array, axis=0).T
            self._critval: dict[int, NDArray[float64]] = dict(zip(r2_grid, cval))
            self._nreps: int = nreps
            self._testsize: list[float] = test_size

        return self._critval

    def find_bubbles(self, alpha: float) -> list[Any] | list[tuple[int, int | None]]:
        stat: dict[int, float] = self.bsadf()
        cval: dict[int, NDArray[float64]] = self.critval()
        alpha_index: int = self._testsize.index(alpha)
        bubble_bool: list[bool] = [stat[i] > cval[i][alpha_index] for i in stat.keys()]
        bubble_index: list[tuple[int, int | None]] = list(
            find_bubble_dates(bubble_bool, self.minlength)
        )
        if hasattr(self, "index") and self.index is not None:
            bubble_dates = []
            for start, end in bubble_index:
                idx_start: Any = self.index[start]
                idx_end: Any = self.index[end] if end else None
                bubble_dates.append((idx_start, idx_end))
            return bubble_dates
        return bubble_index


def check_bubble_exists(bubble_bool: list[bool]) -> bool:
    return bubble_bool.count(True) > 0


def check_bubble_ends(bubble_bool: list[bool]) -> bool:
    return bubble_bool.count(False) > 0


def is_bubble_long_enough(bubble_bool: list[bool], minlength: int) -> bool:
    return bubble_bool[minlength]


def find_bubble_dates(
    bubble_bool: list[bool], minlength: int
) -> Generator[tuple[int, int | None], None, None]:
    i0 = 0
    while len(bubble_bool) > minlength:
        if not check_bubble_exists(bubble_bool):
            break
        start: int = bubble_bool.index(True)
        if is_bubble_long_enough(bubble_bool[start:], minlength):
            if not check_bubble_ends(bubble_bool[start:]):
                yield (start + i0, None)
                break
            end: int = bubble_bool[start:].index(False) + start
            yield (start + i0, end + i0)
        bubble_bool = bubble_bool[end:]
        i0 += end
