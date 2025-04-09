from numpy.typing import NDArray, ArrayLike
from numpy import object_
from numpy import float64, quantile, apply_along_axis, bool_, array, arange, floor
from psytest.utils.functions import r0_default, minlength_default
from psytest.sadftest import bsadf_stat_all_series, bsadfuller_critval
from collections.abc import Callable, Generator
from typing import Generic, TypeVar, Any


class PSYBubbles:
    def __init__(
        self,
        y: NDArray[float64],
        r0: int | None = None,
        rstep: float | None = None,
        kmax: int = 0,
        minlength: int | None = None,
        delta: float | None = None,
    ) -> None:
        self.y: NDArray[float64] = y
        self.nobs: int = len(y)
        self.index: NDArray | None = None
        self.r0: int = r0 or r0_default(self.nobs)
        self.rstep: float = rstep or 1 / self.nobs
        self.kmax: int = kmax
        if minlength:
            self.minlength: int = minlength
            self.delta: float | None = None
        elif delta:
            self.minlength: int = minlength_default(self.nobs, delta)
            self.delta: float | None = delta
        else:
            raise ValueError("Either `minlength` or `delta` must be provided")

    def r2grid(self) -> NDArray[float64]:
        return arange(self.r0, 1 + 1e-16, self.rstep)

    def teststat(self, force: bool = False) -> dict[int, float]:
        if force or not hasattr(self, "_teststat"):
            # Calculate if forced or nonexistant
            stat: NDArray[float64] = bsadf_stat_all_series(
                self.y, self.r0, self.rstep, self.kmax
            )
            self._teststat: dict[int, float] = dict(zip(self.r2grid(), stat))

        return self._teststat

    def critval(
        self,
        nreps: int,
        force: bool = False,
        test_size: list[float] | float = [0.10, 0.05, 0.01],
    ) -> dict[int, NDArray[float64]]:
        if (
            force
            or not hasattr(self, "_critval")
            or getattr(self, "_nreps", None) != nreps
            or getattr(self, "_testsize", None) != test_size
        ):
            cval: NDArray[float64] = bsadfuller_critval(
                self.r0, self.rstep, nreps, self.nobs, test_size
            ).T
            self._critval: dict[int, NDArray[float64]] = dict(zip(self.r2grid(), cval))
            self._nreps: int = nreps
            self._testsize: list[float] = test_size

        return self._critval

    def find_bubbles(
        self, nreps: int, alpha: float
    ) -> list[Any] | list[tuple[int, int | None]]:
        # TODO: Terminar isso com r2 e r1 float (possivelmente truncado fazendo interpol)
        stat: dict[int, float] = self.teststat()
        cval: dict[int, NDArray[float64]] = self.critval(nreps=nreps, test_size=alpha)
        bubble_bool: list[NDArray[bool_]] = [stat[i] > cval[i] for i in stat.keys()]
        bubble_r2index: NDArray[object_] = array(
            list(find_bubble_dates(bubble_bool, self.minlength))
        )
        bubble_index: NDArray[float64] = array(
            [
                int(floor(self.nobs * self.r2grid()[i])) if i is not None else None
                for i in bubble_r2index.flatten()
            ]
        ).reshape((-1, 2))
        if hasattr(self, "index") and self.index is not None:
            bubble_dates: list = []
            for start, end in bubble_index:
                idx_start: Any = self.index[start]
                idx_end: Any = self.index[end] if end is not None else None
                bubble_dates.append((idx_start, idx_end))
            return array(bubble_dates)
        return bubble_index


def check_bubble_exists(bubble_bool: list[bool]) -> bool:
    return bubble_bool.count(True) > 0


def check_bubble_ends(bubble_bool: list[bool]) -> bool:
    return bubble_bool.count(False) > 0


def is_bubble_long_enough(bubble_bool: list[bool], minlength: int) -> bool:
    try:
        return bubble_bool[minlength]
    except IndexError:
        return False


def find_bubble_dates(
    bubble_bool: list, minlength: int
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


# TODO: Make a class method to call from Pandas Series
# TODO: Write docstringsa
