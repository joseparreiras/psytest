from psytest.utils.constants import KMAX, NREPS
from psytest.utils.functions import r0_default
from psytest.bsadftest import bsadfuller_test
from numpy.typing import NDArray
from numpy import float64, arange, int64
from typing import Self
from collections.abc import Iterable
from pandas import Series


class PSYBubbles:
    def __init__(self, y: NDArray[float64]):
        self.y = y
        self.nobs: int = len(y)
        self._index: Iterable = arange(self.nobs)

    @property
    def index(self) -> Iterable:
        return self._index

    @index.setter
    def index(self, value: Iterable) -> None:
        self._index = value

    @classmethod
    def from_pandas(cls, series: Series) -> "PSYBubbles":
        psyb: Self = cls(series.to_numpy())
        psyb.index = series.index
        return psyb

    def critval(self, nreps: int = NREPS): ...

    def teststat(
        self, r0: int | None, kmax: int = KMAX, force: bool = False
    ) -> NDArray[float64]:
        if r0 is None:
            r0 = r0_default(self.nobs)
        if (
            force
            or not hasattr(self, "_teststat")
            or getattr(self, "_r0", None) != r0
            or getattr(self, "_kmax", None) != kmax
        ):
            r2_grid: NDArray[int64] = arange(r0, self.nobs)
            bsadf_stat: NDArray[float64] = bsadfuller_test(self.y, r0, r2_grid, kmax)
            self._teststat: NDArray[float64] = bsadf_stat
            self._r0 = r0
            self._kmax: int = kmax

        return self._teststat

    def find_bubbles(self, kmax: int = KMAX): ...
