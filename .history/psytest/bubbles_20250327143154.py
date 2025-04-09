from psytest.utils.constants import KMAX, NREPS
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

    def teststat(self, r0: int | None, kmax: int = KMAX) -> NDArray[float64]:
        r2_grid: NDArray[int64] = arange(r0, self.nobs)
        bsadf_stat: NDArray[float64] = basdfuller_stat(self.y, r0, r2_grid, kmax)
        ...

    def find_bubbles(self, kmax: int = KMAX): ...
