from psytest.utils.constants import KMAX, NREPS
from numpy.typing import NDArray
from numpy import float64, arange, int64
from typing import Self
from pandas import Series


class PSYBubbles:
    def __init__(self, y: NDArray[float64]):
        self.y = y
        self.nobs: int = len(y)
        self._index: NDArray[int64] = arange(self.nobs)

    @property
    def index(self) -> NDArray[int64]:
        return self._index

    @index.setter
    def index(self, value: NDArray[int64]):
        self._index = value

    def critval(self, nreps: int = NREPS): ...

    def find_bubbles(self, kmax: int = KMAX): ...

    @classmethod
    def from_pandas(cls, series: pd.Series) -> "PSYBubbles":
        psyb: Self = cls(series.values)
        psyb.index = series.index
        return psyb
