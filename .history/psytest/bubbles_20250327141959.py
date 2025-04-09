from psytest.utils.constants import KMAX, NREPS
from numpy.typing import NDArray
from numpy import float64


class PSYBubbles:
    def __init__(self, y: NDArray[float64]):
        self.y = y
        self.nobs: int = len(y)
        self.index: NDArray[int] = arange(self.nobs)

    def critval(self, nreps: int = NREPS):
        ...
        
    def find_bubbles(self, kmax: int = KMAX):
        ...