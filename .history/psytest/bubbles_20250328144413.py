from numpy.typing import NDArray
from numpy import float64

class PSYBubbles():
    def __init__(self, y:NDArray[float64], r0: int | None = None, kmax: int = 0):
        self.y: NDArray[float64] = y
        self.nobs: int = len(y)