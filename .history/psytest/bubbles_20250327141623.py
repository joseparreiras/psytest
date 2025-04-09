from numpy.typing import NDArray
from numpy import float64

class PSYBubbles():
    def __init__(self, y:NDArray[float64]):
        self.y = y
        self.nobs = len(y)