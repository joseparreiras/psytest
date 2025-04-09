from typing import Any
from numpy.typing import NDArray
from numba import njit, prange

class HypothesisTest:
    def __init__(self, y: NDArray[float], **kwargs) -> None:
        self.y: NDArray[float] = y
        self.nobs: int = len(y)
        self.kwargs: dist[str, Any] = kwargs
