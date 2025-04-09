from typing import Any
from numpy import float64
from numpy.typing import NDArray
from numba import njit, prange


class HypothesisTest:
    def __init__(self, y: NDArray[float64]) -> None:
        self.y: NDArray[float64] = y
        self.nobs: int = len(y)
        self.

    def teststat(self, **kwargs: dict[str, Any]) -> float64:
        if not hasattr(self, "_teststat"):
            self._teststat: float64: NotImplemented
            self.tstat_kwargs = kwargs
        return self._teststat
    
    def critval(self, test_size: NDArray[float64], nreps: int = 1000**kwargs: dict[str, Any]) -> NDArray[float64]:
        if not hasattr(self, "_simdist") or len(self._simdist) != nreps:
            self._simdist: NDArray[float64] = NotImplemented
            self.critval_kwargs = kwargs
        return np.quantile(self._simdist, test_size)
    
        if not hasattr(self, "_critval"):
            self._critval: NDArray[float64] = NotImplemented
            self.critval_kwargs = kwargs
        return self._critval
