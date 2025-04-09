from typing import Any
from numpy import float64, quantile
from numpy.typing import NDArray
from numba import njit, prange
from collections.abc import Callable


class HypothesisTest:
    def __init__(self, y: NDArray[float64], teststat_func: Callable, critval_func: Callable) -> None:
        self.y: NDArray[float64] = y
        self.nobs: int = len(y)
        self.__teststat_func__: Callable = teststat_func
        self.__critval_func__: Callable = critval_func

    def teststat(self, **kwargs: dict[str, Any]) -> float64:
        if not hasattr(self, "_teststat"):
            self.__teststat__: float64: self.__teststat
            self.tstat_kwargs = kwargs
        return self._teststat
    
    def critval(self, test_size: NDArray[float64], nreps: int = 1000, **kwargs: dict[str, Any]) -> NDArray[float64]:
        if not hasattr(self, "_simdist") or len(self._simdist) != nreps:
            self._simdist: NDArray[float64] = NotImplemented
            self.critval_kwargs = kwargs
        return quantile(self._simdist, test_size)