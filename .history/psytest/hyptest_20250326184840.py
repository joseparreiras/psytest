from typing import Any
from numpy import float64, quantile
from numpy.typing import NDArray
from numba import njit, prange
from collections.abc import Callable


class HypothesisTest:
    def __init__(
        self, y: NDArray[float64], teststat_func: Callable, simdist_func: Callable, **kwargs
    ) -> None:
        self.y: NDArray[float64] = y
        self.nobs: int = len(y)
        
        self.__teststat_func__: Callable = teststat_func
        self.__simdist_func__: Callable = simdist_func

    def teststat(self, **kwargs: dict[str, Any]) -> float64:
        if not hasattr(self, "__teststat__"):
            self.__teststat__: float64 = self.__teststat_func__(self.y, **kwargs)
            self.tstat_kwargs: dict[str, Any] = kwargs
        return self.__teststat__

    def critval(
        self, test_size: NDArray[float64], nreps: int = 1000, **kwargs: dict[str, Any]
    ) -> NDArray[float64]:
        if not hasattr(self, "__simdist__") or len(self.__simdist__) != nreps:
            self.__simdist__: NDArray[float64] = self.__simdist_func__(
                self.nobs, nreps=nreps, **kwargs
            )
            self.critval_kwargs: dict[str, Any] = kwargs
        return quantile(self.__simdist__, test_size)
