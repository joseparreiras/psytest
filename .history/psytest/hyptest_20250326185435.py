from typing import Any
from numpy import float64, quantile
from numpy.typing import NDArray
from numba import njit, prange
from collections.abc import Callable

TEST_SIZE: NDArray[float64] = float64([0.1, 0.05, 0.01])

class HypothesisTest:
    def __init__(
        self, y: NDArray[float64], teststat_func: Callable, simdist_func: Callable
    ) -> None:
        self.y: NDArray[float64] = y
        self.nobs: int = len(y)
        self.__teststat_func__: Callable = teststat_func
        self.__simdist_func__: Callable = simdist_func

    def teststat(self, **kwargs: dict[str, Any]) -> float64:
        """
        Calculates the test statistics for the hypothesis test.

        Args:
            **kwargs (dict[str, Any]): Keyword arguments for the test statistics function.

        Returns:
            float64: The test statistics.
        """
        if not hasattr(self, "__teststat__"):
            self.__teststat__: float64 = self.__teststat_func__(self.y, **kwargs)
            self.tstat_kwargs: dict[str, Any] = kwargs
        return self.__teststat__

    def critval(
        self, test_size: NDArray[float64] = TEST_SIZE, nreps: int = 1000, **kwargs: dict[str, Any]
    ) -> NDArray[float64]:
        """
        Calculates the critical values for the hypothesis test.

        Args:
            test_size (NDArray[float64]): The test sizes (alpha) to calculate the critical values.
            nreps (int, optional): Number of simulations to perform for the Monte Carlo. Defaults to 1000.

        Returns:
            NDArray[float64]: Critical values, on the order of the test sizes.
        """
        if not hasattr(self, "__simdist__") or len(self.__simdist__) != nreps:
            self.__simdist__: NDArray[float64] = self.__simdist_func__(
                self.nobs, nreps=nreps, **kwargs
            )
            self.critval_kwargs: dict[str, Any] = kwargs
        return quantile(self.__simdist__, test_size)

    @property
    def simdist(self) -> NDArray[float64]:
        """
        Returns the simulated distribution of the test statistics.

        Returns:
            NDArray[float64]: The simulated distribution of the test statistics.
        """
        if not hasattr(self, "__simdist__"):
            raise AttributeError(
                "Simulated distribution not available. Run the `critval` method first."
            )
        return self.__simdist__
