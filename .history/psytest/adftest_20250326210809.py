"""
Implementation of the Augmented Dickey-Fuller test for unit roots.
"""

import numpy as np
from numba import njit, prange
from typing import Any
from numpy import float64, quantile, array, sum, square, diag, diff, empty, ones, sqrt
from numpy.linalg import inv
from numpy.typing import NDArray
from numba import njit, prange
from collections.abc import Callable
from typing import TypeAlias, ParamSpec, Concatenate
from .utils.functions import simulate_random_walks

# Type aliases
NumArray: TypeAlias = NDArray[float64]
Param = ParamSpec("Param")
TStatFunc: TypeAlias = Callable[Concatenate[NumArray, int, Param], float64]
SimDistFunc: TypeAlias = Callable[Concatenate[int, int, int, Param], NumArray]

# Global variables
TEST_SIZE: NumArray = array([0.1, 0.05, 0.01])
NREPS: int = 1000
KMAX: int = 0


@njit(parallel=False)
def adfuller_stat(y: NumArray, kmax: int) -> float64:
    """
    Calculates the test statistics for the Augmented Dickey-Fuller test.

    Args:
        y (NumArray): The time series data.
        kmax (int): Maximum lag to use in the test.

    Returns:
        float64: The test statistics.
    """
    nobs: int = len(y)
    y_diff: NumArray = diff(y)
    X: NumArray = empty((nobs - kmax - 1, 2 + kmax))
    X[:, 0] = ones(nobs - kmax - 1)  # Constant
    for k in prange(1, kmax + 1):
        X[:, k] = y_diff[kmax - k : -k]
    X[:, -1] = y[kmax:-1]
    y_diff = y_diff[kmax:]
    beta: NumArray = inv(X.T @ X) @ X.T @ y_diff
    fit: NumArray = X @ beta
    resid: NumArray = y_diff - fit
    ssr: float64 = sum(square(resid))
    sigma_sq_hat: float64 = ssr / (nobs - kmax - 2)

    coef: float64 = beta[-1]
    coef_var: float64 = sigma_sq_hat * diag(np.linalg.inv(X.T @ X))[-1]
    return coef / sqrt(coef_var)


def adfuller_dist(nobs: int, nreps: int, kmax: int) -> NumArray:
    """
    Simulates tha asymptotic distribution of the Augmented Dickey-Fuller test.

    Args:
        nobs (int): Number of observations in the time series.
        nreps (int): Number of simulations to perform.
        kmax (int): Maximum lag to use in the test.

    Returns:
        NumArray: The simulated distribution of the test statistics.
    """
    random_walks: NumArray = simulate_random_walks(nreps, nobs)
    adf_dist: NumArray = empty(nreps)
    for i in prange(nreps):
        y: NumArray = random_walks[i]
        adf_stat: float64 = adfuller_stat(y, kmax)
        adf_dist[i] = adf_stat
    return adf_dist


# ADFuller class


class ADFtest:
    def __init__(
        self,
        y: NumArray,
        teststat_func: TStatFunc = adfuller_stat,
        simdist_func: SimDistFunc = adfuller_dist,
        kmax: int = KMAX,
        **kwargs: Any
    ) -> None:
        """
        Basic class for the initialization of the ADF tests.

        Args:
            y (NDArray[float64]): The time series data.
            teststat_func (Callable): The function to calculate the test statistics.
            simdist_func (Callable): The function to simulate the distribution of the test statistics.
            kmax (int, optional): The maximum lag to use in the ADF test. Defaults to KMAX.
            **kwargs (dict[str, Any]): Additional keyword arguments for the test.
        """
        self.y: NumArray = y
        self.nobs: int = len(y)
        self.__teststat_func__: TStatFunc = teststat_func
        self.__simdist_func__: SimDistFunc = simdist_func
        self.kmax: int = kmax
        self.kwargs: dict[str, Any] = kwargs

    def teststat(self, **kwargs: Any) -> Any:
        """
        Calculates the test statistics for the hypothesis test.

        Args:
            **kwargs (dict[str, Any]): Keyword arguments for the test statistics function.

        Returns:
            float64: The test statistics.
        """
        if not hasattr(self, "__teststat__") or (
            hasattr(self, "tstat_kwargs") and self.tstat_kwargs != kwargs
        ):
            self.__teststat__: float64 = self.__teststat_func__(
                self.y, kmax=self.kmax, **kwargs, **self.kwargs
            )
            self.tstat_kwargs: dict[str, Any] = kwargs
        return self.__teststat__

    def critval(
        self, test_size: NumArray = TEST_SIZE, nreps: int = NREPS, **kwargs: Any
    ) -> NDArray[Any]:
        """
        Calculates the critical values for the hypothesis test.

        Args:
            test_size (NDArray[float64]): The test sizes (alpha) to calculate the critical values.
            nreps (int, optional): Number of simulations to perform for the Monte Carlo. Defaults to 1000.

        Returns:
            NDArray[float64]: Critical values, on the order of the test sizes.
        """
        if (
            not hasattr(self, "__simdist__")
            or len(self.__simdist__) != nreps
            or (hasattr(self, "critval_kwargs") and self.critval_kwargs != kwargs)
        ):
            self.__simdist__: NumArray = self.__simdist_func__(
                nobs=self.nobs, nreps=nreps, kmax=self.kmax, **kwargs, **self.kwargs
            )
            self.critval_kwargs: dict[str, Any] = kwargs
        return quantile(self.__simdist__, test_size)

    @property
    def simdist(self) -> NumArray:
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
