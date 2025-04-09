"""
Implementation of the Augmented Dickey-Fuller test for unit roots.
"""

from psytest.hyptest import HypothesisTest
import numpy as np
from numba import njit, prange


@njit(parallel=False)
def adfuller_stat(y: np.ndarray, kmax: int) -> np.float64:
    nobs: int = len(y)
    y_diff: np.ndarray = np.diff(y)
    X: np.ndarray = np.zeros((nobs - kmax - 1, 2 + kmax))
    X[:, 0] = np.ones(nobs - kmax - 1)  # Constant
    for k in prange(1, kmax + 1):
        X[:, k] = y_diff[kmax - k : -k]
    X[:, -1] = y[kmax:-1]
    y_diff = y_diff[kmax:]
    beta: np.ndarray = np.linalg.inv(X.T @ X) @ X.T @ y_diff
    fit: np.ndarray = X @ beta
    resid: np.ndarray = y_diff - fit
    ssr: np.float64 = np.sum(np.square(resid))
    sigma_sq_hat: np.float64 = ssr / (nobs - kmax - 2)

    coef: np.float64 = beta[-1]
    coef_var: np.float64 = sigma_sq_hat * np.diag(np.linalg.inv(X.T @ X))[-1]
    return coef / np.sqrt(coef_var)


def adfuller_dist(nobs: int, nreps: int, kmax: int) -> np.ndarray:
    random_walks: np.ndarray = np.cumsum(np.random.normal(size=(nreps, nobs)), axis=1)
    adf_dist: np.ndarray = np.zeros(nreps)
    for i in prange(nreps):
        y: np.ndarray = random_walks[i]
        adf_stat: np.float64 = adfuller_stat(y, kmax)
        adf_dist[i] = adf_stat
    return adf_dist


# ADFuller class


class ADFuller(HypothesisTest):
    def __init__(self, y: np.ndarray) -> None:
        super().__init__(y, adfuller_stat, adfuller_dist)


from typing import Any
from numpy import float64, quantile, array
from numpy.typing import NDArray
from numba import njit, prange
from collections.abc import Callable

TEST_SIZE: NDArray[float64] = array([0.1, 0.05, 0.01])
KMAX = 0


class BaseADFTest:
    def __init__(
        self,
        y: NDArray[float64],
        teststat_func: Callable = adfuller_stat,
        simdist_func: Callable = adfuller_dist,
        kmax: int = KMAX,
        **kwargs: dict[str, Any]
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
        self.y: NDArray[float64] = y
        self.nobs: int = len(y)
        self.__teststat_func__: Callable = teststat_func
        self.__simdist_func__: Callable = simdist_func
        self.kmax: int = kmax
        self.kwargs: dict[str, Any] = kwargs

    def teststat(self, **kwargs: dict[str, Any]) -> Any:
        """
        Calculates the test statistics for the hypothesis test.

        Args:
            **kwargs (dict[str, Any]): Keyword arguments for the test statistics function.

        Returns:
            float64: The test statistics.
        """
        if (
            not hasattr(self, "__teststat__")
            or (hasattr(self, "tstat_kwargs")
            and self.tstat_kwargs != kwargs)
        ):
            self.__teststat__: float64 = self.__teststat_func__(
                self.y, kmax=self.kmax, **kwargs
            )
            self.tstat_kwargs: dict[str, Any] = kwargs
        return self.__teststat__

    def critval(
        self,
        test_size: NDArray[float64] = TEST_SIZE,
        nreps: int = 1000,
        **kwargs: dict[str, Any]
    ) -> NDArray[Any]:
        """
        Calculates the critical values for the hypothesis test.

        Args:
            test_size (NDArray[float64]): The test sizes (alpha) to calculate the critical values.
            nreps (int, optional): Number of simulations to perform for the Monte Carlo. Defaults to 1000.

        Returns:
            NDArray[float64]: Critical values, on the order of the test sizes.
        """
        if not hasattr
        if (
            not hasattr(self, "__simdist__")
            or len(self.__simdist__) != nreps
            or self.critval_kwargs != kwargs
        ):
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
