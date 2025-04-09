from .adftest import adfuller_stat, ADFtest, NumArray, KMAX, TEST_SIZE, NREPS
from .utils.functions import (
    r0_default,
    simulate_random_walks,
)
from numpy import float64, inf, apply_along_axis, quantile
from numpy.typing import NDArray
from numba import njit, prange
from typing import Any


@njit(parallel=True)
def gsadfuller_stat(y: NumArray, r0: int, kmax: int) -> float64:
    """
    Calculates the test statistics for the Generalized Sup Augmented Dickey-Fuller test.

    Args:
        y (NumArray): The time series data.
        r0 (int): Initial period to start the test.
        kmax (int): Maximum lag to use in the test.

    Returns:
        float64: The test statistics.
    """
    nobs: int = len(y)
    stat: float64 = float64(-inf)
    for r2 in prange(r0, nobs + 1):
        for r1 in prange(0, r2 - r0):
            stat = max(stat, adfuller_stat(y[r1:r2], kmax))
    return stat


@njit(parallel=True)
def gsadfuller_asymp_dist(w: NumArray, r0: int) -> float64:
    """
    Generates one iteration from the asymptotic distribution of the Generalized Sup Augmented Dickey-Fuller test.

    Args:
        w (NumArray): The basic Wiener process.
        r0 (int): The initial period to start the test.

    Returns:
        float64: The test statistics.
    """
    nobs: int = len(w)
    stat: float = 0.0
    for r2 in prange(r0, nobs):
        for r1 in prange(r0, r2 - r0):
            rw: int = r2 - r1
            num: float = 1 / 2 * rw * (w[r2] ** 2 - w[r1] ** 2 - rw) - sum(w[r1:r2]) * (
                w[r2] - w[r1]
            )
            den: float = (
                rw**0.5 * (rw * sum(w[r1:r2] ** 2) - (sum(w[r1:r2])) ** 2) ** 0.5
            )
            stat = max(stat, num / den)
    return float64(stat)


def gsadfuller_dist(nobs: int, nreps: int, r0: int) -> NumArray:
    """
    Simulates the asymptotic distribution of the Generalized Sup Augmented Dickey-Fuller test.

    Args:
        nobs (int): Number of observations in the time series.
        nreps (int): Number of simulations to perform.
        r0 (int): Initial period to start the test.

    Returns:
        NumArray: The simulated distribution of the test statistics.
    """
    random_walks: NumArray = simulate_random_walks(nreps, nobs)
    return apply_along_axis(gsadfuller_asymp_dist, 1, random_walks, r0)


class GSADFtest(ADFtest):
    def __init__(self, y: NumArray, r0: int | None = None, kmax: int = KMAX) -> None:
        """
        Generalized Sup ADF test class.

        Args:
            y (NumArray): Time series data
            r0 (int | None, optional): Initial period to start the test. Defaults to None.
        """
        super().__init__(y=y, kmax=kmax)
        self.r0: int = r0 or r0_default(len(y))

    def teststat(self) -> float64:
        """
        Calculates the test statistics for the Generalized Sup Augmented Dickey-Fuller test.

        Returns:
            float64: The test statistics.
        """
        if not hasattr(self, "__teststat__"):
            self.__teststat__: float64 = gsadfuller_stat(
                y=self.y, r0=self.r0, kmax=self.kmax
            )
        return self.__teststat__

    def critval(
        self, test_size: NumArray = TEST_SIZE, nreps: int = NREPS
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
            self.__simdist__: NumArray = gsadfuller_dist(
                nobs=self.nobs, nreps=nreps, r0=self.r0
            )
            self.critval_kwargs: dict[str, Any] = {
                "test_size": test_size,
                "nreps": nreps,
            }
        return quantile(self.__simdist__, test_size)
