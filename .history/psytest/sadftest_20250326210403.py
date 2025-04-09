from .adftest import adfuller_stat, ADFtest, NumArray
from .utils.functions import (
    r0_default,
    simulate_random_walks,
)
from numpy import float64, inf, apply_along_axis
from numba import njit, prange


# Sup ADF Test
@njit(parallel=True)
def sadfuller_stat(y: NumArray, r0: int, kmax: int) -> float64:
    """
    Calculates the test statistics for the Sup Augmented Dickey-Fuller test.

    Args:
        y (NumArray): The time series data.
        r0 (int): Initial period to start the test.
        kmax (int): Maximum lag to use in the test.

    Returns:
        float64: The test statistics.
    """
    nobs: int = len(y)
    stat: float64 = float64(-inf)
    for r in prange(r0, nobs + 1):
        stat = max(stat, adfuller_stat(y[:r], kmax))
    return stat


@njit(parallel=True)
def sadfuller_asymp_dist(w: NumArray, r0: int) -> float64:
    """
    Generates one iteration from the asymptotic distribution of the Sup ADF test.

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


def sadfuller_dist(nobs: int, nreps: int, r0: int) -> NumArray:
    """
    Simulates the asymptotic distribution of the Sup Augmented Dickey-Fuller test.

    Args:
        nobs (int): Number of observations in the time series.
        nreps (int): Number of simulations to perform.
        r0 (int): Initial period to start the test.

    Returns:
        NumArray: The simulated distribution of the test statistics.
    """
    random_walks: NumArray = simulate_random_walks(nreps, nobs)
    return apply_along_axis(sadfuller_asymp_dist, 1, random_walks, r0)


class SADFtest(ADFtest):
    def __init__(self, y: NumArray, kmax: int = KMAX, r0: int | None = None) -> None:
        """
        Sup ADF test class.

        Args:
            y (np.ndarray): Time series data.
            r0 (int | None, optional): Initial period to start the test. Defaults to None.
        """
        super().__init__(y, sadfuller_stat, sadfuller_dist)
        self.r0: int = r0 or r0_default(len(y))
