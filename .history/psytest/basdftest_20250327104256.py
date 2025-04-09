from .adftest import adfuller_stat, ADFtest, NumArray, KMAX, TEST_SIZE, NREPS
from .utils.functions import (
    r0_default,
    simulate_random_walks,
)
from numpy.typing import NDArray
from numpy import float64, inf, apply_along_axis, quantile
from numba import njit, prange
from typing import Any


# Backward Sup ADF Test
def basdfuller_stat(y: NumArray, r0: int, kmax: int) -> float64:
    """
    Calculates the test statistics for the Backward Sup Augmented Dickey-Fuller test.

    Args:
        y (NumArray): The time series data.
        r0 (int): Initial period to start the test.
        kmax (int): Maximum lag to use in the test.

    Returns:
        float64: The test statistics.
    """
