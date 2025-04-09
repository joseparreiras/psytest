from psytest.adftest import (
    rolling_adfuller_stat,
)
from psytest.utils.functions import random_walk, size_rgrid
from psytest.utils.constants import TEST_SIZE, KMAX
from numpy.typing import NDArray
from numpy import (
    float64,
    inf,
    zeros,
    repeat,
    int64,
    arange,
    empty,
    array,
    append,
    quantile,
    floor,
)
from collections.abc import Iterable
from deprecation import deprecated
from numba import njit, prange


# Sup ADF Test
@deprecated
@njit(parallel=True)
def sadfuller_stat(y: NDArray[float64], r0: float, rstep: float, kmax: int) -> float:
    stat: float = -inf
    for r in prange(r0, 1 + rstep, rstep):
        stat = max(stat, rolling_adfuller_stat(y, 0, r, kmax))
    return stat


@njit(parallel=True)
def __sadfuller_dist_from_random_walks__(
    random_walks: NDArray[float64], r0: float, rstep: float
) -> NDArray[float64]:
    """
    Calculates the asymptotic distribution of the Sup ADF test statistics based on a series of simulated random walks.

    Args:
        random_walks (NDArray[float64]): Simulated random walks of size (`nreps`, `nobs`).
        r0 (float): Minimum index to evaluate the test statistics.
        rstep (float): Step size for the index.

    Returns:
        NDArray[float64]: Array of size (`nreps`, int(floor((1 - r0) / rstep) + 1)) containing the test statistics.
    """
    nreps: int = random_walks.shape[0]
    r1r2_grid: NDArray[float64] = __r1r2_combinations__(r0, rstep)
    ntups: int = len(r1r2_grid)
    nstat: int = int(floor((1 - r0) / rstep) + 1)
    stats: NDArray[float64] = repeat(-inf, nreps * nstat)
    stats = stats.reshape((nreps, nstat))
    for j in range(nreps):
        for i in prange(ntups):
            r1: int = r1r2_grid[i][0]
            r2: int = r1r2_grid[i][1]
            idx: int = int(floor((r2 - r1 - r0) / rstep))
            stats[j, idx] = max(
                stats[j, idx], rolling_adfuller_stat(random_walks[j], r1, r2)
            )
    return stats


# Backward Sup ADF Test


@njit(parallel=True)
def bsadf_stat(y: NDArray[float64], r0: float, r2: float, kmax: int) -> float:
    """
    Calculates the Backward Sup ADF test statistic.

    Args:
        y (NDArray[float64]): Values of the time series.
        r0 (float): Minimum index to evaluate the test statistics.
        r2 (float): Index to evaluate the test statistics.
        kmax (int): Maximum lag to use in the test.

    Returns:
        float: The Backward Sup ADF test statistic.
    """
    stat: float = -inf
    for r1 in prange(r2 - r0 + 1):
        stat = max(stat, rolling_adfuller_stat(y, r1, r2, kmax))
    return stat


@njit(parallel=True)
def bsadf_stat_all_series(
    y: NDArray[float64], r0: float, rstep: float, kmax: int
) -> NDArray[float64]:
    """
    Calculates the Backward Sup ADF test statistics for all possible combinations of r1 and r2.

    Args:
        y (NDArray[float64]): Values of the time series.
        r0 (float): Minimum index to evaluate the test statistics.
        rstep (float): Step size for the index.
        kmax (int): Maximum lag to use in the test.

    Returns:
        NDArray[float64]: Array of size (`int(floor((1 - r0) / rstep) + 1)`,) containing the test statistics.
    """
    r1r2_grid: NDArray[float64] = __r1r2_combinations__(r0, rstep)
    ntups: int = len(r1r2_grid)
    nstat: int = int(floor((1 - r0) / rstep)) + 1
    stat: NDArray[float64] = repeat(-inf, nstat)
    for i in prange(ntups):
        r1: int = r1r2_grid[i][0]
        r2: int = r1r2_grid[i][1]
        i: int = int(floor((r2 - r1 - r0) / rstep))
        stat[i] = max(stat[i], rolling_adfuller_stat(y, r1, r2, kmax))
    return stat


@njit(parallel=False)
def __r1r2_combinations__(r0: float, rstep: float) -> NDArray[float64]:
    """
    Creates a vector with all possible combinations of (r1, r2) to evaluate the BSADF test. `r2` ranges from `r0` to `1`, and `r1` ranges from `0` to `r2 - r0`.

    Args:
        r0 (float): Minimum index to evaluate the test statistics.
        rstep (float): Step size for the index.

    Notes:
        - The final vector has size equal to `n * (n + 1) / 2`, where `n` is the number of steps from `r0` to `1` with step size `rstep`, or `n = int(floor((1 - r0) / rstep) + 1)`.

    Returns:
        NDArray[float64]: Vector containing the combinations of (r1, r2).
    """
    n: int = int(floor((1 - r0) / rstep) + 1)
    size = n * (n + 1) // 2
    result: NDArray[float64] = empty(shape=(size, 2), dtype=float64)
    idx: int = 0
    for r2 in arange(r0, 1 + rstep, rstep):
        for r1 in arange(0, r2 - r0 + 1e-16, rstep):
            result[idx, 0] = r1
            result[idx, 1] = r2
            idx += 1
    return result


def bsadfuller_critval(
    r0: float,
    rstep: float,
    nreps: int,
    nobs: int | None = None,  # type: ignore[assignment]
    test_size: list[float] | float = TEST_SIZE,
) -> NDArray[float64]:
    """
    Calculates the critical values of the Backward Sup ADF test from Monte Carlo simulations.

    Args:
        r0 (float): Minimum index to evaluate the test statistics.
        rstep (float): Step size for the index.
        nreps (int): Number of Monte Carlo simulations to perform.
        nobs (int | None, optional): Number of observations to use in the Monte Carlo Simulation. Defaults to None, using `1 / rstep`.
        testsize (list[float] | float, optional): Significance levels to use for the critical values. Defaults to TEST_SIZE (see `psytest.constants`).

    Returns:
        NDArray[float64]: Vector of size (`int(floor((1 - r0) / rstep) + 1)`, `len(testsize)`) containing the critical values for the test statistics.
    """
    if nobs is None:
        nobs: int = int(1 / rstep)
    rw: NDArray[float64] = random_walk(nreps, nobs)
    sadf_dist: NDArray[float64] = __sadfuller_dist_from_random_walks__(rw, r0, rstep)
    if isinstance(test_size, float):
        quant_float: float = 1 - test_size
        critval: NDArray[float64] = quantile(sadf_dist, quant_float, axis=0)
    elif isinstance(test_size, Iterable):
        quant_list: list[float] = [1 - q for q in test_size]
        critval: NDArray[float64] = quantile(sadf_dist, quant_list, axis=0)
    return critval
