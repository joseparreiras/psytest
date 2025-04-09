from psytest.adftest import (
    rolling_adfuller_stat,
)
from psytest.utils.functions import random_walk, size_rgrid
from psytest.utils.constants import TEST_SIZE, KMAX
from numpy.typing import NDArray
from numpy import (
    float64,
    inf,
    repeat,
    int64,
    arange,
    empty,
    quantile,
)
from collections.abc import Iterable
from deprecation import deprecated
from numba import njit, prange


@deprecated
@njit(parallel=True)
def sadfuller_stat(y: NDArray[float64], r0: float, rstep: float, kmax: int) -> float:
    """
    Calculates the Sup ADF statistic by rolling over the interval [r0, 1].

    .. math::
        \\sup_{r \\in [r_0, 1]} \\text{ADF}(y_{0:r})

    Args:
        y (NDArray[float64]): Time series values.
        r0 (float): Minimum index.
        rstep (float): Step size.
        kmax (int): Max lag.

    Returns:
        float: Sup ADF statistic.
    """
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
        random_walks (NDArray[float64]): Simulated random walks of size (nreps, nobs).
        r0 (float): Minimum index.
        rstep (float): Step size.

    Returns:
        NDArray[float64]: Distribution matrix of Sup ADF statistics.
    """
    nreps: int = random_walks.shape[0]
    r1r2_grid: NDArray[float64] = __r1r2_combinations__(r0, rstep)
    ntups: int = len(r1r2_grid)
    nstat: int = size_rgrid(r0, rstep)
    stats: NDArray[float64] = repeat(-inf, nreps * nstat).reshape((nreps, nstat))
    for j in range(nreps):
        for i in prange(ntups):
            r1: int = r1r2_grid[i][0]
            r2: int = r1r2_grid[i][1]
            idx: int = int((r2 - r1 - r0) / rstep)
            stats[j, idx] = max(
                stats[j, idx], rolling_adfuller_stat(random_walks[j], r1, r2)
            )
    return stats


@njit(parallel=True)
def bsadf_stat(y: NDArray[float64], r0: float, r2: float, kmax: int) -> float:
    """
    Calculates the Backward Sup ADF test statistic.

    .. math::
        \\text{BSADF}(r_2) = \\max_{r_1 \\in [0, r_2 - r_0]} \\text{ADF}(y, r_1, r_2)

    Args:
        y (NDArray[float64]): Time series values.
        r0 (float): Minimum index.
        r2 (float): End index.
        kmax (int): Max lag.

    Returns:
        float: BSADF statistic.
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
    Calculates BSADF statistics over all possible (r1, r2) combinations.

    Args:
        y (NDArray[float64]): Time series values.
        r0 (float): Minimum index.
        rstep (float): Step size.
        kmax (int): Max lag.

    Returns:
        NDArray[float64]: Array of test statistics.
    """
    r1r2_grid: NDArray[float64] = __r1r2_combinations__(r0, rstep)
    ntups: int = len(r1r2_grid)
    nstat: int = size_rgrid(r0, rstep)
    stat: NDArray[float64] = repeat(-inf, nstat)
    for i in prange(ntups):
        r1: int = r1r2_grid[i][0]
        r2: int = r1r2_grid[i][1]
        i: int = int((r2 - r1 - r0) / rstep)
        stat[i] = max(stat[i], rolling_adfuller_stat(y, r1, r2, kmax))
    return stat


@njit(parallel=False)
def __r1r2_combinations__(r0: float, rstep: float) -> NDArray[float64]:
    """
    Creates a grid of all (r1, r2) index pairs to evaluate BSADF.

    .. math::
        r_2 \\in [r_0, 1], \\quad r_1 \\in [0, r_2 - r_0]

    Args:
        r0 (float): Minimum index.
        rstep (float): Step size.

    Notes:
        - Vector size: :math:`n(n+1)/2` where :math:`n = \\lfloor (1 - r_0)/rstep \\rfloor + 1`

    Returns:
        NDArray[float64]: Grid of (r1, r2) pairs.
    """
    n: int = size_rgrid(r0, rstep)
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
    test_size: Iterable | float = TEST_SIZE,
) -> NDArray[float64]:
    """
    Calculates critical values of BSADF statistics from Monte Carlo simulations.

    .. math::
        \\text{CV}_{i,\\alpha} = \\text{Quantile}_{1 - \\alpha}(\\text{BSADF}_i)

    Args:
        r0 (float): Minimum index.
        rstep (float): Step size.
        nreps (int): Number of replications.
        nobs (int | None): Number of observations. Defaults to :math:`1/rstep`.
        test_size (Iterable | float): Significance levels.

    Returns:
        NDArray[float64]: Critical values matrix.
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
