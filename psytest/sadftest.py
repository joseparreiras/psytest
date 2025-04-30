"""psytest.sadftest
=================

This module contains the functions related to the Sup ADF test and the Backward Sup ADF test. The functions allow us to calculate both the test statistic and the asymptotic distribution of the test statistic

See :mod:`psytest.adfstat` for the basic Augmented Dickey-Fuller test.
"""

from numpy.typing import NDArray
from numpy import float64, inf, repeat, arange, array, quantile, round
from collections.abc import Iterable
from deprecation import deprecated
from numba import njit, prange

from .adftest import rolling_adfuller_stat
from .utils.functions import random_walk, size_rgrid


@deprecated
@njit(parallel=True)
def sadfuller_stat(y: NDArray[float64], r0: float, rstep: float, kmax: int) -> float:
    """Calculates the Sup ADF statistic by rolling over the interval [0, 1].

    Parameters
    ----------
    y : NDArray[float64]
        Time series values.
    r0 : float
        Minimum index.
    rstep : float
        Step size.
    kmax : int
        Max lag.

    Returns
    -------
    teststat: float
        Sup ADF statistic.

    Notes
    -----
    The Sup ADF statistic is calculated as:
    .. math::

        \\text{SADF} = \\max_{r \\in [r_0, 1]} \\text{ADF}(y_{0:r})

    where :math:`\\text{ADF}(y_{0:r})` is the Augmented Dickey-Fuller test statistic for the series :math:`y_{0:r}` (see :func:`psytest.adfstat.rolling_adfuller_stat`).
    """
    stat: float = -inf
    for r in prange(r0, 1 + rstep, rstep):
        stat = max(stat, rolling_adfuller_stat(y, kmax=kmax, r1=0, r2=r))
    return stat


@njit(parallel=True)
def __sadfuller_dist_from_random_walks__(
    random_walks: NDArray[float64], r0: float, rstep: float, kmax: int
) -> NDArray[float64]:
    """Calculates the asymptotic distribution of the Sup ADF test statistics based on a series of simulated random walks.

    Parameters
    ----------
    random_walks : NDArray[float64]
        Simulated random walks of size (nreps, nobs).
    r0 : float
        Minimum index.
    rstep : float
        Step size.

    Returns
    -------
    distribution: NDArray[float64]
        Distribution matrix of Sup ADF statistics.
    """
    nreps: int = random_walks.shape[0]
    r1r2_grid: NDArray[float64] = make_r1r2_combinations(r0, rstep)
    ntups: int = len(r1r2_grid)
    r2_grid: NDArray[float64] = make_r2_grid(r0, rstep)
    nstat: int = len(r2_grid)
    stats: NDArray[float64] = repeat(-inf, nreps * nstat).reshape((nreps, nstat))
    for j in range(nreps):
        for i in prange(ntups):
            r1: int = r1r2_grid[i][0]
            r2: int = r1r2_grid[i][1]
            idx: int = int(round((r2 - r0) / rstep, 0))
            stats[j, idx] = max(
                stats[j, idx],
                rolling_adfuller_stat(random_walks[j], kmax=kmax, r1=r1, r2=r2),
            )
    return stats


@njit(parallel=True)
def bsadf_stat(y: NDArray[float64], r0: float, r2: float, kmax: int) -> float:
    """Calculates the Backward Sup ADF test statistic.

    Parameters
    ----------
    y : NDArray[float64]
        Time series values.
    r0 : float
        Minimum index.
    r2 : float
        End index.
    kmax : int
        Max lag.

    Returns
    -------
    teststat: float
        BSADF statistic.

    Notes
    -----
    The Backward Sup ADF statistic is calculated as:
    .. math::

        \\text{BSADF}(r_2) = \\max_{r_1 \\in [0, r_2 - r_0]} \\text{ADF}(y_{r_1:r_2})

    where :math:`\\text{ADF}(y_{r_1:r_2})` is the Augmented Dickey-Fuller test statistic for the series :math:`y_{r_1:r_2}` (see :func:`psytest.adfstat.rolling_adfuller_stat`).
    """
    stat: float = -inf
    for r1 in prange(r2 - r0 + 1):
        stat = max(stat, rolling_adfuller_stat(y, kmax=kmax, r1=r2, r2=r2))
    return stat


@njit(parallel=True)
def bsadf_stat_all_series(
    y: NDArray[float64], r0: float, rstep: float, kmax: int
) -> NDArray[float64]:
    """Calculates BSADF statistics over all possible (r1, r2) combinations.

    Parameters
    ----------
    y : NDArray[float64]
        Time series values.
    r0 : float
        Minimum index.
    rstep : float
        Step size.
    kmax : int
        Max lag.

    Returns
    -------
    teststat_array: NDArray[float64]
        Array of test statistics.
    """
    r1r2_grid: NDArray[float64] = make_r1r2_combinations(r0, rstep)
    ntups: int = len(r1r2_grid)
    nstat: int = size_rgrid(r0, rstep)
    stat: NDArray[float64] = repeat(-inf, nstat)
    for i in prange(ntups):
        r1: int = r1r2_grid[i][0]
        r2: int = r1r2_grid[i][1]
        j: int = int(round((r2 - r0) / rstep, 0))
        stat[j] = max(stat[j], rolling_adfuller_stat(y, kmax=kmax, r1=r1, r2=r2))
    return stat


@njit
def r2_index(r2: float, r0: float, rstep: float) -> int:
    """Get the index of :paramref:`r2` in the grid from :paramref:`r0` to 1 with step size :paramref:`rstep`"""
    return int((r2 - r0) // rstep)


@njit
def make_r2_grid(r0: float, rstep: float) -> NDArray[float64]:
    """Creates the grid of all possible `r2` values."""
    return arange(r0, 1 + rstep, rstep)


@njit
def make_r1_grid(r2: float, r0: float, rstep: float) -> NDArray[float64]:
    """Creates the grid of all possible `r1` values given `r2`."""
    return arange(0, r2 - r0 + rstep, rstep)


@njit(parallel=False)
def make_r1r2_combinations(r0: float, rstep: float) -> NDArray[float64]:
    """Creates a grid of all (r1, r2) index pairs to evaluate BSADF.

    Parameters
    ----------
    r0 : float
        Minimum index.
    rstep : float
        Step size.

    Returns
    -------
    [float64]
        Grid of (r1, r2) pairs.

    Notes
    -----
    The grid is defined as:
    .. math::

        \\text{grid} = \\{(r_1, r_2) : r_0 \\leq r_2 \\leq 1, \\quad r_0 \\leq r_1 \\leq r_2 - r_0, \\\\
        \\quad rstep = (r_2 - r_0)/n, \\\\
        \\quad n = \\lfloor (1 - r_0)/rstep \\rfloor + 1\\}

    with :math:`r_0` being the minimum index, :math:`r_2` being the end index, and :math:`r_1` being the start index. The values of :math:`r_1` and :math:`r_2` are created from a grid with increments of :paramref:`rstep`.
    """
    result: list[tuple[float, float]] = []
    for r2 in make_r2_grid(r0=r0, rstep=rstep):
        for r1 in make_r1_grid(r2=r2, r0=r0, rstep=rstep):
            result.append((r1, r2))
    return array(result, dtype=float64)


def bsadfuller_critval(
    r0: float,
    rstep: float,
    nreps: int,
    nobs: int,
    alpha: Iterable | float,
    kmax: int,
) -> NDArray[float64]:
    """Calculates critical values of BSADF statistics from Monte Carlo simulations.

    Parameters
    ----------
    r0 : float
        Minimum index.
    rstep : float
        Step size.
    nreps : int
        Number of replications.
    nobs : int | None
        Number of observations.
    alpha : Iterable | float
        Significance levels.

    Returns
    -------
    critval: NDArray[float64]
        Critical values matrix.

    Notes
    -----
    The critical values are calculated as:
    .. math::

        \\text{CV}_{i,\\alpha} = \\text{Quantile}_{1 - \\alpha}(\\text{BSADF}_i)

    where :math:`\\text{BSADF}_i` is the :math:`i`-th simulated BSADF statistic (see :func:`psytest.adfstat.bsadf_stat`) and :math:`\\text{Quantile}_{1 - \\alpha}` is the quantile function for the distribution of the BSADF statistic.
    """
    rw: NDArray[float64] = random_walk(nreps, nobs)
    sadf_dist: NDArray[float64] = __sadfuller_dist_from_random_walks__(
        random_walks=rw, r0=r0, rstep=rstep, kmax=kmax
    )
    if isinstance(alpha, float):
        quant_float: float = 1 - alpha
        critval: NDArray[float64] = quantile(sadf_dist, quant_float, axis=0)
    elif isinstance(alpha, Iterable):
        quant_list: list[float] = [1 - q for q in alpha]
        critval: NDArray[float64] = quantile(sadf_dist, quant_list, axis=0)
    return critval
