"""psytest.utils.functions
==========================
This module contains utility functions for the psytest package.
It includes functions for generating random walks, simulating Markov processes, and calculating default parameters for bubble detection.
It should NOT be used directly by users and is solely intended for internal use within the psytest package and testing purposes.
"""

from numba import njit
from numpy import cumsum, float64, zeros, log, array, int64, ndarray, float64
from numpy.typing import NDArray
from numpy.random import normal, uniform
from collections.abc import Sequence
from typing import Any
from .defaults import KMAX
from ..info_criteria import find_optimal_kmax


def r0_default(nobs: int) -> int:
    """Calculates the default :code:`r0` parameter following Phillips, Shi & Yu (2015) as a function of the number of observations.

    Parameters
    ----------
    nobs : int
        Number of observations

    Returns
    -------
    default_r0 : int
        The default :code:`r0` parameter

    Notes
    -----
    .. math::
        r_0 = 0.01 \\times 0.08 \\times \\sqrt{n}
    """
    return 0.01 * 0.08 * nobs**0.5


def minlength_default(nobs: int, delta: float) -> int:
    """Calculates the minimum bubble length based on the number of observations.

    Parameters
    ----------
    nobs : int
        Number of observations
    delta : float
        Multiplier parameter for bubble length

    Returns
    -------
    minlength : int
        Minimum bubble length

    Notes
    -----
    .. math::
        \\text{min\\_length} = \\frac{\\delta \\log(n)}{n}
    """
    return delta * log(nobs) / nobs


def random_walk(nreps: int, nobs: int) -> NDArray[float64]:
    """Generates a monte carlo simulation of random walks.

    Parameters
    ----------
    nreps : int
        Number of repetitions.
    nobs : int
        Number of observations.

    Returns
    -------
    random_walk_matrix: NDArray[float64]
        Matrix of shape (:paramref:`nreps`, :paramref:`nobs`) with the random walks.

    Notes
    -----
    .. math::

        RW_{i, t} = \\sum_{s=1}^{t} \\frac{\\varepsilon_{i, s}}{\\sqrt{n}}
    """
    rw: NDArray[float64] = zeros((nreps, nobs))
    rw[:, 1:] = nobs ** (-0.5) * normal(size=(nreps, nobs - 1))
    return cumsum(rw, axis=1)


def simulate_markov(
    nobs: int, p=0.975, beta_list: list[float] = [1.01, 1]
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Simulates a two regime AR(1) process with a Markov switching beta.

    Parameters
    ----------
    nobs : int
        Number of observations for the process
    p : float, optional
        Probability of staying in the same regime. Defaults to 0.975.
    beta_list : list[float], optional
        List of :math:`\\beta` values for the regimes. Defaults to [1.01, 1].

    Returns
    -------
    [list[float], list[float]]: A tuple containing two lists
        the first one with the beta values and the second one with the simulated process values.

    Notes
    -----
    The process is defined as:
    .. math::

        y_t = \\beta_t y_{t-1} + \\varepsilon_t

    where :math:`\\varepsilon_t \\sim \\mathcal{N}(0, 1)` and :math:`\\beta_t` is a Markov switching variable that takes values from :paramref:`beta_list`.
    The transition probability is defined by :paramref:`p`, which is the probability of staying in the same regime.

    Raises
    ------
    TypeError
        If :paramref:`beta_list` is not a list; :paramref:`nobs` is not an integer; :paramref:`p` is not a float.
    ValueError
        If :paramref:`beta_list` does not have length 2; :paramref:`nobs` is less than 1; :paramref:`p` is not between 0 and 1.
    """
    if not isinstance(beta_list, Sequence):
        raise TypeError("`beta_list` must be a Sequence")
    if len(beta_list) != 2:
        raise ValueError("`beta_list` must have length 2")
    if not isinstance(nobs, int):
        raise TypeError("`nobs` must be an integer")
    if nobs < 1:
        raise ValueError("`nobs` must be greater than 0")
    if not isinstance(p, float):
        raise TypeError("`p` must be a float")
    if p < 0 or p > 1:
        raise ValueError("`p` must be between 0 and 1")
    err: NDArray[float64] = normal(size=nobs - 1)
    y: list[float] = [0.0]
    beta: list[float] = [min(beta_list)]
    for t in range(1, nobs):
        cur_beta: float = beta[t - 1]
        change_regime: bool = uniform() < 1 - p
        if change_regime:
            cur_beta_idx: int = beta_list.index(cur_beta)
            next_beta: float = beta_list[1 - cur_beta_idx]
        else:
            next_beta: float = cur_beta
        next_y: float = next_beta * y[t - 1] + err[t - 1]
        beta.append(next_beta)
        y.append(next_y)
        y[t] = beta[t] * y[t - 1] + err[t - 1]
    return (array(beta), array(y))


@njit
def size_rgrid(r0: float, rstep: float) -> int:
    """Calculates the size of the rgrid starting at :paramref:`r0` and with step :paramref:`rstep`.

    Parameters
    ----------
    r0 : float
        Minimum index to evaluate the test statistics.
    rstep : float
        Step size for the index.

    Returns
    -------
    size : int
        Size of the rgrid.

    Notes
    -----
    The size is calculated as:
    .. math::

        \\text{size} = \\left\\lfloor \\frac{1 - r_0}{\\text{rstep}} \\right\\rfloor + 1
    """
    return int((1 - r0) / rstep) + 1


def parse_psy_arguments(**kwargs) -> dict[str, Any]:
    """Parses the arguments for the :class:`psytest.bubbles.PSYBubbles` class and raises errors if they are invalid.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing the parameters to be validated.
        - data: 1D array-like of numbers
        - r0: float, default is calculated using :func:`r0_default`
        - rstep: float, default is 1 / len(data)
        - kmax: int | None. If none, finds the optimal value using :func:`psytest.info_criteria.find_optimal_kmax`.
        - minlength: float, default is calculated using `minlength_default`
        - delta: float, default is None

    Returns
    -------
    dict[str, Any]
        A dictionary with the validated parameters.

    Raises
    -------
    TypeError
        If any of the parameters are of the wrong type.
    ValueError
        If :code:`data` is not a 1D array, if :code:`r0` is not in the range [0, 1], if :code:`rstep` is not in the range (0, 1], if :code:`kmax` is not an integer or is out of bounds, if :code:`minlength` or :code:`delta` are not valid floats.
    """
    kwargs: dict[str, Any] = {k: v for k, v in kwargs.items() if v is not None}
    data: Any = kwargs.get("data")
    if data is None:
        raise ValueError("`data` must be provided")
    if not isinstance(data, ndarray):
        data: NDArray = array(data)
    if data.ndim != 1:
        raise ValueError("`data` must be a 1D array")
    if data.dtype not in [float64, int64]:
        raise ValueError("`data` must be a number array")
    if len(data) < 2:
        raise ValueError("`data` must have at least 2 elements")
    r0: Any = kwargs.get("r0", r0_default(len(data)))
    if not isinstance(r0, float):
        raise TypeError("`r0` must be a float")
    if not 0 <= r0 <= 1:
        raise ValueError("`r0` must be in the range [0, 1]")
    rstep: Any = kwargs.get("rstep", 1 / len(data))
    if not isinstance(rstep, float):
        raise TypeError("`rstep` must be a float")
    if not 0 < rstep <= 1:
        raise ValueError("`rstep` must be in the range (0, 1]")
    kmax: Any = kwargs.get("kmax", KMAX)
    if kmax is None:
        kmax: int = find_optimal_kmax(y=data, klimit=KMAX)
    elif not isinstance(kmax, int):
        raise TypeError("`kmax` must be an integer or None")
    if kmax < 0:
        raise ValueError("`kmax` must be greater than or equal to 0")
    if kmax > len(data) - 1:
        raise ValueError("`kmax` must be less than the length of `data`")
    minlength: Any = kwargs.get("minlength")
    delta: Any = kwargs.get("delta")
    if minlength is not None and delta is not None:
        raise ValueError("Only one of `minlength` or `delta` should be provided")
    if delta is not None:
        if not isinstance(delta, float):
            raise TypeError("`delta` must be a float")
        if delta <= 0:
            raise ValueError("`delta` must be greater than 0")
        minlength = minlength_default(nobs=len(data), delta=delta)
    if minlength is not None:
        if not isinstance(minlength, float):
            raise TypeError("`minlength` must be a float")
        if not 0 < minlength <= 1:
            raise ValueError("`minlength` must be in the range (0, 1]")
    return {
        "data": data,
        "r0": r0,
        "rstep": rstep,
        "kmax": kmax,
        "minlength": minlength,
        "delta": delta,
    }
