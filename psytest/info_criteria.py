"""psytest.info_criteria_functions
========================================

This module contains functions to calculate various information criteria which allow us to choose the parameters for the ADF test.
"""

from numpy.typing import NDArray
from numpy import float64, var, log, int64, diff, array, arange
from collections.abc import Callable
from typing import Literal
from .adftest import adfuller_fit


def bic(y: NDArray[float64], k: int) -> float:
    """Bayes Information Criterion (BIC)

    Parameters
    ----------
    y : :class:`numpy.ndarray`
        Fitted error values.
    k : :class:`int`
        Number of parameters in the model.

    Returns
    -------
    bic : :class:`float`
        The BIC value.
    """
    sigma_sq: float64 = var(y, ddof=k)
    n: int = len(y)
    return float(n * log(sigma_sq) + k * log(n))


def aic(y: NDArray[float64], k: int) -> float:
    """Akaike Information Criterion (AIC)

    Parameters
    ----------
    y : :class:`numpy.ndarray`
        Fitted error values.
    k : :class:`int`
        Number of parameters in the model.

    Returns
    -------
    aic : :class:`float`
        The AIC value.
    """

    sigma_sq: float64 = var(y, ddof=k)
    n: int = len(y)
    return float(2 * k - 2 * log(sigma_sq))


def aicc(y: NDArray[float64], k: int) -> float:
    """Corrected Akaike Information Criterion (AICc)

    Parameters
    ----------
    y : :class:`numpy.ndarray`
        Fitted error values.
    k : :class:`int`
        Number of parameters in the model.

    Returns
    -------
    aicc : :class:`float`
        The AICc value.
    """
    n: int = len(y)
    return aic(y, k) + 2 * k * (k + 1) / (n - k - 1)


def find_optimal_kmax(
    y: NDArray[float64],
    klimit: int,
    criteria: Literal["aic", "bic", "aicc"] = "bic",
) -> int:
    """Find the best model based on the given criteria.

    Parameters
    ----------
    error : list of :class:`numpy.ndarray`
        List of `m` fitted error values for each model.
    k_list : :class:`numpy.ndarray`
        List of `m` values for the number of parameters of each model.
    criteria : :class:`InfoCriteriaFunc`
        Enum member representing the information criterion to use.

    Raises
    ------
    ValueError
        If :paramref:`k_list` and :paramref:`error` have different lengths.

    Returns
    -------
    best_model : :class:`int`
        The optimal value of KMAX.
    """
    match criteria:
        case "aic":
            func: Callable[[NDArray[float64], int], float] = aic
        case "aicc":
            func: Callable[[NDArray[float64], int], float] = aicc
        case "bic":
            func: Callable[[NDArray[float64], int], float] = bic
        case _:
            raise ValueError(f"Unknown criteria: {criteria}")
    if not isinstance(klimit, int):
        raise TypeError("klimit must be an integer")
    if klimit < 0 or klimit > len(y) - 1:
        raise ValueError("klimit must be greater than or equal to 0")

    y_diff: NDArray[float64] = diff(y)
    klist: NDArray[int64] = arange(0, klimit + 1)
    kmax: int = klimit + 1
    y_diff_fitted: NDArray[float64] = array(
        [adfuller_fit(y=y, kmax=k)[kmax - k :] for k in klist]
    )
    error: NDArray[float64] = y_diff - y_diff_fitted
    criteria_values: list[float] = [func(y, k + 1) for y, k in zip(error, klist)]
    koptimal: int = klist[criteria_values.index(min(criteria_values))]
    return koptimal
