"""psytest.bubbles
=========================

This module contains the main class for the PSY test (see :class:`psytest.bubbles.PSYBubbles`). This class handles the frontend of the test, receiving the data and parameters, and providing methods to run the test and retrieve results. It also includes methods for parsing arguments, checking conditions, and finding bubble dates.
"""

from numpy.typing import NDArray
from numpy import float64, int64, generic, bool_, array, arange, vstack, ndarray
from collections.abc import Generator
from typing import Any, Literal, overload, TypeVar, Generic
from functools import lru_cache
from pandas import Series
import logging

from .critval import critval_tabulated, is_available_param
from .utils.functions import r0_default, minlength_default
from .utils.defaults import ALPHA_LIST, NREPS, LAGMAX
from .sadftest import bsadf_stat_all_series, bsadfuller_critval
from .info_criteria import find_optimal_kmax

logging.basicConfig(level=logging.CRITICAL, format="%(levelname)s: %(message)s")

IndexType = TypeVar("IndexType", bound=generic)


class PSYBubbles(Generic[IndexType]):
    """Class to perform the Phillips, Shi & Yu (2015) test for bubbles in time series data.

    Parameters
    ----------
    data : :class:`numpy.ndarray` of dtype :class:`numpy.float64`
        Time series values.
    minwindow : int | None, optional
        Minimum window size for the estimation. Defaults to using :func:`psytest.utils.functions.r0_default`.
    minlength : int | None, optional
        Minimum bubble length. Defaults to :func:`psytest.utils.functions.minlength_default`.
    lagmax : int, optional
        Maximum lag :math:`k_{\\max}`. If none, uses :func:`psytest.info_criteria.find_optimal_kmax` to find the optimal value.
    rstep : float | None, optional
        Step size :math:`r_{\\text{step}}`. Defaults to :math:`1/n` where :math:`n` is the size of :paramref:`data`.
    delta: float, optional
        Default is 1.0. The parameter to determine the minimum length of bubbles. Used only if :paramref:`minlength` is None.

    Raises
    -------
    TypeError
        If arguments are not of the expected type.
    ValueError
        If :paramref:`r0` and :paramref:`rstep` are not in the range (0, 1) or if :paramref:`kmax` is not a positive integer.
    """

    def __init__(
        self,
        data: NDArray[float64],
        minwindow: int | None = None,
        minlength: int | None = None,
        lagmax: int | None = None,
        rstep: float | None = None,
        delta: float = 1.0,
    ) -> None:
        parsed_args: dict[str, Any] = __parse_psy_arguments__(
            data=data,
            minwindow=minwindow,
            rstep=rstep,
            lagmax=lagmax,
            minlength=minlength,
            delta=delta,
        )
        self.data: NDArray[float64] = parsed_args["data"]
        self.nobs: int = len(self.data)
        self.index: NDArray[IndexType] | None = None
        self.minwindow: int = parsed_args["minwindow"]
        self.minlength: int = parsed_args["minlength"]
        self.lagmax: int = parsed_args["lagmax"]
        self.r0: float = self.minwindow / self.nobs
        self.rstep: float = parsed_args["rstep"]
        self.r2grid: NDArray[float64] = arange(self.r0, 1 + 1e-16, self.rstep)

    @lru_cache(maxsize=128)
    def teststat(self, force: bool = False) -> dict[float | IndexType, float]:
        """Retrieves the BSADF test statistic.

        Parameters
        ----------
        force : bool, optional
            If :literal:`True`, forces recalculation.

        Returns
        -------
        teststat : dictionary with float keys and float values
            Test statistics for each observation. If the object contains an index. It will be used as keys for the dictionary. Otherwise, the function returns the test statistic as a function of :math:`r_2` in :func:`psytest.PSYBubbles.r2grid`.

        Notes
        -----
        The test statistic is defined as:

        .. math::

            \\text{BSADF}_{r_2} = \\max_{r_1 \\in [0, r_2 - r_0]} \\text{ADF}(y_{r_1:r_2})

        See the original paper for more details.
        """
        if not isinstance(force, bool):
            raise TypeError("`force` must be a boolean")
        if force or not hasattr(self, "__teststat"):
            stat: NDArray[float64] = bsadf_stat_all_series(
                self.data, self.r0, self.rstep, self.lagmax
            )
            self.__teststat: dict[float, float] = dict(zip(self.r2grid, stat))
        if hasattr(self, "index") and self.index is not None:
            i0: int = int(self.nobs * self.r0)
            return dict(zip(self.index[i0:], self.__teststat.values()))
        else:
            return self.__teststat

    @overload
    def critval(
        self, alpha: float | list[float], fast: Literal[True]
    ) -> dict[float | IndexType, NDArray[float64]]: ...

    @overload
    def critval(
        self,
        alpha: float | list[float],
        fast: Literal[False],
        nreps: int,
        nobs: int | None,
    ) -> dict[float | IndexType, NDArray[float64]]: ...

    @lru_cache(maxsize=5)
    def critval(
        self,
        alpha: list[float] | float = ALPHA_LIST,
        fast: Literal[True, False] = True,
        nreps: int = NREPS,
        nobs: int | None = None,
    ) -> dict[float | IndexType, NDArray[float64]]:
        """Retrieves BSADF critical values using Monte Carlo Simulations.

        Parameters
        ----------
        alpha : list[float] | float, optional
            Significance levels :math:`\\alpha`. Defaults to ALPHA_LIST (see :mod:`psytest.utils.defaults`)
        fast : bool, optional
            If :literal:`True` (Default), uses tabulated critical values. Otherwise, simulates them using :paramref:`nreps` simulations of size :paramref:`nobs`.
        nreps : int, optional
            Number of simulations (required if :paramref:`fast` is :literal:`True`).
        nobs : int | None, optional
            Number of observations (used if :paramref:`fast` is :literal:`False`). Defaults to None, setting it to :paramref:`psytest.PSYBubbles.nobs`.

        Returns
        -------
        critval : dict
            Dictionary with critical values for each :math:`r_2` in :func:`psytest.PSYBubbles.r2grid`. The keys are the :math:`r_2` values and the values are the critical values for the given significance level.
            If :paramref:`alpha` is a list, the keys are the :math:`r_2` values and the values are an array of critical values for each significance level.
            If :paramref:`alpha` is a float, the keys are the :math:`r_2` values and the values are the critical values for the given significance level.

        Raises
        -------
        TypeError
            If :paramref:`alpha` is not a float or list of floats, or if :paramref:`fast` is not a boolean.
        ValueError
            If :paramref:`alpha` is not in the range (0, 1) or if :paramref:`nreps` is not a positive integer.

        Notes
        -----
        This function uses :code:`functools.lru_cache` to cache the results for faster access. So after the first call, subsequent calls with the same parameters will utilize the cached results to reduce computation time.
        """
        if not isinstance(fast, bool):
            raise TypeError("`fast` must be a boolean")
        if isinstance(alpha, float):
            if alpha <= 0 or alpha >= 1:
                raise ValueError("`alpha` must be in the range (0, 1)")
        elif isinstance(alpha, list):
            for size in alpha:
                if not isinstance(size, float):
                    raise TypeError("`alpha` must be a list of floats")
                if size <= 0 or size >= 1:
                    raise ValueError("`alpha` must be in the range (0, 1)")
        else:
            raise TypeError("`alpha` must be a list of floats or a float")
        if fast:
            statvals: dict[float, NDArray[float64]] = self._critval_tabulated(
                alpha=alpha
            )
        else:
            statvals: dict[float, NDArray[float64]] = self._critval_simulated(
                alpha=alpha, nreps=nreps, nobs=nobs
            )
        if hasattr(self, "index") and self.index is not None:
            i0: int = int(self.nobs * self.r0)
            return dict(zip(self.index[i0:], statvals.values()))
        else:
            return statvals

    def _critval_tabulated(
        self, alpha: float | list[float]
    ) -> dict[float, NDArray[float64]]:
        kmax: int = self.lagmax
        r0: float = self.r0
        r2grid: NDArray[float64] = self.r2grid
        if not is_available_param(kmax=self.lagmax, r0=self.r0):
            raise ValueError(
                f"Parameters kmax={kmax!r} and r0={r0!r} are not available for tabulated critical values."
            )
        if isinstance(alpha, float):
            cval: NDArray[float64] | float = critval_tabulated(
                r2grid, alpha=alpha, kmax=self.lagmax, r0=self.r0
            )
        elif isinstance(alpha, list):
            cval: NDArray[float64] | float = vstack(
                [
                    critval_tabulated(r2grid, alpha=alpha, kmax=self.lagmax, r0=self.r0)
                    for alpha in alpha
                ]
            ).T
        return dict(zip(r2grid, cval))

    def _critval_simulated(
        self, alpha: float | list[float], nreps: int, nobs: int | None
    ) -> dict[float, NDArray[float64]]:
        if not isinstance(nreps, int):
            raise TypeError("`nreps` must be an integer")
        if nreps <= 0:
            raise ValueError("`nreps` must be greater than 0")
        if nobs is None:
            nobs = self.nobs
        else:
            if not isinstance(nobs, int):
                raise TypeError("`nobs` must be an integer")
            if nobs <= 0:
                raise ValueError("`nobs` must be greater than 0")
        r2grid: NDArray[float64] = self.r2grid
        cval: NDArray[float64] = bsadfuller_critval(
            r0=self.r0,
            rstep=self.rstep,
            nreps=nreps,
            nobs=nobs,
            alpha=alpha,
            kmax=self.lagmax,
        )
        return dict(zip(r2grid, cval))

    @overload
    def find_bubbles(self, alpha: float, fast: Literal[True]) -> NDArray: ...

    @overload
    def find_bubbles(
        self, alpha: float, fast: Literal[False], nreps: int, nobs: int | None
    ) -> NDArray: ...

    def find_bubbles(
        self,
        alpha: float = 0.05,
        fast: Literal[True] | Literal[False] = True,
        nreps: int = NREPS,
        nobs: int | None = None,
    ) -> NDArray:
        """Identifies the bubble periods in the time series.

        Parameters
        ----------
        alpha : float
            Significance level :math:`\\alpha`. Defaults to 0.05.
        fast : bool, optional
            If :literal:`True` (Default), uses tabulated critical values. Otherwise, simulates them using :paramref:`nreps` simulations of size :paramref:`nobs`.
        nreps : int, optional
            Number of simulations (required if :paramref:`fast` is :literal:`True`).
        nobs : int | None, optional
            Number of observations (used if :paramref:`fast` is :literal:`False`). Defaults to None, setting it to :paramref:`psytest.PSYBubbles.nobs`.

        Returns
        -------
        bubbles : :class:`numpy.ndarray`
            Array with 2 columns. The first element contains the start time of the bubble, given by the index of the data. The second element contains the end time of the data. If the bubble has not ended by the end of the data, it is set to :class:`None`.

        Notes
        -----
        The start of a bubble is defined as the point :math:`r_s` where the test statistic exceeds the critical value:

        .. math::
            \\text{BSADF}_{r_s} > \\text{CV}_{r_s, \\alpha}

        And the end of a bubble is defined as the point :math:`r_e > r_s` where the test statistic falls below the critical value:

        .. math::
            \\text{BSADF}_{r_e} < \\text{CV}_{r_e, \\alpha}

        Following the original paper, we require that the end period be at least :math:`\\text{minlength}` periods after the start period. Therefore, if a bubble start is detected at :math:`r_s`, but ends before :math:`r_s + \\text{minlength}`, it is disregarded.
        """
        if not isinstance(alpha, float):
            raise TypeError("`alpha` must be a float")
        if alpha <= 0 or alpha >= 1:
            raise ValueError("`alpha` must be in the range (0, 1)")
        if not isinstance(fast, bool):
            raise TypeError("`fast` must be a boolean")
        stat: dict[float, float] = self.teststat()
        cval: dict[float, NDArray[float64]] = self.critval(
            alpha=alpha,
            fast=fast,  # type: ignore[assignment]
            nreps=nreps,
            nobs=nobs,
        )
        bubble_bool: list[NDArray[bool_]] = [stat[i] >= cval[i] for i in stat.keys()]
        minlength: int = self.minlength
        bubble_r2index: NDArray = array(
            list(PSYBubbles._find_bubble_dates(bubble_bool, minlength))
        )
        bubble_index: NDArray = array(
            [
                int(self.nobs * self.r2grid[i]) if i is not None else None
                for i in bubble_r2index.flatten()
            ]
        ).reshape((-1, 2))
        if hasattr(self, "index") and self.index is not None:
            bubble_dates: list[tuple[IndexType, IndexType | None]] = []
            for start, end in bubble_index:
                idx_start: Any = self.index[start]
                idx_end: Any = self.index[end] if end is not None else None
                bubble_dates.append((idx_start, idx_end))
            return array(bubble_dates)
        return bubble_index

    @classmethod
    def from_pandas(
        cls,
        data: Series,
        minwindow: int | None = None,
        minlength: int | None = None,
        lagmax: int = 0,
        rstep: float | None = None,
    ) -> "PSYBubbles":
        """Creates a PSYBubbles object from a :class:`pandas.Series`.

        Parameters
        ----------
        data : :class:`pandas.Series` of dtype :class:`numpy.float64`
            Time series data.
        minwindow : int | None, optional
            Minimum window size to calculate the test. Defaults to `r0_default`.
        minlength : int | None, optional
            Minimum bubble length.
        lagmax : int, optional
            Maximum lag :math:`k_{\\max}`. Defaults to 0.
        rstep : float | None, optional
            Step size :math:`r_{\\text{step}}`. Defaults to :math:`1/n`.

        Returns
        -------
        psybubbles: PSYBubbles
            An instance of the PSYBubbles class with :paramref:`psytest.bubbles.PSYBubbles.index` set to the provided :paramref:`pandas.Series.index`.

        Raises
        -------
        TypeError
            If :paramref:`data` is not a :class:`pandas.Series` or if :paramref:`data` dtype is not of type :class:`numpy.float64`.
        """
        if not isinstance(data, Series):
            raise TypeError("Data must be a pandas Series")
        if not data.dtype == float64:
            raise TypeError("Data must be of type float64")
        data_values: NDArray = data.to_numpy()
        data_index: NDArray | None = data.index.to_numpy()
        obj = cls(
            data=data_values,
            minwindow=minwindow,
            minlength=minlength,
            lagmax=lagmax,
            rstep=rstep,
        )
        obj.index = data_index
        return obj

    @staticmethod
    def _check_bubble_exists(bubble_bool: list[bool]) -> bool:
        return bubble_bool.count(True) > 0

    @staticmethod
    def _check_bubble_ends(bubble_bool: list[bool]) -> bool:
        return bubble_bool.count(False) > 0

    @staticmethod
    def _is_bubble_long_enough(bubble_bool: list[bool], minlength: int) -> bool:
        try:
            return all(bubble_bool[: minlength - 1])
        except IndexError:
            return False

    @staticmethod
    def _find_bubble_dates(
        bubble_bool: list, minlength: int
    ) -> Generator[tuple[int, int | None], None, None]:
        i0 = 0
        while len(bubble_bool) > minlength:
            if not PSYBubbles._check_bubble_exists(bubble_bool):
                logging.info("No bubble found")
                break
            start: int = bubble_bool.index(True)
            logging.info(f"Potential bubble found at {start}")
            if PSYBubbles._is_bubble_long_enough(bubble_bool[start:], minlength):
                if not PSYBubbles._check_bubble_ends(bubble_bool[start:]):
                    logging.info(
                        f"Bubble starting at {start} does not end on the last observation"
                    )
                    yield (start + i0, None)
                    break
                end: int = bubble_bool[start:].index(False) + start
                logging.info(f"Bubble starting at {start} ends at {end}")
                yield (start + i0, end + i0)
                bubble_bool = bubble_bool[end:]
                i0 += end
            else:
                logging.info("Bubble is not long enough")
                bubble_bool = bubble_bool[start + minlength :]
                i0 += start + minlength


def __parse_psy_arguments__(**kwargs) -> dict[str, Any]:
    """Parses the arguments for the :class:`psytest.bubbles.PSYBubbles` class and raises errors if they are invalid.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing the parameters to be validated.
        - data: 1D array-like of numbers
        - minwindow: float, default is calculated using :func:`r0_default`
        - rstep: float, default is 1 / len(data)
        - lagmax: int | None. If none, finds the optimal value using :func:`psytest.info_criteria.find_optimal_kmax`.
        - minlength: float, default is calculated using `minlength_default`
        - delta: float, default is 1.0. The parameter to determine the minimum length of bubbles. Used only if minlength is None.

    Returns
    -------
    dict[str, Any]
        A dictionary with the validated parameters.

    Raises
    -------
    TypeError
        If any of the parameters are of the wrong type.
    ValueError
        If :code:`data` is not a 1D array, if :code:`r0` is not in the range [0, 1], if :code:`rstep` is not in the range (0, 1], if :code:`kmax` is not an integer or is out of bounds, if :code:`minlength` is not valid floats.
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
    minwindow: Any = kwargs.get("minwindow", r0_default(len(data)))
    if not isinstance(minwindow, int):
        raise TypeError("`minwindow` must be a int")
    elif not 0 <= minwindow:
        raise ValueError("`minwindow` must be positive")
    elif minwindow >= len(data):
        raise ValueError("`minwindow` must be less than the length of `data`")
    rstep: Any = kwargs.get("rstep", 1 / len(data))
    if not isinstance(rstep, float):
        raise TypeError("`rstep` must be a float")
    if not 0 < rstep <= 1:
        raise ValueError("`rstep` must be in the range (0, 1]")
    lagmax: Any = kwargs.get("lagmax", LAGMAX)
    if lagmax is None:
        lagmax: int = find_optimal_kmax(y=data, klimit=LAGMAX)
    elif not isinstance(lagmax, int):
        raise TypeError("`lagmax` must be an integer or None")
    if lagmax < 0:
        raise ValueError("`lagmax` must be greater than or equal to 0")
    if lagmax > len(data) - 1:
        raise ValueError("`lagmax` must be less than the length of `data`")
    minlength: Any = kwargs.get("minlength")
    delta: Any = kwargs.get("delta", 1.0)
    if minlength is None:
        if not isinstance(delta, float):
            raise TypeError("`delta` must be a float")
        if delta <= 0:
            raise ValueError("`delta` must be greater than 0")
        minlength: float = minlength_default(nobs=len(data), delta=delta)
    elif not isinstance(minlength, int):
        raise TypeError("`minlength` must be a int")
    return {
        "data": data,
        "minwindow": minwindow,
        "rstep": rstep,
        "lagmax": lagmax,
        "minlength": minlength,
    }
