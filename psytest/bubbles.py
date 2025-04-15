"""psytest.bubbles
=========================

This module contains the main class for the PSY test (see :class:`psytest.bubbles.PSYBubbles`). This class handles the frontend of the test, receiving the data and parameters, and providing methods to run the test and retrieve results. It also includes methods for parsing arguments, checking conditions, and finding bubble dates.
"""

from numpy.typing import NDArray
from numpy import object_, float64, bool_, array, arange, vstack
from collections.abc import Generator
from typing import Any, Literal, overload
from .critval import critval_tabulated, is_available_param
from .utils.functions import parse_psy_arguments
from .utils.defaults import KMAX, TEST_SIZE, NREPS
from .sadftest import bsadf_stat_all_series, bsadfuller_critval
from functools import lru_cache
from pandas import Series


class PSYBubbles:
    """Class to perform the Phillips, Shi & Yu (2015) test for bubbles in time series data.

    Parameters
    ----------
    data : :class:`numpy.ndarray` of dtype :class:`numpy.float64`
        Time series values.
    r0 : float | None, optional
        Minimum window size :math:`r_0`. Defaults to :func:`psytest.utils.functions.r0_default`.
    rstep : float | None, optional
        Step size :math:`r_{\\text{step}}`. Defaults to :math:`1/n` where :math:`n` is the size of :paramref:`data`.
    kmax : int, optional
        Maximum lag :math:`k_{\\max}`. Defaults to KMAX (see :mod:`psytest.utils.defaults`).
    minlength : float | None, optional
        Minimum bubble length.
    delta : float | None, optional
        Used to compute default minlength via :math:`\\delta \\log(n)/n`.

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
        r0: float | None = None,
        rstep: float | None = None,
        kmax: int = KMAX,
        minlength: float | None = None,
        delta: float | None = None,
    ) -> None:
        parsed_args: dict[str, Any] = parse_psy_arguments(
            data=data, r0=r0, rstep=rstep, kmax=kmax, minlength=minlength, delta=delta
        )
        self.data: NDArray[float64] = parsed_args["data"]
        self.nobs: int = len(self.data)
        self.index: NDArray | None = None
        self.r0: float = parsed_args["r0"]
        self.rstep: float = parsed_args["rstep"]
        self.kmax: int = parsed_args["kmax"]
        self.minlength: float = parsed_args["minlength"]
        self.delta: float | None = parsed_args["delta"]
        self.r2grid: NDArray[float64] = arange(self.r0, 1 + 1e-16, self.rstep)

    @lru_cache(maxsize=128)
    def teststat(self, force: bool = False) -> dict[float, float]:
        """Retrieves the BSADF test statistic.

        Parameters
        ----------
        force : bool, optional
            If :literal:`True`, forces recalculation.

        Returns
        -------
        teststat : dictionary with float keys and float values
            Test statistics for each :math:`r_2` in :func:`psytest.PSYBubbles.r2grid`.

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
                self.data, self.r0, self.rstep, self.kmax
            )
            self.__teststat: dict[float, float] = dict(zip(self.r2grid, stat))
        return self.__teststat

    @overload
    def critval(
        self, test_size: float | list[float], fast: Literal[True]
    ) -> dict[float, NDArray[float64]]: ...

    @overload
    def critval(
        self,
        test_size: float | list[float],
        fast: Literal[False],
        nreps: int,
        nobs: int | None,
    ) -> dict[float, NDArray[float64]]: ...

    @lru_cache(maxsize=5)
    def critval(
        self,
        test_size: list[float] | float = TEST_SIZE,
        fast: Literal[True, False] = True,
        nreps: int = NREPS,
        nobs: int | None = None,
    ) -> dict[float, NDArray[float64]]:
        """Retrieves BSADF critical values using Monte Carlo Simulations.

        Parameters
        ----------
        test_size : list[float] | float, optional
            Significance levels :math:`\\alpha`. Defaults to TEST_SIZE (see :mod:`psytest.utils.defaults`)
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
            If :paramref:`test_size` is a list, the keys are the :math:`r_2` values and the values are an array of critical values for each significance level.
            If :paramref:`test_size` is a float, the keys are the :math:`r_2` values and the values are the critical values for the given significance level.

        Raises
        -------
        TypeError
            If :paramref:`test_size` is not a float or list of floats, or if :paramref:`fast` is not a boolean.
        ValueError
            If :paramref:`test_size` is not in the range (0, 1) or if :paramref:`nreps` is not a positive integer.

        Notes
        -----
        This function uses :code:`functools.lru_cache` to cache the results for faster access. So after the first call, subsequent calls with the same parameters will utilize the cached results to reduce computation time.
        """
        if not isinstance(fast, bool):
            raise TypeError("`fast` must be a boolean")
        if isinstance(test_size, float):
            if test_size <= 0 or test_size >= 1:
                raise ValueError("`test_size` must be in the range (0, 1)")
        elif isinstance(test_size, list):
            for size in test_size:
                if not isinstance(size, float):
                    raise TypeError("`test_size` must be a list of floats")
                if size <= 0 or size >= 1:
                    raise ValueError("`test_size` must be in the range (0, 1)")
        else:
            raise TypeError("`test_size` must be a list of floats or a float")
        if fast:
            return self._critval_tabulated(test_size=test_size)
        else:
            return self._critval_simulated(test_size=test_size, nreps=nreps, nobs=nobs)

    def _critval_tabulated(
        self, test_size: float | list[float]
    ) -> dict[float, NDArray[float64]]:
        kmax: int = self.kmax
        r0: float = self.r0
        r2grid: NDArray[float64] = self.r2grid
        if not is_available_param(kmax=self.kmax, r0=self.r0):
            raise ValueError(
                f"Parameters kmax={kmax!r} and r0={r0!r} are not available for tabulated critical values."
            )
        if isinstance(test_size, float):
            cval: NDArray[float64] | float = critval_tabulated(
                r2grid, alpha=test_size, kmax=self.kmax, r0=self.r0
            )
        elif isinstance(test_size, list):
            cval: NDArray[float64] | float = vstack(
                [
                    critval_tabulated(r2grid, alpha=alpha, kmax=self.kmax, r0=self.r0)
                    for alpha in test_size
                ]
            ).T
        return dict(zip(r2grid, cval))

    def _critval_simulated(
        self, test_size: float | list[float], nreps: int, nobs: int | None
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
            test_size=test_size,
            kmax=self.kmax,
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
            test_size=alpha,
            fast=fast,  # type: ignore[assignment]
            nreps=nreps,
            nobs=nobs,
        )
        bubble_bool: list[NDArray[bool_]] = [stat[i] > cval[i] for i in stat.keys()]
        minlength: int = int(self.nobs * self.minlength)
        bubble_r2index: NDArray = array(
            list(self._find_bubble_dates(bubble_bool, minlength))
        )
        bubble_index: NDArray = array(
            [
                int(self.nobs * self.r2grid[i]) if i is not None else None
                for i in bubble_r2index.flatten()
            ]
        ).reshape((-1, 2))
        if hasattr(self, "index") and self.index is not None:
            bubble_dates: list[tuple[Any, Any | None]] = []
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
        r0: float | None = None,
        rstep: float | None = None,
        kmax: int = 0,
        minlength: int | None = None,
        delta: float | None = None,
    ) -> "PSYBubbles":
        """Creates a PSYBubbles object from a :class:`pandas.Series`.

        Parameters
        ----------
        data : :class:`pandas.Series` of dtype :class:`numpy.float64`
            Time series data.
        r0 : float | None, optional
            Minimum window size :math:`r_0`. Defaults to `r0_default`.
        rstep : float | None, optional
            Step size :math:`r_{\\text{step}}`. Defaults to :math:`1/n`.
        kmax : int, optional
            Maximum lag :math:`k_{\\max}`. Defaults to 0.
        minlength : int | None, optional
            Minimum bubble length.
        delta : float | None, optional
            Used to compute default minlength via :math:`\\delta \\log(n)/n`.

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
            r0=r0,
            rstep=rstep,
            kmax=kmax,
            minlength=minlength,
            delta=delta,
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
            return all(bubble_bool[:minlength])
        except IndexError:
            return False

    @staticmethod
    def _find_bubble_dates(
        bubble_bool: list, minlength: int
    ) -> Generator[tuple[int, int | None], None, None]:
        i0 = 0
        while len(bubble_bool) > minlength:
            bubble_bool = bubble_bool[i0:]
            if not PSYBubbles._check_bubble_exists(bubble_bool):
                break
            start: int = bubble_bool.index(True)
            if PSYBubbles._is_bubble_long_enough(bubble_bool[start:], minlength):
                if not PSYBubbles._check_bubble_ends(bubble_bool[start:]):
                    yield (start + i0, None)
                    break
                end: int = bubble_bool[start:].index(False) + start
                yield (start + i0, end + i0)
                i0 += end
            else:
                i0 += start + 1
