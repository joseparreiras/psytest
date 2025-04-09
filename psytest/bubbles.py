from numpy.typing import NDArray
from numpy import object_
from numpy import float64, int64, bool_, array, arange, floor, ndarray
from psytest.utils.functions import r0_default, minlength_default
from psytest.sadftest import bsadf_stat_all_series, bsadfuller_critval
from collections.abc import Generator
from typing import Any, Self


class PSYBubbles:
    def __init__(
        self,
        y: NDArray[float64],
        r0: float | None = None,
        rstep: float | None = None,
        kmax: int = 0,
        minlength: int | None = None,
        delta: float | None = None,
    ) -> None:
        """
        Class to perform the Phillips, Shi & Yu (2015) test for bubbles in time series data.

        Args:
            y (NDArray[float64]): Values of the time series to be tested.
            r0 (float | None, optional): Minimum window size for the test. Defaults to None, using the default value from `r0_default` (see paper).
            rstep (float | None, optional): Step size to evaluate the test. Defaults to None, using 1 / nobs.
            kmax (int, optional): Maximum number of lags to include in the ADF test. Defaults to 0.
            minlength (int | None, optional): Minimum bubble length. Defaults to None, using the default value from `minlength_default` (see paper). Must be provided if `delta` is not provided.
            delta (float | None, optional): Parameter to calculate the default `minlength`. Defaults to None. Must be provided if `minlength` is not provided.

        Raises:
            ValueError: If `y` is not a 1D array or if `y` is not a number array.
            ValueError: If `y` has less than 2 elements.
            ValueError: If `r0` is not in the range [0, 1].
            ValueError: If `rstep` is not in the range (0, 1].
            ValueError: If `kmax` is less than 0.
            ValueError: If neither `minlength` nor `delta` is provided.
        """
        if not isinstance(y, ndarray):
            y = array(y)
        if y.ndim != 1:
            raise ValueError("`y` must be a 1D array")
        if y.dtype not in [float64, int64]:
            raise ValueError("`y` must be a number array")
        if len(y) < 2:
            raise ValueError("`y` must have at least 2 elements")
        if r0 is not None and not (0 <= r0 <= 1):
            raise ValueError("`r0` must be in the range [0, 1]")
        if rstep is not None and not (0 < rstep <= 1):
            raise ValueError("`rstep` must be in the range (0, 1]")
        if kmax < 0:
            raise ValueError("`kmax` must be greater than or equal to 0")

        self.y: NDArray[float64] = y
        self.nobs: int = len(y)
        self.index: NDArray | None = None
        self.r0: float = r0 or r0_default(self.nobs)
        self.rstep: float = rstep or 1 / self.nobs
        self.kmax: int = kmax
        if minlength is not None:
            self.minlength: int = minlength
            self.delta: float | None = None
        elif delta is not None:
            self.minlength: int = minlength_default(self.nobs, delta)
            self.delta: float | None = delta
        else:
            raise ValueError("Either `minlength` or `delta` must be provided")

    def r2grid(self) -> NDArray[float64]:
        """
        Grid of r2 to be used in the BSADF test

        Returns:
            NDArray[float64]: Grid of r2 values
        """
        return arange(self.r0, 1 + 1e-16, self.rstep)

    def teststat(self, force: bool = False) -> dict[int, float]:
        """
        Retrieves the BSADF test statistic.

        Args:
            force (bool, optional): Force recalculation of the test statistic. Defaults to False.

        Notes:
            - If this command has been run before and `force` is set to False, the cached value will be returned.

        Raises:
            TypeError: If `force` is not a boolean.
        Returns:
            dict[int, float]: Dictionary with the test statistic for each r2 value.
        """
        if not isinstance(force, bool):
            raise TypeError("`force` must be a boolean")

        if force or not hasattr(self, "_teststat"):
            # Calculate if forced or nonexistant
            stat: NDArray[float64] = bsadf_stat_all_series(
                self.y, self.r0, self.rstep, self.kmax
            )
            self._teststat: dict[int, float] = dict(zip(self.r2grid(), stat))

        return self._teststat

    def critval(
        self,
        nreps: int,
        force: bool = False,
        test_size: list[float] | float = [0.10, 0.05, 0.01],
    ) -> dict[int, NDArray[float64]]:
        """
        Retrieves the critical values for the BSADF test.

        Args:
            nreps (int): Number of repetitions for the Monte Carlo simulation.
            force (bool, optional): Force recalculation of the critical values. Defaults to False.
            test_size (list[float] | float, optional): One or many test sizes to calculate the critical values. Defaults to [0.10, 0.05, 0.01].

        Notes:
            - If this command has been run before the cached value will be returned unless `force` is set to True or `test_size` contains differnt values than previously used.

        Raises:
            TypeError: If `nreps` is not an integer.
            ValueError: If `nreps` is less than 1.
            TypeError: If `force` is not a boolean.
            TypeError: If `test_size` is not a list or a float.
            ValueError: If any of the provided `test_size` are not in the range (0, 1).

        Returns:
            dict[int, NDArray[float64]]: Dictionary with the critical values for each r2 value.
        """
        if not isinstance(nreps, int):
            raise TypeError("`nreps` must be an integer")
        if nreps < 1:
            raise ValueError("`nreps` must be greater than 0")
        if not isinstance(force, bool):
            raise TypeError("`force` must be a boolean")
        if not isinstance(test_size, (list, float)):
            raise TypeError("`test_size` must be a list or a float")
        if isinstance(test_size, float) and (test_size <= 0 or test_size >= 1):
            raise ValueError("`test_size` must be in the range (0, 1)")
        if isinstance(test_size, list):
            for size in test_size:
                if not isinstance(size, float):
                    raise TypeError("`test_size` must be a list of floats")
                if size <= 0 or size >= 1:
                    raise ValueError("`test_size` must be in the range (0, 1)")
            test_size = sorted(test_size)

        if (
            force
            or not hasattr(self, "_critval")
            or getattr(self, "_nreps", None) != nreps
            or getattr(self, "_testsize", None) != test_size
        ):
            cval: NDArray[float64] = bsadfuller_critval(
                self.r0, self.rstep, nreps, self.nobs, test_size
            ).T
            self._critval: dict[int, NDArray[float64]] = dict(zip(self.r2grid(), cval))
            self._nreps: int = nreps
            self._testsize: list[float] | float = test_size

        return self._critval

    def find_bubbles(self, alpha: float, nreps: int | None = None) -> NDArray[object_]:
        """
        Finds the index of the bubbles in the series.

        Args:
            alpha (float): The significance level for the test.
            nreps (int | None, optional): Number of repetitions for the Monte Carlo Simulation. Defaults to None, using the value set in `critval`.

        Notes:
            - This function can be run before or after the `teststat` and `critval` functions. If `teststat` and `critval` have been run before, the cached values will be used. Otherwise, the function will calculate them and `nreps` needs to be provided.
            - The function will return the start and end index of the bubbles in the series. If the bubble is still ongoing, the end index will be None.

        Raises:
            - TypeError: If `alpha` is not a float.
            - ValueError: If `alpha` is not in the range (0, 1).
            - TypeError: If `nreps` is not an integer.
            - ValueError: If `nreps` is less than 1.
            - ValueError: If `nreps` is not provided and has not been set in `critval`.

        Returns:
            NDArray[object_]: Array with the start and end index of each detected bubble.
        """
        if not isinstance(alpha, float):
            raise TypeError("`alpha` must be a float")
        if alpha <= 0 or alpha >= 1:
            raise ValueError("`alpha` must be in the range (0, 1)")
        if nreps is not None:
            if not isinstance(nreps, int):
                raise TypeError("`nreps` must be an integer")
            if nreps < 1:
                raise ValueError("`nreps` must be greater than 0")
        else:
            if not hasattr(self, "_nreps"):
                raise ValueError("`nreps` must be provided or set in `critval`")
            nreps: int = self._nreps

        stat: dict[int, float] = self.teststat()
        cval: dict[int, NDArray[float64]] = self.critval(nreps=nreps, test_size=alpha)
        bubble_bool: list[NDArray[bool_]] = [stat[i] > cval[i] for i in stat.keys()]
        bubble_r2index: NDArray[object_] = array(
            list(find_bubble_dates(bubble_bool, self.minlength))
        )
        bubble_index: NDArray[object_] = array(
            [
                int(floor(self.nobs * self.r2grid()[i])) if i is not None else None
                for i in bubble_r2index.flatten()
            ]
        ).reshape((-1, 2))
        # Retrieve the start and end in terms of the original index
        if hasattr(self, "index") and self.index is not None:
            bubble_dates: list[int64 | None] = []
            for start, end in bubble_index:
                idx_start: Any = self.index[start]
                idx_end: Any = self.index[end] if end is not None else None
                bubble_dates.append((idx_start, idx_end))
            return array(bubble_dates)
        return bubble_index

    @classmethod
    def from_pandas(
        cls,
        y: NDArray[float64],
        index: NDArray | None = None,
        r0: float | None = None,
        rstep: float | None = None,
        kmax: int = 0,
        minlength: int | None = None,
        delta: float | None = None,
    ) -> "PSYBubbles":
        """
        Creates a PSYBubbles object from a Pandas Series.

        Args:
            y (NDArray[float64]): Values of the time series to be tested.
            index (NDArray | None, optional): Index of the time series. Defaults to None.
            r0 (float | None, optional): Minimum window size for the test. Defaults to None, using the default value from `r0_default` (see paper).
            rstep (float | None, optional): Step size to evaluate the test. Defaults to None, using 1 / nobs.
            kmax (int, optional): Maximum number of lags to include in the ADF test. Defaults to 0.
            minlength (int | None, optional): Minimum bubble length. Defaults to None, using the default value from `minlength_default` (see paper). Must be provided if `delta` is not provided.
            delta (float | None, optional): Parameter to calculate the default `minlength`. Defaults to None. Must be provided if `minlength` is not provided.

        Returns:
            PSYBubbles: PSYBubbles object with the specified parameters.
        """
        obj: Self = cls(
            y=y,
            r0=r0,
            rstep=rstep,
            kmax=kmax,
            minlength=minlength,
            delta=delta,
        )
        obj.index = index
        return obj


def check_bubble_exists(bubble_bool: list[bool]) -> bool:
    """
    Checks if the list contains any bubbles

    Args:
        bubble_bool (list[bool]): List of booleans indicating the rejection of the null hypothesis.

    Returns:
        bool: True if there are any bubbles, False otherwise.
    """
    return bubble_bool.count(True) > 0


def check_bubble_ends(bubble_bool: list[bool]) -> bool:
    """
    Check if the list contains any non-bubble state.

    Args:
        bubble_bool (list[bool]): List of booleans indicating the rejection of the null hypothesis where the first element is the start of the bubble.

    Returns:
        bool: True if there are any non-bubble states, False otherwise.
    """
    return bubble_bool.count(False) > 0


def is_bubble_long_enough(bubble_bool: list[bool], minlength: int) -> bool:
    """
    Check if the current bubble has a length greater than the minimum length.

    Args:
        bubble_bool (list[bool]): List of booleans indicating the rejection of the null hypothesis where the first element is the start of the bubble.
        minlength (int): Minimum length of the bubble.

    Returns:
        bool: True if the bubble is long enough, False otherwise.
    """
    try:
        return bubble_bool[minlength]
    except IndexError:
        return False


def find_bubble_dates(
    bubble_bool: list, minlength: int
) -> Generator[tuple[int, int | None], None, None]:
    """
    Finds the index corresponding to the start and end of the bubbles from the vector of booleans indicating the rejection of the null hypothesis.

    Args:
        bubble_bool (list): List of booleans indicating the rejection of the null hypothesis.
        minlength (int): Minimum length of the bubble.

    Yields:
        Generator[tuple[int, int | None], None, None]: Generator of tuples with the start and end index of the bubbles.
    """
    i0 = 0
    while len(bubble_bool) > minlength:
        if not check_bubble_exists(bubble_bool):
            break
        start: int = bubble_bool.index(True)
        if is_bubble_long_enough(bubble_bool[start:], minlength):
            if not check_bubble_ends(bubble_bool[start:]):
                yield (start + i0, None)
                break
            end: int = bubble_bool[start:].index(False) + start
            yield (start + i0, end + i0)
        bubble_bool = bubble_bool[end:]
        i0 += end
