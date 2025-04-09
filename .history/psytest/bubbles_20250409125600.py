from numpy.typing import NDArray
from numpy import object_, float64, int64, bool_, array, arange, ndarray
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
        minlength: float | None = None,
        delta: float | None = None,
    ) -> None:
        """
        Class to perform the Phillips, Shi & Yu (2015) test for bubbles in time series data.

        .. math::
            r_0 = \\text{min window}, \\quad r_{\\text{step}} = \\text{step size}, \\quad k_{\max} = \\text{max lag}

        Args:
            y (NDArray[float64]): Time series values.
            r0 (float | None, optional): Minimum window size :math:`r_0`. Defaults to `r0_default`.
            rstep (float | None, optional): Step size :math:`r_{\\text{step}}`. Defaults to :math:`1/n`.
            kmax (int, optional): Maximum lag :math:`k_{\\max}`. Defaults to 0.
            minlength (int | None, optional): Minimum bubble length.
            delta (float | None, optional): Used to compute default minlength via :math:`\\delta \\log(n)/n`.

        Raises:
            TypeError: For invalid input types.
            ValueError: For invalid input values.
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
            if not isinstance(minlength, float):
                raise TypeError("`minlength` must be a float")
            if not (0 < minlength <= 1):
                raise ValueError("`minlength` must be in the range (0, 1]")
            self.minlength: int = minlength
            self.delta: float | None = None
        elif delta is not None:
            if not isinstance(delta, float):
                raise TypeError("`delta` must be a float")
            if delta <= 0:
                raise ValueError("`delta` must be greater than 0")
            self.minlength: int = minlength_default(self.nobs, delta)
            self.delta: float | None = delta
        else:
            raise ValueError("Either `minlength` or `delta` must be provided")

    def r2grid(self) -> NDArray[float64]:
        """
        Grid of :math:`r_2` values for the BSADF test.

        Returns:
            NDArray[float64]: Grid :math:`\\{r_2\\}` from :math:`r_0` to 1 with step :math:`r_{\\text{step}}`.
        """
        return arange(self.r0, 1 + 1e-16, self.rstep)

    def teststat(self, force: bool = False) -> dict[int, float]:
        """
        Retrieves the BSADF test statistic.

        .. math::
            \\text{BSADF}_{r_2} = \\max_{r_1 \\in [0, r_2 - r_0]} \\text{ADF}(y_{r_1:r_2})

        Args:
            force (bool, optional): If True, forces recalculation.

        Returns:
            dict[int, float]: Test statistic by :math:`r_2`.
        """
        if not isinstance(force, bool):
            raise TypeError("`force` must be a boolean")

        if force or not hasattr(self, "__teststat"):
            stat: NDArray[float64] = bsadf_stat_all_series(
                self.y, self.r0, self.rstep, self.kmax
            )
            self.__teststat: dict[int, float] = dict(zip(self.r2grid(), stat))

        return self.__teststat

    def critval(
        self,
        nreps: int,
        force: bool = False,
        test_size: list[float] | float = [0.10, 0.05, 0.01],
    ) -> dict[int, NDArray[float64]]:
        """
        Retrieves BSADF critical values using Monte Carlo Simulations.

        Args:
            nreps (int): Number of simulations.
            force (bool, optional): Force recalculation.
            test_size (list[float] | float, optional): Significance levels :math:`\\alpha`.

        Returns:
            dict[int, NDArray[float64]]: Critical values for each :math:`r_2`.
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
            or not hasattr(self, "__critval")
            or getattr(self, "nreps", None) != nreps
            or getattr(self, "testsize", None) != test_size
        ):
            cval: NDArray[float64] = bsadfuller_critval(
                self.r0, self.rstep, nreps, self.nobs, test_size
            ).T
            self.__critval: dict[int, NDArray[float64]] = dict(zip(self.r2grid(), cval))
            self.nreps: int = nreps
            self.testsize: list[float] | float = test_size

        return self.__critval

    def find_bubbles(self, alpha: float, nreps: int | None = None) -> NDArray[object_]:
        """
        Identifies the bubble periods in the time series.

        A bubble exists when:

        .. math::
            \\text{BSADF}_{r_2} > \\text{CV}_{r_2, \\alpha}

        Args:
            alpha (float): Significance level :math:`\\alpha`.
            nreps (int | None, optional): Number of simulations (required if no cache).

        Returns:
            NDArray[object_]: Array of bubble start and end indices.
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
            if not hasattr(self, "nreps"):
                raise ValueError("`nreps` must be provided or set in `critval`")
            nreps: int = self.nreps

        stat: dict[int, float] = self.teststat()
        cval: dict[int, NDArray[float64]] = self.critval(nreps=nreps, test_size=alpha)
        bubble_bool: list[NDArray[bool_]] = [stat[i] > cval[i] for i in stat.keys()]
        minlength: int = int(self.nobs * self.minlength)
        bubble_r2index: NDArray[object_] = array(
            list(self._find_bubble_dates(bubble_bool, minlength))
        )
        bubble_index: NDArray[object_] = array(
            [
                int(self.nobs * self.r2grid()[i]) if i is not None else None
                for i in bubble_r2index.flatten()
            ]
        ).reshape((-1, 2))
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
        Creates a PSYBubbles object from a pandas Series.

        Args:
            y (NDArray[float64]): Time series values.
            index (NDArray | None, optional): Index.
            r0, rstep, kmax, minlength, delta: See PSYBubbles constructor.

        Returns:
            PSYBubbles: Configured object.
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

    @staticmethod
    def _check_bubble_exists(bubble_bool: list[bool]) -> bool:
        return bubble_bool.count(True) > 0

    @staticmethod
    def _check_bubble_ends(bubble_bool: list[bool]) -> bool:
        return bubble_bool.count(False) > 0

    @staticmethod
    def _is_bubble_long_enough(bubble_bool: list[bool], minlength: int) -> bool:
        try:
            return bubble_bool[minlength]
        except IndexError:
            return False

    @staticmethod
    def _find_bubble_dates(
        bubble_bool: list, minlength: int
    ) -> Generator[tuple[int, int | None], None, None]:
        i0 = 0
        while len(bubble_bool) > minlength:
            if not PSYBubbles._check_bubble_exists(bubble_bool):
                break
            start: int = bubble_bool.index(True)
            if PSYBubbles._is_bubble_long_enough(bubble_bool[start:], minlength):
                if not PSYBubbles._check_bubble_ends(bubble_bool[start:]):
                    yield (start + i0, None)
                    break
                end: int = bubble_bool[start:].index(False) + start
                yield (start + i0, end + i0)
            bubble_bool = bubble_bool[end:]
            i0 += end
