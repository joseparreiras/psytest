from numpy.typing import NDArray
from numpy import object_, float64, int64, bool_, array, arange, ndarray, vstack
from collections.abc import Generator
from typing import Any, Self
from .critval import critval_tabulated, is_available_param
from .utils.functions import r0_default, minlength_default
from .utils.defaults import KMAX, TEST_SIZE, NREPS
from .sadftest import bsadf_stat_all_series, bsadfuller_critval


def __parse_psy_arguments(**kwargs) -> dict[str, Any]:
    """Parses the arguments for the PSYBubbles class and raises errors if they are invalid."""
    y: Any = kwargs.get("y")
    if not isinstance(y, ndarray):
        y: NDArray = array(y)
    if y.ndim != 1:
        raise ValueError("`y` must be a 1D array")
    if y.dtype not in [float64, int64]:
        raise ValueError("`y` must be a number array")
    if len(y) < 2:
        raise ValueError("`y` must have at least 2 elements")
    r0: Any = kwargs.get("r0", r0_default(len(y)))
    if not isinstance(r0, float):
        raise TypeError("`r0` must be a float")
    if not (0 <= r0 <= 1):
        raise ValueError("`r0` must be in the range [0, 1]")
    rstep: Any = kwargs.get("rstep", 1 / len(y))
    if not isinstance(rstep, float):
        raise TypeError("`rstep` must be a float")
    if not (0 < rstep <= 1):
        raise ValueError("`rstep` must be in the range (0, 1]")
    kmax: Any = kwargs.get("kmax", KMAX)
    if not isinstance(kmax, int):
        raise TypeError("`kmax` must be an integer")
    if kmax < 0:
        raise ValueError("`kmax` must be greater than or equal to 0")
    if kmax > len(y) - 1:
        raise ValueError("`kmax` must be less than the length of `y`")
    minlength: Any = kwargs.get("minlength")
    delta: Any = kwargs.get("delta")
    if minlength is not None and delta is not None:
        raise ValueError("Only one of `minlength` or `delta` should be provided")
    if delta is not None:
        if not isinstance(delta, float):
            raise TypeError("`delta` must be a float")
        if delta <= 0:
            raise ValueError("`delta` must be greater than 0")
        minlength = minlength_default(nobs=len(y), delta=delta)
    if minlength is not None:
        if not isinstance(minlength, float):
            raise TypeError("`minlength` must be a float")
        if not (0 < minlength <= 1):
            raise ValueError("`minlength` must be in the range (0, 1]")
    return {
        "y": y,
        "r0": r0,
        "rstep": rstep,
        "kmax": kmax,
        "minlength": minlength,
        "delta": delta,
    }


class PSYBubbles:
    def __init__(
        self,
        y: NDArray[float64],
        r0: float | None = None,
        rstep: float | None = None,
        kmax: int = KMAX,
        minlength: float | None = None,
        delta: float | None = None,
    ) -> None:
        """
        Class to perform the Phillips, Shi & Yu (2015) test for bubbles in time series data.

        .. math::
            r_0 = \\text{min window}, \\quad r_{\\text{step}} = \\text{step size}, \\quad k_{\\max} = \\text{max lag}

        Args:
            y (NDArray[float64]): Time series values.
            r0 (float | None, optional): Minimum window size :math:`r_0`. Defaults to `r0_default`.
            rstep (float | None, optional): Step size :math:`r_{\\text{step}}`. Defaults to :math:`1/n`.
            kmax (int, optional): Maximum lag :math:`k_{\\max}`. Defaults to KMAX (see :ref:`utils.defaults`).
            minlength (float | None, optional): Minimum bubble length.
            delta (float | None, optional): Used to compute default minlength via :math:`\\delta \\log(n)/n`.

        Raises:
            TypeError: For invalid input types.
            ValueError: For invalid input values.
        """
        parsed_args: dict[str, Any] = __parse_psy_arguments(
            y=y, r0=r0, rstep=rstep, kmax=kmax, minlength=minlength, delta=delta
        )
        self.y: NDArray[float64] = parsed_args["y"]
        self.nobs: int = len(self.y)
        self.index: NDArray | None = None
        self.r0: float = parsed_args["r0"]
        self.rstep: float = parsed_args["rstep"]
        self.kmax: int = parsed_args["kmax"]
        self.minlength: float | None = parsed_args["minlength"]
        self.delta: float | None = parsed_args["delta"]

    def r2grid(self) -> NDArray[float64]:
        """
        Grid of :math:`r_2` values for the BSADF test.

        Returns:
            NDArray[float64]: Grid :math:`\\{r_2\\}` from :math:`r_0` to 1 with step :math:`r_{\\text{step}}`.
        """
        return arange(self.r0, 1 + 1e-16, self.rstep)

    def teststat(self, force: bool = False) -> dict[float, float]:
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
            self.__teststat: dict[float, float] = dict(zip(self.r2grid(), stat))

        return self.__teststat

    def critval(
        self,
        test_size: list[float] | float = TEST_SIZE,
        fast: bool = True,
        **sim_kwargs: Any,
    ) -> dict[float, NDArray[float64]]:
        """
        Retrieves BSADF critical values using Monte Carlo Simulations.

        Args:
            test_size (list[float] | float, optional): Significance levels :math:`\\alpha`. Defaults to TEST_SIZE (see .utils.constants)
            fast (bool, optional): If True, uses tabulated critical values. Defaults to True.
            **sim_kwargs: Additional arguments for `bsadfuller_critval`. Used if `fast` is False. See :ref:`bsadfuller_critval`.

        Raises:
            TypeError: If `fast` is not a boolean.
            TypeError: If `test_size` is not a list or a float.
            ValueError: If `test_size` is not in the range (0, 1).

        Returns:
            dict[int, NDArray[float64]]: Critical values for each :math:`r_2`.
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

        r2grid: NDArray[float64] = self.r2grid()
        if fast:
            if not is_available_param(kmax=self.kmax, r0=self.r0):
                raise ValueError(
                    "Parameters `kmax` and `r0` are not available for tabulated critical values."
                )
            else:
                if isinstance(test_size, float):
                    cval: NDArray[float64] = critval_tabulated(
                        r2grid, alpha=test_size, kmax=self.kmax, r0=self.r0
                    )
                else:
                    cval: NDArray[float64] = vstack(
                        [
                            critval_tabulated(
                                r2grid, alpha=alpha, kmax=self.kmax, r0=self.r0
                            )
                            for alpha in test_size
                        ]
                    ).T
        else:
            parsed_sim_kwargs: dict[str, Any] = {
                "r0": sim_kwargs.get("r0", self.r0),
                "rstep": sim_kwargs.get("rstep", self.rstep),
                "nreps": sim_kwargs.get("nreps", NREPS),
                "nobs": sim_kwargs.get("nobs", self.nobs),
                "test_size": test_size,
            }
            cval: NDArray[float64] = bsadfuller_critval(**parsed_sim_kwargs).T
        return dict(zip(r2grid, cval))

    def find_bubbles(
        self, alpha: float = 0.05, fast=True, **sim_kwargs
    ) -> NDArray[object_]:
        """
        Identifies the bubble periods in the time series.

        A bubble starts at `r_s` if:

        .. math::
            \\text{BSADF}_{r_s} > \\text{CV}_{r_s, \\alpha}

        And ends at :math:`r_e > r_f` if:

        .. math::
            \\text{BSADF}_{r_e} < \\text{CV}_{r_e, \\alpha}

        Args:
            alpha (float): Significance level :math:`\\alpha`. Defaults to 0.05.
            nreps (int | None, optional): Number of simulations (required if no cache).

        Returns:
            NDArray[object_]: Array of bubble start and end indices.
        """
        if not isinstance(alpha, float):
            raise TypeError("`alpha` must be a float")
        if alpha <= 0 or alpha >= 1:
            raise ValueError("`alpha` must be in the range (0, 1)")
        if not isinstance(fast, bool):
            raise TypeError("`fast` must be a boolean")

        stat: dict[float, float] = self.teststat()
        cval: dict[float, NDArray[float64]] = self.critval(
            test_size=alpha, fast=fast, **sim_kwargs
        )
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
