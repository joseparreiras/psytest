from .critval_parameters import AVAILABLE_CRITICAL_VALUE_PARAMETERS, is_available_param
from importlib.resources import files
from importlib.resources.abc import Traversable
from io import StringIO
from pandas import read_csv, DataFrame
from numpy import array, float64, interp
from numpy.typing import NDArray
from collections.abc import Iterable
from typing import overload


def make_colname_from_alpha(alpha: float) -> str:
    """
    Create a column name from the alpha value.
    """
    return f"cv{alpha * 100:0>2.0f}"


def find_best_table(kmax: int, r0: float) -> str:
    """
    Finds the table with the best match for the given parameters. It sorts tables by their nreps (ascending), nobs (ascending), and rstep (descending). It then returns the one with the best value of these that satisfies the condition of having the given `kmax` and `r0` as low as given.

    Args:
        kmax (int): Maximum lag for the critical value table.
        r0 (float): Minimum index to evaluate the test statistics.

    Returns:
        str | None: The name of the table with the best match for the given parameters, or None if no match is found.
    """
    best_match: AVAILABLE_CRITICAL_VALUE_PARAMETERS | None = max(
        (
            table
            for table in AVAILABLE_CRITICAL_VALUE_PARAMETERS
            if table.value.kmax == kmax and table.value.r0 <= r0
        ),
        key=lambda table: (table.value.nreps, table.value.nobs, -table.value.rstep),
        default=None,
    )
    if best_match is None:
        raise ValueError("No matching table found for the given parameters.")
    else:
        return best_match.name


def load_critval(kmax: int, r0: float) -> DataFrame:
    """
    Loads the critical values from the CSV file in the `data` directory.

    Args:
        kmax (int): Maximum lag for the critical value table.
        r0 (float): Minimum index to evaluate the test statistics.

    Returns:
        DataFrame: A DataFrame containing the critical values. The first column correspond to the `r2` parameter. The remaining columns give, for each `r2` the critical values with significance levels 0.10, 0.05, and 0.01 respectively.
    """
    if not is_available_param(kmax, r0):
        raise ValueError(
            f"Critical value table for kmax={kmax} and r0={r0} not available in the stored data."
        )

    # Find the table with the specified kmax and r0, and the largest nreps
    table_name: str = find_best_table(kmax, r0)
    data_path: Traversable = files("psytest.data") / table_name
    data_stringio: StringIO = StringIO(data_path.read_text())
    return read_csv(data_stringio, index_col=0)


@overload
def critval_tabulated(
    r2_eval: float, kmax: int, r0: float, alpha: float = 0.05
) -> float: ...
@overload
def critval_tabulated(
    r2_eval: Iterable[float], kmax: int, r0: float, alpha: float = 0.05
) -> NDArray[float64]: ...


def critval_tabulated(
    r2_eval: float | Iterable[float], kmax: int, r0: float, alpha: float = 0.05
) -> NDArray[float64] | float:
    """
    Returns the critical values for a given `r2` or an array of `r2` from the tabulated values in the `critval.csv` file.

    Args:
        r2_eval (float | Iterable[float]): The `r2` value or an array of `r2` values.
        alpha (float, optional): The significance level. Defaults to 0.05.
        kmax (int): Maximum lag for the critical value table.
        r0 (float): Minimum index to evaluate the test statistics.

    Returns:
        float64 | NDArray[float64]: The critical values for the given `r2` or an array of critical values.

    Raises:
        ValueError: If `r2_eval` is not between 0 and 1.
        TypeError: If `r2_eval` is not a float or an iterable.
        ValueError: If `alpha` is not between 0 and 1.
        TypeError: If `alpha` is not a float.
        ValueError: If the critical value for the given `alpha` is not found in the table.

    See Also:
        :obj:`critval`: The original table with critical values from Monte Carlo simulation.
    """
    if isinstance(r2_eval, float):
        if not (0 <= r2_eval <= 1):
            raise ValueError("`r2_eval` must be between 0 and 1")
    elif isinstance(r2_eval, Iterable):
        for r2 in r2_eval:
            if isinstance(r2, float):
                if not (0 <= r2 <= 1):
                    raise ValueError("`r2_eval` must be between 0 and 1")
            else:
                raise TypeError(
                    f"Invalid type {type(r2)} in `r2_eval`. Expected float."
                )
        r2_eval: NDArray[float64] = array(r2_eval)
    else:
        raise TypeError(
            f"Invalid type {type(r2_eval)} for `r2_eval`. Expected float or iterable."
        )

    if not isinstance(alpha, float):
        raise TypeError("`alpha` must be a float")
    elif not (0 < alpha < 1):
        raise ValueError("`alpha` must be between 0 and 1")

    if not isinstance(kmax, int):
        raise TypeError("`kmax` must be an integer")
    elif kmax < 0:
        raise ValueError("`kmax` must be a non-negative integer")
    if not isinstance(r0, float):
        raise TypeError("`r0` must be a float")
    elif not (0 < r0 < 1):
        raise ValueError("`r0` must be between 0 and 1")

    # Load critical values table
    critval_table: DataFrame = load_critval(kmax=kmax, r0=r0)
    col_name: str = make_colname_from_alpha(alpha)
    try:
        critval_values: NDArray[float64] = critval_table[col_name].values.astype(float)
        r2grid: NDArray[float64] = critval_table.index.values.astype(float)
        cval_interp: float64 | NDArray[float64] = interp(
            r2_eval, r2grid, critval_values
        )
        return cval_interp
    except KeyError:
        raise ValueError(
            f"Critical value for alpha {alpha:.0%} not found in the table."
        )
