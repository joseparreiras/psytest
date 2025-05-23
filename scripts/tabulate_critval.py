from pandas import DataFrame

from numpy.random import seed
from numpy.typing import NDArray
from numpy import float64, arange
from numba import set_num_threads
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psytest.sadftest import bsadfuller_critval, make_r2_grid
from psytest.critval.critval import make_colname_from_alpha
from psytest.utils.defaults import ALPHA_LIST

# Parameters -------------------------------------------------------------------

# PSY parameters
R0: float = 0.1  # Minimum index
KMAX_RANGE: list[int] = list(range(7))  # Max lag
RSTEP: float = 1 / 1000  # Step size
NREPS: int = 2_000  # Number of repetitions
NOBS: int = 2_000  # Number of observations

# CPU Threads
NTHREADS: int = 4

# Random Seed
SEED: int = 19210201

# Output path
OUTPATH: str = "psytest/critval/data"

# ------------------------------------------------------------------------------


def safe_float_str(value: float, precision: int) -> str:
    """
    Convert a float to a string with a fixed precision, replacing '.' with 'p'.
    """
    return f"{value:.{precision}f}".replace(".", "p")


def main() -> None:
    # Make sure the output directory exists
    os.makedirs(OUTPATH, exist_ok=True)
    # Set threads
    set_num_threads(NTHREADS)
    # Set seed
    seed(SEED)

    # Calculate critical values
    r2grid: NDArray[float64] = make_r2_grid(R0, RSTEP)
    for kmax in KMAX_RANGE:
        critval: NDArray[float64] = bsadfuller_critval(
            r0=R0,
            rstep=RSTEP,
            nreps=NREPS,
            nobs=NOBS,
            alpha=ALPHA_LIST,
            kmax=kmax,
        ).T
        # Convert to DataFrame
        col_names: list[str] = [make_colname_from_alpha(s) for s in ALPHA_LIST]
        df: DataFrame = DataFrame(critval, columns=col_names, index=r2grid)
        fname: str = (
            f"critval_"
            f"kmax{kmax}_"
            f"r0{safe_float_str(R0, 2)}_"
            f"rstep{safe_float_str(RSTEP, 4)}_"
            f"nreps{NREPS}_"
            f"nobs{NOBS}.csv"
        )
        fpath: str = os.path.join(OUTPATH, fname)
        df.to_csv(fpath)


if __name__ == "__main__":
    main()
    sys.exit(0)
