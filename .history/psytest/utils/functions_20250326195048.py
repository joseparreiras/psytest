from numpy import ceil, sqrt, cumsum, float64
from numpy.typing import NDArray
from numpy.random import normal
from collections.abc import Iterable, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from deprecation import deprecated


def r0_default(nobs: int) -> int:
    """
    Calculates the default r0 parameter following Phillips, Shi & Yu (2015) as a function of the number of observations.

    Args:
        nobs (int): Number of observations

    Returns:
        int: The default r0 parameter
    """
    return int(ceil((0.01 * 0.08 * sqrt(nobs)) * nobs))


def index_combinations(start: int, stop: int) -> list[tuple[int, int]]:
    """
    Generates the combinations of two indices, where the first index is less than the second index. The result is a tuple of indices (r1, r2) such as:
    * r1 = 0, ..., r2 - start
    * r2 = start, ..., stop

    Args:
        start (int): The initial index to start the combinations
        stop (int): The final index to stop the combinations

    Returns:
        list[tuple[int, int]]: The list of index combinations
    """
    return [(r1, r2) for r2 in range(start, stop + 1) for r1 in range(r2 - start + 1)]


@deprecated
def parallel_apply(func: Callable, iterable: Iterable, **kwargs) -> list:
    """
    Parallelizes the application of a function to an iterable using ThreadPoolExecutor.

    Args:
        func (Callable): _description_
        iterable (Iterable): _description_

    Returns:
        list: _description_
    """
    with ThreadPoolExecutor() as executor:
        futures: list = [executor.submit(func, x, **kwargs) for x in iterable]
        results: list = [f.result() for f in as_completed(futures)]
        executor.shutdown(wait=True)
    return results


def simulate_random_walks(nreps: int, nobs: int) -> NDArray[float64]:
    return cumsum(normal(size=(nreps, nobs)), axis=1)
