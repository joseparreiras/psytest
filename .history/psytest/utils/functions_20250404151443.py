from numpy import ceil, cumsum, float64, zeros, log
from numpy.typing import NDArray
from numpy.random import normal, uniform
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
    return int(ceil((0.01 * 0.08 * nobs**0.5) * nobs))


def minlength_default(nobs: int, delta: float) -> int:
    """
    Calculates the minimum bubble length based on the number of observations

    Args:
        nobs (int): Number of observations
        delta (float): Multiplier parameter for bubble length

    Returns:
        int: Minimum bubble length
    """
    return int(delta * log(nobs))


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


@deprecated(details="Function unused by the package")
def parallel_apply(func: Callable, iterable: Iterable, **kwargs) -> list:
    """
    Parallelizes the application of a function to an iterable using ThreadPoolExecutor.
    """
    with ThreadPoolExecutor() as executor:
        futures: list = [executor.submit(func, x, **kwargs) for x in iterable]
        results: list = [f.result() for f in as_completed(futures)]
        executor.shutdown(wait=True)
    return results


def random_walk(nreps: int, nobs: int) -> NDArray[float64]:
    """
    Generates a monte carlo simulation of random walks.

    Args:
        nreps (int): number of repetitions.
        nobs (int): number of observations.

    Returns:
        NDArray[float64]: Matrix of shape (nreps, nobs) with the random walks.
    """
    rw: NDArray[float64] = zeros((nreps, nobs))  # Set initial value to 0
    rw[:, 1:] = nobs ** (-0.5) * normal(size=(nreps, nobs - 1))  # Simulate for t > 0
    return cumsum(rw, axis=1)


def simulate_markov(nobs: int, p=0.99) -> tuple[list[float], list[float]]:
    err: NDArray[float64] = normal(size=nobs - 1)
    y: list[float] = [0.0]
    beta_list: list[float] = [1.05, 1]
    beta: list[float] = [1]
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
    return beta, y
