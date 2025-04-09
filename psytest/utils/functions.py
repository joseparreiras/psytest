from numpy import cumsum, float64, zeros, log, floor
from numpy.typing import NDArray
from numpy.random import normal, uniform
from collections.abc import Iterable, Callable, Sequence
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
    return 0.01 * 0.08 * nobs**0.5


def minlength_default(nobs: int, delta: float) -> int:
    """
    Calculates the minimum bubble length based on the number of observations

    Args:
        nobs (int): Number of observations
        delta (float): Multiplier parameter for bubble length

    Returns:
        int: Minimum bubble length
    """
    return delta * log(nobs) / nobs


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


def simulate_markov(
    nobs: int, p=0.975, beta_list: list[float] = [1.01, 1]
) -> tuple[list[float], list[float]]:
    """
    Simulates a markov process with two regimes. The two regimes differ in their beta parameter, one potentially explosive (beta > 1) and the other stationary (beta < 1). The markov matrix is set as constant with the parameter `p` giving the probability of remaining in the same regime. The process is simulated using a normal distribution for the error term. The process is defined as:
    .. math::

        y_t = beta * y_{t-1} + e_t
    where :math:`e_t` is a normal error term and :math:`beta` is the parameter of the process. The process starts at 0 and the first error term is generated from a normal distribution.

    Args:
        nobs (int): Number of observations for the process
        p (float, optional): Probability of switch. Defaults to 0.975.
        beta_list (list[float], optional): List of betas for the process. Defaults to [1.01, 1].

    Raises:
        - TypeError: If `p` is not a float.
        - ValueError: If `p` is not between 0 and 1.
        - TypeError: If `beta_list` is not a Sequence.
        - ValueError: If `beta_list` does not have length 2.
        - TypeError: If `nobs` is not an integer.
        - ValueError: If `nobs` is less than 1.

    Returns:
        tuple[list[float], list[float]]: A tuple containing two lists: the first one with the beta values and the second one with the simulated process values.
    """
    if not isinstance(beta_list, Sequence):
        raise TypeError("`beta_list` must be a Sequence")
    if len(beta_list) != 2:
        raise ValueError("`beta_list` must have length 2")
    if not isinstance(nobs, int):
        raise TypeError("`nobs` must be an integer")
    if nobs < 1:
        raise ValueError("`nobs` must be greater than 0")
    if not isinstance(p, float):
        raise TypeError("`p` must be a float")
    if p < 0 or p > 1:
        raise ValueError("`p` must be between 0 and 1")

    err: NDArray[float64] = normal(size=nobs - 1)
    y: list[float] = [0.0]
    beta: list[float] = [min(beta_list)]
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

def size_rgrid(r0: float, rstep:float) -> int:
    """
    Calculates the size of the rgrid starting at `r0` and with step `rstep`.

    Args:
        r0 (float): Minimum index to evaluate the test statistics.
        rstep (float): Step size for the index.
    """
    return int(floor((1 - r0) / rstep) + 1)