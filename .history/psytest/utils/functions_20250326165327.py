import numpy as np
from collections.abc import Iterable, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


def r0_default(y: np.ndarray) -> int:
    nobs = len(y)
    return np.ceil((0.01 * 0.08 * np.sqrt(nobs)) * nobs)


def index_combinations(index_start: int, index_end: int) -> list[tuple[int, int]]:
    return [
        (r1, r2)
        for r2 in range(index_start, index_end)
        for r1 in range(r2 - index_start)
    ]


def parallel_apply(func: Callable, iterable: Iterable, **kwargs) -> list:
    with ThreadPoolExecutor() as executor:
        futures: list = [executor.submit(func, x, **kwargs) for x in iterable]
        results: list = [f.result() for f in as_completed(futures)]
        executor.shutdown(wait=True)
    return results
