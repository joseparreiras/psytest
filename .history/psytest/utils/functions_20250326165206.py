import numpy as np


def r0_default(y: np.ndarray) -> int:
    nobs = len(y)
    return np.ceil((0.01 * 0.08 * np.sqrt(nobs)) * nobs)


def index_combinations(index_start: int, index_end: int):
    return [
        (r1, r2)
        for r2 in range(index_start, index_end)
        for r1 in range(r2 - index_start)
    ]