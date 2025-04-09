import numpy as np
def r0_default(y: np.ndarray) -> int:
    nobs = len(y)
    return np.ceil((0.01 * 0.08 * np.sqrt(nobs)) * nobs)
