from .adftest import adfuller_dist, adfuller_stat
from .utils.functions import r0_default, index_combinations, parallel_apply
import numpy as np

KMAX = 0

def sadfuller(y: np.ndarray, r0: int | None) -> np.flaot64:
    r0 = r0 or r0_default(y)