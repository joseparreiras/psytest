from .adftest import adfuller_dist, adfuller_stat
import numpy as np

KMAX = 0

def sadfuller(y: np.ndarray, r0: int | None) -> np.flaot64