"""
Implementation of the Augmented Dickey-Fuller test for unit roots.
"""

import numpy as np
from numba import njit, prange