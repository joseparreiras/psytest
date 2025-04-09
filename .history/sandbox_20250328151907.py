from numpy.random import normal
import numpy as np
from numpy import empty
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from psytest.utils.functions import random_walk

nreps = 2_000
nobs = 100
r0 = 0.190
i0 = int(r0 * nobs)

rw = random_walk(nreps, nobs)

dist = sadfuller_dist(nobs, nreps, i0)


from numba import njit
import numpy as np
