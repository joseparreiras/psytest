from numpy.random import normal
import numpy as np
from numpy import empty
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from psytest.utils.functions import random_walk, simulate_markov
from psytest.bubbles import PSYBubbles

nreps = 2_000
nobs = 100
r0 = 0.190
r0 = int(r0 * nobs)

y = simulate_markov(nobs)

from numba import njit
import numpy as np

