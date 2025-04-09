from numpy.random import normal
import numpy as np
import matplotlib.pyplot as plt
form itertools import product
from tqdm import tqdm

nreps = 2_000
nobs = 100
r0 = 0.190
i0 = int(r0 * nobs)

rw = random_walk(nreps, nobs)

gsadf_dist = np.repeat(-np.inf, nreps)