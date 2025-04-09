from numpy.random import normal
import numpy as np
import matplotlib.pyplot as plt

nreps = 2_000
nobs = 100
r0 = 0.190
i0 = int(r0 * nobs)

rw = random_walk(nreps, nobs)

gsadf_dist = np.empty(nreps)

for j in range(nreps):
    for r2 in range(i0, nobs + 1):
        for r1 in range(0, r2 - i0 + 1)
        