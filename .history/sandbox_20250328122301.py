from numpy.random import normal
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

nreps = 2_000
nobs = 100
r0 = 0.190
i0 = int(r0 * nobs)

rw = random_walk(nreps, nobs)

gsadf_dist = np.repeat(-np.inf, nreps)

iterable = product(range(nreps), range(i0, nobs + 1), range(0, nobs + 1))

for j, r2, r1 in tqdm(iterable, total=nreps * (nobs - i0) * (nobs - i0)):
    if r1 <= r2 - i0:
        gsadf_dist[j] = max(gsadf_dist[j], rolling_adfuller_stat(rw[j], r1, r2, kmax=0))

plt.hist(gsadf_dist, bins=50)
print(np.quantile(gsadf_dist, [0.90, 0.95, 0.99]))

make_iterable((10, 10))