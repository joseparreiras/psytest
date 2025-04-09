from numpy.random import normal
import numpy as np
from numpy import empty
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from psytest.utils.functions import random_walk, simulate_markov
from psytest.bubbles import PSYBubbles

nreps = 2_000
nobs = 500
r0 = 100

np.random.seed(19210201)
beta, y = simulate_markov(nobs)
start = (np.diff(beta) > 0).astype(int)
end = (np.diff(beta) < 0 * 1).astype(int)

plt.plot(beta)
plt.show()

plt.plot(y)
plt.show()


psy = PSYBubbles(y)

stat = psy.bsadf()

fig, ax = plt.subplots(2, 1)
ax[0].plot(stat.keys(), stat.values())
ax[1].plot(beta)
plt.show()

