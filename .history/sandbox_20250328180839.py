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

plt.plot(beta)
plt.show()

psy = PSYBubbles(np.array(y), minlength = 12, r0 = r0)


stat = psy.bsadf()
cval = psy.critval(nreps)

bubbles = psy.find_bubbles(alpha = 0.05)

fig, ax = plt.subplots(2,1, figsize = (12,10))
ax[0].plot(y)
ax[1].plot(beta)
for (start, end) in bubbles:
    ax[0].axvspan(start, end, alpha = .2)
    ax[1].axvspan(start, end, alpha = .2)
plt.show()

