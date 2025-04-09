import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from psytest.utils.functions import simulate_markov
from psytest.bubbles import PSYBubbles

nreps = 2_000
nobs = 2_000
r0 = None

np.random.seed(19210201)
beta, y = simulate_markov(nobs, p=0.98, beta_list = [1.02, 0.98])

plt.plot(y)
plt.show()

psy = PSYBubbles(np.array(y), minlength=12, r0=r0)
psy.rstep = 1 / 100
stat: dict[int, float] = psy.teststat()
cval: dict[int, NDArray[np.float64]] = psy.critval(nreps, test_size = 0.05)

plt.plot(stat.keys(), stat.values())
plt.plot(cval.keys(), cval.values())
plt.show()

bubbles = psy.find_bubbles(alpha=0.05)

fig, ax = plt.subplots(2, 1, figsize=(12, 10))
ax[0].plot(y)
ax[1].plot(beta)
for start, end in bubbles:
    ax[0].axvspan(start, end, alpha=0.2)
    ax[1].axvspan(start, end, alpha=0.2)
plt.show()
