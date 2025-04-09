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

psy = PSYBubbles(y, minlength = 252)

stat = psy.bsadf()
cval = psy.critval(nreps)

psy.find_bubbles(alpha = 0.05)
