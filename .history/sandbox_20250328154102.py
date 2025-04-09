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

y = simulate_markov(nobs)

psy = PSYBubbles(y)

psy.bsadf()
