from psytest import *
import numpy as np
import pandas as pd

nobs: int = 10_000
nreps: int = 1_000

corridor_adfuller_dist = np.empty(nreps)
