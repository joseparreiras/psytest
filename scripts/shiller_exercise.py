from psytest import PSYBubbles
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from psytest.adftest import adfuller_stat, rolling_adfuller_stat
from psytest.sadftest import bsadf_stat

data = pd.read_csv(
    filepath_or_buffer="tests/data/shiller_data.csv",
    parse_dates=["date"],
    index_col="date",
)
nobs: int = len(data) - 1

pdratio = data["sp500"] / data["dividends"]
pdratio = pdratio.dropna()

plt.plot(pdratio)
plt.show()

psy = PSYBubbles.from_pandas(data=pdratio, r0=None, minlength=None, rstep=None, kmax=0)

stat = psy.teststat()
# cval = psy.critval(test_size=0.05, fast=False, nreps=2000, nobs=2000)

plt.plot(stat.keys(), stat.values(), label="Test Stat")
# plt.plot(cval.keys(), cval.values(), linestyle="--", label="95% Critval")
plt.show()
