from psytest import PSYBubbles
import pandas as pd
import matplotlib.pyplot as plt
import logging
from numpy import datetime64

logging.basicConfig(level=logging.DEBUG)

data = pd.read_csv(
    filepath_or_buffer="scripts/data/shiller_data.csv",
    parse_dates=["date"],
    index_col="date",
)
nobs: int = len(data) - 1

pdratio = data["sp500"] / data["dividends"]
pdratio = pdratio.dropna()

plt.plot(pdratio)
plt.show()

psy: PSYBubbles[datetime64] = PSYBubbles.from_pandas(
    data=pdratio, minwindow=None, minlength=None, lagmax=0
)


stat: dict[datetime64, float] = psy.teststat()
cval: dict[datetime64, float] = psy.critval(alpha=0.05, fast=True)


bubbles: list = psy.find_bubbles(alpha=0.05)


plt.figure(figsize=(12, 6))
plt.plot(stat.keys(), stat.values(), label="Test Stat")
plt.plot(cval.keys(), cval.values(), linestyle="--", label="95% Critval")
for b in bubbles:
    plt.axvspan(b[0], b[1], color="gray", alpha=0.5, zorder=-1)
plt.show()

bubbles_table = pd.DataFrame(bubbles, columns=["start", "end"])
print(bubbles_table)
