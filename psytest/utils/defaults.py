"""psytest.utils.defaults
========================

Defaults used in the :mod:`psytest` package for the test significance levels, the number of repetitions for the random walk simulations, and the maximum lag used in the Augmented Dickey-Fuller test.
"""

# Global variables
TEST_SIZE: list[float] = [0.1, 0.05, 0.01]
NREPS: int = 1000
KMAX: int = 0