from pandas import read_csv, DataFrame

# Global variables
TEST_SIZE: list[float] = [0.1, 0.05, 0.01]
NREPS: int = 1000
KMAX: int = 0