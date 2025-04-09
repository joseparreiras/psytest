# 1. Load the modules
import numpy as np

# 2. Read the data (Markov switching DGP)

nobs: int = 100
betas: list[float] = [0.5, 1.0]
p_matrix: np.ndarray = np.array([[0.9, 0.1], [0.1, 0.9]])
test_error: np.ndarray = np.random.normal(size=nobs)


def simulate_markov_y(
    nobs: int, betas: np.ndarray, p_matrix: np.ndarray, test_error: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y: np.ndarray = np.zeros(nobs)
    beta_array: np.ndarray = np.zeros(nobs)
    beta_array[0] = betas[0]

    for t in range(1, nobs):
        last_beta = beta_array[t - 1]
        idx = 0 if last_beta == betas[0] else 1
        beta_t = np.random.choice(betas, p=p_matrix[idx])
        beta_array[t] = beta_t
        y[t] = beta_t * y[t - 1] + test_error[t]

    return y, beta_array


y, beta_array = simulate_markov_y(nobs, np.array(betas), p_matrix, test_error)
r0 = 10
nreps = 10


gasdf = GSADFuller(y)

gasdf.teststat(r0 = r0)