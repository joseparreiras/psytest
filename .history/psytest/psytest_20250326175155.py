import numpy as np



class PSYbubbles:
    def __init__(self, y: np.ndarray, min_duration: int | None = None):
        self.y: np.ndarray = y
        self.nobs: int = len(y)
