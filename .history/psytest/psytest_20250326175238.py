import numpy as np



class PSYbubbles:
    def __init__(self, y: np.ndarray, min_duration: int | None = None):
        self.y: np.ndarray = y
        self.nobs: int = len(y)
        if min_duration is None:
            min_duration = int(np.ceil(np.log(self.nobs)))
