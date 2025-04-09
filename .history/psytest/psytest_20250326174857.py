

class PSYbubbles:
    def __init__(self, y: np.ndarray):
        self.y: np.ndarray = y
        self.nobs: int = len(y)
