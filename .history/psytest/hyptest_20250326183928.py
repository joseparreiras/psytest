class HypothesisTest():
    def __init__(self, y: np.ndarray, **kwargs) -> None:
        self.y: np.ndarray = y
        self.nobs: int = len(y)
        self.kwargs = kwargs