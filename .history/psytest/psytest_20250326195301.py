#TODO: Create!
import numpy as np


class PSYbubbles:
    def __init__(
        self, y: np.ndarray, min_duration: int | None = None, sig_level: float = 0.05
    ) -> None:
        self.y: np.ndarray = y
        self.nobs: int = len(y)
        if min_duration is None:
            self.min_duration: int = int(np.ceil(np.log(self.nobs)))
        else:
            self.min_duration: int = min_duration
        self.sig_level: float = sig_level


class PWYbubbles:
    def __init__(self, y: np.ndarray, sig_level: float = 0.05) -> None:
        self.y: np.ndarray = y
        self.nobs: int = len(y)
        self.sig_level: float = sig_level
