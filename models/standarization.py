from typing import List, Self
from numbers import Number
import numpy as np


class ZScoreStandardize:
    def __init__(self: Self, dataset: np.ndarray):
        self._values = dataset
        self._means: np.ndarray
        self._scales: np.ndarray
        self._standard_values: np.ndarray
        self._constants: List[int]
        self._zscale_values()

    @property
    def means(self: Self) -> List[float]:
        return self._means.tolist()

    @property
    def scales(self: Self) -> List[float]:
        return self._scales.tolist()

    @property
    def constants(self: Self) -> List[int]:
        return list(self._constants)

    @property
    def standard_values(self: Self) -> np.ndarray:
        return self._standard_values.copy()
    
    def _zscale_values(self: Self):
        self._means, self._scales = self._values.mean(axis=0), self._values.std(axis=0)
        self._constants = [idx for idx, scale in enumerate(self._scales) if scale == 0]
        self._scales[self._scales == 0] = 1.0
        self._standard_values = self.standarize(self._values) # type: ignore

    def standarize(self: Self, values: np.ndarray) -> np.ndarray:
        "Scales values in dataset so that they have means=0 and deviations=1"
        return (values - self._means) / self._scales

    def unstandardize(self: Self, values: np.ndarray) -> np.ndarray:
        "Returns a standardized value back to its original distribution"
        return (values * self._scales) + self._means

    def __len__(self: Self) -> int:
        return len(self._standard_values)

    def __getitem__(self, key):
        return self._standard_values[key]