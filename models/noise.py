from typing import Self, override

import numpy as np


class Noise:
    def __init__(self: Self, rng: np.random.Generator):
        self.rng = rng
    def __call__(self: Self, input_values: np.ndarray, noise_level: float) -> np.ndarray:
        raise NotImplementedError()

class GaussianNoise(Noise):
    @override
    def __call__(self: Self, input_values: np.ndarray, noise_level: float) -> np.ndarray:
        return self.rng.normal(input_values, noise_level)

class SaltAndPepperNoise(Noise):
    def __init__(self: Self, *args, noise_ratio: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.ratio=noise_ratio

    @override
    def __call__(self: Self, input_values: np.ndarray, noise_level: float) -> np.ndarray:
        mask = self.rng.uniform(0, 1, input_values.shape) < noise_level
        noisy = input_values.copy()
        noisy[mask] = 1 - noisy[mask]
        return noisy
