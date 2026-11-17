import pathlib
from typing import Dict, List, Self, Tuple, override

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from data.parser import dumb_parse_c_array
from models.activation import Tanh
from models.autoencoder import Autoencoder
from models.noise import GaussianNoise, Noise, SaltAndPepperNoise


class DenoisingAutoencoder(Autoencoder):
    def __init__(self: Self, *args, noise_generator: Noise, noise_level: float = 0.1, **kwargs):
        self.noise_level = noise_level
        self.noise = noise_generator
        super().__init__(*args, **kwargs)

    def transform_input_data(self: Self, input_data: np.ndarray, noise_level: float) -> np.ndarray:
        return self.noise(input_data, noise_level)

    @override
    def _transform_input_data(self: Self, input_data: np.ndarray) -> np.ndarray:
        # Gaussian noise bc it's easy
        return self.transform_input_data(input_data, self.noise_level/2)

    def evaluate_noise_performance(self: Self, input_data: np.ndarray, target_noise_levels: List[float]) -> Dict[float, Tuple[float, float]]:
        ans = {}
        for noise_level in target_noise_levels:
            noisy_inputs = self.transform_input_data(input_data=input_data, noise_level=noise_level) if noise_level > 0 else input_data
            reconstructed = self.reconstruct(noisy_inputs) >= 0.5
            wrong_pixels = np.count_nonzero(reconstructed != input_data, axis=1)
            ans[noise_level] = (wrong_pixels.mean(), wrong_pixels.std())
        return ans

if __name__ == '__main__':
    inputs, _ = dumb_parse_c_array(pathlib.Path(__file__).parent.resolve() / '..' / 'data' / 'font.h')
    rng = np.random.default_rng(42)

    model = DenoisingAutoencoder(
        topology=[35, 16, 2],
        eta=0.005,
        train_inputs=inputs,
        encoder_activation=Tanh(),
        decoder_activation=Tanh(),
        epochs=50000,
        batch_size=32,
        rng=rng,
        noise_level=0.1,
        noise_generator=SaltAndPepperNoise(rng, noise_ratio=0.5)
    )

    noise_performance_sp = model.evaluate_noise_performance(inputs, [0, 0.05, 0.1, 0.25, 0.5, 1])
    print("SALT AND PEPPER")
    for noise, metrics in noise_performance_sp.items():
        print(f"Wrong pixels per image for noise level {noise}:\t{metrics[0]:5.2f} +- {metrics[1]:5.2f}")

    model = DenoisingAutoencoder(
        topology=[35, 16, 2],
        eta=0.005,
        train_inputs=inputs,
        encoder_activation=Tanh(),
        decoder_activation=Tanh(),
        epochs=50000,
        batch_size=32,
        rng=rng,
        noise_level=0.1,
        noise_generator=GaussianNoise(rng)
    )

    print("GAUSS")
    noise_performance_gauss = model.evaluate_noise_performance(inputs, [0, 0.05, 0.1, 0.25, 0.5, 1])
    for noise, metrics in noise_performance_gauss.items():
        print(f"Wrong pixels per image for noise level {noise}:\t{metrics[0]:5.2f} +- {metrics[1]:5.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    sp_plot: Axes = ax[0]
    sp_plot.errorbar(list(noise_performance_sp.keys()), list(x[0] for x in noise_performance_sp.values()), list(x[1] for x in noise_performance_sp.values()))
    gauss_plot: Axes = ax[1]
    gauss_plot.errorbar(list(noise_performance_gauss.keys()), list(x[0] for x in noise_performance_gauss.values()), list(x[1] for x in noise_performance_gauss.values()))
    plt.show()