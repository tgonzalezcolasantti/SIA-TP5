import pathlib
from typing import Dict, List, Self, Tuple, override

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from data.parser import dumb_parse_c_array
from models.activation import ReLU, Tanh
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

def save_sample_grid(
    left: List[np.ndarray],
    left_labels: List[str],
    right: List[np.ndarray],
    right_labels: List[str],
    left_title: str,
    right_title: str
) -> None:

    figure = plt.figure(figsize=(10, 6), layout="constrained")
    left_panel, right_panel = figure.subfigures(1, 2) # type: ignore
    left_panel.suptitle(left_title)
    right_panel.suptitle(right_title)

    for panel, images, labels in ((left_panel, left, left_labels), (right_panel, right, right_labels)):
        axes = panel.subplots(4, 4).flat
        for index, ax in enumerate(axes):
            ax.set_axis_off()
            if index >= len(images):
                continue
            ax.imshow(
                images[index].reshape(7, 5),
                cmap="gray_r",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.set_title(labels[index], fontsize=10)

    plt.show()

if __name__ == '__main__':
    inputs, _ = dumb_parse_c_array(pathlib.Path(__file__).parent.resolve() / '..' / 'data' / 'font.h')
    rng = np.random.default_rng(42)

    target_noise_levels = [0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]

    taskprogress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    )
    table = Table(box=None)
    table.add_row(taskprogress)
    with Live(table, refresh_per_second=10):
        task = taskprogress.add_task(
            "Running salt-and-pepper noise training",
            start=False,
            total=75000,
            visible=False,
            is_task=True,
        )
        model = DenoisingAutoencoder(
            topology=[35, 16, 2],
            eta=0.005,
            train_inputs=inputs,
            encoder_activation=Tanh(),
            decoder_activation=Tanh(),
            epochs=75000,
            batch_size=32,
            rng=rng,
            noise_level=0.1,
            noise_generator=SaltAndPepperNoise(rng),
            task=task,
            progress=taskprogress
        )

        noise_performance_sp = model.evaluate_noise_performance(inputs, target_noise_levels)
        print("SALT AND PEPPER")
        for noise, metrics in noise_performance_sp.items():
            print(f"Wrong pixels per image for noise level {noise}:\t{metrics[0]:5.2f} +- {metrics[1]:5.2f}")

        task = taskprogress.add_task(
            "Running gaussian noise training",
            start=False,
            total=50000,
            visible=False,
            is_task=True,
        )
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
            noise_generator=GaussianNoise(rng),
            task=task,
            progress=taskprogress
        )

        print("GAUSS")
        noise_performance_gauss = model.evaluate_noise_performance(inputs, target_noise_levels)
        for noise, metrics in noise_performance_gauss.items():
            print(f"Wrong pixels per image for noise level {noise}:\t{metrics[0]:5.2f} +- {metrics[1]:5.2f}")

        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        sp_plot: Axes = ax[0]
        x = list(noise_performance_sp.keys())
        y = list(x[0] for x in noise_performance_sp.values())
        yerr = list(x[1] for x in noise_performance_sp.values())
        sp_plot.plot(x, y)
        # sp_plot.errorbar(x, y, yerr)
        sp_plot.fill_between(x, [y1-yerr1 for y1, yerr1 in zip(y, yerr)], [y1+yerr1 for y1, yerr1 in zip(y, yerr)], alpha=0.2, interpolate=True)
        sp_plot.set_xscale('log')
        sp_plot.set_title("Pixeles incorrectos por imagen segun nivel de ruido salt-and-pepper")
        sp_plot.set_xlabel("Nivel de ruido")
        sp_plot.set_ylabel("Pixeles incorrectos por imagen")
        gauss_plot: Axes = ax[1]
        x = list(noise_performance_gauss.keys())
        y = list(x[0] for x in noise_performance_gauss.values())
        yerr = list(x[1] for x in noise_performance_gauss.values())
        gauss_plot.plot(x, y)
        # gauss_plot.errorbar(x, y, yerr)
        gauss_plot.fill_between(x, [y1-yerr1 for y1, yerr1 in zip(y, yerr)], [y1+yerr1 for y1, yerr1 in zip(y, yerr)], alpha=0.2, interpolate=True)
        gauss_plot.set_xscale('log')
        gauss_plot.set_title("Pixeles incorrectos por imagen segun nivel de ruido gaussiano")
        gauss_plot.set_xlabel("Nivel de ruido")
        gauss_plot.set_ylabel("Pixeles incorrectos por imagen")
        plt.show()

        sp_noise = SaltAndPepperNoise(rng)
        gauss_noise = GaussianNoise(rng)
        sp_noisy_inputs = []
        gauss_noisy_inputs = []
        # Print noise levels
        for noise_level in target_noise_levels:
            sp_noisy_inputs.append(sp_noise(input_values=inputs[2], noise_level=noise_level))
            gauss_noisy_inputs.append(gauss_noise(input_values=inputs[2], noise_level=noise_level))
        save_sample_grid(
            sp_noisy_inputs,
            [str(x) for x in target_noise_levels],
            gauss_noisy_inputs,
            [str(x) for x in target_noise_levels],
            "Niveles de ruido salt-and-pepper",
            "Niveles de ruido gaussiano"
        )