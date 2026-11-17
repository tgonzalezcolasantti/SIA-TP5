import pathlib
from typing import List, Optional, Self, Dict, override
import numpy as np
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskID, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from matplotlib import pyplot as plt

from data.parser import dumb_parse_c_array
from models.activation import Activation, Adaline, ReLU, Tanh
from models.neural_network import Perceptron
from models.optimization import Adam, Optimization

def sigmoid(logits: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -logits))

class Autoencoder:
    def __init__(
            self: Self,
            topology: List[int],
            eta: float,
            train_inputs: np.ndarray,
            rng: np.random.Generator,
            encoder_activation:Activation = Tanh(),
            decoder_activation:Activation = Tanh(),
            epochs: int = 1000,
            batch_size: int = 1,
            optimization: type[Optimization] = Adam,
            decoder_topology: List[int] | None = None,
            task: Optional[TaskID] = None,
            progress: Optional[Progress] = None
    ):
        if not decoder_topology:
            decoder_topology = topology.copy()
            decoder_topology.reverse()
        self.rng = rng
        self.encoder = Perceptron(topology, encoder_activation, eta, optimization=optimization(), rng=self.rng, output_activation=Adaline())
        self.decoder = Perceptron(decoder_topology, decoder_activation, eta, optimization=optimization(), rng=self.rng, output_activation=Adaline())
        self.epochs = epochs
        self.epoch = 0
        self.inputs = train_inputs
        self.batch_size = batch_size
        self.progress=progress
        self.task = task
        if self.progress is not None and self.task is not None:
            self.progress.start_task(self.task)
            self.progress.update(self.task, visible=True)
        self.train()
        if self.progress is not None and self.task is not None:
            self.progress.remove_task(self.task)

    def encode(self: Self, input_value: np.ndarray)-> np.ndarray:
        return self.encoder.classify(input_value).copy()

    def decode(self: Self, latent: np.ndarray) -> np.ndarray:
        return sigmoid(self.decoder.classify(latent))

    def reconstruct(self: Self, input_value: np.ndarray) -> np.ndarray:
        return self.decode(self._apply_intermediate_layer(self.encode(input_value)))

    def train_single(self: Self, input_data: np.ndarray):
        # Forward pass
        transformed_input = self._transform_input_data(input_data)
        latent_code = self.encoder.classify(transformed_input)
        latent_code = self._apply_intermediate_layer(latent_code)
        decoder_output = self.decoder.classify(latent_code)

        # Back pass
        latent_gradient = self.decoder.update_weights((sigmoid(decoder_output) - input_data).T).T
        parameter_gradients = self._apply_intermediate_derivative(latent_gradient)
        self.encoder.update_weights(parameter_gradients.T)

    def _apply_intermediate_derivative(self: Self, latent_gradient: np.ndarray) -> np.ndarray:
        'Helper function for VAE, override it to apply stochastic layer derivative to decoder error output'
        return latent_gradient

    def _apply_intermediate_layer(self: Self, latent_code: np.ndarray) -> np.ndarray:
        'Helper function for VAE, override it to apply stochastic layer to encoder output'
        return latent_code

    def _transform_input_data(self: Self, input_data: np.ndarray) -> np.ndarray:
        'Helper function for denoiser autoencoder, override it to apply noise to the output'
        return input_data

    def _print_stats(self: Self, epoch: int):
        if self.progress is not None and self.task is not None:
            self.progress.update(self.task, completed=epoch)
        else:
            print(f"Epoch {epoch}", flush=True)

    def train(
        self: Self,
    ):
        with np.errstate(invalid='raise'):
            if self.epochs < 0 or self.batch_size == 0 or self.batch_size < -1:
                raise ValueError("epochs must be nonnegative and batch size positive or -1")
            effective_batch_size = len(self.inputs) if self.batch_size == -1 else self.batch_size
            for i in range(self.epoch, self.epochs):
                # print(f"Epoch {i}", flush=True)
                order = self.rng.permutation(len(self.inputs))
                for start in range(0, len(self.inputs), effective_batch_size):
                    indices = order[start:start + effective_batch_size]
                    self.train_single(self.inputs[indices])
                self.epoch += 1
                if self.epoch == 1 or self.epoch % 100 == 0 or self.epoch == self.epochs:
                    self._print_stats(self.epoch)
                # self.save()

class MonitoringAutoencoder(Autoencoder):
    def __init__(self: Self, *args, **kwargs):
        self.stats: Dict[int, float] = {}
        super().__init__(*args, **kwargs)

    @override
    def _print_stats(self: Self, epoch: int):
        reconstructed = self.reconstruct(self.inputs) >= 0.5
        wrong_pixels = np.count_nonzero(reconstructed != self.inputs, axis=1)
        self.stats[epoch] = np.sum(wrong_pixels)
        super()._print_stats(epoch)

    def get_stats(self: Self) -> Dict[int, float]:
        return self.stats

def main():
    inputs, labels = dumb_parse_c_array(pathlib.Path(__file__).parent.resolve() / '..' / 'data' / 'font.h')
    rng = np.random.default_rng(42)
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
            "Running autoencoder training",
            start=False,
            total=50000,
            visible=False,
            is_task=True,
        )
        model = Autoencoder(
            topology=[35, 16, 2],
            eta=0.005,
            train_inputs=inputs,
            encoder_activation=ReLU(),
            decoder_activation=ReLU(),
            rng=rng,
            epochs=50000,
            batch_size=32,
            optimization=Adam,
            task=task,
            progress=taskprogress
        )
        reconstructed = model.reconstruct(inputs) >= 0.5
        wrong_pixels = np.count_nonzero(reconstructed != inputs, axis=1)

        print(f"Pixeles erroneos por imagen {np.mean(wrong_pixels)}\nPixeles erroneos totales {np.sum(wrong_pixels)}")

        # Scatterplot de codigo latente
        encoded = model.encode(inputs)
        # print(encoded)
        figure, ax = plt.subplots()
        ax.scatter(encoded[:,0], encoded[:,1])
        for i, value in enumerate(encoded):
            ax.annotate(labels[i], (value[0], value[1]), fontsize=12)
        ax.set_xlabel("1er componente codigo latente")
        ax.set_ylabel("2da componente codigo latente")
        ax.set_title("Distribucion de caracteres segun componentes de codigo latente")
        plt.show()

        # Generacion de nuevas letras (?)

        latent_codes = []
        countx = []
        county = []
        for x in range(-25, 26, 1):
            countx.append(x)
            county = []
            for y in range(-10, 30, 1):
                county.append(y)
                latent_codes.append([x, y])
        latent_codes = np.array(latent_codes)
        # print(latent_codes)
        # print(countx)
        # print(county)
        generated = (model.decode(latent_codes) >= 0.5).astype(float)
        result = np.zeros((8*len(county), 6*len(countx)))

        for image, latent_code in zip(generated, latent_codes):
            index_x = countx.index(latent_code[0])
            index_y = county.index(latent_code[1])
            result[index_y*8:(index_y + 1) * 8 - 1, index_x*6:(index_x+1)*6 - 1] = image.reshape(7,5)

        plt.figure(figsize=(10, 10))

        plt.imshow(result, cmap='Greys_r')
        plt.title("Mapa del codigo latente del autoencoder")
        plt.show()

if __name__ == '__main__':
    main()
