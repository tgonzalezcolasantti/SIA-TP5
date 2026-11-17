import pathlib
import time
from typing import List, Self, override
import numpy as np

from data.parser import dumb_parse_c_array
from models.activation import Activation, Adaline, Tanh
from models.neural_network import Perceptron
from models.optimization import Adam, Optimization

from matplotlib import pyplot as plt

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
        self.train()

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

class VariationalAutoencoder(Autoencoder):

    def __init__(
        self: Self,
        topology: List[int],
        eta: float,
        rng: np.random.Generator,
        encoder_activation:Activation,
        decoder_activation:Activation,
        train_inputs: np.ndarray,
        epochs: int = 1000,
        batch_size: int = 1,
        optimization: type[Optimization] = Adam
    ):
        decoder_topology = topology.copy()
        decoder_topology.reverse()
        topology[-1] = topology[-1] * 2
        super().__init__(
            topology=topology,
            decoder_topology=decoder_topology,
            eta=eta,
            rng=rng,
            encoder_activation=encoder_activation,
            decoder_activation=decoder_activation,
            train_inputs=train_inputs,
            epochs=epochs,
            batch_size=batch_size,
            optimization=optimization
        )
        self.last_inputs: np.ndarray
        self.latent_offset: np.ndarray

    def split_mean_variance(self: Self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, var = np.split(inputs, 2, axis=1)
        return mean, var
    
    @override
    def _apply_intermediate_layer(self: Self, latent_code: np.ndarray) -> np.ndarray:
        mean, var = self.split_mean_variance(latent_code)
        self.last_inputs = latent_code
        epsilon = self.rng.standard_normal(mean.shape)
        self.latent_offset = np.exp(0.5 * var) * epsilon
        return mean + self.latent_offset

    @override
    def _apply_intermediate_derivative(self: Self, latent_gradient: np.ndarray) -> np.ndarray:
        mean, var = self.split_mean_variance(self.last_inputs)
        mean_gradient = latent_gradient + mean
        variance_minus_one = np.expm1(var)
        log_variance_gradient = 0.5 * (latent_gradient * self.latent_offset + variance_minus_one)
        return np.concatenate((mean_gradient, log_variance_gradient), axis=1)

    @override
    def _print_stats(self: Self, epoch: int):
        # Sum across pixels and latent dimensions, then average across images.
        latent_code = self.encoder.classify(self.inputs)
        mean, var = self.split_mean_variance(latent_code)
        logits = self.decoder.classify(mean)
        reconstruction = np.sum(np.logaddexp(0, logits) - self.inputs * logits, axis=1)
        variance_minus_one = np.expm1(var)
        kl = 0.5 * np.sum(mean**2 + variance_minus_one - var, axis=1)
        reconstruction, kl = float(reconstruction.mean()), float(kl.mean())
        print(
            f"Epoch {epoch:4d} | BCE {reconstruction:7.3f} | KL {kl:6.3f}"
            f" | Total {reconstruction + kl:7.3f}"
        )

    def reconstruct(self, input_value: np.ndarray) -> np.ndarray:
        mean, _ = self.split_mean_variance(
            self.encode(
                input_value
            )
        )
        print(mean.shape)
        return self.decode(mean)

if __name__ == '__main__':
    inputs, labels = dumb_parse_c_array(pathlib.Path(__file__).parent.resolve() / '..' / 'data' / 'font.h')
    rng = np.random.default_rng(42)
    model = Autoencoder(
        topology=[35, 16, 2],
        eta=0.0025,
        train_inputs=inputs,
        encoder_activation=Tanh(),
        decoder_activation=Tanh(),
        rng=rng,
        epochs=50000,
        batch_size=32,
        optimization=Adam
    )
    reconstructed = model.reconstruct(inputs) >= 0.5
    print(f"Input {inputs[1].reshape((7,5))} \nproduced\n{reconstructed[1].reshape((7,5))}")
    print("\n\n")

    # Scatterplot de codigo latente
    encoded = model.encode(inputs)
    print(encoded)
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
    print(latent_codes)
    print(countx)
    print(county)
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

    # inputs = load_emojis()
    # model = VariationalAutoencoder(
    #     train_inputs=inputs,
    #     eta=0.0025,
    #     encoder_activation=Tanh(),
    #     decoder_activation=Tanh(),
    #     topology=[16*16, 64, 2],
    #     batch_size=len(inputs),
    #     rng=rng,
    #     epochs=2000,
    #     optimization=Adam
    # )
    # reconstructed = model.reconstruct(inputs) >= 0.5
    # wrong_pixels = np.count_nonzero(reconstructed != inputs, axis=1)
    # print(f"Reconstruction from the posterior mean: {wrong_pixels.mean():.2f} wrong pixels/image")
    # latent = model.rng.standard_normal((16, 2))
    # generated = model.decode(latent) >= 0.5
    # _, first_indices = np.unique(generated, axis=0, return_index=True)
    # generated = generated[np.sort(first_indices)]
    # differences = np.count_nonzero(generated[:, None, :] != inputs[None, :, :], axis=2)
    # distances = differences.min(axis=1)
    # novel = distances > 0
    # if not np.any(novel):
    #     raise SystemExit("No novel emojis were generated. Try another seed.")
    # save_emoji_grid(pathlib.Path(__file__).parent.resolve() / 'pito.png', inputs, generated[novel], distances[novel])
    # print(f"Saved {np.count_nonzero(novel)} distinct new emojis to pito.png")
