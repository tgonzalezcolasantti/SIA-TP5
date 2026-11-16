import pathlib
from typing import List, Self, Tuple
import numpy as np

from data.parser import dumb_parse_c_array
from models.activation import Activation, Adaline, ReLU, Step, Tanh
from models.neural_network import Perceptron
from models.optimization import Adam, NoOptimization, Optimization
from models.standarization import ZScoreStandardize



class Autoencoder:
    def __init__(
            self: Self,
            topology: List[int],
            eta: float,
            encoder_activation:Activation,
            decoder_activation:Activation,
            train_inputs: ZScoreStandardize,
            epochs: int = 1000,
            batch_size: int = 1,
            seed: int = 42,
            optimization: Optimization = NoOptimization()):
        decoder_topo = topology.copy()
        decoder_topo.reverse()
        self.encoder = Perceptron(topology, encoder_activation, eta, optimization=optimization)
        self.decoder = Perceptron(decoder_topo, decoder_activation, eta, optimization=optimization)
        self.epochs = epochs
        self.epoch = 0
        self.inputs = train_inputs
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.train()

    def get_latent_code(self: Self, input_value: np.ndarray)-> np.ndarray:
        return self.encoder.classify(input_value.reshape((1, -1))).reshape((-1))

    def get_output_from_latent_code(self: Self, input_value: np.ndarray) -> np.ndarray:
        return self.inputs.unstandardize(self.decoder.classify(input_value.reshape(1, -1))) # type: ignore

    def train_single(self: Self, input_data: np.ndarray):
        # Forward pass
        transformed_input = self._transform_input_data(input_data)
        latent_code = self.encoder.classify(transformed_input)
        decoder_output = self.decoder.classify(latent_code)
        # Back pass
        # print(f"LATENT ENCODING {latent_code}")
        # print(f"OUTPUT{decoder_output}")
        # print(f"INPUT{input_data}")
        # print(f"DELTA{decoder_output - input_data}\n\n")
        decoder_propagated_error = self.decoder.update_weights((decoder_output - input_data).T)
        # print(decoder_propagated_error)
        decoder_propagated_error = self._apply_intermediate_derivative(decoder_propagated_error)
        self.encoder.update_weights(decoder_propagated_error)

    def _apply_intermediate_derivative(self: Self, decoder_error: np.ndarray) -> np.ndarray:
        'Helper function for VAE, override it to apply stochastic layer derivative to decoder error output'
        return decoder_error

    def _apply_intermediate_layer(self: Self, decoder_output: np.ndarray) -> np.ndarray:
        'Helper function for VAE, override it to apply stochastic layer to encoder output'
        return decoder_output

    def _transform_input_data(self: Self, input_data: np.ndarray) -> np.ndarray:
        'Helper function for denoiser autoencoder, override it to apply noise to the output'
        return input_data

    def train(
        self: Self,
    ):
        with np.errstate(invalid='raise'):
            if self.epochs < 0 or self.batch_size == 0 or self.batch_size < -1:
                raise ValueError("epochs must be nonnegative and batch size positive or -1")
            # if not inputs:
            #     raise ValueError("training_data must not be empty")
            # if expected.shape != (len(inputs), self.last_layer_count):
            #     raise ValueError("expected output shape does not match the last layer")
            effective_batch_size = len(self.inputs) if self.batch_size == -1 else self.batch_size
            for i in range(self.epoch, self.epochs):
                print(f"Epoch {i}", flush=True)
                order = self.rng.permutation(len(self.inputs))
                for start in range(0, len(self.inputs), effective_batch_size):
                    indices = order[start:start + effective_batch_size]
                    self.train_single(self.inputs[indices])
                self.epoch += 1
                # self.save()

if __name__ == '__main__':
    inputs = dumb_parse_c_array(pathlib.Path(__file__).parent.resolve() / '..' / 'data' / 'font.h')
    encoder = Autoencoder([35, 30, 25, 20, 15, 10, 5, 2], 0.0005, ReLU(), Tanh(), ZScoreStandardize(inputs), epochs=5000, batch_size=1, optimization=Adam())
    print(f"Input {inputs[1].reshape((7,5))} \nproduced\n{encoder.get_output_from_latent_code(encoder.get_latent_code(inputs[1])).reshape((7,5))}")

    # print("\n\n")
    # for i in range(0, 120, 10):
    #     for j in range(0, 127, 10):
    #         print(f"Output for ({i},{j}): {encoder.get_output_from_latent_code(np.asarray([i, j])).round(0)}")
