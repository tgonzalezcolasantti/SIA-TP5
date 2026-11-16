from typing import List, Self
import numpy as np

from models.activation import Activation, Adaline, ReLU, Step
from models.neural_network import Perceptron
from models.optimization import NoOptimization, Optimization



class Autoencoder:
    def __init__(self: Self, topology: List[int], activation:Activation, optimization: Optimization, eta: float):
        self.encoder = Perceptron(topology, activation, eta, optimization=optimization)
        decoder_topo = topology.copy()
        decoder_topo.reverse()
        self.decoder = Perceptron(decoder_topo, activation, eta, optimization=optimization)

    def get_latent_code(self: Self, input_value: np.ndarray)-> np.ndarray:
        return self.encoder.classify(input_value.reshape((1, -1))).reshape((-1))

    def get_output_from_latent_code(self: Self, input_value: np.ndarray) -> np.ndarray:
        return self.decoder.classify(input_value.reshape(1, -1)).reshape((-1))

    def train_single_sample(self: Self, input_data: np.ndarray):
        # Forward pass
        transformed_input = self._transform_input_data(input_data)
        latent_code = self.get_latent_code(transformed_input)
        decoder_output = self.get_output_from_latent_code(latent_code)
        # Back pass
        decoder_propagated_error = self.decoder.update_weights((decoder_output - input_data).reshape((-1,1)))
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

