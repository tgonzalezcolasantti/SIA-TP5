from pathlib import Path
import pickle
import sys
from typing import List, Optional, Self, Tuple, override

import numpy as np

from models.optimization import NoOptimization, Optimization

from .activation import Activation


class Perceptron:
    def __init__(
        self: Self,
        neuron_topology: List[int],
        activation: Activation,
        learning_rate: float,
        save: Path | None = None,
        epoch: int = 0,
        weights: np.ndarray | None = None,
        optimization: Optimization = NoOptimization(),
        model_name: Optional[str] = None,
    ):
        self.model_name = model_name
        self.topology = neuron_topology
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimization = optimization
        self.epoch = epoch
        self.error_history = []

        if save:
            if save.is_dir():
                save = save / self._suggested_filename()
            if save.exists():
                unpickled: Perceptron = self.load(save)
                weights = unpickled.weights
                neuron_topology = unpickled.topology
                self.activation = unpickled.activation
                self.optimization = unpickled.optimization
                self.epoch = unpickled.epoch
                self.learning_rate = unpickled.learning_rate
                self.error_history = unpickled.error_history
        else:
            if len(neuron_topology) < 2 or any(count <= 0 for count in neuron_topology):
                raise ValueError(
                    "neuron_topology must contain at least two positive layer sizes"
                )

        total = sum(neuron_topology)
        self.weights = np.zeros((total, total)) if weights is None else weights
        self.outputs = np.zeros(total)
        self.h = np.zeros(total)
        self.deltas = np.zeros(total)
        self.gradient = np.zeros(self.weights.shape)
        self.last_layer_count = neuron_topology[-1]
        self.save_path = save
        boundaries = np.cumsum([0, *neuron_topology])
        self.layer_slices = [
            slice(boundaries[i], boundaries[i + 1]) for i in range(len(neuron_topology))
        ]
        if weights is None:
            self._init_weights(neuron_topology)

    def _init_weights(self: Self, neuron_topology: List[int]):
        for layer_idx in range(1, len(neuron_topology)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            limit = np.sqrt(
                6 / (neuron_topology[layer_idx - 1] + neuron_topology[layer_idx])
            )
            self.weights[current, previous] = np.random.uniform(
                -limit,
                limit,
                (neuron_topology[layer_idx], neuron_topology[layer_idx - 1]),
            )

    def classify(self: Self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        if inputs.ndim != 2 or inputs.shape[1] != self.topology[0]:
            raise ValueError("input size does not match the first layer")
        return self._calculate_outputs(inputs).T

    def _calculate_outputs(self: Self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        shape = (self.weights.shape[0], len(inputs))
        if self.outputs.shape != shape:
            self.outputs = np.zeros(shape)
            self.h = np.zeros(shape)
        self.outputs[self.layer_slices[0]] = inputs.T
        for layer_idx in range(1, len(self.layer_slices)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            bias = np.diagonal(self.weights[current, current])[:, None]
            self.h[current] = self.weights[current, previous] @ self.outputs[previous] + bias
            self.outputs[current] = self.activation.excite(self.h[current])
        return self.outputs[self.layer_slices[-1]]

    def _error(self: Self, output_errors: np.ndarray) -> float:
        return np.sum(output_errors ** 2) / 2

    def _global_errors(
        self: Self, results: np.ndarray, expected: np.ndarray
    ) -> Tuple[float, float]:
        errors = results - expected
        mse = np.mean(errors**2)
        mae = np.mean(np.abs(errors))
        print(f"MSE: {mse}\tMAE: {mae}")
        return mse, mae

    def update_weights(self: Self, output_errors: np.ndarray):
        batch_count = output_errors.shape[1]
        output_layer = self.layer_slices[-1]
        output_derivatives = self.activation.derivative(self.h[output_layer])
        if self.deltas.shape != (self.weights.shape[0], output_errors.shape[1]):
            self.deltas = np.zeros((self.weights.shape[0], output_errors.shape[1]))
        else:
            self.deltas[:] = 0
        self.deltas[output_layer] = output_errors * output_derivatives

        #backpropagation for derivatives
        for layer_idx in range(len(self.layer_slices) - 2, 0, -1):
            current = self.layer_slices[layer_idx]
            following = self.layer_slices[layer_idx + 1]
            derivatives = self.activation.derivative(self.h[current])
            self.deltas[current] = (
                self.weights[following, current].T @ self.deltas[following]
            ) * derivatives
        delta_propagation = self.weights[self.layer_slices[1], self.layer_slices[0]].T @ self.deltas[self.layer_slices[1]]
        # with np.printoptions(precision=2, threshold=sys.maxsize):
        #     print(self.deltas)
        for layer_idx in range(1, len(self.layer_slices)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            self.gradient[current, previous] = self.deltas[current] @ self.outputs[previous].T / batch_count
        # gradient = self.deltas @ self.outputs.T / batch_count
        self.gradient[range(len(self.weights)), range(len(self.weights))] = np.sum(self.deltas, axis=1) / output_errors.shape[1]
        self.weights += self.optimization.apply(
            self.learning_rate, self.gradient
        )
        return delta_propagation

    @classmethod
    def load(cls:type[Perceptron], save_path: Path):
        with open(save_path, 'rb+') as file:
            return pickle.load(file)

    def save(self: Self):
        if self.save_path:
            with open(self.save_path, '+wb') as file:
                pickle.dump(self, file)

    def _suggested_filename(self: Self) -> str:
        return f"{self.model_name + ' ' if self.model_name else None}"+\
               f"[{','.join(str(x) for x in self.topology)}]"+\
               f" {self.learning_rate:.6g} {str(self.activation)} {str(self.optimization)}.model"


class SimplePerceptron(Perceptron):
    def __init__(
        self: Self,
        inputs: int,
        activation: Activation,
        learning_rate: float,
        weight_range: Tuple[float, float] = (0, 1),
        epoch: int = 0,
        save: Optional[Path] = None,
        model_name: Optional[str] = None
    ):
        self.min_weight = min(weight_range)
        self.max_weight = max(weight_range)
        super().__init__(
            neuron_topology=[inputs, 1],
            activation=activation,
            learning_rate=learning_rate,
            save=save,
            model_name=model_name,
            epoch=epoch
        )

    @override
    def _init_weights(self: Self, neuron_topology: List[int]):
        for layer_idx in range(1, len(neuron_topology)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            self.weights[current, previous] = np.random.uniform(
                self.min_weight,
                self.max_weight,
                (neuron_topology[layer_idx], neuron_topology[layer_idx - 1]),
            )


class MultiLayerPerceptron(Perceptron):
    pass
