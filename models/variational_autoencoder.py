import argparse

import numpy as np

from models.activation import Adaline, Tanh
from models.neural_network import Perceptron
from models.optimization import Adam


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -logits))


class VariationalAutoencoder:
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        latent_dim: int = 2,
        learning_rate: float = 0.005,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.latent_dim = latent_dim
        self.encoder = Perceptron(
            [input_size, hidden_size, 2 * latent_dim],
            Tanh(),
            learning_rate,
            optimization=Adam(),
            output_activation=Adaline(),
            rng=self.rng,
        )
        self.decoder = Perceptron(
            [latent_dim, hidden_size, input_size],
            Tanh(),
            learning_rate,
            optimization=Adam(),
            output_activation=Adaline(),
            rng=self.rng,
        )

    def encode(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the mean and log-variance of each input's latent distribution."""
        parameters = self.encoder.classify(inputs).copy()
        mean, log_variance = np.split(parameters, 2, axis=1)
        return mean, log_variance

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Return the probability of a black pixel for each output position."""
        return sigmoid(self.decoder.classify(latent))

    def reconstruct(self, inputs: np.ndarray) -> np.ndarray:
        mean, _ = self.encode(inputs)
        return self.decode(mean)

    def train_step(self, inputs: np.ndarray) -> tuple[float, float]:
        """Update both networks and return mean reconstruction loss and mean KL."""
        inputs = np.atleast_2d(np.asarray(inputs, dtype=float))
        if not inputs.size or not np.all((0 <= inputs) & (inputs <= 1)):
            raise ValueError("Expected a nonempty batch with pixels in [0, 1]")

        mean, log_variance = self.encode(inputs)
        noise = self.rng.standard_normal(mean.shape)
        latent_offset = np.exp(0.5 * log_variance) * noise
        logits = self.decoder.classify(mean + latent_offset)

        # Sum across pixels and latent dimensions, then average across images.
        reconstruction = np.sum(np.logaddexp(0, logits) - inputs * logits, axis=1)
        variance_minus_one = np.expm1(log_variance)
        kl = 0.5 * np.sum(mean**2 + variance_minus_one - log_variance, axis=1)

        # Perceptron averages its parameter updates over the batch.
        latent_gradient = self.decoder.update_weights((sigmoid(logits) - inputs).T).T
        mean_gradient = latent_gradient + mean
        log_variance_gradient = 0.5 * (latent_gradient * latent_offset + variance_minus_one)
        parameter_gradients = np.concatenate((mean_gradient, log_variance_gradient), axis=1)
        self.encoder.update_weights(parameter_gradients.T)
        return float(reconstruction.mean()), float(kl.mean())


def main() -> None:
    from data.emoji_loader import load_emojis

    parser = argparse.ArgumentParser(
        description="Train a variational autoencoder on binary emojis."
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    inputs = load_emojis()
    model = VariationalAutoencoder(input_size=inputs.shape[1], seed=args.seed)
    print(f"Training with {len(inputs)} emojis and {model.latent_dim} latent dimensions")
    for epoch in range(1, args.epochs + 1):
        reconstruction, kl = model.train_step(inputs)
        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:4d} | BCE {reconstruction:7.3f} | KL {kl:6.3f}"
                f" | Total {reconstruction + kl:7.3f}"
            )

    reconstructed = model.reconstruct(inputs) >= 0.5
    wrong_pixels = np.count_nonzero(reconstructed != inputs, axis=1)
    print(f"Reconstruction from the posterior mean: {wrong_pixels.mean():.2f} wrong pixels/image")


if __name__ == "__main__":
    main()
