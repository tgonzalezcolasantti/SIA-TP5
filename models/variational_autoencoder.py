import argparse
from pathlib import Path
from typing import List, Self, override

import numpy as np

from models.activation import Activation, Tanh
from models.autoencoder import Autoencoder
from models.optimization import Adam, Optimization


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -logits))


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

def save_emoji_grid(
    output: Path,
    inputs: np.ndarray,
    samples: np.ndarray,
    distances: np.ndarray,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 6), layout="constrained")
    FigureCanvasAgg(figure)
    training_panel, generated_panel = figure.subfigures(1, 2) # type: ignore
    training_panel.suptitle("Emojis de entrenamiento")
    generated_panel.suptitle("Emojis generados\nPíxeles distintos respecto del más cercano")

    for panel, images in ((training_panel, inputs), (generated_panel, samples)):
        axes = panel.subplots(4, 4).flat
        for index, ax in enumerate(axes):
            ax.set_axis_off()
            if index >= len(images):
                continue
            ax.imshow(
                images[index].reshape(16, 16),
                cmap="gray_r",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            if panel is generated_panel:
                ax.set_title(f"{index + 1} · {distances[index]} px", fontsize=10)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)


def main() -> None:
    from data.emoji_loader import load_emojis

    parser = argparse.ArgumentParser(
        description="Train a variational autoencoder on binary emojis."
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generate", type=Path, metavar="OUTPUT.png",
        help="Save a PNG grid of training examples and novel generated emojis",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.generate and args.generate.suffix.lower() != ".png":
        parser.error("--generate requires a .png output path")

    rng = np.random.default_rng(args.seed)
    inputs = load_emojis()
    model = VariationalAutoencoder(
        topology=[inputs.shape[1], 64, 2],
        eta=0.005,
        encoder_activation=Tanh(),
        decoder_activation=Tanh(),
        rng=rng,
        train_inputs=inputs,
        epochs=args.epochs,
        batch_size=len(inputs),
        optimization=Adam
    )

    reconstructed = model.reconstruct(inputs) >= 0.5
    wrong_pixels = np.count_nonzero(reconstructed != inputs, axis=1)
    print(f"Reconstruction from the posterior mean: {wrong_pixels.mean():.2f} wrong pixels/image")

    if args.generate:
        latent = model.rng.standard_normal((16, 2))
        generated = model.decode(latent) >= 0.5
        _, first_indices = np.unique(generated, axis=0, return_index=True)
        generated = generated[np.sort(first_indices)]
        differences = np.count_nonzero(generated[:, None, :] != inputs[None, :, :], axis=2)
        distances = differences.min(axis=1)
        novel = distances > 0
        if not np.any(novel):
            raise SystemExit("No novel emojis were generated. Try another seed.")
        save_emoji_grid(args.generate, inputs, generated[novel], distances[novel])
        print(f"Saved {np.count_nonzero(novel)} distinct new emojis to {args.generate}")


if __name__ == "__main__":
    main()
