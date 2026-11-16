from pathlib import Path

import numpy as np
from matplotlib.image import imread


def load_emojis(directory: Path = Path(__file__).with_name("emojis")) -> np.ndarray:
    """Load black pixels as 1 and white or transparent pixels as 0."""
    images = []
    for path in sorted(directory.glob("*.png")):
        pixels = imread(path)
        if pixels.shape != (16, 16, 4):
            raise ValueError(f"Expected a 16 × 16 RGBA PNG: {path}")
        black = np.all(pixels[..., :3] == 0, axis=-1)
        opaque = pixels[..., 3] == 1
        images.append((black & opaque).ravel())

    if not images:
        raise ValueError(f"No emoji PNGs found in {directory}")
    return np.asarray(images, dtype=float)
