import pathlib

from matplotlib import pyplot as plt
import numpy as np

from data.parser import dumb_parse_c_array


def plot_image(img: np.ndarray, title: str):
    plt.imshow(img.reshape(7,5), interpolation='nearest')
    plt.colorbar(label='Value')  # Add a color scale
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    images, labels = dumb_parse_c_array(pathlib.Path(__file__).parent.resolve() / 'font.h')
    plot_image(images[2], labels[2])