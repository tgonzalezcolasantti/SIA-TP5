import pathlib

from matplotlib import pyplot as plt
import numpy as np

from data.parser import dumb_parse_c_array


def plot_image(img: np.ndarray):
    plt.imshow(img.reshape(7,5), interpolation='nearest')
    plt.colorbar(label='Value')  # Add a color scale
    plt.title('2D Array Heatmap')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

if __name__ == '__main__':
    images = dumb_parse_c_array(pathlib.Path(__file__).parent.resolve() / 'font.h')
    for i in images:
        plot_image(i)