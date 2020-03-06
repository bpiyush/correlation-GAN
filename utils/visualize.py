import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored
from scipy.stats import pearsonr as correlation

def colored_print(string, color='yellow'):
    print(colored(string, color))


def fig2data(fig):
    """
    brief Convert a Matplotlib figure to a 4D numpy array with RGBA
    channels and return it
    @param fig: a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf.reshape(h, w, 4)


def fig2im(fig):
    """
    convert figure to ndarray
    """
    img_data = fig2data(fig).astype(np.int32)
    plt.close()
    return img_data[:, :, :3] / 255.


def plot_original_vs_generated(original_data, generated_data):
    """Predicted vs ground truth weight"""

    assert len(original_data.shape) == 2 and len(generated_data.shape) == 2
    original_correlation = np.round(correlation(original_data[:, 0], original_data[:, 1])[0], 3)
    generated_correlation = np.round(correlation(generated_data[:, 0], generated_data[:, 1])[0], 3)

    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.grid()
    ax.scatter(original_data[:, 0], original_data[:, 1], label=r'Original ($\rho = {}$)'.format(original_correlation))
    ax.scatter(generated_data[:, 0], generated_data[:, 1], label=r'Generated ($\rho = {}$)'.format(generated_correlation))
    ax.legend()

    ax.set_xlabel('X1: Number of points = {}'.format(original_data.shape[0]))
    ax.set_ylabel('X2')

    return fig2im(fig)