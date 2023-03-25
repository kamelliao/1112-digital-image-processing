import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

# convolution
def convolve(img, kernel):
    img = img.astype(np.float64)

    kh, kw = kernel.shape
    img_padded = np.pad(img, kh-1, mode='symmetric')
    res = sliding_window_view(img_padded, (kh, kw)) * kernel
    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
            img[j, k] = res[j, k].sum()
    return img

def gaussian_kernel(sigma, r):
    def gaussian_kernel_generator(i, j):
        kernel = (1 / (2*np.pi*(sigma**2)))*np.exp(-(i**2+j**2)/(2*(sigma**2)))
        return kernel
    kernel_generator = np.vectorize(gaussian_kernel_generator)
    
    kernel = np.fromfunction(kernel_generator, (r, r))
    kernel /= kernel.sum()
    return kernel

# neighbor finding
class NeighborFinder:
    def __init__(self, h, w, mode='eight'):
        self.h = h
        self.w = w
        self.mode = mode

    def __call__(self, points: np.ndarray):
        '''
        Parameters
        ----------
        points : ndarray of shape (N, 2)

        Returns
        -------
        neighbors : ndarray of shape (N_neighbors, 2)
        '''
        if self.mode == 'eight':
            offsets = np.array(np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))).T.reshape(9, 2)
        else:
            offsets = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])  # four neighbor
        
        neighbors = np.expand_dims(points, axis=1) + offsets
        neighbors = neighbors.reshape(neighbors.shape[0]*neighbors.shape[1], 2)

        # validate coordinates
        neighbors = neighbors[
            (neighbors[:, 0] >= 0)  # neighbors[:, 0]===neighbors.T[0] coorespond to j
            & (neighbors[:, 0] < self.h)
            & (neighbors[:, 1] >= 0)  # neighbors[:, 1]===neighbors.T[1] coorespond to k
            & (neighbors[:, 1] < self.w)
        ]
        
        return neighbors

# plot
def plot_histogram(img):
    counts, bins = np.histogram(img, bins=256)
    fig, ax = plt.subplots()
    ax.stairs(counts, bins, fill=True)
    return fig

def plot_cdf(img):
    counts, bins = np.histogram(img, bins=256, range=[0, 256])
    counts = counts.cumsum()
    fig, ax = plt.subplots()
    ax.stairs(counts, bins, fill=True)
    return fig