import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

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


# plot
def plot_histogram(img):
    counts, bins = np.histogram(img, bins=256)

    fig, ax = plt.subplots()
    ax.stairs(counts, bins, fill=True)
    return fig