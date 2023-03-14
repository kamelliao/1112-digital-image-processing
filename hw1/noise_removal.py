import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# spatial filtering
def spatial_filter(img):
    img = img.astype(np.int64)
    kernel = create_kernel(1, 7)
    res = convolve(img, kernel)
    return res

def create_kernel(sigma, ws):
    '''Create a Gaussian kernel.

    Parameters
    ----------
    sigma: float
        Sigma value.
    ws: int
        Window size.
    
    Returns
    -------
    filter : array-like of shape (2r+1, 2r+1)
    '''
    def textbook_kernel_generator(i, j):
        if i > r:
            i %= r
        if j > r:
            j %= r
        return sigma**(i+j)
    
    def gaussian_kernel_generator(i, j):
        kernel = (1 / (2*np.pi*(sigma**2)))*np.exp(-(i**2+j**2)/(2*(sigma**2)))
        return kernel

    kernel_generator = np.vectorize(gaussian_kernel_generator)

    kernel = np.fromfunction(kernel_generator, (ws, ws))
    kernel /= kernel.sum()
    return kernel

def convolve(img, kernel):
    kh, kw = kernel.shape
    img_padded = np.pad(img, kh-1, mode='symmetric')
    res = sliding_window_view(img_padded, (kh, kw)) * kernel
    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
            img[j, k] = res[j, k].sum()
    return img

# outlier detection

# median filtering
def pmed_filter(img, ws=7):
    '''
    Findings:
        window size affect performance
        - 3~5: mild-noise
        - 7 up: low-noise, but blur the image 
    '''
    img_padded = np.pad(img, ws-1, mode='symmetric')
    windows = sliding_window_view(img_padded, (ws, ws))
    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
            img[j, k] = mask_pmed(windows[j, k], ws)
    return img

def mask_pmed(block, ws):
    center = int(ws/2)
    seq_x = block[center, :]
    seq_y = block[:, center]
    pmed_x = max(op_maxmin(seq_x), op_maxmin(seq_y))
    pmed_y = min(op_minmax(seq_x), op_minmax(seq_y))
    pmed = 0.5*pmed_x + 0.5*pmed_y  # don't write 0.5*(pmed_x + pmed_y) as it may lead to overflow
    return pmed

def op_maxmin(seq):
    windows = sliding_window_view(seq, 3)
    return max(windows.min(axis=1))

def op_minmax(seq):
    windows = sliding_window_view(seq, 3)
    return min(windows.max(axis=1))

# PSNR
def psnr(img1, img2):
    # note: np.uint8 may cause overflow
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    mse = np.square(img1 - img2).mean()
    psnr_value = 10*np.log10((255**2)/mse)
    return psnr_value