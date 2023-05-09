import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_dist2center(img_shape):
    h, w = img_shape
    cj, ck = h//2, w//2
    js, ks = np.ogrid[:h, :w]
    dist = np.sqrt(np.square(js - cj) + np.square(ks - ck))
    return dist


def lowpass_filter(img_shape, d0, n=5, mode='gaussian'):
    dist = get_dist2center(img_shape)
    if mode == 'ideal':
        mask = (dist <= d0)
        kernel = np.full(img_shape, 1e-6)
        kernel[mask] = 1 - 1e-6
    elif mode == 'butter':
        kernel = 1 / (1 + np.power(dist / d0, 2*n))
    else:
        kernel = np.exp(-np.square(dist) / (2*np.square(d0)))

    return kernel


def highpass_filter(img_shape, d0, n=5, mode='gaussian'):
    dist = get_dist2center(img_shape)
    if mode == 'ideal':
        kernel = 1 - lowpass_filter(img_shape, d0, mode='ideal')
    elif mode == 'butter':
        kernel = 1 / (1 + np.power(d0 / dist, 2*n))
    else:
        kernel = 1 - lowpass_filter(img_shape, d0, mode='gaussian')
    return kernel


def band_reject_filter(img_shape, dl, dh, n=5, mode='gaussian'):
    width = dh - dl
    center = (dh + dl) / 2
    dist = get_dist2center(img_shape)

    if mode == 'ideal':
        kernel = np.ma.masked_outside(dist, dl, dh).mask
    elif mode == 'butter':
        kernel = kernel = 1 / (1 + np.power((dist*width) / (np.square(dist)-np.square(center)), 2*n))
    else:
        kernel = 1 - np.exp(-np.square((np.square(dist)-np.square(center))/(dist*width)))

    return kernel


def notch_reject_filter(img_shape, d0, u0, v0):
    h, w = img_shape
    cj, ck = h//2, w//2
    js, ks = np.ogrid[:h, :w]
    dist1 = np.sqrt(np.square(js - cj - u0) + np.square(ks - ck - v0))
    dist2 = np.sqrt(np.square(js - cj + u0) + np.square(ks - ck + v0))
    mask = (np.logical_and(dist1 > d0, dist2 > d0)).astype(np.float32)

    return mask


def vertical_notch_reject_filter(img_shape, width=1):
    cj, ck = img_shape[0]//2, img_shape[1]//2
    hw = width // 2
    kernel = np.ones(img_shape)
    kernel[:, ck-hw:ck+hw+1] = 1e-5
    kernel[cj-50:cj+50, ck-hw:ck+hw+1] = 1

    return kernel


def frequency_domain_filtering(image, kernel):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    rshift = kernel * fshift

    r = np.fft.ifftshift(rshift)
    result = np.fft.ifft2(r)
    result = np.abs(result)
    return result


def plot_mag(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fmag = np.log(np.abs(fshift))

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image, cmap='gray')
    axes[1].imshow(fmag, cmap='gray')
    plt.imshow(fmag, cmap='gray')
    plt.show()


def plot_filtering(image, kernel):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fmag = np.log(np.abs(fshift))

    rshift = kernel * fshift
    rmag = np.log(np.abs(rshift))
    r = np.fft.ifftshift(rshift)
    result = np.fft.ifft2(r)
    result = np.abs(result)

    # plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('sample2.png')
    axes[0, 1].imshow(fmag, cmap='gray')
    axes[0, 1].set_title('Magnitude spectrum')
    axes[0, 2].set_axis_off()
    axes[1, 0].imshow(result, cmap='gray')
    axes[1, 0].set_title('result6.png')
    axes[1, 1].imshow(rmag, cmap='gray')
    axes[1, 1].set_title('Magnitude spectrum')
    axes[1, 2].imshow(kernel, cmap='gray')
    axes[1, 2].set_title('Kernel')
    plt.tight_layout()
    return fig
