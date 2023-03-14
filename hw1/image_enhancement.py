import cv2
import numpy as np
from tqdm import tqdm

# amplitude scaling
def dec_brightness(img: np.ndarray):
    return (img / 3).astype(np.uint8)

def inc_brightness(img: np.ndarray):
    return (img * 3).astype(np.uint8)

# histogram equalization
def build_hist_equalize_map(img):
    hist, _ = np.histogram(img, 256, [0, 256])
    cdf = hist.cumsum()
    inv_slope = 255 / cdf[-1]
    hemap =  (cdf * inv_slope).astype(np.uint8)
    return hemap

def global_hist_equalize(img):
    hemap = build_hist_equalize_map(img)
    return hemap[img]

def local_hist_equalize(img, ws=100):
    height, width = img.shape
    img_padded = np.pad(img, ws-1, mode='symmetric')
    windows = np.lib.stride_tricks.sliding_window_view(img_padded, (ws, ws))
    for j in tqdm(range(height)):
        for k in range(width):
            hemap = build_hist_equalize_map(windows[j, k])
            img[j, k] = hemap[img[j, k]]
    return img

def pizer_local_hist_equalize(img, ws=200):
    height, width = img.shape
    rows, cols = int(height/ws), int(width/ws)

    # calculate histogram
    hemaps = []
    for i in range(rows):
        for j in range(cols):
            jt, jb = i*ws, (i+1)*ws
            kl, kr = j*ws, (j+1)*ws
            block = img[jt:jb, kl:kr]
            hemap = build_hist_equalize_map(block)
            hemaps.append(hemap)

    hemaps = np.array(hemaps).reshape(rows, cols, 256)

    # perform equalization
    coord_j = lambda j0: (j0 + 0.5)*ws - 1
    coord_k = lambda k0: (k0 + 0.5)*ws - 1

    for j in range(height):
        for k in range(width):
            j0 = (j - 0.5*ws) / ws
            k0 = (k - 0.5*ws) / ws

            # corner cases
            if j0 < 0 and k0 < 0:  # corner-tl
                img[j, k] = hemaps[0, 0][img[j, k]]
            elif j0 < 0 and k0 >= (cols - 1):  # corner-tr
                img[j, k] = hemaps[0, cols-1][img[j, k]]
            elif j0 >= (rows - 1) and k0 < 0:  # corner-bl
                img[j, k] = hemaps[rows-1, 0][img[j, k]]
            elif j0 >= (rows - 1) and k0 >= (cols - 1):  # corner-br
                img[j, k] = hemaps[rows-1, cols-1][img[j, k]]
            # border cases
            elif j0 < 0 and (0 <= k0 < (cols-1)):  # border-t
                j0, k0 = int(j0), int(k0)
                m10 = hemaps[0, k0][img[j, k]]
                m11 = hemaps[0, k0+1][img[j, k]]
                a = (k - coord_k(k0)) / ws
                img[j, k] = a*m10 + (1-a)*m11
            elif j0 >= (rows-1) and (0 <= k0 < (cols-1)):  # border-b
                j0, k0 = int(j0), int(k0)
                m00 = hemaps[rows-1, k0][img[j, k]]
                m01 = hemaps[rows-1, k0+1][img[j, k]]
                a = (k - coord_k(k0)) / ws
                img[j, k] = a*m00 + (1-a)*m01
            elif (0 <= j0 < (rows-1)) and k0 < 0:  # border-l
                j0, k0 = int(j0), int(k0)
                m01 = hemaps[j0, 0][img[j, k]]
                m11 = hemaps[j0+1, 0][img[j, k]]
                b = (j - coord_j(j0)) / ws
                img[j, k] = b*m01 + (1-b)*m11
            elif (0 <= j0 < (rows-1)) and (k0 >= (cols-1)):  # border-r
                j0, k0 = int(j0), int(k0)
                m00 = hemaps[j0, cols-1][img[j, k]]
                m10 = hemaps[j0+1, cols-1][img[j, k]]
                b = (j - coord_j(j0)) / ws
                img[j, k] = b*m00 + (1-b)*m10
            # center cases
            else:
                a = (k - coord_k(k0)) / ws
                b = (j - coord_j(j0)) / ws
                j0, k0 = int(j0), int(k0)

                m00 = hemaps[j0, k0][img[j, k]]
                m10 = hemaps[j0+1, k0][img[j, k]]
                m01 = hemaps[j0, k0+1][img[j, k]]
                m11 = hemaps[j0+1, k0+1][img[j, k]]
                
                img[j, k] = a*(b*m00+(1-b)*m10) + (1-a)*(b*m01+(1-b)*m11)

    return img

# transfer functions
def transfer_func(img):
    img = img.astype(np.float64)
    # a, b, c = -0.28, 16, 100
    # func_quadratic = lambda x: a*((x-b)**2)+c
    a, b, cutoff_point = 15, 30, 120
    func_log2 = lambda x: a*np.log2(x) + b
    img = np.piecewise(img,
        condlist=[img <= cutoff_point, img > cutoff_point],
        funclist=[func_log2, lambda x: x]
    )
    return img.astype(np.uint8)