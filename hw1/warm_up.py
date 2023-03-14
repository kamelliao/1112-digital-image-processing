import numpy as np

def grayscale(img: np.ndarray):
    R_WT = 0.299
    G_WT = 0.587
    B_WT = 0.114
    return (img[:, :, 0]*R_WT + img[:, :, 1]*G_WT + img[:, :, 2]*B_WT).astype(np.uint8)

def flip(img: np.ndarray):
    # 0: vertical, 1: horizontal
    return np.flip(img, axis=0)
