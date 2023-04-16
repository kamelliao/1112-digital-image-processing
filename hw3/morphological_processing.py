import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

from utils import segmentation

def morphological_operation(func):
    def wrap(img, *args):
        img = (img / 255).astype(int)
        result = func(img, *args)
        result = (result * 255).astype(int)
        return result
    return wrap

def get_structuring_element(conn=None):
    if conn == 'four':
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    elif conn == 'eight':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    return np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]])

# basic operations
def dilation(img, se):
    windows = sliding_window_view(np.pad(img, (se.shape[0]//2, se.shape[1]//2)), se.shape)
    result = np.any(((windows & se) == 1), axis=(-1, -2))
    return result.astype(int)

def erosion(img, se):
    img = np.pad(img, (se.shape[0]//2, se.shape[1]//2))
    img = np.logical_not(img)
    windows = sliding_window_view(img, se.shape)
    result = np.all((windows & se) == 0, axis=(-1, -2))
    return result.astype(int)

def open_operator(img, se):
    img = erosion(img, se)
    img = dilation(img, se)
    return img

def close_operator(img, se):
    img = dilation(img, se)
    img = erosion(img, se)
    return img

# applications
@morphological_operation
def boundary_extraction(img):
    se = get_structuring_element('four')
    img_erosed = erosion(img, se)
    result = img - img_erosed
    return result

@morphological_operation
def hole_filling(img):
    se = get_structuring_element('four')
    img_invert = np.logical_not(img).astype(int)
    components = segmentation(img, foreground=0)

    # note: the first component is background
    seeds = np.array([se[0] for se in components[1:]])

    result = np.zeros(img.shape, dtype=img.dtype)
    result[seeds.T[0], seeds.T[1]] = 1

    prev_result = result
    while True:
        result = np.logical_and(dilation(result, se), img_invert)
        if np.array_equal(prev_result, result):
            break
        prev_result = result
        
    result = np.logical_or(result, img)
    return result

@morphological_operation
def connected_component_labeling(img, seed):
    se = get_structuring_element('eight')

    result = np.zeros(img.shape, dtype=img.dtype)
    result[seed[0], seed[1]] = 1

    prev_result = result
    while True:
        result = np.logical_and(dilation(result, se), img)
        if np.array_equal(prev_result, result):
            break
        prev_result = result
    return result    

@morphological_operation
def open_op(img):
    se = get_structuring_element('four')
    result = open_operator(img, se)
    return result

@morphological_operation
def close_op(img):
    se = get_structuring_element('four')
    result = close_operator(img, se)
    return result

def object_counting(img):
    n_objects = 0
    target = img.copy()
    labels = np.full(img.shape, np.nan)
    while (target==255).sum() > 0:
        # print(f'n_objects = {n_objects}')
        whites = np.where(target==255)
        seed_j, seed_k = whites[0][0], whites[1][0]
        res = connected_component_labeling(img, [seed_j, seed_k])
        # cv2.imwrite('test.png', res)
        
        w_j, w_k = np.where(res==255)
        target[w_j, w_k] = 0
        labels[w_j, w_k] = n_objects + 1
        n_objects += 1

    fig, ax = plt.subplots()
    ax.imshow(labels)
    ax.axis('off')
    return n_objects, fig