from functools import reduce
import cv2
import numpy as np
from utils import NeighborFinder


# segmentation
def segmentation(img):
    h, w = img.shape
    get_neighbors = NeighborFinder(h, w, mode='four')

    components = []
    visited = np.zeros(img.shape).astype(bool)
    for j, k in np.ndindex(img.shape):
        if (visited[j, k]) or (img[j, k] == 255):
            continue
        # perform dfs to find connected component
        comps = [[j, k]]
        cands = [[j, k]]
        while cands:
            curr = cands.pop()
            neighbors = get_neighbors(np.array([curr]))
            neighbors = neighbors[
                (img[neighbors.T[0], neighbors.T[1]] < 255)
                & (visited[neighbors.T[0], neighbors.T[1]] == False)
            ]
            visited[curr[0], curr[1]] = True
            visited[neighbors.T[0], neighbors.T[1]] = True
            comps.extend(neighbors)
            cands.extend(neighbors)
        if comps:
            components.append(np.array(comps))
    return components

# geometric transformation
def geometric_transformation(func):
    '''
    A function wrapper that deals with the common operations,
    such as coordinate transform and backward treatment, for
    all kinds of geometric transformations.

    Parameters
    ----------
    func : function
        The transformation function (by backward treatment).
        Input : np.array of shape (2, h*w) that indicates the Cartesian coordinates,
            where I[j, k] = [x, y].
        Output : np.array of shape (2, h*w) that indicates the Cartesian coordinates,
            where O[j, k] = [u, v].

    Returns
    -------
    new_img : np.array
        The result image after transformation.
    '''
    def wrapper(img, *args):
        h, w = img.shape
        org_coord = np.indices((h, w)).reshape(2, -1)
        org_coord = np.vstack((org_coord, np.ones(h*w)))
        # image coord -> Cartesian coord
        org_coord = convert_coordinate(org_coord, h, w, mode='i2c')

        # perform transformation
        new_coord = func(org_coord, *args)
        new_coord = new_coord.T.reshape(img.shape[1], img.shape[0], 2)
        
        new_img = np.full(img.shape, 255)
        for j, k in np.ndindex(img.shape):
            u, v = new_coord[j, k]
            if (
                u < 0
                or v < 0
                or u >= img.shape[1]
                or v >= img.shape[0]
            ):
                new_img[j, k] = 255
            else:
                # Cartesian coord -> image coord
                p, q = img.shape[0] - 1 - v, u
                new_img[j, k] = img[p, q]
        return new_img.astype(np.uint8)
    return wrapper

# linear transformation
@geometric_transformation
def linear_transformation(org_coord, trans):
    if isinstance(trans, list):
        trans.reverse()
        trans = reduce(np.dot, trans)
    # TODO: instead of .astype(int), could do interpolation
    new_coord = np.dot(trans, org_coord).astype(int)[:2]
    return new_coord

# popcat transformation
@geometric_transformation
def popcat(org_coord, center):
    org_coord = org_coord[:2,]
    vec = center - org_coord.T
    norm = np.linalg.norm(vec, axis=1)
    orien = np.arctan2(vec.T[1], vec.T[0])

    # note: clip to center
    new_coord = np.vstack([
        org_coord[0] + (np.clip(8*norm.max()/(norm+1), 0, a_max=norm))*np.cos(orien),
        org_coord[1] + (np.clip(8*norm.max()/(norm+1), 0, a_max=norm))*np.sin(orien)
    ]).astype(int)
    return new_coord

# utils and basic transformations
def convert_coordinate(coord, h, w, mode='i2c'):
    # image coordinate -> Cartesian coordinate
    if mode == 'i2c':
        coord[0], coord[1] = coord[1], h - 1 - coord[0]
        return coord
    # Cartesian coordinate -> image coordinate
    elif mode == 'c2i':
        coord[0], coord[1] = w - 1 - coord[1], coord[0]
        return coord

def translation_bt(tx, ty):
    return np.array([
        [1, 0, -tx],
        [0, 1, -ty],
        [0, 0,  1]
    ])

def scaling_bt(sx, sy):
    return np.array([
        [1/sx, 0, 0],
        [0, 1/sy, 0],
        [0,  0, 1]
    ])

def rotation_bt(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, s, 0],
        [-s,  c, 0],
        [0, 0, 1]
    ])