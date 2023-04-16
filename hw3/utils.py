import numpy as np

class NeighborFinder:
    def __init__(self, h, w, mode='four'):
        self.h = h
        self.w = w
        self.mode = mode
        if mode == 'four':
            self.offsets = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
        else:
            self.offsets = offsets = np.array(np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))).T.reshape(9, 2)

    def __call__(self, points):
        '''
        Parameters
        ----------
        points : array-like of shape (2,)

        Returns
        -------
        neighbors : ndarray of shape (N_neighbors, 2)
        '''
        neighbors = points + self.offsets
        neighbors = neighbors[
            (neighbors[:, 0] >= 0)  # neighbors[:, 0]===neighbors.T[0] coorespond to j
            & (neighbors[:, 0] < self.h)
            & (neighbors[:, 1] >= 0)  # neighbors[:, 1]===neighbors.T[1] coorespond to k
            & (neighbors[:, 1] < self.w)
        ]
        return neighbors


def segmentation(img, foreground):
    h, w = img.shape
    get_neighbors = NeighborFinder(h, w, mode='four')

    components = []
    visited = np.zeros(img.shape).astype(bool)
    for j, k in np.ndindex(img.shape):
        if (visited[j, k]) or (img[j, k] != foreground):
            continue

        comps = []
        cands = [(j, k)]
        while cands:
            curr = cands.pop()
            neighbors = get_neighbors(curr)
            neighbors = neighbors[
                (img[neighbors.T[0], neighbors.T[1]] == foreground)
                & (visited[neighbors.T[0], neighbors.T[1]] == False)
            ]
            visited[curr[0], curr[1]] = True
            visited[neighbors.T[0], neighbors.T[1]] = True
            comps.extend(neighbors)
            cands.extend(neighbors)
        if comps:
            components.append(np.array(comps))
    components = sorted(components, key=lambda x: x.shape[0], reverse=True)
    return components