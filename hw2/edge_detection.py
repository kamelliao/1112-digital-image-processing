import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from utils import convolve, gaussian_kernel, plot_histogram

class EdgeDetection:
    def __init__(self):
        pass

    def thresholding(self):
        pass

# Sobel edge detection
def compute_gradient(img):
    img_padded = np.pad(img, 1, mode='constant')
    windows = sliding_window_view(img_padded, (3, 3))

    grad_magn = np.zeros(img.shape)
    grad_orien = np.zeros(img.shape)
    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
            magn, orien = sobel_operator(windows[j, k])
            grad_magn[j, k] = magn
            grad_orien[j, k] = orien

    return grad_magn, grad_orien

def sobel_operator(w):
    MASK_ROW_GRAD = np.array([[-1, 0, 1], [-2, 0, 2], [-1,  0,  1]])
    MASK_COL_GRAD = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1, -2, -1]])

    col_grad = (MASK_COL_GRAD * w).sum() / 3
    row_grad = (MASK_ROW_GRAD * w).sum() / 3

    magn = np.hypot(col_grad, row_grad)
    theta = np.arctan2(col_grad, row_grad)
    return magn, theta


# Canny edge detection
class CannyEdgeDetector:
    def __init__(self):
        self.th_cand = 50
        self.th_edge = 100

        self.PIXEL_EDGE = 255
        self.PXIEL_CAND = 100
        self.PIXEL_NONE = 0

    def __call__(self, img, th_cand=50, th_edge=100):
        self.th_cand = th_cand
        self.th_edge = th_edge

        magn, orien = compute_gradient(img)
        magn = self.non_maximal_suppression(magn, orien)
        magn = self.hysteretic_thresholding(magn)
        magn = self.connected_component_labeling(magn)
        return magn

    def non_maximal_suppression(self, magn, orien):
        magn_suppresed = magn.copy()

        for j in range(magn.shape[0]):
            for k in range(magn.shape[1]):
                nn1, nn2 = self._nearest_neighbors(np.array([j, k]), orien[j, k])
                if (
                    (magn[j, k] <= magn[nn1[0], nn1[1]])
                    or (magn[j, k] <= magn[nn2[0], nn2[1]])
                ):
                    magn_suppresed[j, k] = 0
        return magn_suppresed

    def hysteretic_thresholding(self, magn):
        bins = [self.th_cand, self.th_edge]
        quantized_values = np.array([self.PIXEL_NONE, self.PXIEL_CAND, self.PIXEL_EDGE])

        bins_idx = np.digitize(magn, bins=bins)
        magn = quantized_values[bins_idx]
        return magn

    def connected_component_labeling(self, img):
        h, w = img.shape
        
        # get coordinates of edge points
        j_ep, k_ep = np.where(img == self.PIXEL_EDGE)

        get_neighbors = NeighborFinder(h, w, mode='eight')
        neighbors = get_neighbors(np.array([[j, k] for j, k in zip(j_ep, k_ep)]))
        neighbors = neighbors[img[neighbors.T[0], neighbors.T[1]] == 100].tolist()  # candidate neighbors

        # perform DFS
        visited = np.zeros((h, w)).astype(bool)
        while neighbors:
            j, k = neighbors.pop()
            img[j, k] = self.PIXEL_EDGE
            visited[j, k] = True
            nbrs = get_neighbors([[j, k]])
            nbrs = nbrs[
                (img[nbrs.T[0], nbrs.T[1]]==self.PXIEL_CAND)
                & (visited[nbrs.T[0], nbrs.T[1]]==False)
            ].tolist()
            neighbors.extend(nbrs)

        # set un-connected candidate points to zero
        img[np.where(img==self.PXIEL_CAND)] = self.PIXEL_NONE
        return img

    def _nearest_neighbors(self, p, orien):
        o = np.array([np.cos(orien), np.sin(orien)])
        nn1 = (p + o).astype(np.uint8)
        nn2 = (p - o).astype(np.uint8)
        return nn1, nn2


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


# Laplacian of Gaussian edge detection
laplacian_4nbrs = (1/4)*np.array([[ 0, -1,  0], [-1, 4, -1], [ 0, -1,  0]])
laplacian_8nbrs_nonsep = (1/8)*np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
laplacian_8nbrs_sep = (1/8)*np.array([[-2,  1, -2], [ 1, 4,  1], [-2,  1, -2]])

def laplacian_of_gaussian(img):
    gaussian = gaussian_kernel(3, 5)
    img = convolve(img, gaussian)
    img = convolve(img, laplacian_8nbrs_sep)

    # visiualize gradient map (second-order)
    # img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # zero-crossing
    # 1 plot hitogram
    hist = plot_histogram(img)
    hist.savefig('hist.png')

    # 2 quantization
    threshold = 2
    quantized_value = np.array([-1, 0, 1])
    idx = np.digitize(img, bins=[-threshold, threshold])
    img = quantized_value[idx]

    # 3 check zero-crossing
    # note: the padding and sliding_window_view is for ease of dealing boundary cases
    img_padded = np.pad(img, (1, 1))
    windows = sliding_window_view(img_padded, (3, 3))
    
    edge_img = np.zeros(img.shape)
    zp_j, zp_k = np.where(img==0)
    for j, k in zip(zp_j, zp_k):
        w = windows[j, k]
        if (
            w[0, 1]*w[2, 1] < 0
            or w[1, 0]*w[1, 2] < 0
            or w[0, 0]*w[2, 2] < 0
            or w[0, 2]*w[2, 0] < 0
        ):
            edge_img[j, k] = 255

    return edge_img

# edge crispening
def edge_crispening(img, L=3, c=0.7):
    '''Perform edge crispening by unsharp masking.

    Parameters
    ----------
    img : ndarray of shape (H, W)
    L : int
        Size of low-pass filter
    c : float
        Weight value.
    '''
    lowpass_filter = gaussian_kernel(1, L)
    low_component = convolve(img, lowpass_filter)
    
    img = (c/(2*c-1))*img - ((1-c)/(2*c-1))*low_component
    return img

# Hough transform

if __name__ == '__main__':
    # img = cv2.imread('hw2_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
    # result = laplacian_of_gaussian(img)
    # cv2.imwrite('result4.jpg', result)
    img = cv2.imread('hw2_sample_images/sample2.png', cv2.IMREAD_GRAYSCALE)
    result = edge_crispening(img)
    cv2.imwrite('result5.jpg', result)
