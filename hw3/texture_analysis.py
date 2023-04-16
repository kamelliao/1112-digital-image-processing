import numpy as np
import matplotlib.pyplot as plt

from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from kmeans import KMeans
# from skimage.filters.rank import entropy
# from skimage.morphology import disk

def convolve(img, kernel):
    img = img.astype(np.float64)
    kh, kw = kernel.shape
    img_padded = np.pad(img, kh-1, mode='symmetric')
    res = sliding_window_view(img_padded, (kh, kw)) * kernel
    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
            img[j, k] = res[j, k].sum()
    return img

def pooling(img, r, func):
    img = img.astype(np.float64)
    img_padded = np.pad(img, r//2, mode='symmetric')
    windows = sliding_window_view(img_padded, (r, r))
    for j, k in np.ndindex(img.shape):
        img[j, k] = func(windows[j, k])
    return img

def correlation(img):
    js, ks = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
    return (js * ks * img).mean()

def inertia(img):
    js, ks = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
    return (np.square(js - ks) * img).mean()

class LawsMicroStructures:
    _pool_methods = {
        'mean': np.mean,
        'std': np.std,
        'var': np.var,
        'median': np.median,
        'min': np.min,
        'max': np.max,
        'l2': np.linalg.norm,
        'l1': lambda x: np.sum(np.abs(x)),
        'correlation': correlation,
        'inertia': inertia,
        'energy': lambda x: np.divide(np.square(x).sum(), np.square(x.size)),
        # 'entropy': lambda x: entropy(x, disk(17//2 + 1)),
    }

    def __init__(self):
        self.v1 = (1/18)*np.array([1, 4, 1])
        self.v2 = (1/2)*np.array([-1, 0, 1])
        self.v3 = (1/2)*np.array([1, -2, 1])
        self.basis = np.vstack((self.v1, self.v2, self.v3))
        self.structures = np.array([
            np.outer(self.basis[i], self.basis[j]) for i, j in np.ndindex(self.basis.shape[0], self.basis.shape[0])
        ])

    def __call__(self, img, r, pool, pos=False):
        # microstructure inpulse response
        features = [convolve(img, kernel) for kernel in tqdm(self.structures)]
        
        # normalize
        features = [((feat - feat.min()) / (feat.max() - feat.min())) for feat in tqdm(features)]

        # energy computation
        if pool != 'raw':
            pool_func = LawsMicroStructures._pool_methods.get(pool, np.identity)
            features = [pooling(feat, r=r, func=pool_func) for feat in tqdm(features)]

        # consider position
        if pos:
            js, ks = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
            features.append(0.7*(js-js.min())/(js.max()-js.min()))
            features.append(0.3*(ks-ks.min())/(ks.max()-ks.min()))

        return np.array(features).transpose(1, 2, 0)

def texture_segmentation(img, args=None):
    laws = LawsMicroStructures()
    feats = laws(img, r=args.r, pool=args.pool, pos=args.pos)
    
    kmeans = KMeans(args.n_clusters)
    kmeans.fit(feats.reshape(feats.shape[0]*feats.shape[1], feats.shape[-1]))

    fig = plot_segmentation(img, kmeans.labels)
    labels = kmeans.labels.reshape(img.shape)
    return fig, labels

def plot_segmentation(img, labels):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray', alpha=0.9)
    ax.imshow(labels.reshape(img.shape), alpha=0.5)
    ax.set_axis_off()
    
    return fig

# print grid of (3, 3)
def plot_features(feat):
    fig, ax = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            im = ax[i, j].imshow(feat[:, :, i*3+j], cmap='gray')
            fig.colorbar(im)
    plt.axis('off')
    plt.savefig('feats.png')
