import numpy as np
from tqdm import trange

class KMeans:
    def __init__(self, n_clusters, max_iter = 300, init = 'kmeans++'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.labels = None
        self.centroids = None
        self.n_iter = 0

    def fit(self, X):
        # initialization
        self.labels = np.zeros(X.shape[0])
        self.centroids = self._init_centroids(X, self.init)

        # iteration
        prev_inertia = 0
        for i in trange(self.max_iter):
            self.inertia = 0

            # assign
            ssds = np.square(np.linalg.norm(X[:, None] - self.centroids, axis=2))  # (n_samples, n_clusters)

            self.labels = ssds.argmin(axis=-1)
            self.inertia += ssds.min(axis=-1).sum()

            # stopping criterion
            if abs(prev_inertia - self.inertia) < 1:
                break
            
            # update
            for k in range(self.n_clusters):
                cluster = X[self.labels==k]
                self.centroids[k] = cluster.mean(axis=0)
            
            prev_inertia = self.inertia
            self.n_iter += 1

        # compute for each cluster
        # self.cluster_intertia = np.zeros(self.n_clusters)
        # for k in range(self.n_clusters):
        #     cluster = X[self.labels==k]
        #     self.cluster_intertia[k] = np.square(np.linalg.norm(cluster - self.centroids[k]))

    def _init_centroids(self, X, method):
        if method == 'random':
            return self._init_centroids_random(X)
        elif method == 'kmeans++':
            return self._init_centroids_kmeanspp(X)
    
    def _init_centroids_random(self, X):
        cids = np.random.choice(X.shape[0], self.n_clusters)
        return X[cids]

    def _init_centroids_kmeanspp(self, X):
        # centroids = X[np.random.choice(X.shape[0], 1)]
        centroids = X[1126]
        for k in range(self.n_clusters - 1):
            dist = np.square(np.linalg.norm(X[:, None] - centroids, axis=2)).min(axis=-1)
            cid = dist.argmax()
            centroids = np.vstack((centroids, X[cid]))
        return centroids


def test_kmeans():
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    np.random.seed(1126)

    X, y = make_blobs(n_samples=200, n_features=2, centers=5, cluster_std=0.7)
    kmeans = KMeans(5)
    kmeans.fit(X)
    plt.scatter(X.T[0], X.T[1], c=kmeans.labels, alpha=0.6, s=10)
    plt.savefig('kmeans.png')