import numpy as np


def expand(mat, r):
    if not np.log2(r).is_integer():
        raise ValueError('New size should be power of 2.')

    mat_expanded = mat.copy()
    while mat_expanded.shape[0] < r:
        mat_expanded = np.block([
            [4*mat_expanded+mat[0, 0], 4*mat_expanded+mat[0, 1]],
            [4*mat_expanded+mat[1, 0], 4*mat_expanded+mat[1, 1]]
        ])

    return mat_expanded


def normalize(dm):
    return dm / dm.size


def add_noise(img, loc=0, scale=10):
    img = img.astype(np.float32)
    noise = np.random.normal(loc, scale, (img.shape[0], img.shape[1]))

    return img + noise


class HalftoneDithering:
    def __init__(self, matrix, r=None):
        matrix = np.asarray(matrix)
        if r:
            matrix = expand(matrix, r)
        self.matrix = normalize(matrix)

    def __call__(self, image, noise=None):
        image = image.astype(np.float32)
        if noise:
            image = add_noise(image, noise[0], noise[1])
        image = image / 255
        n = int(np.ceil(image.shape[0] / self.matrix.shape[0]))
        dm_complete = np.tile(self.matrix, (n, n))[:image.shape[0], :image.shape[1]]
        res = (image >= dm_complete)

        return (res * 255).astype(np.uint8)


class HalftoneErrorDiffusion:
    def __init__(self, mask):
        self.mask = mask / mask.sum()

    def __call__(self, image):
        radius = self.mask.shape[0]//2
        img_padded = np.pad(image, radius).astype(np.float32) / 255
        for j in range(radius, radius + image.shape[0]):
            for k in range(radius, radius + image.shape[1]):
                new_val = np.digitize(img_padded[j, k], [0.5])
                error = img_padded[j, k] - new_val
                img_padded[j, k] = new_val
                img_padded[j-radius:j+radius+1, k-radius:k+radius+1] += error*self.mask

        result = img_padded[radius:image.shape[0]+radius, radius:image.shape[1]+radius]
        return (result * 255).astype(np.uint8)
