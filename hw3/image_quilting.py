import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from utils import segmentation

class ImageQuilting:
    def __init__(self, texture, patch_size=50, overlap_ratio=1/6, tolerance=0.1):
        self.texture = texture
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.tolerance = tolerance

        self.overlap_len = int(self.patch_size*self.overlap_ratio)
        patch_src = sliding_window_view(texture, (patch_size, patch_size))
        self.patch_src = patch_src.reshape((patch_src.shape[0]*patch_src.shape[1], patch_size, patch_size)).astype(np.float64)
        self.patch_sets = {
            part: np.array([self._get_overlap(x, part) for x in self.patch_src]).astype(np.float64)
            for part in ['right', 'left', 'top', 'bottom']
        }

    def __call__(self, shape, transfer_src=None):
        n_rows = int(np.ceil((shape[0] - self.patch_size) / (self.patch_size - self.overlap_len)))
        n_cols = int(np.ceil((shape[1] - self.patch_size) / (self.patch_size - self.overlap_len)))
        nonov_len = self.patch_size - self.overlap_len
        if (n_rows-1)*(nonov_len)+self.patch_size < shape[0]:
            n_rows += 1
        if (n_cols-1)*(nonov_len)+self.patch_size < shape[1]:
            n_cols += 1
        print(f'Rows = {n_rows}, Cols = {n_cols}')

        # Step 1. Select candidate patches
        if transfer_src is None:
            patch_ids = self._select_patches(n_rows, n_cols)
        else:
            # TODO:
            patch_ids = self._select_patches_transfer(n_rows, n_cols, transfer_src)

        # Step 2. Find optimal cut and fuse patches
        # first row
        newshape = ((n_rows-1)*(nonov_len)+self.patch_size, (n_cols-1)*(nonov_len)+self.patch_size)
        result = np.zeros(newshape)
        result[:self.patch_size, :self.patch_size] = self.patch_src[patch_ids[0, 0]]
        for k in range(1, n_cols):
            js = 0
            ks = k*(self.patch_size-self.overlap_len)
            offsets = self._get_patch_mask(0, k, patch_ids, 'l')
            result[js+offsets[0], ks+offsets[1]] = self.patch_src[patch_ids[0, k]][offsets[0], offsets[1]]

        # start from second row
        for j in range(1, n_rows):
            for k in range(n_cols):
                js = j*(self.patch_size-self.overlap_len)
                ks = k*(self.patch_size-self.overlap_len)
                mode = 't' if k % n_cols == 0 else 'lt'
                offsets = self._get_patch_mask(j, k, patch_ids, mode)
                result[js+offsets[0], ks+offsets[1]] = self.patch_src[patch_ids[j, k]][offsets[0], offsets[1]]

        # Step 3. Post-processing: crop to original size
        js, ks = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        result = result[js, ks]

        return result

    def _select_patches(self, n_rows, n_cols):
        patch_ids = np.zeros((n_rows, n_cols), dtype=int)

        # first row
        patch_ids[0, 0] = np.random.choice(len(self.patch_src))
        for k in range(1, n_cols):
            error = self._get_overlap_error(0, k, patch_ids, 'l')
            candidates = np.where(error <= error.min()*(1+self.tolerance))[0]
            patch_ids[0, k] = np.random.choice(candidates)

        # start from second row
        for j in range(1, n_rows):
            for k in range(n_cols):
                mode = 't' if k % n_cols == 0 else 'lt'
                error = self._get_overlap_error(j, k, patch_ids, mode)
                candidates = np.where(error <= error.min()*(1+self.tolerance))[0]
                patch_ids[j, k] = np.random.choice(candidates)

        return patch_ids

    def _select_patches_transfer(self, n_rows, n_cols, transfer_src):
        # TODO:
        patch_ids = np.zeros((n_rows, n_cols), dtype=int)

        # first row
        patch_ids[0, 0] = np.random.choice(len(self.patch_src))
        for k in range(1, n_cols):
            js, ks = 0, k*(self.patch_size-self.overlap_len)
            js, ks = np.meshgrid(np.arange(js, js+self.patch_size), np.arange(ks, ks+self.patch_size), indexing='ij')
            error = self._get_overlap_error(0, k, patch_ids, 'l')
            error += self._get_correspondence_error(transfer_src[js, ks])
            candidates = np.where(error <= error.min()*(1+self.tolerance))[0]
            patch_ids[0, k] = np.random.choice(candidates)

        # start from second row
        for j in range(1, n_rows):
            for k in range(n_cols):
                js, ks = j*(self.patch_size-self.overlap_len), k*(self.patch_size-self.overlap_len)
                js, ks = np.meshgrid(np.arange(js, js+self.patch_size), np.arange(ks, ks+self.patch_size), indexing='ij')
                mode = 't' if k % n_cols == 0 else 'lt'
                error = self._get_overlap_error(j, k, patch_ids, mode)
                error += self._get_correspondence_error(transfer_src[js, ks])
                candidates = np.where(error <= error.min()*(1+self.tolerance))[0]
                patch_ids[j, k] = np.random.choice(candidates)

        return patch_ids

    def _get_overlap(self, x, part):
        if part == 'right':
            return x[:, :self.overlap_len]
        elif part == 'left':
            return x[:, -self.overlap_len:]
        elif part == 'top':
            return x[:self.overlap_len, :]
        elif part == 'bottom':
            return x[-self.overlap_len:, :]

    def _get_overlap_error(self, j, k, patch_ids, mode):
        if 'l' in mode:
            pl = self.patch_sets['right'][patch_ids[j, k - 1]].flatten()
            candl = self.patch_sets['left'].reshape(len(self.patch_src), self.patch_size*self.overlap_len)
        
        if 't' in mode:
            pt = self.patch_sets['bottom'][patch_ids[j - 1, k]].flatten()
            candt = self.patch_sets['top'].reshape(len(self.patch_src), self.patch_size*self.overlap_len)
        
        if mode == 'l':
            return np.linalg.norm(pl - candl, axis=-1)
        elif mode == 't':
            return np.linalg.norm(pt - candt, axis=-1)
        else:
            return np.linalg.norm(np.hstack([(pl - candl), (pt - candt)]), axis=-1)

    def _get_correspondence_error(self, patch_og):
        # TODO:
        return np.linalg.norm(patch_og.flatten() - self.patch_src.reshape(len(self.patch_src), self.patch_size*self.patch_size), axis=-1)

    def _get_patch_mask(self, j, k, patch_ids, mode):
        mask1 = np.full((self.patch_size, self.patch_size), True, dtype=bool)
        mask2 = np.full((self.patch_size, self.patch_size), True, dtype=bool)

        if 'l' in mode:
            ov_l = self.patch_sets['left'][patch_ids[j, k]]
            ov_nbr_l = self.patch_sets['right'][patch_ids[j, k-1]]
            _, offsets_l = self._optimal_cut(ov_l, ov_nbr_l, orien='vertical')

            mask1 = np.zeros((self.patch_size, self.patch_size), dtype=bool)
            for pj in range(self.patch_size):
                mask1[pj, np.arange(offsets_l[pj], self.patch_size)] = 1

        if 't' in mode:
            ov_t = self.patch_sets['top'][patch_ids[j, k]]
            ov_nbr_t = self.patch_sets['bottom'][patch_ids[j-1, k]]
            _, offsets_t = self._optimal_cut(ov_t, ov_nbr_t, orien='horizontal')

            mask2 = np.zeros((self.patch_size, self.patch_size), dtype=bool)
            for pk in range(self.patch_size):
                mask2[np.arange(offsets_t[pk], self.patch_size), pk] = 1

        offsets = np.where((mask1 & mask2)==True)
        return offsets

    def _optimal_cut(self, ov1, ov2, orien='vertical'):
        return optimal_cut(ov1, ov2, self.patch_size, self.overlap_len, orien)

def optimal_cut(ov1, ov2, patch_size, overlap_len, orien='vertical'):
    if orien == 'horizontal':
        ov1 = ov1.T
        ov2 = ov2.T

    error = np.square(ov1 - ov2)
    memo = np.zeros((patch_size, overlap_len), dtype=int)
    memo[0] = np.arange(0, overlap_len)
    for i in range(1, patch_size):
        for j in range(overlap_len):
            opt_val = error[i-1, j]
            opt_pred = j
            if j == 0:
                if opt_val >= error[i-1, j+1]:
                    opt_val = error[i-1, j+1]
                    opt_pred = j+1
            elif j == overlap_len - 1:
                if opt_val >= error[i-1, j-1]:
                    opt_val = error[i-1, j-1]
                    opt_pred = j-1
            else:
                if opt_val >= error[i-1, j-1]:
                    opt_val = error[i-1, j-1]
                    opt_pred = j-1
                if opt_val >= error[i-1, j+1]:
                    opt_val = error[i-1, j+1]
                    opt_pred = j+1

            error[i, j] += opt_val
            memo[i, j] = opt_pred
    
    # back tracking
    path = np.zeros(patch_size, dtype=int)
    path[-1] = error[-1].argmin()
    for k in range(patch_size - 2, -1, -1):
        path[k] = memo[k][path[k+1]]

    return error, path


if __name__ == '__main__':
    img = cv2.imread('hw3_sample_images/sample2.png', cv2.IMREAD_GRAYSCALE)
    texture = cv2.imread('hw3_sample_images/sample3.png', cv2.IMREAD_GRAYSCALE)
    labels = np.load('labels.npy').reshape(img.shape)  # perform problem 2-(c) to get this file

    ocean = segmentation(labels, foreground=1)[0]
    ul = ocean.min(axis=0)
    lr = ocean.max(axis=0)
    mask =  ocean - ul
    
    # image quilting
    imgquilter = ImageQuilting(texture, patch_size=50, overlap_ratio=1/6)
    result = imgquilter((lr - ul + 1))
    img[ocean.T[0], ocean.T[1]] = result[mask.T[0], mask.T[1]]
    
    # texture transfer
    # src = np.zeros((lr - ul + 1))
    # src[mask.T[0], mask.T[1]] = img[ocean.T[0], ocean.T[1]]
    # result = imgquilter((lr - ul + 1), transfer_src=src)

    cv2.imwrite('result7.png', img)
