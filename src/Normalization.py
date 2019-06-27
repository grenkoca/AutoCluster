from __future__ import division
import time
import cv2
import spams
import numpy as np
from multiprocessing import Pool
from functools import partial


def od2rgb(OD):
    return (255 * np.exp(-1 * OD)).astype(np.uint8)


def rgb2od(I):
    I = remove_zeros(I)
    return -1 * np.log(I / 255)


def standardize_brightness(I):
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


def remove_zeros(I):
    mask = I == 0
    I[mask] = 1
    return I


def notwhite_mask(I, thresh=0.8):
    """
    Get a binary mask where true denotes 'not white'
    :param I:
    :param thresh:
    :return:
    """
    I_LAB = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0
    return L < thresh


def get_stain_matrix(I, threshold=0.8, lamda=0.1):
    """
    Get 2x3 stain matrix. First row H and second row E
    :param I:
    :param threshold:
    :param lamda:
    :return:
    """
    mask = notwhite_mask(I, thresh=threshold).reshape((-1,))
    OD = rgb2od(I).reshape((-1, 3))
    OD = OD[mask]
    if len(OD) == 0:
        return np.array([[0, 0, 0], [0, 0, 0]])
    dictionary = spams.trainDL(
        OD.T,
        K=2,
        lambda1=lamda,
        mode=2,
        modeD=0,
        posAlpha=True,
        posD=True,
        verbose=False,
    ).T
    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    dictionary = normalize_rows(dictionary)
    return dictionary


def normalize_rows(A):
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    if np.all(A == 0.0):  # Check for 0s
        return A
    return A / np.linalg.norm(A, axis=1)[:, None]


###


def get_concentrations(I, stain_matrix, lamda=0.01):
    """
    Get concentrations, a npix x 2 matrix
    :param I:
    :param stain_matrix: a 2x3 six
    :return:
    """
    OD = rgb2od(I).reshape((-1, 3))
    return (
        spams.lasso(OD.T, D=stain_matrix.T, mode=2, lambda1=lamda, pos=True).toarray().T
    )


class Normalizer(object):
    """
    A stain normalization object
    """

    def __init__(self):
        self.stain_matrix_target = None

    def set_target_matrix(self, mat):
        self.stain_matrix_target = mat

    def get_target_matrix(self):
        return self.stain_matrix_target

    def fit(self, target):
        target = standardize_brightness(target)
        self.stain_matrix_target = get_stain_matrix(target)

    def target_stains(self):
        return od2rgb(self.stain_matrix_target)

    def transform(self, I):
        I = standardize_brightness(I)
        stain_matrix_source = get_stain_matrix(I)
        if stain_matrix_source.any() == 0:
            return np.full_like(I, 255)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        return (
            255
            * np.exp(
                -1
                * np.dot(source_concentrations, self.stain_matrix_target).reshape(
                    I.shape
                )
            )
        ).astype(np.uint8)

    def transform_set(self, tiles, n_samples=50, njobs=1):
        tile_interval = int(len(tiles) / n_samples)
        print("Calculating stain matrices")
        start = time.time()

        pool = Pool(processes=njobs)
        standardized_tiles = [
            standardize_brightness(tile[1]) for tile in tiles[::tile_interval]
        ]
        print(len(standardized_tiles))
        stain_matrices = pool.map(get_stain_matrix, standardized_tiles)
        print(stain_matrices)

        flat_matrices = None
        for matrix in stain_matrices:
            if np.all(matrix > 240):
                continue
            if flat_matrices is None:
                flat_matrices = np.array([matrix])
            else:
                flat_matrices = np.append(flat_matrices, [matrix], axis=0)

        stain_matrix_source = np.mean(stain_matrices, axis=0)

        print(f"Stain vectors found. Took {time.time() - start} seconds")
        print(stain_matrix_source)

        print("Transforming tiles")
        start = time.time()
        transform_partial = partial(
            transform_tile, stain_matrix_source, self.stain_matrix_target
        )
        transformed_tiles = pool.map(transform_partial, tiles)
        print(f"All tiles found. Took {time.time() - start} seconds")

        return transformed_tiles

    def fit_set(self, targets):
        target_stains = None
        for target in targets:
            target = standardize_brightness(target)
            if target_stains is None:
                target_stains = [get_stain_matrix(target).astype(np.float32)]
            else:
                target_stains = np.append(
                    target_stains, [get_stain_matrix(target).astype(np.float32)], axis=0
                )
        self.stain_matrix_target = np.mean(target_stains, axis=0)


def transform_tile(stain_matrix_source, stain_matrix_target, tile):
    if np.all(get_stain_matrix(tile[1]) == 0):
        return [tile[0], np.full_like(tile[1], 255)]
    else:
        source_concentrations = get_concentrations(tile[1], stain_matrix_source)
        return [
            tile[0],
            (
                255
                * np.exp(
                    -1
                    * np.dot(source_concentrations, stain_matrix_target).reshape(
                        tile[1].shape
                    )
                )
            ).astype(np.uint8),
        ]
