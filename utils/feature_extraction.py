import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]


def ekstrak_fitur_glcm(image_grid):
    gray_grid = cv2.cvtColor(image_grid, cv2.COLOR_RGB2GRAY)

    glcm = graycomatrix(
        gray_grid,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()

    n_angles = glcm.shape[3]
    entropies = []

    for angle_idx in range(n_angles):
        glcm_single = glcm[:, :, 0, angle_idx]
        glcm_nonzero = glcm_single[glcm_single > 0]
        entropy_val = -np.sum(glcm_nonzero * np.log2(glcm_nonzero))
        entropies.append(entropy_val)

    entropy = np.mean(entropies)

    return {
        'X1_Entropy': entropy,
        'X2_Contrast': contrast,
        'X3_Homogeneity': homogeneity,
        'X4_Energy': energy,
        'X5_Correlation': correlation
    }


def hitung_indeks_vegetasi_exg(image_grid):
    grid_float = image_grid.astype(np.float32)

    R = grid_float[:, :, 0]
    G = grid_float[:, :, 1]
    B = grid_float[:, :, 2]

    matriks_exg = (2 * G) - R - B
    rata_rata_exg = np.mean(matriks_exg)

    return rata_rata_exg


def hitung_indeks_exr_exgr(img_rgb):
    img = img_rgb.astype(np.float32)

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    total = R + G + B + 1e-6
    r = R / total
    g = G / total
    b = B / total

    exg = 2 * g - r - b
    exr = 1.4 * r - g
    exgr = exg - exr

    return np.mean(exr), np.mean(exgr)