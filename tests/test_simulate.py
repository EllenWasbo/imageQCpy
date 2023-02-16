# -*- coding: utf-8 -*-
"""
Tests simulating calculations with known input to validate known output.

@author: ewas
"""
import random
import numpy as np
from scipy import ndimage

from imageQC.scripts.input_main_auto import InputMain
import imageQC.scripts.mini_methods_calculate as mmcalc
import imageQC.scripts.calculate_qc as calculate_qc


def add_noise(input_signal, random_seed=0, dark_noise=2):
    """Simulate noise and add to input signal (image)."""
    rs = np.random.RandomState(random_seed)
    poisson_noise = rs.poisson(input_signal, size=input_signal.shape)
    dark_noise = rs.normal(scale=dark_noise, size=input_signal.shape)
    noisy_image = poisson_noise + dark_noise

    return noisy_image


def simulate_line_source_3d(matrix_shape, offset_xy=(5.3, 13.2), rotation=(5., 10.),
                            background_signal=5.0, amplitude=100., sigma=10.,
                            added_noise=True):
    """Simulate 3d line source or wire.

    Parameters
    ----------
    matrix_shape : (tuple of int)
        z, y, x size of resulting matrix
    offset_xy: (tuple of float)
        distance from center to simulate signal hitting center of pixels differently
    rotation_xy : (tuple of float)
        degree of rotation in x and y direction (if isotropic voxels).
        The default is (5., 10.)
    background_signal : float, optional
        Level of background signal. The default is 0.
    amplitude : float, optional
        Amplitude of un-blurred signal. The default is 1..
    sigma : float, optional
        Sigma of gaussian blur. The default is 1..
    added_noise : bool, optional
        The default is True.

    Returns
    -------
    matrix : np.3darray
    """
    # slice with gaussian function to replicate in z - unrotated
    dist_map = mmcalc.get_distance_map_point(
        matrix_shape[1:], center_dx=offset_xy[0], center_dy=offset_xy[1])
    dists_flat = dist_map.flatten()
    gauss_vals_flat = mmcalc.gauss_4param(
        dists_flat, background_signal, amplitude, 0, sigma)
    gauss_vals = gauss_vals_flat.reshape(matrix_shape[1:])

    if add_noise:
        matrix = np.zeros(matrix_shape)
        rand_seeds = [
            random.randint(0, matrix_shape[0]) for i in range(matrix_shape[0])]
        for i in range(matrix_shape[0]):
            matrix[i] = add_noise(gauss_vals, random_seed=rand_seeds[i])
    else:
        matrix = np.broadcast_to(gauss_vals, (matrix_shape))
    # TODO
    matrix = np.zeros(matrix_shape)

    return matrix

def test_MTF_3d_line():
    matrix = simulate_line_source_3d((10, 100, 100))

    assert 1 == 1
