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
    noisy_image = input_signal + poisson_noise + dark_noise

    return noisy_image


def simulate_disc(matrix_shape, radius, sigma=5,
                  center_xy=(0., 0.), inner_signal=1., outer_signal=0.,
                  added_noise=False):
    """Generate image of disc."""
    image = np.full(matrix_shape, outer_signal)
    dist_map = mmcalc.get_distance_map_point(
        matrix_shape, center_dx=center_xy[0], center_dy=center_xy[1])
    image[dist_map < radius] = inner_signal
    image = ndimage.gaussian_filter(image, sigma=sigma)
    if added_noise:
        image = add_noise(image, dark_noise=0.05*np.max(image))

    return image


def test_center_disc():
    image_size = (100, 100)
    matrix_size = (75, 75)
    radius = 23.3
    image = simulate_disc(image_size, radius, sigma=3., center_xy=(0.5, 0.3),
                          inner_signal=100., outer_signal=10, added_noise=False)
    centers = [(0, 0), (-10.8, 0), (0, -10.5), (10.2, 10.3), (-10.4, -10.7)]
    found_centers = []
    dists = []
    vals = []
    for center in centers:
        dy = image_size[1] // 2 - round(center[1]) - matrix_size[1] // 2
        dx = image_size[0] // 2 -round(center[0]) - matrix_size[0] // 2
        matrix = image[dy:dy+matrix_size[1], dx:dx+matrix_size[0]]
        center_xy = mmcalc.center_xy_of_disc(matrix)
        found_centers.append([
            center_xy[0] - 0.5 * matrix_size[1],
            center_xy[1] - 0.5 * matrix_size[0]
            ])
        dist_map = mmcalc.get_distance_map_point(
            matrix_size,
            center_dx=center_xy[0] - 0.5 * matrix_size[1],
            center_dy=center_xy[1] - 0.5 * matrix_size[0])
        #masked_img = np.copy(matrix)
        #masked_img[dist_map > 0.3 * matrix.shape[0]] = 0
        dists_flat = dist_map.flatten()
        sort_idxs = np.argsort(dists_flat)
        dists.append(dists_flat[sort_idxs])
        vals.append(matrix.flatten()[sort_idxs])

    breakpoint()
    #plt.plot(dists[0], vals[0], '.', markersize=2, color='red')
    assert 1==1


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
