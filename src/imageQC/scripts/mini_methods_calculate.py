#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection of small functions used in ImageQC.

@author: Ellen Wasbo
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter


def gauss_4param(x, H, A, x0, sigma):
    """Calculate gaussian curve from x values and parameters."""
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gauss_4param_fit(x, y):
    """Fit x,y to gaussian curve."""
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss_4param, x, y, p0=[min(y), max(y), mean, sigma])
    return popt


def gauss(x, A, sigma):
    """Calculate gaussian curve from x values and parameters."""
    return A * np.exp(-(x ** 2) / (2 * sigma ** 2))


def gauss_fit(x, y, fwhm=0):
    """Fit x,y to gaussian curve."""
    if fwhm == 0:
        width, center = get_width_center_at_threshold(y, np.max(y)/2)
        if width is not None:
            fwhm = width * (x[1] - x[0])
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    A = max(y)
    popt, pcov = curve_fit(
        gauss, x, y, p0=[A, sigma],
        bounds=([0.5*A, 0.5*sigma], [2*A, 2*sigma])
        )
    return popt


def gauss_double(x, A1, sigma1, A2, sigma2):
    """Calculate sum of two centered gaussian curves."""
    return (A1 * np.exp(-(x) ** 2 / (2 * sigma1 ** 2)) +
            A2 * np.exp(-(x) ** 2 / (2 * sigma2 ** 2)))


def gauss_double_fit(x, y, fwhm1=0):
    """Fit x,y to double gaussian - (centered, 1 positive and 1 negative amplitude)."""
    A1 = np.max(y) - 2 * np.min(y)
    if fwhm1 == 0:
        width, center = get_width_center_at_threshold(y, np.max(y)/2)
        if width is not None:
            fwhm1 = width * (x[1] - x[0])
    sigma1 = fwhm1 / (2 * np.sqrt(2 * np.log(2)))
    A2 = 2 * np.min(y)
    sigma2 = 2 * sigma1
    if A2 < 0:
        popt, pcov = curve_fit(
            gauss_double, x, y, p0=[A1, sigma1, A2, sigma2],
            bounds=([0.5*A1, 0.5*sigma1, 2*A2, 0.5*sigma2],
                    [2*A1, 2*sigma1, 0, 2*sigma2]),
            )
    else:
        A1 = max(y)
        popt, pcov = curve_fit(
            gauss, x, y, p0=[A1, sigma1],
            bounds=([0.5*A1, 0.5*sigma1], [2*A1, 2*sigma1]),
            )
    return popt


def get_MTF_gauss(LSF, dx=1., prefilter_sigma=None, gaussfit='single'):
    """Fit LSF to gaussian and calculate gaussian MTF.

    Parameters
    ----------
    LSF : np.array
        LSF 1d
    dx : float, optional
        step_size, default is 1.
    prefilter_sigma : float, optional
        prefiltered with gaussian filter, sigma=prefilter_sigma. The default is None.
    gaussfit : str, optional
        single or double (sum of two gaussian). The default is 'single'.

    Returns
    -------
    dict
        with calculation details
    """
    popt = None
    width, center = get_width_center_at_threshold(LSF, 0.5 * max(LSF))
    if center is not None:
        LSF_x = dx * (np.arange(LSF.size) - center)
        A2 = None
        sigma2 = None
        if gaussfit == 'double':
            popt = gauss_double_fit(LSF_x, LSF, fwhm1=width * dx)
            if len(popt) == 4:
                A1, sigma1, A2, sigma2 = popt
                LSF_fit = gauss_double(LSF_x, *popt)
            else:
                A1, sigma1 = popt
                LSF_fit = gauss(LSF_x, *popt)
        else:  # single
            A1, sigma1 = gauss_fit(LSF_x, LSF)
            LSF_fit = gauss(LSF_x, A1, sigma1)

        n_steps = 200  # sample 20 steps from 0 to 1 stdv MTF curve (stdev = 1/sigma1)
        # TODO user configurable n_steps
        k_vals = np.arange(n_steps) * (10./n_steps) / sigma1
        MTF = gauss(k_vals, A1*sigma1, 1/sigma1)
        if A2 is not None and sigma2 is not None:
            F2 = gauss(k_vals, A2*sigma2, 1/sigma2)
            MTF = np.add(MTF, F2)
        MTF_filtered = None
        if prefilter_sigma is not None:
            if prefilter_sigma > 0:
                F_filter = gauss(k_vals, 1., 1/prefilter_sigma)
                MTF_filtered = 1/MTF[0] * MTF  # for display
                MTF = np.divide(MTF, F_filter)
        MTF = 1/MTF[0] * MTF
        k_vals = k_vals / (2*np.pi)
        details = {
            'LSF_fit_x': LSF_x, 'LSF_fit': LSF_fit, 'LSF_prefit': LSF,
            'LSF_fit_params': popt,
            'MTF_freq': k_vals, 'MTF': MTF, 'MTF_filtered': MTF_filtered
            }
    else:
        details = {}

    return details


def get_MTF_discrete(LSF, dx=1, padding_factor=1):
    """Fourier transform of vector with zero-padding.

    Parameters
    ----------
    LSF : np.array of floats
        line spread function
    dx : float
        stepsize in LSF
    padding_factor: float, optional
        zero-pad each side by padding-factor * length of LSF. Default is 1.

    Returns
    -------
    freq : np.array of float
        frequency axis of MTF
    MTF : np.array of float
        modulation transfer function
    """
    nLSF = round(padding_factor * LSF.size)
    LSF = np.pad(LSF, pad_width=nLSF, mode='constant', constant_values=0)
    MTF = np.abs(np.fft.fft(LSF))
    freq = np.fft.fftfreq(LSF.size, d=dx)
    MTF = MTF[:MTF.size//2]
    freq = freq[:freq.size//2]
    MTF = MTF/MTF[0]

    return {'MTF_freq': freq, 'MTF': MTF}


def point_source_func(x, C, D):
    """Calculate foton fluence from point source measured by detector.

    Based on equation A6 in
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3966082/#x0
    """
    return C * D / ((x**2 + D**2)**(1.5))


def point_source_func_fit(x, y, center_value=0., avg_radius=0.):
    """Fit foton fluence from point source at detector."""
    C = center_value*avg_radius**2
    popt, pcov = curve_fit(point_source_func, x, y,
                           p0=[C, avg_radius])
    return popt


def center_xy_of_disc(matrix2d, threshold=None, roi=None, mode='mean'):
    """Find center of disc (or point) object in image.

    Robust against noise and returning sub pixel precision center.

    Parameters
    ----------
    matrix2d : np.array of float
        2 dimensions
    threshold : float, optional
        value to differ inside/outside disc. The default is None.
        if None use halfmax (value between min, max)
    roi : np.array of bool, optional
        ignore pixels outside roi
    mode : str, optional
        Mean or max. Max better for point, mean better for disc. Default is 'mean'.

    Returns
    -------
    center_x : float
    center_y : float
    """
    center_xy = []
    smoothed_matrix = gaussian_filter(matrix2d, sigma=5)
    if roi is not None:
        smoothed_matrix = np.ma.masked_array(smoothed_matrix, mask=np.invert(roi))
    # smoothed mean or max profile (within ROI) to estimate center
    for ax in [0, 1]:
        if mode == 'mean':
            prof1 = np.mean(smoothed_matrix, axis=ax)
        else:  # assume max
            prof1 = np.max(smoothed_matrix, axis=ax)
        if threshold is None:
            threshold = min(prof1) + 0.5 * (max(prof1) - min(prof1))
        width, center = get_width_center_at_threshold(
            prof1, threshold, get_widest=True)
        center_xy.append(center)

    return center_xy


def get_width_center_at_threshold(
        profile, threshold, get_widest=False, force_above=False):
    """Get width and center of largest group in profile above threshold.

    Parameters
    ----------
    profile : np.array or list
        vector
    threshold : float
    get_widest : bool, optional
        True if ignore inner differences, just match first/last. Default is False
    force_above : bool, optional
        True if center_indexes locked to above threshold. Default is False.

    Returns
    -------
    width : float
        width of profile center values above or below threshold
        if failed -1
    center : float
        center position of center values above or below threshold
        if failed -1
    """
    width = None
    center = None

    if isinstance(profile, list):
        profile = np.array(profile)

    above = np.where(profile > threshold)
    below = np.where(profile < threshold)

    if len(above[0]) > 1 and len(below[0]) > 1:
        if above[0][0] >= 1 or force_above:
            center_indexes = above[0]
        else:
            center_indexes = below[0]

        if center_indexes[0] >= 1 and len(center_indexes) > 1:
            if get_widest:
                first = center_indexes[0]
                last = center_indexes[-1]
            else:
                # find largest group, first/last index
                grouped_indexes = [0]
                for i in range(1, len(center_indexes)):
                    if center_indexes[i] == center_indexes[i-1] + 1:
                        grouped_indexes.append(grouped_indexes[-1])
                    else:
                        grouped_indexes.append(grouped_indexes[-1]+1)
                group_start = []
                group_size = []
                for i in range(grouped_indexes[-1]+1):
                    group_start.append(center_indexes[
                        grouped_indexes.index(i)])
                    group_size.append(grouped_indexes.count(i))
                largest_group = group_size.index(max(group_size))
                first = group_start[largest_group]
                last = first + group_size[largest_group] - 1

            if first == 0 or last == len(profile) - 1:
                width = None
                center = None
            else:
                # interpolate to find more exact width and center
                dy = profile[first] - threshold
                if profile[first] != profile[first-1]:
                    dx = dy / (profile[first] - profile[first-1])
                else:
                    dx = 0
                x1 = first - dx
                dy = profile[last] - threshold
                if profile[last] != profile[last+1]:
                    dx = dy / (profile[last] - profile[last+1])
                else:
                    dx = 0
                x2 = last + dx
                center = 0.5 * (x1+x2)
                width = x2 - x1

    return (width, center)


def optimize_center(image, mask_outer=0, max_from_part=4):
    """Find center and width of object in image.

    Parameters
    ----------
    image : np.2darray
    mask_outer : int, optional (float ok)
        Number of outer pixels to mask when searching. The default is 0.
    max_from_part : float, optional
        1/part of central image to find max from average

    Returns
    -------
    res : tuple of float or None
        center_x, center_y, widht_x, width_y
    """
    # get maximum profiles x and y
    mask_outer = round(mask_outer)
    if mask_outer == 0:
        prof_y = np.max(image, axis=1)
        prof_x = np.max(image, axis=0)
    else:
        prof_y = np.max(image[mask_outer:-mask_outer, mask_outer:-mask_outer], axis=1)
        prof_x = np.max(image[mask_outer:-mask_outer, mask_outer:-mask_outer], axis=0)
    # get width at halfmax and center for profiles
    max_from_part = round(max_from_part)
    if max_from_part == 0:
        max_from_part = 1
    width_x, center_x = get_width_center_at_threshold(
        prof_x, 0.5 * (
            np.mean(prof_x[prof_x.size//max_from_part:-prof_x.size//max_from_part])
            + min(prof_x)
            ),
        force_above=True)
    width_y, center_y = get_width_center_at_threshold(
        prof_y, 0.5 * (
            np.mean(prof_y[prof_y.size//max_from_part:-prof_y.size//max_from_part])
            + min(prof_y)
            ),
        force_above=True)
    if width_x is not None and width_y is not None:
        center_x += mask_outer
        center_y += mask_outer
        res = (center_x, center_y, width_x, width_y)
    else:
        res = None

    return res


def ESF_to_LSF(ESF, prefilter_sigma):
    """Calculate LSF spread function from edge spread function.

    Parameters
    ----------
    ESF : np.1darray
    prefilter_sigma : float
        gaussian filter to be corrected when gaussian MTF is used

    Returns
    -------
    LSF : np.1darray
        line spread function after gaussian filter - for gaussian fit
    LSF_not_filtered : np.1darray
        line spread function without gaussian prefilter - for discrete MTF
    ESF_filtered : np.1darray
        ESF gaussian smoothed
    """
    values_f = gaussian_filter(ESF, sigma=prefilter_sigma)
    LSF = np.diff(values_f)
    LSF_not_filtered = np.diff(ESF)
    if np.abs(np.min(LSF)) > np.abs(np.max(LSF)):  # ensure positive gauss
        LSF = -1. * LSF
        LSF_not_filtered = -1. * LSF_not_filtered

    return (LSF, LSF_not_filtered, values_f)

def cut_and_fade_LSF(profile, center=0, fwhm=0, cut_width=0, fade_width=0):
    """Cut and fade profile fwhm from center to reduce noise in LSF.

    Parameters
    ----------
    profile : list of float
    center : float
        center pos in pix
    fwhm : float
        fwhm in pix
    cut_width : float
        cut profile cut_width*fwhm from halfmax. If zero, also fade width ignored.
    fade_width : float (or None)
        fade profile to background fade*fwhm outside cut_width

    Returns
    -------
    modified_profile : list of float
    cut_dist_pix : float
        distance from center where profile is cut (unit = pix)
    fade_dist_pix : float
        distance from center where profile stops fading (unit = pix)
    """
    modified_profile = profile
    n_fade = 0
    if cut_width > 0:
        cut_width += 0.5  # from halfmax = 0.5 fwhm
        dx = cut_width*fwhm
        first_x = max(round(center - dx), 0)
        last_x = min(round(center + dx), len(profile) - 1)
        if fade_width is None:
            fade_width = 0
        if fade_width > 0:
            n_fade = round(fade_width*fwhm)
            first_fade_x = max(first_x - n_fade, 0)
            last_fade_x = min(last_x + n_fade, len(profile) - 1)
            nn = first_x - first_fade_x
            gradient = np.multiply(
                np.arange(nn)/nn, profile[first_fade_x:first_x])
            modified_profile[first_fade_x:first_x] = gradient
            nn = last_fade_x - last_x
            gradient = np.multiply(
                np.flip(np.arange(nn)/nn), profile[last_x:last_fade_x])
            modified_profile[last_x:last_fade_x] = gradient
            first_x = first_fade_x
            last_x = last_fade_x
        if first_x > 0:
            modified_profile[0:first_x - 1] = 0
        if last_x < len(profile) - 2:
            modified_profile[last_x + 1:] = 0
    return (modified_profile, dx, dx + n_fade)


def rotate_point(xy, rotation_axis_xy, angle):
    """Rotate xy around origin_xy by an angle.

    Parameters
    ----------
    xy : tuple of floats
        coordinates to rotate
    origin_xy: tuple of floats
        coordinate of origin
    angle : float
        rotation angle in degrees

    Returns
    -------
    new_x : float
        rotated x
    new_y : float
        rotated y
    """
    xr, yr = rotation_axis_xy
    x, y = xy
    rad_angle = np.deg2rad(angle)
    new_x = xr + np.cos(rad_angle) * (x - xr) - np.sin(rad_angle) * (y - yr)
    new_y = yr + np.sin(rad_angle) * (x - xr) + np.cos(rad_angle) * (y - yr)
    return (new_x, new_y)


def get_curve_values(x, y, y_values, force_first_below=False):
    """Lookup interpolated x values given y values.

    Parameters
    ----------
    x : array like of float
    y : array like of float
    y_values : array like of float
        input y values to calculate corresponding x values. The default is None.
    force_first_below : bool, optional
        return first x-value where y below given y_value. Default is False.

    Returns
    -------
    result_values : list of float
        corresponding values of input values
    """
    def get_interpolated_x(y_value, x1, x2, y1, y2):
        w = (y_value - y2) / (y1 - y2)
        return w * x1 + (1 - w) * x2

    result_values = []
    for yval in y_values:
        idx_above = np.where(y >= yval)
        idx_below = np.where(y < yval)
        try:
            idxs = [
                idx_below[0][0], idx_below[0][-1], idx_above[0][0], idx_above[0][-1]]
            idxs.sort()
            first = idxs[1]
            last = idxs[2]
            if first + 1 == last:
                result_values.append(
                    get_interpolated_x(yval, x[first], x[last], y[first], y[last]))
            else:
                if force_first_below:
                    result_values.append(x[idx_below[0][0]])
                else:
                    result_values.append(None)
        except IndexError:
            result_values.append(None)

    return result_values


def resample(input_y, input_x, step, first_step=0, n_steps=None):
    """Resample input_y with regular step.

    Parameters
    ----------
    input_y : array like of float
    input_x : array like of float
        assumed to be sorted
    step : float
        resulting step size of x
    first_step : float
        first value of x to output
    n_steps : int, optional
        number of values in new_x / new_y

    Returns
    -------
    new_x : np.array of float
    new_y : np.array of float
    """
    input_y = np.array(input_y)
    input_x = np.array(input_x)
    if n_steps is None:
        n_steps = (np.max(input_x) - first_step) // step
    new_x = step * np.arange(n_steps) + first_step
    new_y = np.interp(new_x, input_x, input_y)

    return (new_x, new_y)

def get_min_max_pos_2d(image, roi_array):
    """Get position of first min and max value in image.

    Parameters
    ----------
    image : np.array
    roi_array : np.array
        dtype bool

    Returns
    -------
    list
        min_idx, max_idx
    """
    arr = np.ma.masked_array(image, mask=np.invert(roi_array))
    min_idx = np.where(arr == np.min(arr))
    max_idx = np.where(arr == np.max(arr))

    return [
        [min_idx[0][0], min_idx[1][0]],
        [max_idx[0][0], max_idx[1][0]]
        ]
