#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection of small functions used in ImageQC.

@author: Ellen Wasbo
"""
import numpy as np
from scipy.optimize import curve_fit


def gauss_4param(x, H, A, x0, sigma):
    """Calculate gaussian curve from x values and parameters."""
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gauss_4param_fit(x, y, cut_width_fwhm=0):
    """Fit x,y to gaussian curve."""
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss_4param, x, y, p0=[min(y), max(y), mean, sigma])
    return popt


def gauss(x, A, sigma):
    """Calculate gaussian curve from x values and parameters."""
    return A * np.exp(-(x ** 2) / (2 * sigma ** 2))


def gauss_fit(x, y, cut_width_fwhm=0):
    """Fit x,y to gaussian curve."""
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[max(y), sigma])
    return popt


def gauss_double(x, A1, sigma1, A2, sigma2):
    """Calculate sum of two centered gaussian curves."""
    return (A1 * np.exp(-(x) ** 2 / (2 * sigma1 ** 2)) +
            A2 * np.exp(-(x) ** 2 / (2 * sigma2 ** 2)))


def gauss_double_fit(x, y, cut_width_fwhm=0, fwhm1=0):
    """Fit x,y to double gaussian - (centered, 1 positive and 1 negative amplitude)."""
    A1 = np.max(y) - np.min(y)
    if fwhm1 == 0:
        fwhm1, center = get_width_center_at_threshold(y, np.max(y)/2)
    sigma1 = fwhm1 / (2 * np.sqrt(2 * np.log(2)))
    A2 = 1.5 * np.min(y)
    fwhm2, center = get_width_center_at_threshold(y, A2/2)
    if A2 < 0 and fwhm2 > fwhm1:
        sigma2 = fwhm2 / (2 * np.sqrt(2 * np.log(2)))
        popt, pcov = curve_fit(gauss_double, x, y, p0=[A1, sigma1, A2, sigma2])
    else:
        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
        popt, pcov = curve_fit(gauss, x, y, p0=[0, max(y), mean, sigma])
    return popt


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


def get_width_center_at_threshold(profile, threshold, get_widest=False):
    """Get width and center of largest group in profile above threshold.

    Parameters
    ----------
    profile : np.array or list
        vector
    threshold : float
    get_widest : bool
        True if ignore inner differences, just match first/last. Default is False

    Returns
    -------
    width : float
        width of profile center values above or below threshold
        if failed -1
    center : float
        center position of center values above or below threshold
        if failed -1
    """
    width = -1
    center = -1

    if isinstance(profile, list):
        profile = np.array(profile)

    above = np.where(profile > threshold)
    below = np.where(profile < threshold)

    if len(above[0]) > 2 and len(below[0]) > 2:
        if above[0][0] >= 1:
            center_indexes = above[0]
        else:
            center_indexes = below[0]

        if center_indexes[0] >= 1 and len(center_indexes) > 1:
            if get_widest:
                first = center_indexes[0]
                last = center_indexes[1]
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

            if first == 0:
                first = 1
            if last == len(profile) - 1:
                last = len(profile) - 2

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
