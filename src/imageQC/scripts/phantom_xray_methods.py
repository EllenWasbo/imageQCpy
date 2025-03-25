# -*- coding: utf-8 -*-
"""
Collection of methods for phantom-specific calculations xray.

@author: ellen
"""
import numpy as np
from scipy.signal import fftconvolve
from skimage.transform import hough_circle, hough_circle_peaks, resize
from skimage import feature


# imageQC block start
from imageQC.scripts.calculate_roi import get_roi_circle
import imageQC.scripts.mini_methods_calculate as mmcalc
# imageQC block end


def calculate_phantom_xray(image, image_info, paramset):
    if paramset.pha_alt == 0:
        res = calculate_tor(image, image_info, paramset)
    return res


def calculate_tor(image, image_info, paramset):
    details_dict = {}
    errmsgs = None
    values = []

    # thresholding variance image to detect phantom using hough transform
    pix = image_info.pix[0]
    roi_sz = 5. // pix
    roi_sz = round(np.max([3, roi_sz]))
    kernel = np.full((roi_sz, roi_sz), 1./(roi_sz ** 2))
    mu = fftconvolve(image, kernel, mode='valid')
    ii = fftconvolve(image ** 2, kernel, mode='valid')
    variance_image = ii - mu**2
    # resize to speed up hough transform
    dscale = 5
    var = resize(
        variance_image, (image.shape[0]//dscale, image.shape[1] //dscale))
    binary = np.zeros(var.shape, dtype=bool)
    threshold = 0.1 * np.max(var)
    binary[var > threshold] = True
    radius_phantom = 77 / (pix * dscale)  # radius of phantom circle approx 77nn
    # also long axis of rectangle with line-pairs same
    hough_radii = np.linspace(radius_phantom*0.9, radius_phantom*1.1, num=9)
    hough_res = hough_circle(binary, hough_radii)
    accums, cx, cy, radi = hough_circle_peaks(
        hough_res, hough_radii, total_num_peaks=3)
    
    if accums > 0.7:  # assume circle found
        dx_dy = (
            dscale * (cx - var.shape[1] / 2),
            dscale * (cy - var.shape[0] / 2))
        roi_circle = get_roi_circle(image.shape, dx_dy, radi * dscale)
    else:
        cx = var.shape[1] / 2
        cy = var.shape[0] / 2

    sub_corners = binary[
        round(cy - radius_phantom / 2):round(cy + radius_phantom / 2),
        round(cx - radius_phantom / 2):round(cx + radius_phantom / 2)]
    

    # roi_array[0] = framed MTF central part, rotation
    # roi_array[1] = list of center MTF parts
    # roi_array[2] = list of center low contrast parts
    return (details_dict, values, errmsgs)


def find_rectangle_object(image_binary, min_dist=0):
    """Detect rectangle in image.

    Parameters
    ----------
    image_binary : np.array
    min_dist: int
        minimum distance between corners

    Returns
    -------
    dict:
        centers_of_edges_xy : list of list
            for each edge (top, right, btm, left) [x, y]
            longest/most central edge if not full rect in image
        corners_xy : list of list
            [toplft, toprgt, btmrgt, btmlft] [x, y]
            None if not 4 corners
    """
    centers_of_edges_xy = None
    corners_xy = None

    corn = feature.corner_peaks(
        feature.corner_fast(image_binary, 10),
        min_distance=min_dist
        )

    '''
    if corn.shape[1] != 2 or corn.shape[0] != 4:  # try negative high signal
        if corn.shape[1] != 2:
            image_binary = np.logical_not(image_binary).astype(int)
        else:  # corn.shape[0] != 4
            image_binary = np.zeros(image.shape)
            image_binary[image < threshold] = 1.
            image_binary[inside == False] = 1.
        corn = feature.corner_peaks(
            feature.corner_fast(image_binary, 10),
            min_distance=10
            )

    if corn.shape[1] == 2:
        ys = np.array([c[0] for c in corn])
        xs = np.array([c[1] for c in corn])
        if corn.shape[0] == 4:
            # sort corners in toplft, toprgt, btmrgt, btmlft
            # first sort top to btm
            ys_sortidx = np.argsort(ys)
            ys_sort = ys[ys_sortidx]
            xs_sort = xs[ys_sortidx]
            # top lft to rgt
            if xs_sort[0] > xs_sort[1]:
                xs_sort = np.append(np.flip(xs_sort[:2]), xs_sort[2:])
                ys_sort = np.append(np.flip(ys_sort[:2]), ys_sort[2:])
            # btm rgt to lft
            if xs_sort[3] > xs_sort[2]:
                xs_sort = np.append(xs_sort[:2], np.flip(xs_sort[2:]))
                ys_sort = np.append(ys_sort[:2], np.flip(ys_sort[2:]))

            xs_diff = np.diff(np.append(xs_sort, xs_sort[0]))
            ys_diff = np.diff(np.append(ys_sort, ys_sort[0]))

            centers_of_edges_xy = [
                [xs_sort[i] + xs_diff[i] // 2, ys_sort[i] + ys_diff[i] // 2]
                for i in range(4)
                ]

            corners_xy = [[xs_sort[i], ys_sort[i]] for i in range(xs_sort.size)]
        else:
            # try fwhm method - assume square (for now)
            try:
                center_x, center_y, width_x, width_y = mmcalc.optimize_center(
                    image, mask_outer=0.05*image.shape[0])
                y0 = round(center_y)
                y1 = round(center_y + 0.1*width_y)
                center_y0_y1 = []
                for yval in [y0, y1]:
                    profile = image[yval]
                    _, center = mmcalc.get_width_center_at_threshold(
                        profile, 0.5*(np.max(profile) + np.min(profile)))
                    center_y0_y1.append(center)
                if None not in center_y0_y1:
                    tan_angle = (center_y0_y1[1] - center_y0_y1[0])/(y1-y0)
                    angle = np.arctan(tan_angle)
                    w_square = width_x * np.cos(angle)
                    centers_of_edges_xy = []
                    rot_angles = np.pi/2 * np.arange(4) - np.pi - angle
                    for i in rot_angles:
                        x, y = mmcalc.rotate_point(
                            [center_x + w_square//2, center_y],
                            [center_x, center_y], np.rad2deg(i))
                        centers_of_edges_xy.append([x, y])
            except TypeError:
                pass

    # TODO find edges also if full rectangle not imag-ed
    '''

    return {'centers_of_edges_xy': centers_of_edges_xy,
            'corners_xy': corners_xy}