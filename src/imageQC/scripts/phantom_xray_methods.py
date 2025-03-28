# -*- coding: utf-8 -*-
"""
Collection of methods for phantom-specific calculations xray.

@author: ellen
"""
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from skimage.transform import hough_circle, hough_circle_peaks, resize
from skimage import feature, filters


# imageQC block start
from imageQC.scripts.calculate_roi import (
    get_roi_circle, get_roi_rectangle, find_edges, find_intercepts)
import imageQC.scripts.mini_methods_calculate as mmcalc
# imageQC block end


def calculate_phantom_xray(image, image_info, roi_array, paramset):
    if paramset.pha_alt == 0:
        res = calculate_tor(image, image_info, roi_array, paramset)
    return res


def calculate_tor(image, image_info, roi_array, paramset):
    details_dict = {}
    errmsgs = None
    values = []
    pix = image_info.pix[0]

    image_center = image*roi_array
    rows = np.max(roi_array, axis=1)
    cols = np.max(roi_array, axis=0)
    image_center = image[rows][:, cols]
    image_filt = ndimage.gaussian_filter(image_center, sigma=2./pix)
    res = find_center_object(image, mask_outer=0, tolerances_width=[None, None], sigma=0)
    cx, cy, wx, wy = res
    breakpoint()

    # finding edges in image to detect phantom using hough transform
    '''
    edge_image = filters.sobel(image)
    binary = np.zeros(image.shape, dtype=bool)
    binary[edge_image > 0.1*np.max(edge_image)] = True

    bin_reduced = resize(
        binary, (image.shape[0]//dscale, image.shape[1] //dscale))
    # resize to speed up hough transform
    radius_phantom = 77 / (pix * dscale)  # radius of phantom circle approx 77mm

    hough_radii = np.linspace(radius_phantom*0.9, radius_phantom*1.1, num=9)
    hough_res = hough_circle(bin_reduced, hough_radii)
    accums, cx, cy, radi = hough_circle_peaks(
        hough_res, hough_radii, total_num_peaks=1)
    cx = cx[0]
    cy = cy[0]

    if accums[0] > 0.7:  # assume circle found
        dx_dy = (
            dscale * (cx - bin_reduced.shape[1] / 2),
            dscale * (cy - bin_reduced.shape[0] / 2))
        roi_circle = get_roi_circle(image.shape, dx_dy, radi * dscale)
    else:
        dx_dy = (0, 0)

    dd = round(80 / pix / 2)  # 8x8cm distance to offset
    cx = round(dx_dy[0]) - image.shape[1] // 2
    cy = round(dx_dy[1]) - image.shape[0] // 2

    min_dist = round(10. / pix)
    lines = find_edges(binary_center, 2, [45-30, 45+30], 0.5, min_dist)
    lines2 = find_edges(binary_center, 2, [-45-30, -45+30], 0.5, min_dist)
    corners_xy = find_intercepts(lines, lines2)
    breakpoint()
    # remove corners outside bin_sub
    corners_xy = [x for x in corners_xy if np.all(
        [np.min(x) > 0,  x[0] < bin_sub.shape[1], x[1] < bin_sub.shape[0]])]
    if len(corners_xy) == 4:
        xs, ys = zip(*corners_xy)
        cx_sub = np.mean(xs)
        cy_sub = np.mean(ys)

        corners_xy.sort(key=lambda x: x[1])
        xs, ys = zip(*corners_xy)
        top = [xs[0], ys[0]]
        btm = [xs[-1], ys[-1]]
        corners_xy.sort(key=lambda x: x[0])
        xs, ys = zip(*corners_xy)
        lft = [xs[0], ys[0]]
        rgt = [xs[-1], ys[-1]]

        points = [top, rgt, btm, lft, top]
        angles = [[], []]
        widths = [[], []]
        angles_widths_1 = []
        for i in range(len(points) - 1):
            angle = np.rad2deg(np.arctan(
                (points[i+1][1] - points[i][1]) /
                (points[i+1][0] - points[i][0])))
            width = np.sqrt(
                (points[i+1][1] - points[i][1])**2 +
                (points[i+1][0] - points[i][0])**2)
            angles[i % 2].append(angle)
            widths[i % 2].append(width)

        off_center_xy = (cx_sub - bin_sub.shape[1] // 2,
                         cy_sub - bin_sub.shape[0] // 2)
        roi_rect = get_roi_rectangle(
            bin_sub.shape,
            roi_width=np.mean(widths[0]), roi_height=np.mean(widths[1]),
            offcenter_xy=off_center_xy)
        roi = mmcalc.rotate2d_offcenter(
            roi_rect.astype(float), -np.mean(angles[0]), off_center_xy)
        roi = np.round(roi)
        roi_center = np.array(roi, dtype=bool)

        image_sub = image[cy - dd:cy + dd, cx - dd:cx + dd] * roi_center
        image_sub_rot = mmcalc.rotate2d_offcenter(
            image_sub, np.mean(angles[0]), off_center_xy)
        rows = np.max(roi_rect, axis=1)
        cols = np.max(roi_rect, axis=0)
        image_center = image_sub_rot[rows][:, cols]
        if image_center.shape[1] < image_center.shape[0]:
            image_center = np.rot90(image_center, k=1)
        dd = int(0.2 * image_center.shape[1])
        margin = dd // 4
        profile1 = np.mean(image_center[:, dd-margin:dd+margin], axis=1)
        profile2 = np.mean(image_center[:, -dd-margin:-dd+margin], axis=1)
        range1 = np.max(profile1) - np.min(profile1)
        range2 = np.max(profile2) - np.min(profile2)
        sz = profile1.size
        if range1 > range2:
            image_center = np.fliplr(image_center)
            prof = profile1
        else:
            prof = profile2
        rangeU = np.max(prof[:sz // 2]) - np.min(prof[:sz // 2])
        rangeD = np.max(prof[sz // 2:]) - np.min(prof[sz // 2:])
        if rangeU > rangeD:
            image_center = np.flipud(image_center)
        profile1 = np.mean(image_center[:, dd-margin:dd+margin], axis=1)
        profile2 = np.mean(image_center[:, dd*2-margin:dd*2+margin], axis=1)
        profile3 = np.mean(image_center[:, dd*3-margin:dd*3+margin], axis=1)
        profile_ref = np.mean(image_center[:, dd*4:dd*4+margin*2], axis=1)
        range_ref = (np.mean(profile_ref[dd-margin:dd+margin]) -
                     np.mean(profile_ref[dd*2-margin:dd*2+margin]))
        
        breakpoint()
    #from matplotlib.lines import AxLine
    #plt.gca().add_artist(AxLine(lines[0][0], None, lines[0][1], color='r'))

    # roi_array[0] = framed MTF central part, rotation
    # roi_array[1] = list of center MTF parts
    # roi_array[2] = list of center low contrast parts
    '''
    return (details_dict, values, errmsgs)
